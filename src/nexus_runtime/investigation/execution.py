"""Submit investigation plans to the existing distributed runtime in DAG-ready waves."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Protocol

from nexus_runtime.distributed.model import (
    TERMINAL_TASK_STATES,
    DistributedTask,
    DistributedTaskState,
    TaskPriority,
)
from nexus_runtime.models import DomainError, utcnow

from .objective import _required_string, _timestamp_from_text, _timestamp_to_text
from .planner import InvestigationPlan


class DistributedRuntimePort(Protocol):
    """Public subset of ``RuntimeApplication`` used by the investigation layer."""

    def submit_task(
        self,
        run_id: str,
        *,
        correlation_id: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        required_capabilities: frozenset[str] = frozenset(),
        metadata: dict[str, Any] | None = None,
    ) -> DistributedTask: ...

    def get_task(self, task_id: str) -> DistributedTask: ...

    def cancel_task(self, task_id: str, principal: str) -> DistributedTask: ...

    def list_tasks(self) -> list[DistributedTask]: ...


@dataclass(slots=True)
class PlanExecution:
    """Persistable bridge between a plan, AgentRuns, and distributed task ids."""

    plan_id: str
    session_id: str
    run_ids: dict[str, str]
    task_ids: dict[str, str] = field(default_factory=dict)
    blocked_investigations: dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)

    def __post_init__(self) -> None:
        if not self.plan_id.strip() or not self.session_id.strip():
            raise DomainError("execution plan_id and session_id are required")
        if any(not key.strip() or not value.strip() for key, value in self.run_ids.items()):
            raise DomainError("execution AgentRun mapping cannot contain empty identifiers")

    @property
    def all_scheduled_or_blocked(self) -> bool:
        return len(self.task_ids) + len(self.blocked_investigations) == len(self.run_ids)

    def to_dict(self) -> dict[str, object]:
        return {
            "plan_id": self.plan_id,
            "session_id": self.session_id,
            "run_ids": dict(self.run_ids),
            "task_ids": dict(self.task_ids),
            "blocked_investigations": dict(self.blocked_investigations),
            "created_at": _timestamp_to_text(self.created_at),
            "updated_at": _timestamp_to_text(self.updated_at),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> PlanExecution:
        run_ids = payload.get("run_ids")
        task_ids = payload.get("task_ids")
        blocked = payload.get("blocked_investigations")
        if (
            not isinstance(run_ids, dict)
            or not isinstance(task_ids, dict)
            or not isinstance(blocked, dict)
        ):
            raise DomainError("malformed PlanExecution")
        for name, value in (
            ("run_ids", run_ids),
            ("task_ids", task_ids),
            ("blocked_investigations", blocked),
        ):
            if any(
                not isinstance(key, str) or not isinstance(item, str) for key, item in value.items()
            ):
                raise DomainError(f"malformed PlanExecution {name}")
        try:
            return cls(
                plan_id=_required_string(payload["plan_id"], "plan_id"),
                session_id=_required_string(payload["session_id"], "session_id"),
                run_ids=dict(run_ids),
                task_ids=dict(task_ids),
                blocked_investigations=dict(blocked),
                created_at=_timestamp_from_text(payload["created_at"], "created_at"),
                updated_at=_timestamp_from_text(payload["updated_at"], "updated_at"),
            )
        except KeyError as exc:
            raise DomainError("malformed PlanExecution") from exc


@dataclass(frozen=True, slots=True)
class ExecutionStatus:
    execution: PlanExecution
    task_states: dict[str, DistributedTaskState]
    terminal: bool
    succeeded: tuple[str, ...]
    failed: tuple[str, ...]
    running: tuple[str, ...]


class PlanExecutionController:
    """Respect plan dependencies without duplicating runtime scheduling semantics."""

    def __init__(self, runtime: DistributedRuntimePort) -> None:
        self._runtime = runtime

    def start(
        self,
        plan: InvestigationPlan,
        run_ids: Mapping[str, str],
    ) -> ExecutionStatus:
        return self.advance(plan, self.prepare(plan, run_ids))

    def prepare(
        self,
        plan: InvestigationPlan,
        run_ids: Mapping[str, str],
    ) -> PlanExecution:
        expected = set(plan.dependencies)
        if set(run_ids) != expected:
            missing = sorted(expected - set(run_ids))
            extra = sorted(set(run_ids) - expected)
            raise DomainError(f"AgentRun mapping mismatch; missing={missing}, extra={extra}")
        return PlanExecution(
            plan_id=plan.plan_id,
            session_id=plan.session_id,
            run_ids=dict(run_ids),
            created_at=plan.created_at,
            updated_at=plan.created_at,
        )

    def advance(
        self,
        plan: InvestigationPlan,
        execution: PlanExecution,
    ) -> ExecutionStatus:
        self._validate(plan, execution)
        states = self._states(execution)
        made_progress = True
        while made_progress:
            made_progress = False
            for investigation_id in sorted(plan.dependencies):
                if (
                    investigation_id in execution.task_ids
                    or investigation_id in execution.blocked_investigations
                ):
                    continue
                parents = plan.dependencies[investigation_id]
                failed_parents = [
                    parent
                    for parent in parents
                    if parent in execution.blocked_investigations
                    or states.get(parent)
                    in {DistributedTaskState.CANCELLED, DistributedTaskState.DEAD_LETTERED}
                ]
                if failed_parents:
                    execution.blocked_investigations[investigation_id] = (
                        "dependency failed: " + ", ".join(sorted(failed_parents))
                    )
                    made_progress = True
                    continue
                if not all(
                    states.get(parent) == DistributedTaskState.SUCCEEDED for parent in parents
                ):
                    continue
                task = self._submit_or_recover(
                    plan, investigation_id, execution.run_ids[investigation_id]
                )
                execution.task_ids[investigation_id] = task.task_id
                states[investigation_id] = task.state
                made_progress = True
        execution.updated_at = utcnow()
        states = self._states(execution)
        succeeded = tuple(
            sorted(key for key, value in states.items() if value == DistributedTaskState.SUCCEEDED)
        )
        failed = tuple(
            sorted(
                set(execution.blocked_investigations)
                | {
                    key
                    for key, value in states.items()
                    if value in {DistributedTaskState.CANCELLED, DistributedTaskState.DEAD_LETTERED}
                }
            )
        )
        running = tuple(
            sorted(key for key, value in states.items() if value not in TERMINAL_TASK_STATES)
        )
        terminal = len(succeeded) + len(failed) == len(plan.investigations)
        return ExecutionStatus(execution, states, terminal, succeeded, failed, running)

    def cancel(self, execution: PlanExecution, principal: str) -> PlanExecution:
        for task_id in execution.task_ids.values():
            task = self._runtime.get_task(task_id)
            if task.state not in TERMINAL_TASK_STATES:
                self._runtime.cancel_task(task_id, principal)
        execution.updated_at = utcnow()
        return execution

    def tasks(self, execution: PlanExecution) -> tuple[DistributedTask, ...]:
        return tuple(self._runtime.get_task(task_id) for task_id in execution.task_ids.values())

    def _submit_or_recover(
        self,
        plan: InvestigationPlan,
        investigation_id: str,
        run_id: str,
    ) -> DistributedTask:
        existing = [
            task
            for task in self._runtime.list_tasks()
            if task.run_id == run_id
            and task.correlation_id == plan.session_id
            and task.metadata.get("plan_id") == plan.plan_id
            and task.metadata.get("investigation_id") == investigation_id
        ]
        if len(existing) > 1:
            raise DomainError(f"duplicate distributed tasks for investigation: {investigation_id}")
        if existing:
            return existing[0]
        investigation = next(
            item for item in plan.investigations if item.investigation_id == investigation_id
        )
        metadata: dict[str, Any] = {
            "plan_id": plan.plan_id,
            "session_id": plan.session_id,
            "investigation_id": investigation_id,
            "gap_id": investigation.gap_id,
            "question": investigation.question,
            "hypothesis": investigation.hypothesis,
            "required_evidence": list(investigation.required_evidence),
            "constraints": list(investigation.constraints),
            "dependency_investigation_ids": list(plan.dependencies[investigation_id]),
        }
        return self._runtime.submit_task(
            run_id,
            correlation_id=plan.session_id,
            priority=self._priority(investigation.priority),
            required_capabilities=frozenset(investigation.capabilities),
            metadata=metadata,
        )

    def _states(self, execution: PlanExecution) -> dict[str, DistributedTaskState]:
        return {
            investigation_id: self._runtime.get_task(task_id).state
            for investigation_id, task_id in execution.task_ids.items()
        }

    @staticmethod
    def _validate(plan: InvestigationPlan, execution: PlanExecution) -> None:
        if execution.plan_id != plan.plan_id or execution.session_id != plan.session_id:
            raise DomainError("execution does not belong to the investigation plan")
        if set(execution.run_ids) != set(plan.dependencies):
            raise DomainError("execution AgentRun mapping does not match the plan")

    @staticmethod
    def _priority(value: float) -> TaskPriority:
        if value >= 0.67:
            return TaskPriority.HIGH
        if value <= 0.33:
            return TaskPriority.LOW
        return TaskPriority.NORMAL
