"""Compile citation work into ready waves for the existing distributed runtime."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from nexus_runtime.distributed.model import (
    TERMINAL_TASK_STATES,
    DistributedTask,
    DistributedTaskState,
    FailureClass,
    RetryPolicy,
    TaskPriority,
)
from nexus_runtime.distributed.worker import (
    HarnessExecutionContext,
    HarnessOutcome,
    HarnessStatus,
)
from nexus_runtime.models import DomainError

from .extraction import ClaimExtractor
from .model import stable_id
from .planning import SearchPlanner


class EvidenceRuntimePort(Protocol):
    def submit_task(
        self,
        run_id: str,
        *,
        correlation_id: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        required_capabilities: frozenset[str] = frozenset(),
        retry_policy: RetryPolicy | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DistributedTask: ...

    def get_task(self, task_id: str) -> DistributedTask: ...

    def cancel_task(self, task_id: str, principal: str) -> DistributedTask: ...

    def list_tasks(
        self, states: frozenset[DistributedTaskState] | None = None
    ) -> list[DistributedTask]: ...


@dataclass(frozen=True, slots=True)
class EvidenceTaskNode:
    node_id: str
    task_type: str
    dependencies: tuple[str, ...]
    payload: dict[str, Any]
    capabilities: frozenset[str]
    priority: TaskPriority = TaskPriority.NORMAL
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)


@dataclass(frozen=True, slots=True)
class EvidenceTaskPlan:
    workflow_id: str
    session_id: str
    draft_hash: str
    nodes: tuple[EvidenceTaskNode, ...]

    @property
    def by_id(self) -> dict[str, EvidenceTaskNode]:
        return {item.node_id: item for item in self.nodes}


@dataclass(slots=True)
class EvidenceExecution:
    workflow_id: str
    session_id: str
    task_ids: dict[str, str] = field(default_factory=dict)
    blocked_nodes: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "workflow_id": self.workflow_id,
            "session_id": self.session_id,
            "task_ids": dict(self.task_ids),
            "blocked_nodes": dict(self.blocked_nodes),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> EvidenceExecution:
        task_ids = payload.get("task_ids")
        blocked = payload.get("blocked_nodes")
        if not isinstance(task_ids, dict) or not isinstance(blocked, dict):
            raise DomainError("malformed evidence execution")
        return cls(
            workflow_id=str(payload["workflow_id"]),
            session_id=str(payload["session_id"]),
            task_ids={str(key): str(value) for key, value in task_ids.items()},
            blocked_nodes={str(key): str(value) for key, value in blocked.items()},
        )


@dataclass(frozen=True, slots=True)
class EvidenceExecutionStatus:
    execution: EvidenceExecution
    task_states: dict[str, DistributedTaskState]
    terminal: bool
    succeeded: tuple[str, ...]
    failed: tuple[str, ...]
    running: tuple[str, ...]


class EvidenceTaskCompiler:
    """Build a deterministic DAG; the runtime still owns all execution semantics."""

    def __init__(
        self,
        extractor: ClaimExtractor | None = None,
        planner: SearchPlanner | None = None,
    ) -> None:
        self._extractor = extractor or ClaimExtractor()
        self._planner = planner or SearchPlanner()

    def compile(self, draft_answer: str, *, session_id: str) -> EvidenceTaskPlan:
        import hashlib

        if not draft_answer.strip():
            raise DomainError("draft answer is required")
        if len(draft_answer.encode("utf-8")) > 32 * 1024:
            raise DomainError("draft answer is too large for distributed task metadata")
        draft_hash = hashlib.sha256(draft_answer.encode()).hexdigest()
        workflow_id = stable_id("evidence_workflow", session_id, draft_hash)
        extraction_id = stable_id("evidence_node", workflow_id, "claim_extraction")
        nodes = [
            EvidenceTaskNode(
                extraction_id,
                "claim_extraction",
                (),
                {"draft_answer": draft_answer, "draft_hash": draft_hash},
                frozenset({"evidence.claim.extract"}),
                TaskPriority.HIGH,
            )
        ]
        verification_nodes: list[str] = []
        claims = self._planner.prioritize(self._extractor.extract(draft_answer))
        for claim in claims:
            retrieval_nodes: list[str] = []
            for query in self._planner.plan(claim):
                node_id = stable_id("evidence_node", workflow_id, "retrieval", query.query_id)
                nodes.append(
                    EvidenceTaskNode(
                        node_id,
                        "evidence_retrieval",
                        (extraction_id,),
                        {"claim": claim.to_dict(), "query": query.to_dict()},
                        frozenset({"evidence.retrieve"}),
                        self._priority(query.priority),
                    )
                )
                retrieval_nodes.append(node_id)
            verification_id = stable_id(
                "evidence_node", workflow_id, "verification", claim.claim_id
            )
            nodes.append(
                EvidenceTaskNode(
                    verification_id,
                    "evidence_verification",
                    tuple(retrieval_nodes) or (extraction_id,),
                    {"claim": claim.to_dict()},
                    frozenset({"evidence.verify"}),
                    self._priority(claim.importance),
                )
            )
            verification_nodes.append(verification_id)
        audit_id = stable_id("evidence_node", workflow_id, "citation_audit")
        nodes.append(
            EvidenceTaskNode(
                audit_id,
                "citation_audit",
                tuple(verification_nodes) or (extraction_id,),
                {"workflow_id": workflow_id},
                frozenset({"evidence.audit"}),
                TaskPriority.HIGH,
            )
        )
        return EvidenceTaskPlan(workflow_id, session_id, draft_hash, tuple(nodes))

    @staticmethod
    def _priority(value: float) -> TaskPriority:
        if value >= 0.8:
            return TaskPriority.HIGH
        if value < 0.45:
            return TaskPriority.LOW
        return TaskPriority.NORMAL


class EvidenceExecutionController:
    def __init__(self, runtime: EvidenceRuntimePort) -> None:
        self._runtime = runtime

    def start(self, plan: EvidenceTaskPlan) -> EvidenceExecutionStatus:
        return self.advance(plan, EvidenceExecution(plan.workflow_id, plan.session_id))

    def advance(
        self, plan: EvidenceTaskPlan, execution: EvidenceExecution
    ) -> EvidenceExecutionStatus:
        if (execution.workflow_id, execution.session_id) != (
            plan.workflow_id,
            plan.session_id,
        ):
            raise DomainError("execution does not belong to this evidence task plan")
        states = self._states(execution)
        made_progress = True
        while made_progress:
            made_progress = False
            for node in plan.nodes:
                if node.node_id in execution.task_ids or node.node_id in execution.blocked_nodes:
                    continue
                failed_parents = [
                    parent
                    for parent in node.dependencies
                    if parent in execution.blocked_nodes
                    or states.get(parent)
                    in {DistributedTaskState.CANCELLED, DistributedTaskState.DEAD_LETTERED}
                ]
                if failed_parents:
                    execution.blocked_nodes[node.node_id] = "dependency failed: " + ", ".join(
                        sorted(failed_parents)
                    )
                    made_progress = True
                    continue
                if not all(
                    states.get(parent) == DistributedTaskState.SUCCEEDED
                    for parent in node.dependencies
                ):
                    continue
                task = self._submit_or_recover(plan, node)
                execution.task_ids[node.node_id] = task.task_id
                states[node.node_id] = task.state
                made_progress = True
        states = self._states(execution)
        succeeded = tuple(
            sorted(key for key, value in states.items() if value == DistributedTaskState.SUCCEEDED)
        )
        failed = tuple(
            sorted(
                set(execution.blocked_nodes)
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
        return EvidenceExecutionStatus(
            execution,
            states,
            len(succeeded) + len(failed) == len(plan.nodes),
            succeeded,
            failed,
            running,
        )

    def cancel(self, execution: EvidenceExecution, principal: str) -> EvidenceExecution:
        for task_id in execution.task_ids.values():
            task = self._runtime.get_task(task_id)
            if task.state not in TERMINAL_TASK_STATES:
                self._runtime.cancel_task(task_id, principal)
        return execution

    def _submit_or_recover(self, plan: EvidenceTaskPlan, node: EvidenceTaskNode) -> DistributedTask:
        existing = [
            task
            for task in self._runtime.list_tasks()
            if task.correlation_id == plan.session_id
            and task.metadata.get("evidence_workflow_id") == plan.workflow_id
            and task.metadata.get("evidence_node_id") == node.node_id
        ]
        if len(existing) > 1:
            raise DomainError(f"duplicate distributed evidence tasks: {node.node_id}")
        if existing:
            return existing[0]
        metadata = {
            "schema_version": 1,
            "evidence_workflow_id": plan.workflow_id,
            "evidence_node_id": node.node_id,
            "evidence_task_type": node.task_type,
            "idempotency_key": stable_id("evidence_idempotency", plan.workflow_id, node.node_id),
            "dependencies": list(node.dependencies),
            "payload": node.payload,
            "source_content_is_untrusted": True,
        }
        return self._runtime.submit_task(
            stable_id("evidence_run", plan.workflow_id, node.node_id),
            correlation_id=plan.session_id,
            priority=node.priority,
            required_capabilities=node.capabilities,
            retry_policy=node.retry_policy,
            metadata=metadata,
        )

    def _states(self, execution: EvidenceExecution) -> dict[str, DistributedTaskState]:
        return {
            node_id: self._runtime.get_task(task_id).state
            for node_id, task_id in execution.task_ids.items()
        }


class EvidenceStageHandler(Protocol):
    """Provider-specific stage work; return a durable, idempotent result reference."""

    def execute(
        self,
        task_type: str,
        payload: Mapping[str, Any],
        context: HarnessExecutionContext,
    ) -> str: ...

    def cancel(self, run_id: str) -> None: ...


class EvidenceHarness:
    """Agent Harness adapter so evidence tasks use the standard Worker unchanged."""

    def __init__(self, handler: EvidenceStageHandler) -> None:
        self._handler = handler

    def execute_or_resume(
        self,
        context: HarnessExecutionContext,
        cancellation_requested: Callable[[], bool],
    ) -> HarnessOutcome:
        if cancellation_requested():
            return HarnessOutcome(HarnessStatus.CANCELLED, failure_class=FailureClass.CANCELLED)
        task_type = context.metadata.get("evidence_task_type")
        payload = context.metadata.get("payload")
        if not isinstance(task_type, str) or not isinstance(payload, Mapping):
            return HarnessOutcome(
                HarnessStatus.FAILED,
                failure_class=FailureClass.PERMANENT,
                error="malformed evidence task metadata",
            )
        try:
            result_ref = self._handler.execute(task_type, payload, context)
        except (ValueError, DomainError) as exc:
            return HarnessOutcome(
                HarnessStatus.FAILED,
                failure_class=FailureClass.PERMANENT,
                error=str(exc),
            )
        return HarnessOutcome(HarnessStatus.SUCCEEDED, result_ref=result_ref)

    def cancel_run(self, run_id: str) -> None:
        self._handler.cancel(run_id)
