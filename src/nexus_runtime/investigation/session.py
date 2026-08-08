"""Explicit, bounded lifecycle for an autonomous investigation session."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum

from nexus_runtime.models import DomainError, InvalidTransition, new_id, utcnow

from .objective import (
    _required_string,
    _timestamp_from_text,
    _timestamp_to_text,
    _validate_timestamp,
)


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError
    return value


def _as_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError
    return float(value)


class SessionState(StrEnum):
    PLANNING = "PLANNING"
    EXECUTING = "EXECUTING"
    EVALUATING = "EVALUATING"
    UPDATING = "UPDATING"
    COMPLETED = "COMPLETED"
    PAUSED = "PAUSED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


class TerminationReason(StrEnum):
    OBJECTIVE_SATISFIED = "objective_satisfied"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_VALUABLE_INVESTIGATION = "no_valuable_investigation"
    CONFIDENCE_THRESHOLD_REACHED = "confidence_threshold_reached"
    MAXIMUM_ITERATIONS_REACHED = "maximum_iterations_reached"
    UNRESOLVABLE_CONTRADICTION = "unresolvable_contradiction"
    USER_CANCELLATION = "user_cancellation"
    SYSTEM_FAILURE = "system_failure"


@dataclass(frozen=True, slots=True)
class InvestigationBudget:
    max_iterations: int
    max_investigations: int
    max_agent_runs: int
    max_cost: float
    max_execution_time: timedelta

    def __post_init__(self) -> None:
        if self.max_iterations < 1:
            raise DomainError("max_iterations must be at least one")
        if self.max_investigations < 1:
            raise DomainError("max_investigations must be at least one")
        if self.max_agent_runs < 1:
            raise DomainError("max_agent_runs must be at least one")
        if self.max_cost <= 0:
            raise DomainError("max_cost must be positive")
        if self.max_execution_time <= timedelta(0):
            raise DomainError("max_execution_time must be positive")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "max_iterations": self.max_iterations,
            "max_investigations": self.max_investigations,
            "max_agent_runs": self.max_agent_runs,
            "max_cost": self.max_cost,
            "max_execution_seconds": self.max_execution_time.total_seconds(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> InvestigationBudget:
        try:
            return cls(
                max_iterations=_as_int(payload["max_iterations"]),
                max_investigations=_as_int(payload["max_investigations"]),
                max_agent_runs=_as_int(payload["max_agent_runs"]),
                max_cost=_as_float(payload["max_cost"]),
                max_execution_time=timedelta(seconds=_as_float(payload["max_execution_seconds"])),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DomainError("malformed InvestigationBudget") from exc


@dataclass(slots=True)
class InvestigationUsage:
    investigations: int = 0
    agent_runs: int = 0
    cost: float = 0.0
    execution_time: timedelta = timedelta(0)

    def __post_init__(self) -> None:
        if self.investigations < 0 or self.agent_runs < 0:
            raise DomainError("usage counters cannot be negative")
        if self.cost < 0 or self.execution_time < timedelta(0):
            raise DomainError("cost and execution time cannot be negative")

    def to_dict(self) -> dict[str, int | float]:
        return {
            "investigations": self.investigations,
            "agent_runs": self.agent_runs,
            "cost": self.cost,
            "execution_seconds": self.execution_time.total_seconds(),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> InvestigationUsage:
        try:
            return cls(
                investigations=_as_int(payload["investigations"]),
                agent_runs=_as_int(payload["agent_runs"]),
                cost=_as_float(payload["cost"]),
                execution_time=timedelta(seconds=_as_float(payload["execution_seconds"])),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DomainError("malformed InvestigationUsage") from exc


_ACTIVE_STATES = frozenset(
    {
        SessionState.PLANNING,
        SessionState.EXECUTING,
        SessionState.EVALUATING,
        SessionState.UPDATING,
    }
)
_TERMINAL_STATES = frozenset({SessionState.COMPLETED, SessionState.FAILED, SessionState.CANCELLED})


@dataclass(slots=True)
class InvestigationSession:
    objective_id: str
    budget: InvestigationBudget
    session_id: str = field(default_factory=lambda: new_id("session"))
    state: SessionState = SessionState.PLANNING
    iteration: int = 0
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    usage: InvestigationUsage = field(default_factory=InvestigationUsage)
    termination_reason: TerminationReason | None = None
    paused_from: SessionState | None = None

    _TRANSITIONS = {
        SessionState.PLANNING: frozenset(
            {
                SessionState.EXECUTING,
                SessionState.COMPLETED,
                SessionState.PAUSED,
                SessionState.FAILED,
                SessionState.CANCELLED,
            }
        ),
        SessionState.EXECUTING: frozenset(
            {
                SessionState.EVALUATING,
                SessionState.PAUSED,
                SessionState.FAILED,
                SessionState.CANCELLED,
            }
        ),
        SessionState.EVALUATING: frozenset(
            {
                SessionState.UPDATING,
                SessionState.PLANNING,
                SessionState.COMPLETED,
                SessionState.PAUSED,
                SessionState.FAILED,
                SessionState.CANCELLED,
            }
        ),
        SessionState.UPDATING: frozenset(
            {
                SessionState.PLANNING,
                SessionState.COMPLETED,
                SessionState.PAUSED,
                SessionState.FAILED,
                SessionState.CANCELLED,
            }
        ),
        SessionState.PAUSED: _ACTIVE_STATES
        | frozenset({SessionState.FAILED, SessionState.CANCELLED}),
        SessionState.COMPLETED: frozenset(),
        SessionState.FAILED: frozenset(),
        SessionState.CANCELLED: frozenset(),
    }

    def __post_init__(self) -> None:
        self.objective_id = self.objective_id.strip()
        self.session_id = self.session_id.strip()
        if not self.objective_id or not self.session_id:
            raise DomainError("objective_id and session_id are required")
        if self.iteration < 0:
            raise DomainError("session iteration cannot be negative")
        if not isinstance(self.created_at, datetime) or not isinstance(self.updated_at, datetime):
            raise DomainError("session timestamps must be datetimes")
        _validate_timestamp(self.created_at, "created_at")
        _validate_timestamp(self.updated_at, "updated_at")
        if self.updated_at < self.created_at:
            raise DomainError("updated_at cannot precede created_at")
        if self.state in _TERMINAL_STATES and self.termination_reason is None:
            raise DomainError("terminal sessions require a termination reason")
        if self.state not in _TERMINAL_STATES and self.termination_reason is not None:
            raise DomainError("active sessions cannot have a termination reason")
        if (self.state == SessionState.PAUSED) != (self.paused_from is not None):
            raise DomainError("paused sessions must record their prior state")
        if self.paused_from is not None and self.paused_from not in _ACTIVE_STATES:
            raise DomainError("paused_from must be an active state")

    def transition(
        self,
        target: SessionState,
        *,
        at: datetime | None = None,
        reason: TerminationReason | None = None,
    ) -> InvestigationSession:
        if target not in self._TRANSITIONS[self.state]:
            raise InvalidTransition(
                f"invalid investigation session transition: {self.state} -> {target}"
            )
        if (
            self.state == SessionState.PAUSED
            and target in _ACTIVE_STATES
            and target != self.paused_from
        ):
            raise InvalidTransition("a paused session must resume its prior active state")
        if target in _TERMINAL_STATES and reason is None:
            raise DomainError("terminal transitions require a termination reason")
        if target not in _TERMINAL_STATES and reason is not None:
            raise DomainError("termination reason is only valid for terminal transitions")
        if target == SessionState.CANCELLED and reason != TerminationReason.USER_CANCELLATION:
            raise DomainError("cancelled sessions require user_cancellation")
        if target == SessionState.FAILED and reason != TerminationReason.SYSTEM_FAILURE:
            raise DomainError("failed sessions require system_failure")
        transition_time = utcnow() if at is None else at
        if not isinstance(transition_time, datetime):
            raise DomainError("transition time must be a datetime")
        _validate_timestamp(transition_time, "transition time")
        if transition_time < self.updated_at:
            raise DomainError("transition time cannot precede updated_at")
        if target == SessionState.PAUSED:
            self.paused_from = self.state
        elif self.state == SessionState.PAUSED:
            self.paused_from = None
        self.state = target
        self.updated_at = transition_time
        self.termination_reason = reason
        return self

    def pause(self, *, at: datetime | None = None) -> InvestigationSession:
        return self.transition(SessionState.PAUSED, at=at)

    def resume(self, *, at: datetime | None = None) -> InvestigationSession:
        if self.state != SessionState.PAUSED or self.paused_from is None:
            raise InvalidTransition("only a paused session can resume")
        return self.transition(self.paused_from, at=at)

    def complete_iteration(self, *, at: datetime | None = None) -> InvestigationSession:
        if self.state != SessionState.UPDATING:
            raise InvalidTransition("an iteration can complete only from UPDATING")
        self.iteration += 1
        return self.transition(SessionState.PLANNING, at=at)

    def record_usage(
        self,
        *,
        investigations: int = 0,
        agent_runs: int = 0,
        cost: float = 0.0,
        execution_time: timedelta = timedelta(0),
    ) -> None:
        if investigations < 0 or agent_runs < 0 or cost < 0 or execution_time < timedelta(0):
            raise DomainError("usage increments cannot be negative")
        self.usage.investigations += investigations
        self.usage.agent_runs += agent_runs
        self.usage.cost += cost
        self.usage.execution_time += execution_time

    def remaining_budget(self) -> dict[str, int | float]:
        return {
            "iterations": max(0, self.budget.max_iterations - self.iteration),
            "investigations": max(0, self.budget.max_investigations - self.usage.investigations),
            "agent_runs": max(0, self.budget.max_agent_runs - self.usage.agent_runs),
            "cost": max(0.0, self.budget.max_cost - self.usage.cost),
            "execution_seconds": max(
                0.0,
                (self.budget.max_execution_time - self.usage.execution_time).total_seconds(),
            ),
        }

    def to_dict(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "objective_id": self.objective_id,
            "state": self.state.value,
            "iteration": self.iteration,
            "created_at": _timestamp_to_text(self.created_at),
            "updated_at": _timestamp_to_text(self.updated_at),
            "budget": self.budget.to_dict(),
            "usage": self.usage.to_dict(),
            "termination_reason": (
                None if self.termination_reason is None else self.termination_reason.value
            ),
            "paused_from": None if self.paused_from is None else self.paused_from.value,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> InvestigationSession:
        try:
            budget = payload["budget"]
            usage = payload["usage"]
            if not isinstance(budget, dict) or not isinstance(usage, dict):
                raise TypeError
            reason_value = payload["termination_reason"]
            paused_value = payload["paused_from"]
            state_value = _required_string(payload["state"], "session state")
            if reason_value is not None and not isinstance(reason_value, str):
                raise TypeError
            if paused_value is not None and not isinstance(paused_value, str):
                raise TypeError
            return cls(
                session_id=_required_string(payload["session_id"], "session_id"),
                objective_id=_required_string(payload["objective_id"], "objective_id"),
                state=SessionState(state_value),
                iteration=_as_int(payload["iteration"]),
                created_at=_timestamp_from_text(payload["created_at"], "created_at"),
                updated_at=_timestamp_from_text(payload["updated_at"], "updated_at"),
                budget=InvestigationBudget.from_dict(budget),
                usage=InvestigationUsage.from_dict(usage),
                termination_reason=(
                    None if reason_value is None else TerminationReason(reason_value)
                ),
                paused_from=None if paused_value is None else SessionState(paused_value),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise DomainError("malformed InvestigationSession") from exc
