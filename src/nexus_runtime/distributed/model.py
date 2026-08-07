"""Distributed task records and the centralized task state machine."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import IntEnum, StrEnum
from typing import Any

from ..models import DomainError, new_id, utcnow


class DistributedTaskState(StrEnum):
    QUEUED = "QUEUED"
    CLAIMED = "CLAIMED"
    RUNNING = "RUNNING"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    RETRY_WAIT = "RETRY_WAIT"
    CANCEL_REQUESTED = "CANCEL_REQUESTED"
    CANCELLED = "CANCELLED"
    DEAD_LETTERED = "DEAD_LETTERED"


TERMINAL_TASK_STATES = frozenset(
    {
        DistributedTaskState.SUCCEEDED,
        DistributedTaskState.CANCELLED,
        DistributedTaskState.DEAD_LETTERED,
    }
)


class TaskPriority(IntEnum):
    LOW = 0
    NORMAL = 50
    HIGH = 100


class FailureClass(StrEnum):
    TRANSIENT = "TRANSIENT"
    PERMANENT = "PERMANENT"
    CANCELLED = "CANCELLED"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    POLICY_VIOLATION = "POLICY_VIOLATION"


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    max_attempts: int = 3
    initial_backoff: timedelta = timedelta(seconds=1)
    max_backoff: timedelta = timedelta(minutes=1)
    backoff_multiplier: float = 2.0
    jitter_ratio: float = 0.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise DomainError("max_attempts must be at least one")
        if self.initial_backoff < timedelta(0) or self.max_backoff < self.initial_backoff:
            raise DomainError("retry backoff bounds are invalid")
        if self.backoff_multiplier < 1.0:
            raise DomainError("backoff_multiplier must be at least one")
        if not 0.0 <= self.jitter_ratio <= 1.0:
            raise DomainError("jitter_ratio must be between zero and one")

    def delay(self, task_id: str, completed_attempt: int) -> timedelta:
        exponent = max(0, completed_attempt - 1)
        raw = self.initial_backoff.total_seconds() * (self.backoff_multiplier**exponent)
        bounded = min(raw, self.max_backoff.total_seconds())
        if self.jitter_ratio:
            digest = hashlib.sha256(f"{task_id}:{completed_attempt}".encode()).digest()
            unit = int.from_bytes(digest[:8], "big") / ((1 << 64) - 1)
            bounded *= 1.0 - self.jitter_ratio + (2.0 * self.jitter_ratio * unit)
        return timedelta(seconds=min(bounded, self.max_backoff.total_seconds()))


@dataclass(frozen=True, slots=True)
class Lease:
    task_id: str
    worker_id: str
    issued_at: datetime
    expires_at: datetime
    lease_id: str = field(default_factory=lambda: new_id("lease"))

    def is_valid(self, now: datetime) -> bool:
        return now < self.expires_at


@dataclass(slots=True)
class TaskAttempt:
    task_id: str
    worker_id: str
    lease_id: str
    number: int
    started_at: datetime
    attempt_id: str = field(default_factory=lambda: new_id("attempt"))
    state: str = DistributedTaskState.CLAIMED.value
    completed_at: datetime | None = None
    failure_class: FailureClass | None = None
    error: str | None = None


@dataclass(slots=True)
class DistributedTask:
    run_id: str
    correlation_id: str
    required_capabilities: frozenset[str] = frozenset()
    priority: TaskPriority = TaskPriority.NORMAL
    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    metadata: dict[str, Any] = field(default_factory=dict)
    available_at: datetime = field(default_factory=utcnow)
    deadline: datetime | None = None
    task_id: str = field(default_factory=lambda: new_id("dtask"))
    state: DistributedTaskState = DistributedTaskState.QUEUED
    attempt: int = 0
    worker_id: str | None = None
    lease: Lease | None = None
    attempts: list[TaskAttempt] = field(default_factory=list)
    result_ref: str | None = None
    last_error: str | None = None
    last_checkpoint_ref: str | None = None
    created_at: datetime = field(default_factory=utcnow)
    updated_at: datetime = field(default_factory=utcnow)
    version: int = 0

    def __post_init__(self) -> None:
        if not self.run_id.strip() or not self.correlation_id.strip():
            raise DomainError("run_id and correlation_id are required")
        if self.deadline is not None and self.deadline <= self.created_at:
            raise DomainError("task deadline must be after creation")
        if any(not value.strip() for value in self.required_capabilities):
            raise DomainError("task capabilities cannot be empty")
        try:
            encoded = json.dumps(self.metadata, sort_keys=True, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise DomainError("task metadata must be JSON serializable") from exc
        if len(encoded.encode()) > 64 * 1024:
            raise DomainError("task metadata exceeds 64 KiB")

    @property
    def lease_id(self) -> str | None:
        return None if self.lease is None else self.lease.lease_id


class TaskStateMachine:
    """The only definition of legal distributed task transitions."""

    _ALLOWED: dict[DistributedTaskState, frozenset[DistributedTaskState]] = {
        DistributedTaskState.QUEUED: frozenset(
            {
                DistributedTaskState.CLAIMED,
                DistributedTaskState.FAILED,
                DistributedTaskState.CANCEL_REQUESTED,
            }
        ),
        DistributedTaskState.CLAIMED: frozenset(
            {
                DistributedTaskState.RUNNING,
                DistributedTaskState.FAILED,
                DistributedTaskState.CANCEL_REQUESTED,
            }
        ),
        DistributedTaskState.RUNNING: frozenset(
            {
                DistributedTaskState.SUCCEEDED,
                DistributedTaskState.FAILED,
                DistributedTaskState.CANCEL_REQUESTED,
            }
        ),
        DistributedTaskState.FAILED: frozenset(
            {DistributedTaskState.RETRY_WAIT, DistributedTaskState.DEAD_LETTERED}
        ),
        DistributedTaskState.RETRY_WAIT: frozenset(
            {
                DistributedTaskState.QUEUED,
                DistributedTaskState.FAILED,
                DistributedTaskState.CANCEL_REQUESTED,
            }
        ),
        DistributedTaskState.CANCEL_REQUESTED: frozenset({DistributedTaskState.CANCELLED}),
        DistributedTaskState.DEAD_LETTERED: frozenset({DistributedTaskState.QUEUED}),
        DistributedTaskState.SUCCEEDED: frozenset(),
        DistributedTaskState.CANCELLED: frozenset(),
    }

    @classmethod
    def transition(cls, task: DistributedTask, target: DistributedTaskState, now: datetime) -> None:
        if target not in cls._ALLOWED[task.state]:
            raise DomainError(f"invalid distributed task transition: {task.state} -> {target}")
        task.state = target
        task.updated_at = now
