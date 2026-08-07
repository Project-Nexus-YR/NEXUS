"""Atomic task coordination ports and local in-memory/SQLite adapters."""

from __future__ import annotations

import builtins
import json
import sqlite3
from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
from dataclasses import fields
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock
from typing import Any, Protocol

from ..models import DomainError
from .model import (
    TERMINAL_TASK_STATES,
    DistributedTask,
    DistributedTaskState,
    FailureClass,
    Lease,
    RetryPolicy,
    TaskAttempt,
    TaskPriority,
    TaskStateMachine,
)

Mutation = Callable[[DistributedTask], None]


class LeaseStore(Protocol):
    """Lease operations that must be atomic with their owning task record."""

    def claim(
        self,
        task_id: str,
        worker_id: str,
        now: datetime,
        lease_duration: timedelta,
    ) -> DistributedTask | None: ...

    def renew_lease(
        self,
        task_id: str,
        worker_id: str,
        lease_id: str,
        now: datetime,
        lease_duration: timedelta,
    ) -> DistributedTask: ...

    def list_expired(self, now: datetime) -> builtins.list[DistributedTask]: ...

    def release_expired(
        self, task_id: str, now: datetime, retry_at: datetime | None
    ) -> DistributedTask: ...


class TaskStore(ABC):
    """Atomic coordination contract expected from a transactional backend."""

    @abstractmethod
    def create(self, task: DistributedTask) -> DistributedTask: ...

    @abstractmethod
    def get(self, task_id: str) -> DistributedTask | None: ...

    @abstractmethod
    def update(self, task: DistributedTask, expected_version: int) -> DistributedTask: ...

    @abstractmethod
    def list(
        self, states: frozenset[DistributedTaskState] | None = None
    ) -> list[DistributedTask]: ...

    @abstractmethod
    def _mutate(self, task_id: str, mutation: Mutation) -> DistributedTask: ...

    def claim(
        self,
        task_id: str,
        worker_id: str,
        now: datetime,
        lease_duration: timedelta,
    ) -> DistributedTask | None:
        def mutation(task: DistributedTask) -> None:
            if task.state != DistributedTaskState.QUEUED or task.available_at > now:
                raise _ClaimRejected
            if task.deadline is not None and task.deadline <= now:
                raise _ClaimRejected
            TaskStateMachine.transition(task, DistributedTaskState.CLAIMED, now)
            task.attempt += 1
            task.worker_id = worker_id
            task.lease = Lease(task.task_id, worker_id, now, now + lease_duration)
            task.attempts.append(
                TaskAttempt(task.task_id, worker_id, task.lease.lease_id, task.attempt, now)
            )

        try:
            return self._mutate(task_id, mutation)
        except _ClaimRejected:
            return None

    def mark_running(
        self, task_id: str, worker_id: str, lease_id: str, now: datetime
    ) -> DistributedTask:
        def mutation(task: DistributedTask) -> None:
            _require_owner(task, worker_id, lease_id, now)
            TaskStateMachine.transition(task, DistributedTaskState.RUNNING, now)
            attempt = _current_attempt(task)
            attempt.state = DistributedTaskState.RUNNING.value

        return self._mutate(task_id, mutation)

    def renew_lease(
        self,
        task_id: str,
        worker_id: str,
        lease_id: str,
        now: datetime,
        lease_duration: timedelta,
    ) -> DistributedTask:
        def mutation(task: DistributedTask) -> None:
            _require_owner(task, worker_id, lease_id, now)
            if task.state not in {
                DistributedTaskState.CLAIMED,
                DistributedTaskState.RUNNING,
            }:
                raise DomainError("only active tasks can renew a lease")
            if task.lease is None:
                raise DomainError("active task has no lease")
            task.lease = Lease(
                task.task_id,
                worker_id,
                task.lease.issued_at,
                now + lease_duration,
                lease_id,
            )
            task.updated_at = now

        return self._mutate(task_id, mutation)

    def complete(
        self,
        task_id: str,
        worker_id: str,
        lease_id: str,
        now: datetime,
        result_ref: str | None,
        attempt_id: str | None = None,
    ) -> DistributedTask:
        def mutation(task: DistributedTask) -> None:
            _require_owner(task, worker_id, lease_id, now, attempt_id)
            TaskStateMachine.transition(task, DistributedTaskState.SUCCEEDED, now)
            attempt = _current_attempt(task)
            attempt.state = DistributedTaskState.SUCCEEDED.value
            attempt.completed_at = now
            task.result_ref = result_ref

        return self._mutate(task_id, mutation)

    def fail(
        self,
        task_id: str,
        worker_id: str,
        lease_id: str,
        now: datetime,
        failure_class: FailureClass,
        error: str,
        retry_at: datetime | None,
        checkpoint_ref: str | None = None,
        attempt_id: str | None = None,
    ) -> DistributedTask:
        def mutation(task: DistributedTask) -> None:
            _require_owner(task, worker_id, lease_id, now, attempt_id)
            if failure_class == FailureClass.CANCELLED:
                if task.state != DistributedTaskState.CANCEL_REQUESTED:
                    TaskStateMachine.transition(task, DistributedTaskState.CANCEL_REQUESTED, now)
                TaskStateMachine.transition(task, DistributedTaskState.CANCELLED, now)
                attempt = _current_attempt(task)
                attempt.state = DistributedTaskState.CANCELLED.value
                attempt.failure_class = failure_class
                attempt.completed_at = now
                _release_ownership(task)
                return
            TaskStateMachine.transition(task, DistributedTaskState.FAILED, now)
            _record_failed_attempt(task, failure_class, error, now, checkpoint_ref)
            if failure_class == FailureClass.TRANSIENT and retry_at is not None:
                TaskStateMachine.transition(task, DistributedTaskState.RETRY_WAIT, now)
                task.available_at = retry_at
            else:
                TaskStateMachine.transition(task, DistributedTaskState.DEAD_LETTERED, now)
            _release_ownership(task)

        return self._mutate(task_id, mutation)

    def release_expired(
        self, task_id: str, now: datetime, retry_at: datetime | None
    ) -> DistributedTask:
        def mutation(task: DistributedTask) -> None:
            if task.state == DistributedTaskState.CANCEL_REQUESTED:
                if task.lease is not None and task.lease.expires_at > now:
                    raise DomainError("task cancellation lease has not expired")
                TaskStateMachine.transition(task, DistributedTaskState.CANCELLED, now)
                if task.attempts:
                    attempt = _current_attempt(task)
                    attempt.state = DistributedTaskState.CANCELLED.value
                    attempt.failure_class = FailureClass.CANCELLED
                    attempt.completed_at = now
                _release_ownership(task)
                return
            if task.state not in {
                DistributedTaskState.CLAIMED,
                DistributedTaskState.RUNNING,
            }:
                raise DomainError("task does not hold an active lease")
            if task.lease is None or task.lease.expires_at > now:
                raise DomainError("task lease has not expired")
            TaskStateMachine.transition(task, DistributedTaskState.FAILED, now)
            _record_failed_attempt(
                task,
                FailureClass.TRANSIENT,
                "lease expired",
                now,
                task.last_checkpoint_ref,
            )
            if retry_at is not None:
                TaskStateMachine.transition(task, DistributedTaskState.RETRY_WAIT, now)
                task.available_at = retry_at
            else:
                TaskStateMachine.transition(task, DistributedTaskState.DEAD_LETTERED, now)
            _release_ownership(task)

        return self._mutate(task_id, mutation)

    def request_cancel(self, task_id: str, now: datetime) -> DistributedTask:
        def mutation(task: DistributedTask) -> None:
            if task.state in TERMINAL_TASK_STATES:
                return
            if task.state == DistributedTaskState.CANCEL_REQUESTED:
                return
            TaskStateMachine.transition(task, DistributedTaskState.CANCEL_REQUESTED, now)
            if task.worker_id is None:
                TaskStateMachine.transition(task, DistributedTaskState.CANCELLED, now)
                _release_ownership(task)

        return self._mutate(task_id, mutation)

    def acknowledge_cancel(
        self,
        task_id: str,
        worker_id: str,
        lease_id: str,
        now: datetime,
        attempt_id: str | None = None,
    ) -> DistributedTask:
        def mutation(task: DistributedTask) -> None:
            _require_owner(task, worker_id, lease_id, now, attempt_id)
            if task.state != DistributedTaskState.CANCEL_REQUESTED:
                raise DomainError("task has no cancellation request")
            TaskStateMachine.transition(task, DistributedTaskState.CANCELLED, now)
            attempt = _current_attempt(task)
            attempt.state = DistributedTaskState.CANCELLED.value
            attempt.failure_class = FailureClass.CANCELLED
            attempt.completed_at = now
            _release_ownership(task)

        return self._mutate(task_id, mutation)

    def manual_retry(self, task_id: str, now: datetime) -> DistributedTask:
        def mutation(task: DistributedTask) -> None:
            if task.state != DistributedTaskState.DEAD_LETTERED:
                raise DomainError("only dead-lettered tasks can be retried manually")
            TaskStateMachine.transition(task, DistributedTaskState.QUEUED, now)
            task.available_at = now
            task.result_ref = None
            _release_ownership(task)

        return self._mutate(task_id, mutation)

    def advance_retry(self, task_id: str, now: datetime) -> DistributedTask:
        def mutation(task: DistributedTask) -> None:
            if task.state != DistributedTaskState.RETRY_WAIT or task.available_at > now:
                raise DomainError("task retry is not yet available")
            TaskStateMachine.transition(task, DistributedTaskState.QUEUED, now)

        return self._mutate(task_id, mutation)

    def expire_deadline(self, task_id: str, now: datetime) -> DistributedTask:
        def mutation(task: DistributedTask) -> None:
            if task.state not in {
                DistributedTaskState.QUEUED,
                DistributedTaskState.RETRY_WAIT,
            }:
                raise DomainError("active or terminal task deadline cannot be expired here")
            if task.deadline is None or task.deadline > now:
                raise DomainError("task deadline has not expired")
            TaskStateMachine.transition(task, DistributedTaskState.FAILED, now)
            task.last_error = "task deadline expired"
            TaskStateMachine.transition(task, DistributedTaskState.DEAD_LETTERED, now)

        return self._mutate(task_id, mutation)

    def list_queued(self, now: datetime) -> builtins.list[DistributedTask]:
        return [
            task
            for task in self.list(frozenset({DistributedTaskState.QUEUED}))
            if task.available_at <= now and (task.deadline is None or task.deadline > now)
        ]

    def list_expired(self, now: datetime) -> builtins.list[DistributedTask]:
        return [
            task
            for task in self.list(
                frozenset(
                    {
                        DistributedTaskState.CLAIMED,
                        DistributedTaskState.RUNNING,
                        DistributedTaskState.CANCEL_REQUESTED,
                    }
                )
            )
            if task.lease is None or task.lease.expires_at <= now
        ]

    def list_by_worker(self, worker_id: str) -> builtins.list[DistributedTask]:
        return [task for task in self.list() if task.worker_id == worker_id]

    def counts(self) -> dict[str, int]:
        result = {state.value: 0 for state in DistributedTaskState}
        for task in self.list():
            result[task.state.value] += 1
        return result


class _ClaimRejected(Exception):
    pass


class InMemoryTaskStore(TaskStore):
    def __init__(self) -> None:
        self._tasks: dict[str, DistributedTask] = {}
        self._lock = RLock()

    def create(self, task: DistributedTask) -> DistributedTask:
        with self._lock:
            if task.task_id in self._tasks:
                raise DomainError(f"task already exists: {task.task_id}")
            self._tasks[task.task_id] = deepcopy(task)
            return deepcopy(task)

    def get(self, task_id: str) -> DistributedTask | None:
        with self._lock:
            task = self._tasks.get(task_id)
            return None if task is None else deepcopy(task)

    def update(self, task: DistributedTask, expected_version: int) -> DistributedTask:
        def mutation(current: DistributedTask) -> None:
            if current.version != expected_version:
                raise DomainError("optimistic task version conflict")
            _validate_safe_update(current, task)
            _copy_task_fields(current, task)

        return self._mutate(task.task_id, mutation)

    def list(self, states: frozenset[DistributedTaskState] | None = None) -> list[DistributedTask]:
        with self._lock:
            return [
                deepcopy(task)
                for task in self._tasks.values()
                if states is None or task.state in states
            ]

    def _mutate(self, task_id: str, mutation: Mutation) -> DistributedTask:
        with self._lock:
            try:
                task = deepcopy(self._tasks[task_id])
            except KeyError as exc:
                raise DomainError(f"unknown distributed task: {task_id}") from exc
            mutation(task)
            task.version += 1
            self._tasks[task_id] = task
            return deepcopy(task)

    def counts(self) -> dict[str, int]:
        with self._lock:
            result = {state.value: 0 for state in DistributedTaskState}
            for task in self._tasks.values():
                result[task.state.value] += 1
            return result


class SQLiteTaskStore(TaskStore):
    """Single-node durable adapter with explicit write transactions."""

    def __init__(self, path: str | Path) -> None:
        path_text = str(path)
        if path_text != ":memory:":
            Path(path_text).parent.mkdir(parents=True, exist_ok=True)
        self._connection = sqlite3.connect(path_text, check_same_thread=False, timeout=30)
        self._connection.row_factory = sqlite3.Row
        self._lock = RLock()
        with self._connection:
            self._connection.execute("PRAGMA journal_mode=WAL")
            self._connection.execute("PRAGMA busy_timeout=30000")
            self._connection.execute(
                """
                CREATE TABLE IF NOT EXISTS distributed_tasks (
                    task_id TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    worker_id TEXT,
                    lease_expires_at TEXT,
                    available_at TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                )
                """
            )
            self._connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_distributed_tasks_state_available "
                "ON distributed_tasks(state, available_at)"
            )

    def create(self, task: DistributedTask) -> DistributedTask:
        with self._lock, self._connection:
            try:
                self._connection.execute(
                    "INSERT INTO distributed_tasks(task_id, state, worker_id, "
                    "lease_expires_at, available_at, version, payload_json) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    _task_row(task),
                )
            except sqlite3.IntegrityError as exc:
                raise DomainError(f"task already exists: {task.task_id}") from exc
        return deepcopy(task)

    def get(self, task_id: str) -> DistributedTask | None:
        with self._lock:
            row = self._connection.execute(
                "SELECT payload_json FROM distributed_tasks WHERE task_id = ?", (task_id,)
            ).fetchone()
        return None if row is None else task_from_dict(json.loads(str(row["payload_json"])))

    def update(self, task: DistributedTask, expected_version: int) -> DistributedTask:
        def mutation(current: DistributedTask) -> None:
            if current.version != expected_version:
                raise DomainError("optimistic task version conflict")
            _validate_safe_update(current, task)
            _copy_task_fields(current, task)

        return self._mutate(task.task_id, mutation)

    def list(self, states: frozenset[DistributedTaskState] | None = None) -> list[DistributedTask]:
        with self._lock:
            if states:
                placeholders = ",".join("?" for _ in states)
                rows = self._connection.execute(
                    f"SELECT payload_json FROM distributed_tasks WHERE state IN ({placeholders})",
                    tuple(state.value for state in states),
                ).fetchall()
            else:
                rows = self._connection.execute(
                    "SELECT payload_json FROM distributed_tasks"
                ).fetchall()
        return [task_from_dict(json.loads(str(row["payload_json"]))) for row in rows]

    def _mutate(self, task_id: str, mutation: Mutation) -> DistributedTask:
        with self._lock:
            cursor = self._connection.cursor()
            cursor.execute("BEGIN IMMEDIATE")
            try:
                row = cursor.execute(
                    "SELECT payload_json FROM distributed_tasks WHERE task_id = ?", (task_id,)
                ).fetchone()
                if row is None:
                    raise DomainError(f"unknown distributed task: {task_id}")
                task = task_from_dict(json.loads(str(row["payload_json"])))
                mutation(task)
                task.version += 1
                cursor.execute(
                    "UPDATE distributed_tasks SET state = ?, worker_id = ?, "
                    "lease_expires_at = ?, available_at = ?, version = ?, payload_json = ? "
                    "WHERE task_id = ?",
                    (*_task_row(task)[1:], task.task_id),
                )
                self._connection.commit()
                return task
            except Exception:
                self._connection.rollback()
                raise

    def counts(self) -> dict[str, int]:
        result = {state.value: 0 for state in DistributedTaskState}
        with self._lock:
            rows = self._connection.execute(
                "SELECT state, COUNT(*) AS count FROM distributed_tasks GROUP BY state"
            ).fetchall()
        for row in rows:
            result[str(row["state"])] = int(row["count"])
        return result

    def close(self) -> None:
        self._connection.close()


def _require_owner(
    task: DistributedTask,
    worker_id: str,
    lease_id: str,
    now: datetime,
    attempt_id: str | None = None,
) -> None:
    if task.worker_id != worker_id or task.lease is None or task.lease.lease_id != lease_id:
        raise DomainError("worker does not own the current task lease")
    if not task.lease.is_valid(now):
        raise DomainError("task lease is stale")
    if attempt_id is not None and _current_attempt(task).attempt_id != attempt_id:
        raise DomainError("attempt does not match the current task lease")


def _current_attempt(task: DistributedTask) -> TaskAttempt:
    if not task.attempts:
        raise DomainError("task has no current attempt")
    return task.attempts[-1]


def _record_failed_attempt(
    task: DistributedTask,
    failure_class: FailureClass,
    error: str,
    now: datetime,
    checkpoint_ref: str | None,
) -> None:
    attempt = _current_attempt(task)
    attempt.state = DistributedTaskState.FAILED.value
    attempt.failure_class = failure_class
    attempt.error = error[:4096]
    attempt.completed_at = now
    task.last_error = error[:4096]
    task.last_checkpoint_ref = checkpoint_ref


def _release_ownership(task: DistributedTask) -> None:
    task.worker_id = None
    task.lease = None


def _copy_task_fields(target: DistributedTask, source: DistributedTask) -> None:
    for item in fields(DistributedTask):
        if item.name not in {"task_id", "version"}:
            setattr(target, item.name, deepcopy(getattr(source, item.name)))


def _validate_safe_update(current: DistributedTask, proposed: DistributedTask) -> None:
    protected = (
        "run_id",
        "correlation_id",
        "state",
        "attempt",
        "worker_id",
        "lease",
        "attempts",
        "result_ref",
        "last_error",
        "created_at",
    )
    if any(getattr(current, name) != getattr(proposed, name) for name in protected):
        raise DomainError("coordination fields require an atomic task operation")


def _task_row(task: DistributedTask) -> tuple[str, str, str | None, str | None, str, int, str]:
    return (
        task.task_id,
        task.state.value,
        task.worker_id,
        None if task.lease is None else task.lease.expires_at.isoformat(),
        task.available_at.isoformat(),
        task.version,
        json.dumps(task_to_dict(task), sort_keys=True),
    )


def task_to_dict(task: DistributedTask) -> dict[str, Any]:
    return {
        "task_id": task.task_id,
        "run_id": task.run_id,
        "correlation_id": task.correlation_id,
        "required_capabilities": sorted(task.required_capabilities),
        "priority": int(task.priority),
        "retry_policy": {
            "max_attempts": task.retry_policy.max_attempts,
            "initial_backoff_seconds": task.retry_policy.initial_backoff.total_seconds(),
            "max_backoff_seconds": task.retry_policy.max_backoff.total_seconds(),
            "backoff_multiplier": task.retry_policy.backoff_multiplier,
            "jitter_ratio": task.retry_policy.jitter_ratio,
        },
        "metadata": task.metadata,
        "available_at": task.available_at.isoformat(),
        "deadline": None if task.deadline is None else task.deadline.isoformat(),
        "state": task.state.value,
        "attempt": task.attempt,
        "worker_id": task.worker_id,
        "lease": None
        if task.lease is None
        else {
            "task_id": task.lease.task_id,
            "worker_id": task.lease.worker_id,
            "issued_at": task.lease.issued_at.isoformat(),
            "expires_at": task.lease.expires_at.isoformat(),
            "lease_id": task.lease.lease_id,
        },
        "attempts": [
            {
                "task_id": attempt.task_id,
                "worker_id": attempt.worker_id,
                "lease_id": attempt.lease_id,
                "number": attempt.number,
                "started_at": attempt.started_at.isoformat(),
                "attempt_id": attempt.attempt_id,
                "state": attempt.state,
                "completed_at": None
                if attempt.completed_at is None
                else attempt.completed_at.isoformat(),
                "failure_class": None
                if attempt.failure_class is None
                else attempt.failure_class.value,
                "error": attempt.error,
            }
            for attempt in task.attempts
        ],
        "result_ref": task.result_ref,
        "last_error": task.last_error,
        "last_checkpoint_ref": task.last_checkpoint_ref,
        "created_at": task.created_at.isoformat(),
        "updated_at": task.updated_at.isoformat(),
        "version": task.version,
    }


def task_from_dict(value: dict[str, Any]) -> DistributedTask:
    retry = value["retry_policy"]
    lease_value = value.get("lease")
    lease = (
        None
        if lease_value is None
        else Lease(
            task_id=str(lease_value["task_id"]),
            worker_id=str(lease_value["worker_id"]),
            issued_at=datetime.fromisoformat(str(lease_value["issued_at"])),
            expires_at=datetime.fromisoformat(str(lease_value["expires_at"])),
            lease_id=str(lease_value["lease_id"]),
        )
    )
    attempts = [
        TaskAttempt(
            task_id=str(item["task_id"]),
            worker_id=str(item["worker_id"]),
            lease_id=str(item["lease_id"]),
            number=int(item["number"]),
            started_at=datetime.fromisoformat(str(item["started_at"])),
            attempt_id=str(item["attempt_id"]),
            state=str(item["state"]),
            completed_at=None
            if item["completed_at"] is None
            else datetime.fromisoformat(str(item["completed_at"])),
            failure_class=None
            if item["failure_class"] is None
            else FailureClass(str(item["failure_class"])),
            error=item["error"],
        )
        for item in value.get("attempts", [])
    ]
    return DistributedTask(
        run_id=str(value["run_id"]),
        correlation_id=str(value["correlation_id"]),
        required_capabilities=frozenset(value.get("required_capabilities", [])),
        priority=TaskPriority(int(value["priority"])),
        retry_policy=RetryPolicy(
            max_attempts=int(retry["max_attempts"]),
            initial_backoff=timedelta(seconds=float(retry["initial_backoff_seconds"])),
            max_backoff=timedelta(seconds=float(retry["max_backoff_seconds"])),
            backoff_multiplier=float(retry["backoff_multiplier"]),
            jitter_ratio=float(retry["jitter_ratio"]),
        ),
        metadata=dict(value.get("metadata", {})),
        available_at=datetime.fromisoformat(str(value["available_at"])),
        deadline=None
        if value.get("deadline") is None
        else datetime.fromisoformat(str(value["deadline"])),
        task_id=str(value["task_id"]),
        state=DistributedTaskState(str(value["state"])),
        attempt=int(value["attempt"]),
        worker_id=value.get("worker_id"),
        lease=lease,
        attempts=attempts,
        result_ref=value.get("result_ref"),
        last_error=value.get("last_error"),
        last_checkpoint_ref=value.get("last_checkpoint_ref"),
        created_at=datetime.fromisoformat(str(value["created_at"])),
        updated_at=datetime.fromisoformat(str(value["updated_at"])),
        version=int(value["version"]),
    )
