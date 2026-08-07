"""Leased, priority-aware scheduler with explicit at-least-once semantics."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from threading import RLock
from typing import Any

from .dag import TaskDAG
from .events import Event, EventBus, InMemoryEventBus
from .models import DomainError, Task, TaskAttempt, TaskState, task_from_snapshot, utcnow
from .persistence import StateStore

Clock = Callable[[], datetime]


@dataclass(slots=True)
class Worker:
    worker_id: str
    capabilities: frozenset[str]
    max_concurrency: int
    registered_at: datetime
    last_heartbeat: datetime
    metadata: dict[str, str] = field(default_factory=dict)
    draining: bool = False


class Scheduler:
    """Owns task state transitions; callers cannot mutate the graph directly."""

    _TRANSITIONS: dict[TaskState, frozenset[TaskState]] = {
        TaskState.CREATED: frozenset({TaskState.READY, TaskState.CANCELLED, TaskState.COMPLETED}),
        TaskState.READY: frozenset({TaskState.LEASED, TaskState.CANCELLED}),
        TaskState.LEASED: frozenset({TaskState.RUNNING, TaskState.RETRYING, TaskState.CANCELLED}),
        TaskState.RUNNING: frozenset(
            {TaskState.COMPLETED, TaskState.RETRYING, TaskState.CANCELLED}
        ),
        TaskState.RETRYING: frozenset({TaskState.READY, TaskState.FAILED, TaskState.CANCELLED}),
        TaskState.COMPLETED: frozenset(),
        TaskState.FAILED: frozenset(),
        TaskState.CANCELLED: frozenset(),
    }

    def __init__(
        self,
        *,
        max_queued_tasks: int = 1_000,
        lease_duration: timedelta = timedelta(seconds=30),
        worker_timeout: timedelta = timedelta(seconds=45),
        event_bus: EventBus | None = None,
        state_store: StateStore | None = None,
        clock: Clock = utcnow,
    ) -> None:
        if max_queued_tasks < 1 or lease_duration <= timedelta(0) or worker_timeout <= timedelta(0):
            raise DomainError("scheduler limits must be positive")
        self._dag = TaskDAG()
        self._workers: dict[str, Worker] = {}
        self._attempts: dict[str, list[TaskAttempt]] = {}
        self._completed_keys: set[str] = set()
        self._max_queued_tasks = max_queued_tasks
        self._lease_duration = lease_duration
        self._worker_timeout = worker_timeout
        self._bus = event_bus or InMemoryEventBus()
        self._store = state_store
        self._clock = clock
        self._lock = RLock()

    def restore(self) -> int:
        """Load persisted task snapshots after a scheduler process restart.

        Workers deliberately are not restored: their old leases are reclaimed on the
        next recovery pass, preserving at-least-once rather than assuming a crashed
        process completed work.
        """
        if self._store is None:
            raise DomainError("scheduler has no persistent state store")
        with self._lock:
            if self._dag.tasks:
                raise DomainError("restore requires an empty scheduler")
            snapshots = self._store.latest_task_snapshots()
            pending = [task_from_snapshot(snapshot) for snapshot in snapshots]
            restored = 0
            while pending:
                before = len(pending)
                for task in tuple(pending):
                    if task.dependencies <= self._dag.tasks.keys():
                        self._dag.add(task)
                        if task.idempotency_key and task.state == TaskState.COMPLETED:
                            self._completed_keys.add(task.idempotency_key)
                        if task.attempt_count and task.worker_id:
                            attempt = TaskAttempt(
                                task.task_id, task.attempt_count, task.worker_id, task.state
                            )
                            self._attempts[task.task_id] = [attempt]
                        pending.remove(task)
                        restored += 1
                if len(pending) == before:
                    raise DomainError("persisted task graph has unresolved dependencies")
            return restored

    @property
    def tasks(self) -> dict[str, Task]:
        with self._lock:
            return self._dag.tasks

    @property
    def attempts(self) -> dict[str, list[TaskAttempt]]:
        with self._lock:
            return {task_id: list(items) for task_id, items in self._attempts.items()}

    def register_worker(
        self,
        worker_id: str,
        capabilities: frozenset[str],
        max_concurrency: int,
        metadata: dict[str, str] | None = None,
    ) -> Worker:
        if max_concurrency < 1:
            raise DomainError("worker concurrency must be positive")
        with self._lock:
            now = self._clock()
            worker = Worker(worker_id, capabilities, max_concurrency, now, now, metadata or {})
            self._workers[worker_id] = worker
            self._emit(
                "worker.registered", {"worker_id": worker_id, "capabilities": sorted(capabilities)}
            )
            return worker

    def heartbeat(self, worker_id: str) -> None:
        with self._lock:
            worker = self._worker(worker_id)
            worker.last_heartbeat = self._clock()
            self._emit("worker.heartbeat", {"worker_id": worker_id})

    def drain_worker(self, worker_id: str) -> None:
        with self._lock:
            self._worker(worker_id).draining = True
            self._emit("worker.draining", {"worker_id": worker_id})

    def enqueue(self, task: Task) -> Task:
        with self._lock:
            unfinished = sum(
                item.state not in {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}
                for item in self._dag.tasks.values()
            )
            if unfinished >= self._max_queued_tasks:
                raise DomainError("scheduler backpressure limit reached")
            self._dag.add(task)
            if task.idempotency_key and task.idempotency_key in self._completed_keys:
                self._transition(task, TaskState.COMPLETED, "duplicate idempotency key")
            else:
                self._persist(task, "created")
                self._emit("task.created", self._task_payload(task))
                self._refresh_ready()
            return task

    def lease_next(self, worker_id: str) -> Task | None:
        with self._lock:
            now = self._clock()
            self.recover(now)
            self._refresh_ready(now)
            worker = self._worker(worker_id)
            if worker.draining or now - worker.last_heartbeat > self._worker_timeout:
                return None
            running = sum(
                task.worker_id == worker_id and task.state in {TaskState.LEASED, TaskState.RUNNING}
                for task in self._dag.tasks.values()
            )
            if running >= worker.max_concurrency:
                return None
            candidates = [
                task
                for task in self._dag.tasks.values()
                if task.state == TaskState.READY
                and task.capability in worker.capabilities
                and (task.next_attempt_at is None or task.next_attempt_at <= now)
            ]
            if not candidates:
                return None
            task = sorted(
                candidates, key=lambda item: (-item.priority, item.created_at, item.task_id)
            )[0]
            task.worker_id = worker_id
            task.attempt_count += 1
            task.lease_expires_at = now + min(self._lease_duration, task.timeout)
            attempt = TaskAttempt(task.task_id, task.attempt_count, worker_id)
            self._attempts.setdefault(task.task_id, []).append(attempt)
            self._transition(task, TaskState.LEASED, "worker lease")
            self._emit(
                "task.leased", {**self._task_payload(task), "attempt_id": attempt.attempt_id}
            )
            return task

    def start(self, worker_id: str, task_id: str) -> Task:
        with self._lock:
            task = self._owned_task(worker_id, task_id)
            self._transition(task, TaskState.RUNNING, "worker started task")
            self._attempt(task).state = TaskState.RUNNING
            self._emit("task.running", self._task_payload(task))
            return task

    def renew_lease(self, worker_id: str, task_id: str) -> None:
        with self._lock:
            task = self._owned_task(worker_id, task_id)
            if task.state not in {TaskState.LEASED, TaskState.RUNNING}:
                raise DomainError("only active tasks can renew leases")
            task.lease_expires_at = self._clock() + min(self._lease_duration, task.timeout)
            task.updated_at = self._clock()
            self._persist(task, "lease renewed")

    def complete(self, worker_id: str, task_id: str, outputs: dict[str, Any]) -> Task:
        with self._lock:
            task = self._owned_task(worker_id, task_id)
            if task.state not in {TaskState.LEASED, TaskState.RUNNING}:
                raise DomainError("only active tasks can complete")
            task.outputs = outputs
            task.lease_expires_at = None
            if task.idempotency_key:
                self._completed_keys.add(task.idempotency_key)
            self._transition(task, TaskState.COMPLETED, "worker completed task")
            attempt = self._attempt(task)
            attempt.state = TaskState.COMPLETED
            attempt.completed_at = self._clock()
            self._emit("task.completed", self._task_payload(task))
            self._refresh_ready()
            return task

    def fail(self, worker_id: str, task_id: str, error: str) -> Task:
        with self._lock:
            task = self._owned_task(worker_id, task_id)
            return self._retry_or_fail(task, error)

    def cancel(self, task_id: str, reason: str = "cancelled by caller") -> Task:
        with self._lock:
            task = self._get_task(task_id)
            if task.state in {TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELLED}:
                return task
            was_active = task.state in {TaskState.LEASED, TaskState.RUNNING}
            self._transition(task, TaskState.CANCELLED, reason)
            if was_active:
                self._attempt(task).state = TaskState.CANCELLED
            self._emit("task.cancelled", self._task_payload(task))
            self._refresh_ready()
            return task

    def recover(self, now: datetime | None = None) -> list[str]:
        """Reclaim expired leases or task work owned by a non-heartbeating worker."""
        with self._lock:
            current = now or self._clock()
            recovered: list[str] = []
            dead_workers = {
                worker_id
                for worker_id, worker in self._workers.items()
                if current - worker.last_heartbeat > self._worker_timeout
            }
            for task in self._dag.tasks.values():
                expired = task.lease_expires_at is not None and task.lease_expires_at <= current
                owner_dead = task.worker_id in dead_workers
                owner_unknown = task.worker_id is not None and task.worker_id not in self._workers
                if task.state in {TaskState.LEASED, TaskState.RUNNING} and (
                    expired or owner_dead or owner_unknown
                ):
                    reason = (
                        "lease expired"
                        if expired
                        else "worker heartbeat expired"
                        if owner_dead
                        else "worker unavailable after restart"
                    )
                    self._retry_or_fail(task, reason)
                    recovered.append(task.task_id)
            self._refresh_ready(current)
            return recovered

    def metrics(self) -> dict[str, int]:
        with self._lock:
            states = {state: 0 for state in TaskState}
            for task in self._dag.tasks.values():
                states[task.state] += 1
            active = sum(not worker.draining for worker in self._workers.values())
            return {
                "workers_active": active,
                "tasks_completed": states[TaskState.COMPLETED],
                "tasks_failed": states[TaskState.FAILED],
                "tasks_ready": states[TaskState.READY],
                "tasks_active": states[TaskState.LEASED] + states[TaskState.RUNNING],
                "retry_count": sum(
                    max(0, task.attempt_count - 1) for task in self._dag.tasks.values()
                ),
            }

    def _refresh_ready(self, now: datetime | None = None) -> None:
        current = now or self._clock()
        for task in self._dag.tasks.values():
            dependency_states = [
                self._get_task(dependency).state for dependency in task.dependencies
            ]
            if task.state == TaskState.CREATED and all(
                state == TaskState.COMPLETED for state in dependency_states
            ):
                self._transition(task, TaskState.READY, "dependencies satisfied")
                self._emit("task.ready", self._task_payload(task))
            elif (
                task.state == TaskState.RETRYING
                and task.next_attempt_at is not None
                and task.next_attempt_at <= current
            ):
                self._transition(task, TaskState.READY, "retry backoff elapsed")
                self._emit("task.ready", self._task_payload(task))
            elif task.state == TaskState.CREATED and any(
                state in {TaskState.FAILED, TaskState.CANCELLED} for state in dependency_states
            ):
                self._transition(task, TaskState.CANCELLED, "dependency did not complete")
                self._emit("task.cancelled", self._task_payload(task))

    def _retry_or_fail(self, task: Task, error: str) -> Task:
        if task.state not in {TaskState.LEASED, TaskState.RUNNING}:
            raise DomainError("only active tasks can fail")
        attempt = self._attempt(task)
        attempt.error = error
        attempt.completed_at = self._clock()
        task.worker_id = None
        task.lease_expires_at = None
        if task.attempt_count < task.retry_policy.max_attempts:
            task.next_attempt_at = self._clock() + task.retry_policy.backoff
            self._transition(task, TaskState.RETRYING, error)
            attempt.state = TaskState.RETRYING
            self._emit("task.retrying", {**self._task_payload(task), "error": error})
        else:
            self._transition(task, TaskState.FAILED, error)
            attempt.state = TaskState.FAILED
            self._emit("task.failed", {**self._task_payload(task), "error": error})
        return task

    def _transition(self, task: Task, target: TaskState, reason: str) -> None:
        if target not in self._TRANSITIONS[task.state]:
            raise DomainError(f"invalid task transition: {task.state} -> {target}")
        task.state = target
        task.updated_at = self._clock()
        self._persist(task, reason)

    def _persist(self, task: Task, reason: str) -> None:
        if self._store:
            self._store.record_task(task, reason)

    def _emit(self, event_type: str, payload: dict[str, Any]) -> None:
        event = Event(
            event_type=event_type,
            payload=payload,
            producer="scheduler",
            trace_id=str(payload.get("trace_id", payload.get("task_id", "scheduler"))),
            correlation_id=str(payload.get("task_id", payload.get("worker_id", "scheduler"))),
        )
        self._bus.publish(event)
        if self._store:
            self._store.record_event(event)

    def _get_task(self, task_id: str) -> Task:
        try:
            return self._dag.tasks[task_id]
        except KeyError as exc:
            raise DomainError(f"unknown task: {task_id}") from exc

    def _worker(self, worker_id: str) -> Worker:
        try:
            return self._workers[worker_id]
        except KeyError as exc:
            raise DomainError(f"unknown worker: {worker_id}") from exc

    def _owned_task(self, worker_id: str, task_id: str) -> Task:
        self._worker(worker_id)
        task = self._get_task(task_id)
        if task.worker_id != worker_id:
            raise DomainError("worker does not own this task lease")
        return task

    def _attempt(self, task: Task) -> TaskAttempt:
        try:
            return self._attempts[task.task_id][-1]
        except (KeyError, IndexError) as exc:
            raise DomainError("task has no active attempt") from exc

    @staticmethod
    def _task_payload(task: Task) -> dict[str, Any]:
        return {
            "task_id": task.task_id,
            "state": task.state.value,
            "capability": task.capability,
            "worker_id": task.worker_id,
            "attempt_count": task.attempt_count,
            "parent_task_id": task.parent_task_id,
        }
