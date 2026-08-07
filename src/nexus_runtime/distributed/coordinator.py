"""Coordinator: task lifecycle, workers, leases, retries, and recovery."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from threading import RLock
from typing import Any

from ..events import Event, EventBus, InMemoryEventBus
from ..models import DomainError
from .clock import Clock, SystemClock
from .metrics import InMemoryMetrics, MetricsSink
from .model import (
    TERMINAL_TASK_STATES,
    DistributedTask,
    DistributedTaskState,
    FailureClass,
    RetryPolicy,
    TaskPriority,
)
from .queue import StoreBackedTaskQueue, TaskQueue
from .scheduler import PriorityAgingScheduler, SchedulingPolicy
from .security import (
    AllowAllRuntimeAuthorizer,
    RuntimeAuthorizer,
    TrustedLocalWorkerAuthenticator,
    WorkerAuthenticator,
    WorkerIdentity,
)
from .store import TaskStore
from .worker_registry import WorkerRecord, WorkerRegistry, WorkerStatus


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    lease_duration: timedelta = timedelta(seconds=30)
    heartbeat_interval: timedelta = timedelta(seconds=10)
    worker_failure_threshold: timedelta = timedelta(seconds=45)
    max_pending_tasks: int = 10_000
    aging_interval: timedelta = timedelta(minutes=1)
    aging_step: int = 1

    def __post_init__(self) -> None:
        if min(
            self.lease_duration,
            self.heartbeat_interval,
            self.worker_failure_threshold,
            self.aging_interval,
        ) <= timedelta(0):
            raise DomainError("runtime durations must be positive")
        if self.heartbeat_interval >= self.worker_failure_threshold:
            raise DomainError("worker failure threshold must exceed heartbeat interval")
        if self.max_pending_tasks < 1 or self.aging_step < 0:
            raise DomainError("runtime capacity and aging step must be valid")


class Coordinator:
    """Coordinates durable work but never executes an AgentRun itself."""

    def __init__(
        self,
        task_store: TaskStore,
        *,
        task_queue: TaskQueue | None = None,
        workers: WorkerRegistry | None = None,
        scheduler: SchedulingPolicy | None = None,
        clock: Clock | None = None,
        event_bus: EventBus | None = None,
        metrics: MetricsSink | None = None,
        worker_authenticator: WorkerAuthenticator | None = None,
        authorizer: RuntimeAuthorizer | None = None,
        config: RuntimeConfig | None = None,
    ) -> None:
        self.task_store = task_store
        self.config = config or RuntimeConfig()
        self._queue = task_queue or StoreBackedTaskQueue(task_store)
        self._workers = workers or WorkerRegistry()
        self._scheduler = scheduler or PriorityAgingScheduler(
            self.config.aging_interval, self.config.aging_step
        )
        self._clock = clock or SystemClock()
        self._events = event_bus or InMemoryEventBus()
        self._metrics = metrics or InMemoryMetrics()
        self._worker_auth = worker_authenticator or TrustedLocalWorkerAuthenticator()
        self._authorizer = authorizer or AllowAllRuntimeAuthorizer()
        self._lock = RLock()
        self._candidate_cache: dict[tuple[frozenset[str], int], deque[str]] = {}

    def submit_task(
        self,
        run_id: str,
        *,
        correlation_id: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        required_capabilities: frozenset[str] = frozenset(),
        retry_policy: RetryPolicy | None = None,
        metadata: dict[str, Any] | None = None,
        available_at: datetime | None = None,
        deadline: datetime | None = None,
    ) -> DistributedTask:
        with self._lock:
            pending = sum(
                count
                for state, count in self.task_store.counts().items()
                if DistributedTaskState(state) not in TERMINAL_TASK_STATES
            )
            if pending >= self.config.max_pending_tasks:
                self._metrics.increment("submission_rejected_backpressure")
                raise DomainError("distributed runtime backpressure limit reached")
            now = self._clock.now()
            task = DistributedTask(
                run_id=run_id,
                correlation_id=correlation_id,
                required_capabilities=required_capabilities,
                priority=priority,
                retry_policy=retry_policy or RetryPolicy(),
                metadata=metadata or {},
                available_at=available_at or now,
                deadline=deadline,
                created_at=now,
                updated_at=now,
            )
            created = self._queue.enqueue(task)
            self._candidate_cache.clear()
            self._metrics.increment("tasks_submitted")
            self._emit("task.created", created)
            self._emit("task.queued", created)
            return created

    def register_worker(
        self,
        identity: WorkerIdentity,
        *,
        version: str,
        max_concurrency: int,
    ) -> WorkerRecord:
        self._worker_auth.verify(identity)
        worker = self._workers.register(identity, version, max_concurrency, self._clock.now())
        self._emit_worker("worker.registered", worker)
        return worker

    def heartbeat(self, identity: WorkerIdentity) -> WorkerRecord:
        self._require_worker_identity(identity)
        now = self._clock.now()
        worker = self._workers.heartbeat(identity.worker_id, now)
        for task_id in tuple(worker.current_tasks):
            task = self.task_store.get(task_id)
            if task is None or task.lease is None:
                self._workers.release(identity.worker_id, task_id)
                continue
            try:
                self.task_store.renew_lease(
                    task_id,
                    identity.worker_id,
                    task.lease.lease_id,
                    now,
                    self.config.lease_duration,
                )
            except DomainError:
                self._workers.release(identity.worker_id, task_id)
        self._metrics.increment("worker_heartbeats")
        self._emit_worker("worker.heartbeat", worker)
        return self._workers.require(identity.worker_id)

    def claim_task(self, identity: WorkerIdentity) -> DistributedTask | None:
        self._require_worker_identity(identity)
        with self._lock:
            worker = self._workers.require(identity.worker_id)
            if worker.status not in {WorkerStatus.READY, WorkerStatus.BUSY}:
                return None
            if worker.available_slots < 1:
                return None
            now = self._clock.now()
            age_bucket = int(now.timestamp() // self.config.aging_interval.total_seconds())
            cache_key = (worker.identity.capabilities, age_bucket)
            candidate_ids = self._candidate_cache.get(cache_key)
            if candidate_ids is None:
                candidates = self._scheduler.rank(self.task_store.list_queued(now), worker, now)
                candidate_ids = deque(task.task_id for task in candidates)
                self._candidate_cache[cache_key] = candidate_ids
            claimed = None
            while candidate_ids and claimed is None:
                claimed = self._queue.claim(
                    [candidate_ids.popleft()],
                    identity.worker_id,
                    now,
                    self.config.lease_duration,
                )
            if claimed is None:
                return None
            self._workers.assign(identity.worker_id, claimed.task_id)
            self._metrics.increment("tasks_claimed")
            self._metrics.observe(
                "task_queue_wait_seconds",
                (self._clock.now() - claimed.created_at).total_seconds(),
            )
            self._emit("task.claimed", claimed)
            return claimed

    def start_task(self, identity: WorkerIdentity, task_id: str, lease_id: str) -> DistributedTask:
        self._require_worker_identity(identity)
        task = self.task_store.mark_running(
            task_id, identity.worker_id, lease_id, self._clock.now()
        )
        self._metrics.increment("tasks_started")
        self._emit("task.started", task)
        return task

    def complete_task(
        self,
        identity: WorkerIdentity,
        task_id: str,
        lease_id: str,
        result_ref: str | None,
        attempt_id: str | None = None,
    ) -> DistributedTask:
        self._require_worker_identity(identity)
        task = self._queue.ack(
            task_id,
            identity.worker_id,
            lease_id,
            self._clock.now(),
            result_ref,
            attempt_id,
        )
        self._workers.release(identity.worker_id, task_id)
        self._metrics.increment("tasks_succeeded")
        self._metrics.observe(
            "task_latency_seconds", (self._clock.now() - task.created_at).total_seconds()
        )
        if task.attempts:
            self._metrics.observe(
                "task_execution_seconds",
                (self._clock.now() - task.attempts[-1].started_at).total_seconds(),
            )
        self._emit("task.completed", task)
        return task

    def fail_task(
        self,
        identity: WorkerIdentity,
        task_id: str,
        lease_id: str,
        failure_class: FailureClass,
        error: str,
        checkpoint_ref: str | None = None,
        attempt_id: str | None = None,
    ) -> DistributedTask:
        self._require_worker_identity(identity)
        current = self.require_task(task_id)
        retry_at = self._retry_at(current, failure_class)
        task = self._queue.nack(
            task_id,
            identity.worker_id,
            lease_id,
            self._clock.now(),
            failure_class,
            error,
            retry_at,
            checkpoint_ref,
            attempt_id,
        )
        self._workers.release(identity.worker_id, task_id)
        self._metrics.increment("tasks_failed")
        self._emit("task.failed", task, {"failure_class": failure_class.value})
        if task.state == DistributedTaskState.RETRY_WAIT:
            self._metrics.increment("task_retries")
            self._emit("task.retried", task)
        elif task.state == DistributedTaskState.DEAD_LETTERED:
            self._metrics.increment("tasks_dead_lettered")
            self._emit("task.dead_lettered", task)
        elif task.state == DistributedTaskState.CANCELLED:
            self._emit("task.cancelled", task)
        return task

    def cancel_task(self, task_id: str, principal: str) -> DistributedTask:
        self._authorizer.authorize(principal, "runtime.task.cancel", task_id)
        before = self.require_task(task_id)
        task = self.task_store.request_cancel(task_id, self._clock.now())
        self._candidate_cache.clear()
        self._emit("task.cancel_requested", task, {"principal": principal})
        if task.state == DistributedTaskState.CANCELLED:
            self._workers.release(before.worker_id, task_id)
            self._metrics.increment("tasks_cancelled")
            self._emit("task.cancelled", task)
        return task

    def acknowledge_cancellation(
        self,
        identity: WorkerIdentity,
        task_id: str,
        lease_id: str,
        attempt_id: str | None = None,
    ) -> DistributedTask:
        self._require_worker_identity(identity)
        task = self.task_store.acknowledge_cancel(
            task_id, identity.worker_id, lease_id, self._clock.now(), attempt_id
        )
        self._workers.release(identity.worker_id, task_id)
        self._metrics.increment("tasks_cancelled")
        self._emit("task.cancelled", task)
        return task

    def retry_task(self, task_id: str, principal: str) -> DistributedTask:
        self._authorizer.authorize(principal, "runtime.task.retry", task_id)
        task = self._queue.requeue(task_id, self._clock.now())
        self._candidate_cache.clear()
        self._metrics.increment("manual_retries")
        self._emit("task.retried", task, {"principal": principal, "manual": True})
        self._emit("task.queued", task)
        return task

    def drain_worker(self, worker_id: str, principal: str) -> WorkerRecord:
        self._authorizer.authorize(principal, "runtime.worker.drain", worker_id)
        worker = self._workers.drain(worker_id)
        event_type = "worker.stopped" if worker.status.value == "STOPPED" else "worker.draining"
        self._emit_worker(event_type, worker)
        return worker

    def recover(self) -> list[DistributedTask]:
        now = self._clock.now()
        recovered: list[DistributedTask] = []
        for worker in self._workers.mark_unhealthy(now, self.config.worker_failure_threshold):
            self._metrics.increment("workers_unhealthy")
            self._emit_worker("worker.unhealthy", worker)
        for task in self.task_store.list(frozenset({DistributedTaskState.RETRY_WAIT})):
            if task.available_at <= now:
                queued = self.task_store.advance_retry(task.task_id, now)
                recovered.append(queued)
                self._emit("task.queued", queued)
        for task in self.task_store.list(
            frozenset({DistributedTaskState.QUEUED, DistributedTaskState.RETRY_WAIT})
        ):
            if task.deadline is not None and task.deadline <= now:
                dead = self.task_store.expire_deadline(task.task_id, now)
                recovered.append(dead)
                self._metrics.increment("tasks_dead_lettered")
                self._emit("task.dead_lettered", dead, {"reason": "deadline"})
        for task in self.task_store.list_expired(now):
            owner = task.worker_id
            retry_at = self._retry_at(task, FailureClass.TRANSIENT)
            released = self.task_store.release_expired(task.task_id, now, retry_at)
            self._workers.release(owner, task.task_id)
            self._metrics.increment("lease_expirations")
            self._emit("task.lease_expired", released)
            if released.state == DistributedTaskState.RETRY_WAIT:
                self._metrics.increment("task_retries")
                self._emit("task.retried", released)
                if released.available_at <= now:
                    released = self.task_store.advance_retry(released.task_id, now)
                    self._emit("task.queued", released)
            elif released.state == DistributedTaskState.DEAD_LETTERED:
                self._metrics.increment("tasks_dead_lettered")
                self._emit("task.dead_lettered", released)
            elif released.state == DistributedTaskState.CANCELLED:
                self._metrics.increment("tasks_cancelled")
                self._emit("task.cancelled", released)
            recovered.append(released)
        if recovered:
            self._candidate_cache.clear()
        return recovered

    def cancellation_requested(self, task_id: str, lease_id: str) -> bool:
        task = self.require_task(task_id)
        return task.state == DistributedTaskState.CANCEL_REQUESTED and task.lease_id == lease_id

    def require_task(self, task_id: str) -> DistributedTask:
        task = self.task_store.get(task_id)
        if task is None:
            raise DomainError(f"unknown distributed task: {task_id}")
        return task

    def list_tasks(
        self, states: frozenset[DistributedTaskState] | None = None
    ) -> list[DistributedTask]:
        return self.task_store.list(states)

    def list_workers(self) -> list[WorkerRecord]:
        return self._workers.list()

    def queue_stats(self) -> dict[str, int]:
        counts = self.task_store.counts()
        counts["depth"] = sum(
            count
            for state, count in counts.items()
            if DistributedTaskState(state) not in TERMINAL_TASK_STATES
        )
        return counts

    def runtime_stats(self) -> dict[str, object]:
        workers = self._workers.list()
        total_capacity = sum(worker.max_concurrency for worker in workers)
        current = sum(len(worker.current_tasks) for worker in workers)
        return {
            "queue": self.queue_stats(),
            "workers": {
                "total": len(workers),
                "capacity": total_capacity,
                "current_concurrency": current,
                "utilization": current / total_capacity if total_capacity else 0.0,
            },
            "metrics": self._metrics.snapshot(),
        }

    def _retry_at(self, task: DistributedTask, failure_class: FailureClass) -> datetime | None:
        if (
            failure_class != FailureClass.TRANSIENT
            or task.attempt >= task.retry_policy.max_attempts
        ):
            return None
        return self._clock.now() + task.retry_policy.delay(task.task_id, task.attempt)

    def _require_worker_identity(self, identity: WorkerIdentity) -> WorkerRecord:
        self._worker_auth.verify(identity)
        worker = self._workers.require(identity.worker_id)
        if worker.identity != identity:
            raise DomainError("worker identity does not match registration")
        return worker

    def _emit(
        self,
        event_type: str,
        task: DistributedTask,
        extra: dict[str, Any] | None = None,
    ) -> None:
        attempt = task.attempts[-1] if task.attempts else None
        self._events.publish(
            Event(
                event_type=event_type,
                payload={
                    "task_id": task.task_id,
                    "run_id": task.run_id,
                    "state": task.state.value,
                    "worker_id": task.worker_id,
                    "attempt_id": None if attempt is None else attempt.attempt_id,
                    "lease_id": task.lease_id,
                    **(extra or {}),
                },
                producer="distributed-coordinator",
                trace_id=task.correlation_id,
                correlation_id=task.correlation_id,
            )
        )

    def _emit_worker(self, event_type: str, worker: WorkerRecord) -> None:
        self._events.publish(
            Event(
                event_type=event_type,
                payload={
                    "worker_id": worker.identity.worker_id,
                    "status": worker.status.value,
                    "capacity": worker.max_concurrency,
                    "current_concurrency": len(worker.current_tasks),
                },
                producer="distributed-coordinator",
                trace_id=worker.identity.worker_id,
                correlation_id=worker.identity.worker_id,
            )
        )
