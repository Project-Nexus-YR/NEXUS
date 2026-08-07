"""Transport-independent application service for the distributed runtime."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .coordinator import Coordinator
from .model import DistributedTask, DistributedTaskState, RetryPolicy, TaskPriority
from .security import WorkerIdentity
from .worker_registry import WorkerRecord


class RuntimeApplication:
    def __init__(self, coordinator: Coordinator) -> None:
        self._coordinator = coordinator

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
        return self._coordinator.submit_task(
            run_id,
            correlation_id=correlation_id,
            priority=priority,
            required_capabilities=required_capabilities,
            retry_policy=retry_policy,
            metadata=metadata,
            available_at=available_at,
            deadline=deadline,
        )

    def get_task(self, task_id: str) -> DistributedTask:
        return self._coordinator.require_task(task_id)

    def cancel_task(self, task_id: str, principal: str) -> DistributedTask:
        return self._coordinator.cancel_task(task_id, principal)

    def retry_task(self, task_id: str, principal: str) -> DistributedTask:
        return self._coordinator.retry_task(task_id, principal)

    def list_tasks(
        self, states: frozenset[DistributedTaskState] | None = None
    ) -> list[DistributedTask]:
        return self._coordinator.list_tasks(states)

    def register_worker(
        self, identity: WorkerIdentity, version: str, max_concurrency: int
    ) -> WorkerRecord:
        return self._coordinator.register_worker(
            identity, version=version, max_concurrency=max_concurrency
        )

    def heartbeat(self, identity: WorkerIdentity) -> WorkerRecord:
        return self._coordinator.heartbeat(identity)

    def drain_worker(self, worker_id: str, principal: str) -> WorkerRecord:
        return self._coordinator.drain_worker(worker_id, principal)

    def list_workers(self) -> list[WorkerRecord]:
        return self._coordinator.list_workers()

    def get_queue_stats(self) -> dict[str, int]:
        return self._coordinator.queue_stats()

    def get_runtime_stats(self) -> dict[str, object]:
        return self._coordinator.runtime_stats()

    def recover(self) -> list[DistributedTask]:
        return self._coordinator.recover()
