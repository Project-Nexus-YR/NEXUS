"""TaskQueue port implemented over atomic TaskStore operations."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Protocol

from .model import DistributedTask, FailureClass
from .store import TaskStore


class TaskQueue(Protocol):
    def enqueue(self, task: DistributedTask) -> DistributedTask: ...

    def claim(
        self,
        candidate_task_ids: list[str],
        worker_id: str,
        now: datetime,
        lease_duration: timedelta,
    ) -> DistributedTask | None: ...

    def ack(
        self,
        task_id: str,
        worker_id: str,
        lease_id: str,
        now: datetime,
        result_ref: str | None,
        attempt_id: str | None = None,
    ) -> DistributedTask: ...

    def nack(
        self,
        task_id: str,
        worker_id: str,
        lease_id: str,
        now: datetime,
        failure_class: FailureClass,
        error: str,
        retry_at: datetime | None,
        checkpoint_ref: str | None,
        attempt_id: str | None = None,
    ) -> DistributedTask: ...

    def requeue(self, task_id: str, now: datetime) -> DistributedTask: ...


class StoreBackedTaskQueue:
    def __init__(self, store: TaskStore) -> None:
        self._store = store

    def enqueue(self, task: DistributedTask) -> DistributedTask:
        return self._store.create(task)

    def claim(
        self,
        candidate_task_ids: list[str],
        worker_id: str,
        now: datetime,
        lease_duration: timedelta,
    ) -> DistributedTask | None:
        for task_id in candidate_task_ids:
            claimed = self._store.claim(task_id, worker_id, now, lease_duration)
            if claimed is not None:
                return claimed
        return None

    def ack(
        self,
        task_id: str,
        worker_id: str,
        lease_id: str,
        now: datetime,
        result_ref: str | None,
        attempt_id: str | None = None,
    ) -> DistributedTask:
        return self._store.complete(task_id, worker_id, lease_id, now, result_ref, attempt_id)

    def nack(
        self,
        task_id: str,
        worker_id: str,
        lease_id: str,
        now: datetime,
        failure_class: FailureClass,
        error: str,
        retry_at: datetime | None,
        checkpoint_ref: str | None,
        attempt_id: str | None = None,
    ) -> DistributedTask:
        return self._store.fail(
            task_id,
            worker_id,
            lease_id,
            now,
            failure_class,
            error,
            retry_at,
            checkpoint_ref,
            attempt_id,
        )

    def requeue(self, task_id: str, now: datetime) -> DistributedTask:
        return self._store.manual_retry(task_id, now)
