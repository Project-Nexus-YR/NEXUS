"""Worker process adapter that keeps execution separate from scheduling semantics."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .models import DomainError, Task
from .scheduler import Scheduler

TaskHandler = Callable[[Task], dict[str, Any]]


class WorkerProcess:
    """A single cooperative worker; real process managers can host this adapter."""

    def __init__(
        self,
        scheduler: Scheduler,
        worker_id: str,
        capabilities: frozenset[str],
        handlers: dict[str, TaskHandler],
        max_concurrency: int = 1,
    ) -> None:
        self._scheduler = scheduler
        self.worker_id = worker_id
        self._handlers = handlers
        self._scheduler.register_worker(worker_id, capabilities, max_concurrency)

    def heartbeat(self) -> None:
        self._scheduler.heartbeat(self.worker_id)

    def run_once(self) -> Task | None:
        task = self._scheduler.lease_next(self.worker_id)
        if task is None:
            return None
        self._scheduler.start(self.worker_id, task.task_id)
        try:
            handler = self._handlers[task.capability]
        except KeyError:
            self._scheduler.fail(self.worker_id, task.task_id, "no handler for capability")
            return task
        try:
            output = handler(task)
        except Exception as exc:
            self._scheduler.fail(self.worker_id, task.task_id, f"handler failure: {exc!r}")
            return task
        if not isinstance(output, dict):
            raise DomainError("task handlers must return structured object outputs")
        self._scheduler.complete(self.worker_id, task.task_id, output)
        return task

    def shutdown(self) -> None:
        """Stop accepting work while allowing the scheduler to reclaim active leases."""
        self._scheduler.drain_worker(self.worker_id)
