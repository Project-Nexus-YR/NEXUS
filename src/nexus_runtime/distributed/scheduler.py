"""Replaceable capability-aware priority scheduling policy."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Protocol

from ..models import DomainError
from .model import DistributedTask
from .worker_registry import WorkerRecord, WorkerStatus


class SchedulingPolicy(Protocol):
    def rank(
        self, tasks: list[DistributedTask], worker: WorkerRecord, now: datetime
    ) -> list[DistributedTask]: ...


class PriorityAgingScheduler:
    def __init__(
        self,
        aging_interval: timedelta = timedelta(minutes=1),
        aging_step: int = 1,
    ) -> None:
        if aging_interval <= timedelta(0) or aging_step < 0:
            raise DomainError("scheduler aging configuration is invalid")
        self._aging_interval = aging_interval
        self._aging_step = aging_step

    def rank(
        self, tasks: list[DistributedTask], worker: WorkerRecord, now: datetime
    ) -> list[DistributedTask]:
        if worker.status not in {WorkerStatus.READY, WorkerStatus.BUSY}:
            return []
        if worker.available_slots < 1:
            return []
        eligible = [
            task
            for task in tasks
            if task.available_at <= now
            and (task.deadline is None or task.deadline > now)
            and task.required_capabilities <= worker.identity.capabilities
        ]

        def key(task: DistributedTask) -> tuple[float, datetime, str]:
            waited = max(0.0, (now - task.created_at).total_seconds())
            age_units = int(waited // self._aging_interval.total_seconds())
            effective_priority = int(task.priority) + age_units * self._aging_step
            return (-float(effective_priority), task.created_at, task.task_id)

        return sorted(eligible, key=key)
