"""Worker liveness, capability identity, draining, and capacity accounting."""

from __future__ import annotations

from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import StrEnum
from threading import RLock

from ..models import DomainError
from .security import WorkerIdentity


class WorkerStatus(StrEnum):
    STARTING = "STARTING"
    READY = "READY"
    BUSY = "BUSY"
    DRAINING = "DRAINING"
    UNHEALTHY = "UNHEALTHY"
    STOPPED = "STOPPED"


@dataclass(slots=True)
class WorkerRecord:
    identity: WorkerIdentity
    version: str
    started_at: datetime
    last_heartbeat: datetime
    max_concurrency: int
    status: WorkerStatus = WorkerStatus.STARTING
    current_tasks: set[str] = field(default_factory=set)
    revision: int = 0

    @property
    def available_slots(self) -> int:
        return max(0, self.max_concurrency - len(self.current_tasks))


class WorkerStore(ABC):
    @abstractmethod
    def put(self, worker: WorkerRecord) -> WorkerRecord: ...

    @abstractmethod
    def get(self, worker_id: str) -> WorkerRecord | None: ...

    @abstractmethod
    def list(self) -> list[WorkerRecord]: ...


class InMemoryWorkerStore(WorkerStore):
    def __init__(self) -> None:
        self._workers: dict[str, WorkerRecord] = {}
        self._lock = RLock()

    def put(self, worker: WorkerRecord) -> WorkerRecord:
        with self._lock:
            value = deepcopy(worker)
            value.revision += 1
            self._workers[value.identity.worker_id] = value
            return deepcopy(value)

    def get(self, worker_id: str) -> WorkerRecord | None:
        with self._lock:
            value = self._workers.get(worker_id)
            return None if value is None else deepcopy(value)

    def list(self) -> list[WorkerRecord]:
        with self._lock:
            return [deepcopy(worker) for worker in self._workers.values()]


class WorkerRegistry:
    def __init__(self, store: WorkerStore | None = None) -> None:
        self._store = store or InMemoryWorkerStore()
        self._lock = RLock()

    def register(
        self,
        identity: WorkerIdentity,
        version: str,
        max_concurrency: int,
        now: datetime,
    ) -> WorkerRecord:
        if max_concurrency < 1:
            raise DomainError("worker capacity must be positive")
        with self._lock:
            existing = self._store.get(identity.worker_id)
            if existing is not None and existing.status not in {
                WorkerStatus.STOPPED,
                WorkerStatus.UNHEALTHY,
            }:
                if existing.identity != identity:
                    raise DomainError("worker identity changed during registration")
                existing.last_heartbeat = now
                existing.version = version
                existing.max_concurrency = max_concurrency
                existing.status = (
                    WorkerStatus.BUSY if existing.current_tasks else WorkerStatus.READY
                )
                return self._store.put(existing)
            worker = WorkerRecord(
                identity=identity,
                version=version,
                started_at=now,
                last_heartbeat=now,
                max_concurrency=max_concurrency,
                status=WorkerStatus.READY,
            )
            return self._store.put(worker)

    def heartbeat(self, worker_id: str, now: datetime) -> WorkerRecord:
        with self._lock:
            worker = self.require(worker_id)
            if worker.status in {WorkerStatus.UNHEALTHY, WorkerStatus.STOPPED}:
                raise DomainError("inactive worker must register again")
            worker.last_heartbeat = now
            if worker.status != WorkerStatus.DRAINING:
                worker.status = WorkerStatus.BUSY if worker.current_tasks else WorkerStatus.READY
            return self._store.put(worker)

    def assign(self, worker_id: str, task_id: str) -> WorkerRecord:
        with self._lock:
            worker = self.require(worker_id)
            if worker.status not in {WorkerStatus.READY, WorkerStatus.BUSY}:
                raise DomainError("worker is not accepting tasks")
            if worker.available_slots < 1:
                raise DomainError("worker has no available capacity")
            worker.current_tasks.add(task_id)
            worker.status = WorkerStatus.BUSY
            return self._store.put(worker)

    def release(self, worker_id: str | None, task_id: str) -> WorkerRecord | None:
        if worker_id is None:
            return None
        with self._lock:
            worker = self._store.get(worker_id)
            if worker is None:
                return None
            worker.current_tasks.discard(task_id)
            if worker.status == WorkerStatus.DRAINING and not worker.current_tasks:
                worker.status = WorkerStatus.STOPPED
            elif worker.status not in {WorkerStatus.UNHEALTHY, WorkerStatus.STOPPED}:
                worker.status = WorkerStatus.BUSY if worker.current_tasks else WorkerStatus.READY
            return self._store.put(worker)

    def drain(self, worker_id: str) -> WorkerRecord:
        with self._lock:
            worker = self.require(worker_id)
            worker.status = WorkerStatus.DRAINING if worker.current_tasks else WorkerStatus.STOPPED
            return self._store.put(worker)

    def mark_unhealthy(self, now: datetime, threshold: timedelta) -> list[WorkerRecord]:
        unhealthy: list[WorkerRecord] = []
        with self._lock:
            for worker in self._store.list():
                if worker.status not in {WorkerStatus.STOPPED, WorkerStatus.UNHEALTHY} and (
                    now - worker.last_heartbeat >= threshold
                ):
                    worker.status = WorkerStatus.UNHEALTHY
                    unhealthy.append(self._store.put(worker))
        return unhealthy

    def require(self, worker_id: str) -> WorkerRecord:
        worker = self._store.get(worker_id)
        if worker is None:
            raise DomainError(f"unknown worker: {worker_id}")
        return worker

    def list(self) -> list[WorkerRecord]:
        return self._store.list()
