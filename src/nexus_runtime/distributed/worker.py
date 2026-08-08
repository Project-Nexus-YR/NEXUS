"""Worker loop that delegates execution to the existing Agent Harness boundary."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from enum import StrEnum
from threading import Event as ThreadEvent
from threading import Thread
from types import MappingProxyType
from typing import Any, Protocol

from ..models import DomainError
from .coordinator import Coordinator
from .model import DistributedTask, DistributedTaskState, FailureClass
from .security import WorkerIdentity


class HarnessStatus(StrEnum):
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"


@dataclass(frozen=True, slots=True)
class HarnessOutcome:
    status: HarnessStatus
    result_ref: str | None = None
    checkpoint_ref: str | None = None
    failure_class: FailureClass = FailureClass.TRANSIENT
    error: str = ""


@dataclass(frozen=True, slots=True)
class HarnessExecutionContext:
    """Immutable lineage supplied by the worker to the Agent Harness."""

    run_id: str
    correlation_id: str
    task_id: str
    attempt_id: str
    lease_id: str
    worker_id: str
    metadata: Mapping[str, Any]


class Harness(Protocol):
    """Small integration port; implementations own all AgentRun execution details."""

    def execute_or_resume(
        self,
        context: HarnessExecutionContext,
        cancellation_requested: Callable[[], bool],
    ) -> HarnessOutcome: ...

    def cancel_run(self, run_id: str) -> None: ...


class Worker:
    def __init__(
        self,
        coordinator: Coordinator,
        harness: Harness,
        identity: WorkerIdentity,
        *,
        version: str = "dev",
        max_concurrency: int = 1,
    ) -> None:
        self._coordinator = coordinator
        self._harness = harness
        self.identity = identity
        self._version = version
        self._max_concurrency = max_concurrency
        self._registered = False
        self._crashed = False

    def register(self) -> None:
        self._coordinator.register_worker(
            self.identity,
            version=self._version,
            max_concurrency=self._max_concurrency,
        )
        self._registered = True
        self._crashed = False

    def heartbeat(self) -> None:
        self._ensure_active()
        self._coordinator.heartbeat(self.identity)

    def poll_once(self) -> DistributedTask | None:
        self._ensure_active()
        self.heartbeat()
        task = self._coordinator.claim_task(self.identity)
        if task is None:
            return None
        if task.lease_id is None:
            raise DomainError("claimed task has no lease")
        if not task.attempts:
            raise DomainError("claimed task has no attempt")
        lease_id = task.lease_id
        attempt_id = task.attempts[-1].attempt_id
        self._coordinator.start_task(self.identity, task.task_id, lease_id)
        heartbeat_stop = ThreadEvent()
        heartbeat_thread = Thread(
            target=self._heartbeat_loop,
            args=(heartbeat_stop,),
            name=f"heartbeat-{self.identity.worker_id}",
            daemon=True,
        )
        heartbeat_thread.start()

        def cancellation_requested() -> bool:
            return self._coordinator.cancellation_requested(task.task_id, lease_id)

        try:
            outcome = self._harness.execute_or_resume(
                HarnessExecutionContext(
                    run_id=task.run_id,
                    correlation_id=task.correlation_id,
                    task_id=task.task_id,
                    attempt_id=attempt_id,
                    lease_id=lease_id,
                    worker_id=self.identity.worker_id,
                    metadata=MappingProxyType(dict(task.metadata)),
                ),
                cancellation_requested,
            )
        except Exception as exc:
            heartbeat_stop.set()
            heartbeat_thread.join(timeout=1)
            self._coordinator.fail_task(
                self.identity,
                task.task_id,
                lease_id,
                FailureClass.TRANSIENT,
                f"harness execution raised: {exc!r}",
                attempt_id=attempt_id,
            )
            return self._coordinator.require_task(task.task_id)
        heartbeat_stop.set()
        heartbeat_thread.join(timeout=1)
        current = self._coordinator.require_task(task.task_id)
        if current.state == DistributedTaskState.CANCEL_REQUESTED:
            self._harness.cancel_run(task.run_id)
            return self._coordinator.acknowledge_cancellation(
                self.identity, task.task_id, lease_id, attempt_id
            )
        if outcome.status == HarnessStatus.SUCCEEDED:
            return self._coordinator.complete_task(
                self.identity,
                task.task_id,
                lease_id,
                outcome.result_ref,
                attempt_id,
            )
        if outcome.status == HarnessStatus.CANCELLED:
            return self._coordinator.fail_task(
                self.identity,
                task.task_id,
                lease_id,
                FailureClass.CANCELLED,
                outcome.error or "harness cancelled",
                outcome.checkpoint_ref,
                attempt_id,
            )
        return self._coordinator.fail_task(
            self.identity,
            task.task_id,
            lease_id,
            outcome.failure_class,
            outcome.error or "harness failed",
            outcome.checkpoint_ref,
            attempt_id,
        )

    def drain(self, principal: str) -> None:
        self._coordinator.drain_worker(self.identity.worker_id, principal)

    def crash(self) -> None:
        """Failure injection: stop all worker activity without releasing its lease."""
        self._crashed = True

    def _ensure_active(self) -> None:
        if not self._registered:
            raise DomainError("worker is not registered")
        if self._crashed:
            raise DomainError("worker process has crashed")

    def _heartbeat_loop(self, stop: ThreadEvent) -> None:
        interval = self._coordinator.config.heartbeat_interval.total_seconds()
        while not stop.wait(interval):
            try:
                self.heartbeat()
            except DomainError:
                return
