"""Deterministic single-process simulation of real coordinator/worker interfaces."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import cast

from ..models import DomainError
from .clock import ManualClock
from .coordinator import Coordinator, RuntimeConfig
from .model import TERMINAL_TASK_STATES, DistributedTask, TaskPriority
from .security import WorkerIdentity
from .store import InMemoryTaskStore
from .worker import Harness, HarnessOutcome, HarnessStatus, Worker


class DeterministicHarness:
    def __init__(
        self,
        execute: Callable[[str, str, Callable[[], bool]], HarnessOutcome] | None = None,
    ) -> None:
        self.calls: list[str] = []
        self.cancelled: list[str] = []
        self._execute = execute

    def execute_or_resume(
        self,
        run_id: str,
        correlation_id: str,
        cancellation_requested: Callable[[], bool],
    ) -> HarnessOutcome:
        self.calls.append(run_id)
        if self._execute:
            return self._execute(run_id, correlation_id, cancellation_requested)
        if cancellation_requested():
            return HarnessOutcome(HarnessStatus.CANCELLED)
        return HarnessOutcome(HarnessStatus.SUCCEEDED, result_ref=f"result://{run_id}")

    def cancel_run(self, run_id: str) -> None:
        self.cancelled.append(run_id)


@dataclass(frozen=True, slots=True)
class SimulationReport:
    tasks: int
    workers: int
    cycles: int
    succeeded: int
    failed: int
    retries: int
    elapsed_seconds: float
    throughput_per_second: float
    average_queue_wait_seconds: float
    average_execution_seconds: float
    worker_utilization: float


class LocalDistributedSimulator:
    def __init__(
        self,
        *,
        start: datetime | None = None,
        config: RuntimeConfig | None = None,
    ) -> None:
        self.clock = ManualClock(start or datetime(2026, 1, 1, tzinfo=UTC))
        self.store = InMemoryTaskStore()
        self.coordinator = Coordinator(self.store, clock=self.clock, config=config)
        self.workers: list[Worker] = []

    def add_worker(
        self,
        worker_id: str,
        capabilities: frozenset[str],
        harness: Harness,
        *,
        max_concurrency: int = 1,
    ) -> Worker:
        worker = Worker(
            self.coordinator,
            harness,
            WorkerIdentity(worker_id, capabilities, f"local:{worker_id}"),
            max_concurrency=max_concurrency,
        )
        worker.register()
        self.workers.append(worker)
        return worker

    def submit(self, count: int, capability: str = "agent.execute") -> list[DistributedTask]:
        if count < 0:
            raise DomainError("simulation task count cannot be negative")
        return [
            self.coordinator.submit_task(
                f"run-{index}",
                correlation_id=f"trace-{index}",
                required_capabilities=frozenset({capability}),
                priority=TaskPriority.NORMAL,
            )
            for index in range(count)
        ]

    def run_until_terminal(self, max_cycles: int = 100_000) -> SimulationReport:
        started = perf_counter()
        cycles = 0
        while cycles < max_cycles:
            nonterminal = [
                task for task in self.store.list() if task.state not in TERMINAL_TASK_STATES
            ]
            if not nonterminal:
                break
            progressed = False
            for worker in self.workers:
                task = worker.poll_once()
                progressed = progressed or task is not None
            self.coordinator.recover()
            if not progressed:
                future = [
                    task.available_at
                    for task in nonterminal
                    if task.available_at > self.clock.now()
                ]
                if future:
                    self.clock.advance(min(future) - self.clock.now())
                    self.coordinator.recover()
                else:
                    raise DomainError("simulation made no progress")
            cycles += 1
        else:
            raise DomainError("simulation exceeded cycle bound")
        elapsed = max(perf_counter() - started, 1e-9)
        stats = self.coordinator.runtime_stats()
        metrics = cast(dict[str, object], stats["metrics"])
        counters = cast(dict[str, int], metrics["counters"])
        observations = cast(dict[str, dict[str, float]], metrics["observations"])
        workers = cast(dict[str, object], stats["workers"])
        queue_wait = observations.get("task_queue_wait_seconds", {"average": 0.0})
        execution = observations.get("task_execution_seconds", {"average": 0.0})
        counts = self.store.counts()
        total = sum(counts.values())
        executed_attempts = sum(task.attempt for task in self.store.list())
        logical_capacity = cycles * len(self.workers)
        return SimulationReport(
            tasks=total,
            workers=len(self.workers),
            cycles=cycles,
            succeeded=counts["SUCCEEDED"],
            failed=counts["DEAD_LETTERED"],
            retries=int(counters.get("task_retries", 0)),
            elapsed_seconds=elapsed,
            throughput_per_second=total / elapsed,
            average_queue_wait_seconds=float(queue_wait["average"]),
            average_execution_seconds=float(execution["average"]),
            worker_utilization=(
                executed_attempts / logical_capacity
                if logical_capacity
                else cast(float, workers["utilization"])
            ),
        )
