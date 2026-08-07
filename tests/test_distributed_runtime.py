from __future__ import annotations

import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Barrier

import pytest

from nexus_runtime.distributed.cli import main as cli_main
from nexus_runtime.distributed.clock import ManualClock
from nexus_runtime.distributed.coordinator import Coordinator, RuntimeConfig
from nexus_runtime.distributed.model import (
    DistributedTask,
    DistributedTaskState,
    FailureClass,
    RetryPolicy,
    TaskPriority,
    TaskStateMachine,
)
from nexus_runtime.distributed.security import (
    ConfiguredWorkerAuthenticator,
    StaticRuntimeAuthorizer,
    WorkerIdentity,
)
from nexus_runtime.distributed.simulator import (
    DeterministicHarness,
    LocalDistributedSimulator,
)
from nexus_runtime.distributed.store import InMemoryTaskStore, SQLiteTaskStore
from nexus_runtime.distributed.worker import HarnessOutcome, HarnessStatus, Worker
from nexus_runtime.distributed.worker_registry import WorkerStatus
from nexus_runtime.events import InMemoryEventBus
from nexus_runtime.models import DomainError


def make_clock() -> ManualClock:
    return ManualClock(datetime(2026, 1, 1, tzinfo=UTC))


def make_config(max_pending: int = 10_000) -> RuntimeConfig:
    return RuntimeConfig(
        lease_duration=timedelta(seconds=5),
        heartbeat_interval=timedelta(seconds=1),
        worker_failure_threshold=timedelta(seconds=3),
        max_pending_tasks=max_pending,
        aging_interval=timedelta(minutes=1),
    )


def identity(worker_id: str, *capabilities: str) -> WorkerIdentity:
    return WorkerIdentity(worker_id, frozenset(capabilities), f"local:{worker_id}")


def register(coordinator: Coordinator, worker_id: str, *capabilities: str) -> WorkerIdentity:
    value = identity(worker_id, *capabilities)
    coordinator.register_worker(value, version="test", max_concurrency=1)
    return value


def zero_backoff(attempts: int = 3) -> RetryPolicy:
    return RetryPolicy(
        max_attempts=attempts,
        initial_backoff=timedelta(0),
        max_backoff=timedelta(0),
    )


def _process_claim(
    database: str,
    task_id: str,
    worker_id: str,
    now_text: str,
    output: object,
) -> None:
    store = SQLiteTaskStore(database)
    claimed = store.claim(
        task_id,
        worker_id,
        datetime.fromisoformat(now_text),
        timedelta(seconds=5),
    )
    output.put(None if claimed is None else claimed.worker_id)
    store.close()


def _process_worker(database: str, worker_id: str, output: object) -> None:
    store = SQLiteTaskStore(database)
    coordinator = Coordinator(store)
    worker = Worker(
        coordinator,
        DeterministicHarness(),
        identity(worker_id, "agent.execute"),
    )
    worker.register()
    completed: list[str] = []
    for _ in range(2):
        task = worker.poll_once()
        if task is not None:
            completed.append(task.task_id)
    output.put(completed)
    store.close()


class TestDistributedModel:
    def test_invalid_state_transition_is_rejected(self) -> None:
        task = DistributedTask("run-1", "trace-1")
        with pytest.raises(DomainError, match="invalid distributed task transition"):
            TaskStateMachine.transition(task, DistributedTaskState.SUCCEEDED, task.created_at)

    def test_retry_backoff_is_bounded_and_deterministic(self) -> None:
        policy = RetryPolicy(
            max_attempts=5,
            initial_backoff=timedelta(seconds=2),
            max_backoff=timedelta(seconds=5),
            backoff_multiplier=2,
            jitter_ratio=0.25,
        )
        first = policy.delay("task-1", 4)
        assert first == policy.delay("task-1", 4)
        assert timedelta(seconds=3.75) <= first <= timedelta(seconds=5)

    def test_metadata_must_be_bounded_json(self) -> None:
        with pytest.raises(DomainError, match="JSON serializable"):
            DistributedTask("run-1", "trace-1", metadata={"bad": object()})
        with pytest.raises(DomainError, match="64 KiB"):
            DistributedTask("run-1", "trace-1", metadata={"large": "x" * 70_000})


@pytest.mark.parametrize("backend", ["memory", "sqlite"])
def test_atomic_claim_allows_only_one_worker(backend: str, tmp_path: Path) -> None:
    clock = make_clock()
    if backend == "memory":
        stores = [InMemoryTaskStore(), None]
        stores[1] = stores[0]
    else:
        path = tmp_path / "claims.sqlite"
        stores = [SQLiteTaskStore(path), SQLiteTaskStore(path)]
    task = DistributedTask("run-1", "trace-1", available_at=clock.now(), created_at=clock.now())
    stores[0].create(task)
    barrier = Barrier(2)

    def claim(index: int) -> DistributedTask | None:
        barrier.wait()
        return stores[index].claim(
            task.task_id,
            f"worker-{index}",
            clock.now(),
            timedelta(seconds=5),
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(claim, range(2)))

    assert sum(result is not None for result in results) == 1
    stored = stores[0].get(task.task_id)
    assert stored is not None
    assert stored.state == DistributedTaskState.CLAIMED
    if backend == "sqlite":
        stores[0].close()
        stores[1].close()


def test_sqlite_claim_is_atomic_across_processes(tmp_path: Path) -> None:
    clock = make_clock()
    path = tmp_path / "process-claims.sqlite"
    store = SQLiteTaskStore(path)
    task = store.create(
        DistributedTask(
            "run-process",
            "trace-process",
            available_at=clock.now(),
            created_at=clock.now(),
        )
    )
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    processes = [
        context.Process(
            target=_process_claim,
            args=(str(path), task.task_id, f"worker-{index}", clock.now().isoformat(), output),
        )
        for index in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    results = [output.get(timeout=2) for _ in processes]

    assert sum(result is not None for result in results) == 1
    store.close()


def test_two_process_coordinators_and_workers_complete_unique_tasks(tmp_path: Path) -> None:
    path = tmp_path / "process-runtime.sqlite"
    store = SQLiteTaskStore(path)
    coordinator = Coordinator(store)
    submitted = {
        coordinator.submit_task(
            f"run-{index}",
            correlation_id=f"trace-{index}",
            required_capabilities=frozenset({"agent.execute"}),
        ).task_id
        for index in range(4)
    }
    context = multiprocessing.get_context("spawn")
    output = context.Queue()
    processes = [
        context.Process(
            target=_process_worker,
            args=(str(path), f"process-worker-{index}", output),
        )
        for index in range(2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=10)
        assert process.exitcode == 0
    completed = [task_id for _ in processes for task_id in output.get(timeout=2)]

    assert set(completed) == submitted
    assert len(completed) == len(set(completed)) == 4
    assert store.counts()[DistributedTaskState.SUCCEEDED.value] == 4
    store.close()


class TestCoordinatorAndWorker:
    def setup_method(self) -> None:
        self.clock = make_clock()
        self.store = InMemoryTaskStore()
        self.coordinator = Coordinator(self.store, clock=self.clock, config=make_config())

    def test_normal_worker_delegates_to_harness_and_completes(self) -> None:
        harness = DeterministicHarness()
        worker = Worker(
            self.coordinator,
            harness,
            identity("worker-a", "agent.execute"),
        )
        worker.register()
        submitted = self.coordinator.submit_task(
            "run-1",
            correlation_id="trace-1",
            required_capabilities=frozenset({"agent.execute"}),
        )

        completed = worker.poll_once()

        assert completed is not None
        assert completed.state == DistributedTaskState.SUCCEEDED
        assert completed.result_ref == "result://run-1"
        assert harness.calls == ["run-1"]
        assert self.coordinator.require_task(submitted.task_id).attempt == 1

    def test_worker_crash_before_claim_does_not_block_task(self) -> None:
        crashed = Worker(
            self.coordinator,
            DeterministicHarness(),
            identity("worker-a", "agent.execute"),
        )
        crashed.register()
        crashed.crash()
        self.coordinator.submit_task(
            "run-1",
            correlation_id="trace-1",
            required_capabilities=frozenset({"agent.execute"}),
        )
        self.clock.advance(timedelta(seconds=3))
        self.coordinator.recover()
        replacement = Worker(
            self.coordinator,
            DeterministicHarness(),
            identity("worker-b", "agent.execute"),
        )
        replacement.register()

        assert replacement.poll_once().state == DistributedTaskState.SUCCEEDED

    def test_crash_during_execution_recovers_same_run_on_replacement(self) -> None:
        worker_a = register(self.coordinator, "worker-a", "agent.execute")
        task = self.coordinator.submit_task(
            "run-recover",
            correlation_id="trace-recover",
            required_capabilities=frozenset({"agent.execute"}),
            retry_policy=zero_backoff(2),
        )
        claimed = self.coordinator.claim_task(worker_a)
        assert claimed is not None and claimed.lease_id is not None
        self.coordinator.start_task(worker_a, task.task_id, claimed.lease_id)

        self.clock.advance(timedelta(seconds=6))
        self.coordinator.recover()
        self.coordinator.recover()
        harness = DeterministicHarness()
        worker_b = Worker(
            self.coordinator,
            harness,
            identity("worker-b", "agent.execute"),
        )
        worker_b.register()
        completed = worker_b.poll_once()

        assert completed is not None
        assert completed.state == DistributedTaskState.SUCCEEDED
        assert completed.attempt == 2
        assert harness.calls == ["run-recover"]
        assert completed.attempts[0].error == "lease expired"

    def test_ack_loss_causes_duplicate_execution_but_stale_ack_is_rejected(self) -> None:
        worker_a = register(self.coordinator, "worker-a", "agent.execute")
        task = self.coordinator.submit_task(
            "run-duplicate",
            correlation_id="trace-duplicate",
            required_capabilities=frozenset({"agent.execute"}),
            retry_policy=zero_backoff(2),
        )
        first = self.coordinator.claim_task(worker_a)
        assert first is not None and first.lease_id is not None
        old_lease = first.lease_id
        self.coordinator.start_task(worker_a, task.task_id, old_lease)
        executions = ["run-duplicate"]  # execution finished; ACK is deliberately lost

        self.clock.advance(timedelta(seconds=6))
        self.coordinator.recover()
        self.coordinator.recover()
        harness = DeterministicHarness(
            lambda run, trace, cancelled: (
                executions.append(run) or HarnessOutcome(HarnessStatus.SUCCEEDED, "result://winner")
            )
        )
        replacement = Worker(
            self.coordinator,
            harness,
            identity("worker-b", "agent.execute"),
        )
        replacement.register()
        winner = replacement.poll_once()

        assert winner is not None and winner.state == DistributedTaskState.SUCCEEDED
        assert executions == ["run-duplicate", "run-duplicate"]
        with pytest.raises(DomainError, match="does not own"):
            self.coordinator.complete_task(worker_a, task.task_id, old_lease, "result://stale")
        assert self.coordinator.require_task(task.task_id).result_ref == "result://winner"

    def test_transient_failure_retries_and_permanent_failure_dead_letters(self) -> None:
        calls = 0

        def execute(run_id: str, trace_id: str, cancellation_requested: object) -> HarnessOutcome:
            nonlocal calls
            calls += 1
            if calls == 1:
                return HarnessOutcome(
                    HarnessStatus.FAILED,
                    checkpoint_ref="checkpoint://1",
                    failure_class=FailureClass.TRANSIENT,
                    error="temporary",
                )
            return HarnessOutcome(HarnessStatus.SUCCEEDED, "result://ok")

        worker = Worker(
            self.coordinator,
            DeterministicHarness(execute),
            identity("worker-a", "agent.execute"),
        )
        worker.register()
        retrying = self.coordinator.submit_task(
            "run-retry",
            correlation_id="trace-retry",
            required_capabilities=frozenset({"agent.execute"}),
            retry_policy=zero_backoff(2),
        )
        assert worker.poll_once().state == DistributedTaskState.RETRY_WAIT
        self.coordinator.recover()
        assert worker.poll_once().state == DistributedTaskState.SUCCEEDED
        assert (
            self.coordinator.require_task(retrying.task_id).last_checkpoint_ref == "checkpoint://1"
        )

        permanent = self.coordinator.submit_task(
            "run-permanent",
            correlation_id="trace-permanent",
            required_capabilities=frozenset({"agent.execute"}),
        )
        claimed = self.coordinator.claim_task(worker.identity)
        assert claimed is not None and claimed.lease_id is not None
        self.coordinator.start_task(worker.identity, permanent.task_id, claimed.lease_id)
        result = self.coordinator.fail_task(
            worker.identity,
            permanent.task_id,
            claimed.lease_id,
            FailureClass.PERMANENT,
            "invalid input",
        )
        assert result.state == DistributedTaskState.DEAD_LETTERED
        assert result.attempts[-1].failure_class == FailureClass.PERMANENT

    def test_cancellation_before_during_and_without_worker(self) -> None:
        queued = self.coordinator.submit_task("run-queued", correlation_id="trace-q")
        assert (
            self.coordinator.cancel_task(queued.task_id, "operator").state
            == DistributedTaskState.CANCELLED
        )

        def cancel_during(
            run_id: str, trace_id: str, cancellation_requested: object
        ) -> HarnessOutcome:
            active = next(task for task in self.coordinator.list_tasks() if task.run_id == run_id)
            self.coordinator.cancel_task(active.task_id, "operator")
            return HarnessOutcome(HarnessStatus.SUCCEEDED, "result://ignored")

        harness = DeterministicHarness(cancel_during)
        worker = Worker(
            self.coordinator,
            harness,
            identity("worker-a", "agent.execute"),
        )
        worker.register()
        self.coordinator.submit_task(
            "run-during",
            correlation_id="trace-d",
            required_capabilities=frozenset({"agent.execute"}),
        )
        assert worker.poll_once().state == DistributedTaskState.CANCELLED
        assert harness.cancelled == ["run-during"]

        unavailable = self.coordinator.submit_task(
            "run-unavailable",
            correlation_id="trace-u",
            required_capabilities=frozenset({"agent.execute"}),
        )
        claim = self.coordinator.claim_task(worker.identity)
        assert claim is not None and claim.lease_id is not None
        self.coordinator.start_task(worker.identity, unavailable.task_id, claim.lease_id)
        requested = self.coordinator.cancel_task(unavailable.task_id, "operator")
        assert requested.state == DistributedTaskState.CANCEL_REQUESTED
        self.clock.advance(timedelta(seconds=6))
        self.coordinator.recover()
        assert (
            self.coordinator.require_task(unavailable.task_id).state
            == DistributedTaskState.CANCELLED
        )

    def test_capabilities_priority_aging_backpressure_and_draining(self) -> None:
        cpu = register(self.coordinator, "cpu", "cpu")
        gpu = register(self.coordinator, "gpu", "gpu")
        self.coordinator.submit_task(
            "run-low",
            correlation_id="trace-low",
            priority=TaskPriority.LOW,
            required_capabilities=frozenset({"gpu"}),
        )
        self.clock.advance(timedelta(minutes=101))
        high = self.coordinator.submit_task(
            "run-high",
            correlation_id="trace-high",
            priority=TaskPriority.HIGH,
            required_capabilities=frozenset({"gpu"}),
        )
        assert self.coordinator.claim_task(cpu) is None
        aged = self.coordinator.claim_task(gpu)
        assert aged is not None and aged.run_id == "run-low"
        assert self.coordinator.require_task(high.task_id).state == DistributedTaskState.QUEUED
        self.coordinator.drain_worker("cpu", "operator")
        assert self.coordinator.claim_task(cpu) is None

        limited = Coordinator(
            InMemoryTaskStore(), clock=self.clock, config=make_config(max_pending=1)
        )
        limited.submit_task("one", correlation_id="one")
        with pytest.raises(DomainError, match="backpressure"):
            limited.submit_task("two", correlation_id="two")

    def test_stale_lease_cannot_be_renewed(self) -> None:
        worker = register(self.coordinator, "worker-a", "agent.execute")
        task = self.coordinator.submit_task(
            "run-stale",
            correlation_id="trace-stale",
            required_capabilities=frozenset({"agent.execute"}),
        )
        claim = self.coordinator.claim_task(worker)
        assert claim is not None and claim.lease_id is not None
        self.clock.advance(timedelta(seconds=6))
        with pytest.raises(DomainError, match="stale"):
            self.store.renew_lease(
                task.task_id,
                worker.worker_id,
                claim.lease_id,
                self.clock.now(),
                timedelta(seconds=5),
            )

    def test_attempt_identity_must_match_current_lease(self) -> None:
        worker = register(self.coordinator, "worker-attempt", "agent.execute")
        task = self.coordinator.submit_task(
            "run-attempt",
            correlation_id="trace-attempt",
            required_capabilities=frozenset({"agent.execute"}),
        )
        claim = self.coordinator.claim_task(worker)
        assert claim is not None and claim.lease_id is not None
        self.coordinator.start_task(worker, task.task_id, claim.lease_id)

        with pytest.raises(DomainError, match="attempt does not match"):
            self.coordinator.complete_task(
                worker,
                task.task_id,
                claim.lease_id,
                "result://wrong-attempt",
                "attempt_stale",
            )

    def test_heartbeat_extends_lease_and_capacity_is_enforced(self) -> None:
        worker = identity("worker-capacity", "agent.execute")
        self.coordinator.register_worker(worker, version="test", max_concurrency=2)
        for index in range(3):
            self.coordinator.submit_task(
                f"run-{index}",
                correlation_id=f"trace-{index}",
                required_capabilities=frozenset({"agent.execute"}),
            )
        first = self.coordinator.claim_task(worker)
        second = self.coordinator.claim_task(worker)
        assert first is not None and first.lease is not None
        assert second is not None and second.lease is not None
        assert self.coordinator.claim_task(worker) is None

        original_expiry = first.lease.expires_at
        self.clock.advance(timedelta(seconds=4))
        record = self.coordinator.heartbeat(worker)
        renewed = self.coordinator.require_task(first.task_id)
        assert renewed.lease is not None and renewed.lease.expires_at > original_expiry
        assert record.available_slots == 0
        self.coordinator.start_task(worker, first.task_id, renewed.lease.lease_id)
        self.coordinator.complete_task(worker, first.task_id, renewed.lease.lease_id, None)
        assert self.coordinator.claim_task(worker) is not None

    def test_lost_heartbeat_marks_worker_unhealthy(self) -> None:
        register(self.coordinator, "worker-lost", "agent.execute")
        self.clock.advance(timedelta(seconds=3))

        self.coordinator.recover()

        record = next(
            worker
            for worker in self.coordinator.list_workers()
            if worker.identity.worker_id == "worker-lost"
        )
        assert record.status == WorkerStatus.UNHEALTHY

    def test_deadline_and_manual_retry_are_durable(self) -> None:
        task = self.coordinator.submit_task(
            "run-deadline",
            correlation_id="trace-deadline",
            deadline=self.clock.now() + timedelta(seconds=1),
        )
        self.clock.advance(timedelta(seconds=2))
        self.coordinator.recover()
        dead = self.coordinator.require_task(task.task_id)
        assert dead.state == DistributedTaskState.DEAD_LETTERED
        assert dead.last_error == "task deadline expired"

        retried = self.coordinator.retry_task(task.task_id, "operator")
        assert retried.state == DistributedTaskState.QUEUED
        assert retried.last_error == "task deadline expired"


def test_coordinator_restart_recovers_sqlite_task_and_registry_loss(
    tmp_path: Path,
) -> None:
    clock = make_clock()
    path = tmp_path / "runtime.sqlite"
    first_store = SQLiteTaskStore(path)
    first = Coordinator(first_store, clock=clock, config=make_config())
    worker_a = register(first, "worker-a", "agent.execute")
    submitted = first.submit_task(
        "run-restart",
        correlation_id="trace-restart",
        required_capabilities=frozenset({"agent.execute"}),
        retry_policy=zero_backoff(2),
    )
    claim = first.claim_task(worker_a)
    assert claim is not None and claim.lease_id is not None
    first.start_task(worker_a, submitted.task_id, claim.lease_id)
    first_store.close()

    clock.advance(timedelta(seconds=6))
    resumed_store = SQLiteTaskStore(path)
    resumed = Coordinator(resumed_store, clock=clock, config=make_config())
    resumed.recover()
    resumed.recover()
    harness = DeterministicHarness()
    replacement = Worker(
        resumed,
        harness,
        identity("worker-b", "agent.execute"),
    )
    replacement.register()

    completed = replacement.poll_once()
    assert completed is not None and completed.state == DistributedTaskState.SUCCEEDED
    assert completed.attempt == 2
    assert harness.calls == ["run-restart"]
    resumed_store.close()


def test_many_tasks_many_workers_have_exact_accounting() -> None:
    simulator = LocalDistributedSimulator()
    harness = DeterministicHarness()
    for index in range(10):
        simulator.add_worker(f"worker-{index}", frozenset({"agent.execute"}), harness)
    simulator.submit(100)

    report = simulator.run_until_terminal()

    assert report.succeeded == 100
    assert report.failed == 0
    assert report.cycles == 10
    assert len(harness.calls) == len(set(harness.calls)) == 100
    assert sum(simulator.store.counts().values()) == 100


class FailingTaskStore(InMemoryTaskStore):
    def create(self, task: DistributedTask) -> DistributedTask:
        raise OSError("injected store failure")


def test_task_store_failure_is_not_hidden() -> None:
    coordinator = Coordinator(FailingTaskStore(), clock=make_clock(), config=make_config())
    with pytest.raises(OSError, match="injected store failure"):
        coordinator.submit_task("run-1", correlation_id="trace-1")


def test_worker_capabilities_and_control_actions_are_authorized() -> None:
    clock = make_clock()
    authenticator = ConfiguredWorkerAuthenticator({"service:gpu-1": frozenset({"model.gpu"})})
    authorizer = StaticRuntimeAuthorizer(
        {"operator": frozenset({"runtime.task.cancel", "runtime.task.retry"})}
    )
    coordinator = Coordinator(
        InMemoryTaskStore(),
        clock=clock,
        config=make_config(),
        worker_authenticator=authenticator,
        authorizer=authorizer,
    )
    approved = WorkerIdentity("gpu-1", frozenset({"model.gpu"}), "service:gpu-1")
    coordinator.register_worker(approved, version="test", max_concurrency=1)
    with pytest.raises(DomainError, match="not authorized"):
        coordinator.register_worker(
            WorkerIdentity(
                "gpu-2",
                frozenset({"model.gpu", "filesystem.write"}),
                "service:gpu-1",
            ),
            version="test",
            max_concurrency=1,
        )
    task = coordinator.submit_task("run-1", correlation_id="trace-1")
    with pytest.raises(DomainError, match="cannot perform"):
        coordinator.cancel_task(task.task_id, "viewer")
    assert coordinator.cancel_task(task.task_id, "operator").state == DistributedTaskState.CANCELLED


def test_distributed_events_preserve_end_to_end_correlation() -> None:
    bus = InMemoryEventBus()
    coordinator = Coordinator(
        InMemoryTaskStore(),
        clock=make_clock(),
        config=make_config(),
        event_bus=bus,
    )
    worker = Worker(
        coordinator,
        DeterministicHarness(),
        identity("worker-trace", "agent.execute"),
    )
    worker.register()
    coordinator.submit_task(
        "run-trace",
        correlation_id="investigation-trace",
        required_capabilities=frozenset({"agent.execute"}),
    )
    worker.poll_once()

    task_events = [event for event in bus.events if event.event_type.startswith("task.")]
    assert {event.trace_id for event in task_events} == {"investigation-trace"}
    assert all(event.payload["run_id"] == "run-trace" for event in task_events)


def test_cli_uses_durable_application_service(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "cli.sqlite"
    assert (
        cli_main(
            [
                "--db",
                str(path),
                "submit",
                "run-cli",
                "--correlation-id",
                "trace-cli",
                "--capability",
                "agent.execute",
            ]
        )
        == 0
    )
    submitted = json.loads(capsys.readouterr().out)
    assert cli_main(["--db", str(path), "task", submitted["task_id"]]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["run_id"] == "run-cli"
    assert cli_main(["--db", str(path), "queue"]) == 0
    queue = json.loads(capsys.readouterr().out)
    assert queue["QUEUED"] == 1
