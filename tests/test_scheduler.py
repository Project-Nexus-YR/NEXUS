from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

from nexus_runtime.events import InMemoryEventBus
from nexus_runtime.models import DomainError, RetryPolicy, Task, TaskState
from nexus_runtime.scheduler import Scheduler


class Clock:
    def __init__(self) -> None:
        self.now = datetime(2026, 1, 1, tzinfo=UTC)

    def __call__(self) -> datetime:
        return self.now

    def advance(self, duration: timedelta) -> None:
        self.now += duration


class SchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = Clock()
        self.bus = InMemoryEventBus()
        self.scheduler = Scheduler(
            lease_duration=timedelta(seconds=5),
            worker_timeout=timedelta(seconds=8),
            event_bus=self.bus,
            clock=self.clock,
        )
        self.scheduler.register_worker("worker-a", frozenset({"search.execute", "analysis"}), 2)
        self.scheduler.register_worker("worker-b", frozenset({"search.execute", "analysis"}), 2)

    def test_independent_tasks_lease_to_distinct_workers(self) -> None:
        first = self.scheduler.enqueue(Task("first", "search.execute", priority=1))
        second = self.scheduler.enqueue(Task("second", "search.execute"))

        leased_first = self.scheduler.lease_next("worker-a")
        leased_second = self.scheduler.lease_next("worker-b")

        self.assertEqual(first.task_id, leased_first.task_id)
        self.assertEqual(second.task_id, leased_second.task_id)

    def test_dependency_is_not_schedulable_before_parent_completion(self) -> None:
        parent = self.scheduler.enqueue(Task("retrieve", "search.execute"))
        child = self.scheduler.enqueue(Task("analyse", "analysis", dependencies={parent.task_id}))
        self.assertEqual(child.state, TaskState.CREATED)

        self.scheduler.start("worker-a", parent.task_id) if self.scheduler.lease_next(
            "worker-a"
        ) else None
        self.scheduler.complete("worker-a", parent.task_id, {"evidence": "ref"})

        self.assertEqual(child.state, TaskState.READY)

    def test_expired_lease_is_retried_by_another_worker(self) -> None:
        task = self.scheduler.enqueue(
            Task("recover", "search.execute", retry_policy=RetryPolicy(2, timedelta(0)))
        )
        self.assertEqual(self.scheduler.lease_next("worker-a"), task)
        self.scheduler.start("worker-a", task.task_id)

        self.clock.advance(timedelta(seconds=6))
        recovered = self.scheduler.recover()
        retry = self.scheduler.lease_next("worker-b")

        self.assertEqual(recovered, [task.task_id])
        self.assertEqual(retry, task)
        self.assertEqual(task.worker_id, "worker-b")
        self.assertEqual(task.attempt_count, 2)

    def test_duplicate_idempotency_key_is_not_delivered_twice(self) -> None:
        first = self.scheduler.enqueue(Task("effect", "search.execute", idempotency_key="effect:1"))
        self.scheduler.lease_next("worker-a")
        self.scheduler.start("worker-a", first.task_id)
        self.scheduler.complete("worker-a", first.task_id, {"done": True})

        duplicate = self.scheduler.enqueue(
            Task("duplicate", "search.execute", idempotency_key="effect:1")
        )

        self.assertEqual(duplicate.state, TaskState.COMPLETED)
        self.assertIsNone(self.scheduler.lease_next("worker-b"))

    def test_backpressure_prevents_unbounded_queue_growth(self) -> None:
        limited = Scheduler(max_queued_tasks=1, clock=self.clock)
        limited.enqueue(Task("one", "search.execute"))
        with self.assertRaisesRegex(DomainError, "backpressure"):
            limited.enqueue(Task("two", "search.execute"))

    def test_cancel_during_execution_records_cancelled_attempt(self) -> None:
        task = self.scheduler.enqueue(Task("cancel", "search.execute"))
        self.scheduler.lease_next("worker-a")
        self.scheduler.start("worker-a", task.task_id)

        self.scheduler.cancel(task.task_id)

        self.assertEqual(task.state, TaskState.CANCELLED)
        self.assertEqual(self.scheduler.attempts[task.task_id][-1].state, TaskState.CANCELLED)

    def test_subscriber_failure_is_dead_lettered_without_losing_event(self) -> None:
        self.bus.subscribe("task", lambda _: (_ for _ in ()).throw(RuntimeError("disconnect")))

        task = self.scheduler.enqueue(Task("event", "search.execute"))

        self.assertEqual(task.state, TaskState.READY)
        self.assertGreaterEqual(len(self.bus.dead_letters), 1)

    def test_invalid_transition_is_rejected(self) -> None:
        task = self.scheduler.enqueue(Task("invalid", "search.execute"))
        with self.assertRaisesRegex(DomainError, "invalid task transition"):
            self.scheduler._transition(task, TaskState.COMPLETED, "test")
