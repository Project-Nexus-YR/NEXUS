from __future__ import annotations

import unittest
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

from nexus_runtime.events import InMemoryEventBus
from nexus_runtime.models import RetryPolicy, Task
from nexus_runtime.persistence import SQLiteStateStore
from nexus_runtime.replay import RunReplayer
from nexus_runtime.scheduler import Scheduler


class PersistenceReplayTests(unittest.TestCase):
    def test_transition_history_checkpoint_and_safe_replay(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            store = SQLiteStateStore(Path(temporary_directory) / "runtime.sqlite")
            bus = InMemoryEventBus()
            scheduler = Scheduler(state_store=store, event_bus=bus)
            scheduler.register_worker("worker", frozenset({"search.execute"}), 1)
            task = scheduler.enqueue(Task("retrieve", "search.execute"))
            scheduler.lease_next("worker")
            scheduler.start("worker", task.task_id)
            scheduler.complete("worker", task.task_id, {"artifact": "evidence/1"})
            version = store.save_checkpoint("run-1", {"task": task.task_id, "state": "COMPLETED"})

            self.assertEqual(version, 1)
            self.assertEqual(store.load_checkpoint("run-1")["state"], "COMPLETED")
            self.assertGreaterEqual(len(store.task_history(task.task_id)), 3)
            view = RunReplayer().reconstruct(task.task_id, bus.events)
            self.assertEqual(view.task_states[task.task_id], "COMPLETED")
            self.assertTrue(view.events)
            store.close()

    def test_scheduler_restart_reclaims_unacknowledged_work(self) -> None:
        with TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "runtime.sqlite"
            first = Scheduler(state_store=SQLiteStateStore(path))
            first.register_worker("crashed-worker", frozenset({"search.execute"}), 1)
            task = first.enqueue(
                Task("retrieve", "search.execute", retry_policy=RetryPolicy(3, timedelta(0)))
            )
            first.lease_next("crashed-worker")
            first.start("crashed-worker", task.task_id)

            resumed_store = SQLiteStateStore(path)
            resumed = Scheduler(state_store=resumed_store)
            self.assertEqual(resumed.restore(), 1)
            resumed.register_worker("replacement", frozenset({"search.execute"}), 1)
            resumed.recover()
            recovered = resumed.lease_next("replacement")

            self.assertEqual(recovered.task_id, task.task_id)
            self.assertEqual(recovered.attempt_count, 2)
            resumed_store.close()
