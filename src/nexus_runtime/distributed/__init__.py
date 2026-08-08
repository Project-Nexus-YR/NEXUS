"""Fault-tolerant distributed execution for durable AgentRuns.

This package owns placement and coordination only. Agent execution remains behind
the :class:`Harness` port in ``worker``.
"""

from .clock import Clock, ManualClock, SystemClock
from .coordinator import Coordinator, RuntimeConfig
from .model import (
    DistributedTask,
    DistributedTaskState,
    FailureClass,
    Lease,
    RetryPolicy,
    TaskPriority,
)
from .store import InMemoryTaskStore, LeaseStore, SQLiteTaskStore, TaskStore
from .worker import Harness, HarnessExecutionContext, HarnessOutcome, Worker

__all__ = [
    "Clock",
    "Coordinator",
    "DistributedTask",
    "DistributedTaskState",
    "FailureClass",
    "Harness",
    "HarnessExecutionContext",
    "HarnessOutcome",
    "InMemoryTaskStore",
    "Lease",
    "LeaseStore",
    "ManualClock",
    "RetryPolicy",
    "RuntimeConfig",
    "SQLiteTaskStore",
    "SystemClock",
    "TaskPriority",
    "TaskStore",
    "Worker",
]
