"""Thin CLI over RuntimeApplication; all business rules remain in the coordinator."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..models import DomainError
from .coordinator import Coordinator
from .model import TaskPriority
from .service import RuntimeApplication
from .store import SQLiteTaskStore, task_to_dict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nexus-runtime")
    parser.add_argument("--db", type=Path, default=Path(".nexus/runtime.sqlite"))
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit = subparsers.add_parser("submit", help="submit an AgentRun for execution")
    submit.add_argument("run_id")
    submit.add_argument("--correlation-id", required=True)
    submit.add_argument(
        "--priority", choices=[item.name for item in TaskPriority], default="NORMAL"
    )
    submit.add_argument("--capability", action="append", default=[])
    submit.add_argument("--metadata", default="{}", help="JSON object")

    task = subparsers.add_parser("task", help="inspect one distributed task")
    task.add_argument("task_id")

    cancel = subparsers.add_parser("cancel", help="request durable task cancellation")
    cancel.add_argument("task_id")
    cancel.add_argument("--principal", required=True)

    retry = subparsers.add_parser("retry", help="retry a dead-lettered task")
    retry.add_argument("task_id")
    retry.add_argument("--principal", required=True)

    subparsers.add_parser("workers", help="list workers known to this coordinator")
    drain = subparsers.add_parser("worker-drain", help="drain a registered worker")
    drain.add_argument("worker_id")
    drain.add_argument("--principal", required=True)
    subparsers.add_parser("queue", help="show queue state counts")
    subparsers.add_parser("status", help="show structured runtime metrics")
    subparsers.add_parser("recover", help="run one recovery pass")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    application: RuntimeApplication | None = None,
) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    store: SQLiteTaskStore | None = None
    if application is None:
        store = SQLiteTaskStore(arguments.db)
        application = RuntimeApplication(Coordinator(store))
    try:
        result = _dispatch(arguments, application)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    except (DomainError, json.JSONDecodeError) as exc:
        parser.exit(2, f"error: {exc}\n")
    finally:
        if store is not None:
            store.close()
    return 2


def _dispatch(arguments: argparse.Namespace, application: RuntimeApplication) -> object:
    if arguments.command == "submit":
        metadata = json.loads(arguments.metadata)
        if not isinstance(metadata, dict):
            raise DomainError("metadata must be a JSON object")
        task = application.submit_task(
            arguments.run_id,
            correlation_id=arguments.correlation_id,
            priority=TaskPriority[arguments.priority],
            required_capabilities=frozenset(arguments.capability),
            metadata=metadata,
        )
        return task_to_dict(task)
    if arguments.command == "task":
        return task_to_dict(application.get_task(arguments.task_id))
    if arguments.command == "cancel":
        return task_to_dict(application.cancel_task(arguments.task_id, arguments.principal))
    if arguments.command == "retry":
        return task_to_dict(application.retry_task(arguments.task_id, arguments.principal))
    if arguments.command == "workers":
        return [_worker_to_dict(worker) for worker in application.list_workers()]
    if arguments.command == "worker-drain":
        return _worker_to_dict(application.drain_worker(arguments.worker_id, arguments.principal))
    if arguments.command == "queue":
        return application.get_queue_stats()
    if arguments.command == "status":
        return application.get_runtime_stats()
    if arguments.command == "recover":
        return [task_to_dict(task) for task in application.recover()]
    raise DomainError(f"unknown runtime command: {arguments.command}")


def _worker_to_dict(worker: Any) -> dict[str, object]:
    return {
        "worker_id": worker.identity.worker_id,
        "capabilities": sorted(worker.identity.capabilities),
        "version": worker.version,
        "status": worker.status.value,
        "max_concurrency": worker.max_concurrency,
        "current_concurrency": len(worker.current_tasks),
        "available_slots": worker.available_slots,
        "last_heartbeat": worker.last_heartbeat.isoformat(),
    }


if __name__ == "__main__":
    raise SystemExit(main())
