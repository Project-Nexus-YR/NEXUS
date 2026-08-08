"""Thin command-line adapter over ``InvestigationApplication``."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import timedelta
from pathlib import Path

from nexus_knowledge.service.factory import create_engine
from nexus_runtime.models import DomainError

from .application import InvestigationApplication
from .objective import ResearchObjective
from .repository import InvestigationRecord, SQLiteInvestigationRepository
from .session import InvestigationBudget


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="research")
    parser.add_argument("--db", type=Path, default=Path(".nexus/investigations.sqlite"))
    commands = parser.add_subparsers(dest="command", required=True)

    create = commands.add_parser("create", help="create a bounded research session")
    create.add_argument("question")
    create.add_argument("--criterion", action="append", required=True)
    create.add_argument("--scope", action="append", default=[])
    create.add_argument("--constraint", action="append", default=[])
    create.add_argument("--max-iterations", type=int, default=3)
    create.add_argument("--max-investigations", type=int, default=20)
    create.add_argument("--max-agent-runs", type=int, default=20)
    create.add_argument("--max-cost", type=float, default=100.0)
    create.add_argument("--max-execution-seconds", type=float, default=3600.0)

    for name in (
        "status",
        "pause",
        "resume",
        "cancel",
        "explain",
        "evidence",
        "gaps",
        "plan",
        "iterations",
    ):
        command = commands.add_parser(name)
        command.add_argument("session_id")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    application: InvestigationApplication | None = None,
) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    repository: SQLiteInvestigationRepository | None = None
    if application is None:
        repository = SQLiteInvestigationRepository(arguments.db)
        engine = create_engine()
        application = InvestigationApplication(engine, repository=repository)
    try:
        result = _dispatch(arguments, application)
        print(json.dumps(result, indent=2, sort_keys=True, default=str))
        return 0
    except (DomainError, ValueError) as exc:
        parser.exit(2, f"error: {exc}\n")
    finally:
        if repository is not None:
            repository.close()
    return 2


def _dispatch(arguments: argparse.Namespace, application: InvestigationApplication) -> object:
    if arguments.command == "create":
        objective = ResearchObjective(
            question=arguments.question,
            success_criteria=tuple(arguments.criterion),
            scope=tuple(arguments.scope),
            constraints=tuple(arguments.constraint),
        )
        budget = InvestigationBudget(
            max_iterations=arguments.max_iterations,
            max_investigations=arguments.max_investigations,
            max_agent_runs=arguments.max_agent_runs,
            max_cost=arguments.max_cost,
            max_execution_time=timedelta(seconds=arguments.max_execution_seconds),
        )
        return application.create(objective, budget).to_dict()
    session_id = str(arguments.session_id)
    if arguments.command == "status":
        return application.status(session_id).to_dict()
    if arguments.command == "pause":
        return application.pause(session_id).to_dict()
    if arguments.command == "resume":
        return application.resume(session_id).to_dict()
    if arguments.command == "cancel":
        return application.cancel(session_id).to_dict()
    if arguments.command == "explain":
        return application.explain(session_id)
    record = application.status(session_id)
    if arguments.command == "evidence":
        return _artifacts(record, "evidence_set")
    if arguments.command == "gaps":
        snapshots = _artifacts(record, "knowledge_snapshot")
        return [] if not snapshots else snapshots[-1].get("gaps", [])
    if arguments.command == "plan":
        plans = _artifacts(record, "investigation_plan")
        return None if not plans else plans[-1]
    if arguments.command == "iterations":
        return [
            {
                "iteration": iteration,
                "artifacts": [item.to_dict() for item in record.for_iteration(iteration)],
            }
            for iteration in sorted({item.iteration for item in record.artifacts})
        ]
    raise DomainError(f"unknown research command: {arguments.command}")


def _artifacts(record: InvestigationRecord, kind: str) -> list[dict[str, object]]:
    return [dict(item.payload) for item in record.artifacts if item.kind == kind]


if __name__ == "__main__":
    raise SystemExit(main())
