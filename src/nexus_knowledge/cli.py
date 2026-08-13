"""Command-line interface for the knowledge engine.

Deterministic demo entry point: ingest, retrieve, GraphRAG, gap
analysis and investigation scoring without the autonomous runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .domain.source import Source, SourceKind
from .ingestion.adapters import (
    JsonAdapter,
    MarkdownAdapter,
    RepositoryAdapter,
    SourceAdapter,
    TextAdapter,
)
from .persistence.json_codec import load_snapshot, save_snapshot
from .service.factory import Adapters, create_engine

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nexus-knowledge", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="ingest a text file or directory")
    ingest.add_argument("path")
    ingest.add_argument("--title", default="source")
    ingest.add_argument(
        "--kind",
        default=SourceKind.TEXT,
        choices=[SourceKind.TEXT, SourceKind.MARKDOWN, SourceKind.JSON, SourceKind.REPOSITORY],
    )
    ingest.add_argument("--gazetteer", default=None, help="JSON file mapping entity type -> names")
    ingest.add_argument("--output", default=None, help="save the resulting knowledge snapshot")

    retrieve = sub.add_parser("retrieve", help="hybrid retrieval for a query")
    retrieve.add_argument("query")
    retrieve.add_argument("--top-k", type=int, default=10)
    retrieve.add_argument("--data", default=None, help="JSON snapshot to load")

    graphrag = sub.add_parser("graphrag", help="evidence graph for a query")
    graphrag.add_argument("query")
    graphrag.add_argument("--data", default=None)

    gaps = sub.add_parser("gaps", help="detect knowledge gaps")
    gaps.add_argument("--data", default=None)

    score = sub.add_parser("score", help="score candidate investigations")
    score.add_argument("--top-k", type=int, default=20)
    score.add_argument("--data", default=None)

    stats = sub.add_parser("stats", help="graph statistics")
    stats.add_argument("--data", default=None)

    bench = sub.add_parser("bench", help="run the evaluation benchmarks")
    bench.add_argument("--output", default=None, help="write the report to a JSON file")
    return parser


def _gazetteer(path: str | None) -> dict[str, list[str]] | None:
    if not path:
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def _ingest_payload(path: str, kind: str) -> tuple[object, SourceAdapter]:
    source_path = Path(path)
    if kind == SourceKind.REPOSITORY or source_path.is_dir():
        return source_path, RepositoryAdapter()
    payload = source_path.read_text(encoding="utf-8")
    adapter = {
        SourceKind.MARKDOWN: MarkdownAdapter,
        SourceKind.JSON: JsonAdapter,
    }.get(kind, TextAdapter)()
    return payload, adapter


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    repository = load_snapshot(args.data) if getattr(args, "data", None) else None
    adapter_arguments = {}
    if repository is not None:
        adapter_arguments["repository"] = repository
    if hasattr(args, "gazetteer"):
        adapter_arguments["gazetteer"] = _gazetteer(args.gazetteer)
    adapters = Adapters(**adapter_arguments)
    engine = create_engine(adapters)

    if args.command == "ingest":
        payload, adapter = _ingest_payload(args.path, args.kind)
        engine.ingest(
            Source(title=args.title, kind=args.kind, reference=args.path),
            payload,
            adapter,
        )
        print(json.dumps(engine.healthcheck(), indent=2))
        if args.output:
            save_snapshot(engine.repository, args.output)
    elif args.command == "retrieve":
        result = engine.retrieve(args.query, top_k=args.top_k)
        print(json.dumps(result.to_dict(), indent=2, default=str))
    elif args.command == "graphrag":
        evidence = engine.graphrag(args.query)
        print(json.dumps(evidence.to_dict(), indent=2, default=str))
    elif args.command == "gaps":
        gaps = engine.find_knowledge_gaps()
        print(
            json.dumps(
                [
                    {
                        "kind": g.kind,
                        "description": g.description,
                        "priority": round(g.priority, 4),
                        "estimated_cost": g.estimated_cost,
                    }
                    for g in gaps
                ],
                indent=2,
            )
        )
    elif args.command == "score":
        scored = engine.score_investigation(top_k=args.top_k)
        print(json.dumps([s.to_dict() for s in scored], indent=2))
    elif args.command == "stats":
        print(json.dumps(engine.graph_statistics(), indent=2))
    elif args.command == "bench":
        from .eval.benchmarks import run_benchmarks

        report = run_benchmarks()
        text = report.to_json()
        print(text)
        if args.output:
            with open(args.output, "w", encoding="utf-8") as handle:
                handle.write(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
