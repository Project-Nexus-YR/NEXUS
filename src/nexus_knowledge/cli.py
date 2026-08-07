"""Command-line interface for the knowledge engine.

Deterministic demo entry point: ingest, retrieve, GraphRAG, gap
analysis and investigation scoring without the autonomous runtime.
"""

from __future__ import annotations

import argparse
import json
import sys

from .domain.source import Source, SourceKind
from .service.factory import Adapters, create_engine

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nexus-knowledge", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="ingest a text file or directory")
    ingest.add_argument("path")
    ingest.add_argument("--title", default="source")
    ingest.add_argument("--kind", default=SourceKind.TEXT, choices=[
        SourceKind.TEXT, SourceKind.MARKDOWN, SourceKind.JSON, SourceKind.REPOSITORY])
    ingest.add_argument("--gazetteer", default=None, help="JSON file mapping entity type -> names")

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


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    adapters = Adapters(gazetteer=_gazetteer(args.gazetteer)) if hasattr(args, "gazetteer") else Adapters()
    engine = create_engine(adapters)

    if args.command == "ingest":
        engine.ingest(Source(title=args.title, kind=args.kind, reference=args.path), args.path)
        print(json.dumps(engine.healthcheck(), indent=2))
    elif args.command == "retrieve":
        result = engine.retrieve(args.query, top_k=args.top_k)
        print(json.dumps(result.to_dict(), indent=2, default=str))
    elif args.command == "graphrag":
        evidence = engine.graphrag(args.query)
        print(json.dumps(evidence.to_dict(), indent=2, default=str))
    elif args.command == "gaps":
        gaps = engine.find_knowledge_gaps()
        print(json.dumps([
            {
                "kind": g.kind,
                "description": g.description,
                "priority": round(g.priority, 4),
                "estimated_cost": g.estimated_cost,
            }
            for g in gaps
        ], indent=2))
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
