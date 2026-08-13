"""Command-line interface for claim extraction and citation verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nexus_knowledge.persistence.json_codec import load_snapshot
from nexus_knowledge.service.factory import Adapters, create_engine

from .benchmark import run_benchmark
from .extraction import ClaimExtractor
from .retrieval import KnowledgeEvidenceRetriever
from .workflow import EvidenceEngine


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="nexus-evidence", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    extract = commands.add_parser("extract", help="decompose prose into atomic claims")
    _draft_arguments(extract)
    audit = commands.add_parser("audit", help="verify claims against a knowledge snapshot")
    _draft_arguments(audit)
    audit.add_argument("--data", required=True, help="NEXUS knowledge JSON snapshot")
    audit.add_argument("--session-id", default="cli")
    audit.add_argument("--output", help="write the citation report as JSON")
    benchmark = commands.add_parser("bench", help="run the deterministic evidence benchmark")
    benchmark.add_argument("--output", help="write the benchmark report as JSON")
    return parser


def _draft_arguments(parser: argparse.ArgumentParser) -> None:
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--draft", help="draft answer text")
    source.add_argument("--draft-file", help="UTF-8 file containing the draft answer")


def _draft(args: argparse.Namespace) -> str:
    if args.draft_file:
        return Path(args.draft_file).read_text(encoding="utf-8")
    return str(args.draft)


def _emit(payload: object, output: str | None = None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if output:
        Path(output).write_text(rendered + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "extract":
        _emit([item.to_dict() for item in ClaimExtractor().extract(_draft(args))])
        return 0
    if args.command == "bench":
        _emit(run_benchmark().to_dict(), args.output)
        return 0
    repository = load_snapshot(args.data)
    knowledge = create_engine(Adapters(repository=repository))
    report = EvidenceEngine(KnowledgeEvidenceRetriever(knowledge)).verify_answer(
        _draft(args), session_id=args.session_id
    )
    _emit(report.to_dict(), args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
