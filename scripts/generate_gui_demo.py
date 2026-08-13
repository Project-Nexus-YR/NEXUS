"""Generate a clearly separated, deterministic GUI development snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

from nexus_knowledge.domain.source import Source, SourceKind
from nexus_knowledge.eval.fixtures import build_corpus
from nexus_knowledge.persistence.json_codec import save_snapshot
from nexus_knowledge.service.factory import Adapters, create_engine

GAZETTEER = {
    "Company": ["Acme Corp", "Initech", "Umbrella Corp", "Sterling Labs"],
    "Person": ["Ada Lovelace", "Alan Turing", "Grace Hopper", "Nikola Tesla", "Marie Curie"],
    "City": ["London", "Menlo Park", "Manhattan", "Vienna"],
    "Institution": ["Cambridge"],
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", nargs="?", type=Path, default=Path(".nexus/demo-knowledge.json"))
    arguments = parser.parse_args()

    engine = create_engine(Adapters(gazetteer=GAZETTEER))
    for reference, text in build_corpus().documents:
        engine.ingest(
            Source(title=reference, kind=SourceKind.TEXT, reference=f"demo:{reference}"),
            text,
        )
    engine.detect_contradictions()
    engine.find_knowledge_gaps()
    save_snapshot(engine.repository, arguments.output)
    print(f"Wrote NEXUS GUI development snapshot to {arguments.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
