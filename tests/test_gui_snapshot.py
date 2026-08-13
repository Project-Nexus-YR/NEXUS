"""Snapshot compatibility required by the local visual server."""

import json

from nexus_knowledge.cli import main as knowledge_main
from nexus_knowledge.domain.contradiction import Contradiction
from nexus_knowledge.domain.knowledge_gap import Investigation, KnowledgeGap
from nexus_knowledge.persistence.json_codec import load_snapshot, save_snapshot
from nexus_knowledge.service.explorer import KnowledgeExplorer
from nexus_knowledge.service.factory import Adapters, create_engine


def test_snapshot_v2_restores_analysis_records_enums_and_graph(ingested_engine, tmp_path):
    claim = ingested_engine.repository.claims.all()[0]
    ingested_engine.repository.contradictions.save(
        Contradiction(
            kind="stale_claim",
            claim_a_id=claim.id,
            claim_b_id=claim.id,
            description="Claim needs refresh",
            id="contra_snapshot",
        )
    )
    gap = KnowledgeGap(
        kind="missing_evidence",
        description="Independent evidence is missing",
        reason="Only one source family is present",
        affected_claims=[claim.id],
        id="gap_snapshot",
    )
    ingested_engine.repository.gaps.save(gap)
    ingested_engine.repository.investigations.save(
        Investigation(gap_id=gap.id, description="Find a second source", id="inv_snapshot")
    )
    path = tmp_path / "knowledge.json"

    save_snapshot(ingested_engine.repository, path)
    payload = json.loads(path.read_text())
    restored = load_snapshot(path)
    engine = create_engine(Adapters(repository=restored))
    projection = KnowledgeExplorer(engine).graph()

    assert payload["version"] == 2
    assert restored.gaps.get(gap.id).description == gap.description
    assert restored.contradictions.get("contra_snapshot").description == "Claim needs refresh"
    assert restored.investigations.get("inv_snapshot").gap_id == gap.id
    assert restored.claims.get(claim.id).verification_state.value == claim.verification_state.value
    assert engine.graph_statistics()["num_entities"] > 0
    assert claim.id in {node.id for node in projection.nodes}


def test_cli_ingestion_persists_file_content_for_gui(tmp_path):
    source = tmp_path / "research.txt"
    source.write_text("Ada Lovelace works at Acme Corp.", encoding="utf-8")
    gazetteer = tmp_path / "gazetteer.json"
    gazetteer.write_text(
        json.dumps({"Person": ["Ada Lovelace"], "Company": ["Acme Corp"]}),
        encoding="utf-8",
    )
    snapshot = tmp_path / "nested" / "knowledge.json"

    exit_code = knowledge_main(
        [
            "ingest",
            str(source),
            "--gazetteer",
            str(gazetteer),
            "--output",
            str(snapshot),
        ]
    )
    restored = load_snapshot(snapshot)

    assert exit_code == 0
    assert restored.documents.all()[0].text == "Ada Lovelace works at Acme Corp."
    assert restored.relations.all()
