"""Section 16: closed-loop learning against the real knowledge engine.

Investigation A learns a verified claim through the real engine's commit
boundary; the engine's own verifier then confirms it, and a subsequent query
(B) retrieves the grounding corpus the learned claim stands on.
"""

from __future__ import annotations

from nexus_knowledge.domain.source import Source, SourceKind
from nexus_runtime.investigation.evaluation import EvidenceEvaluator
from nexus_runtime.investigation.evidence import (
    ClaimStatement,
    Evidence,
    EvidenceSet,
)
from nexus_runtime.investigation.knowledge_update import KnowledgeUpdateIntegrator
from nexus_runtime.investigation.provenance import EvidenceProvenance
from nexus_runtime.investigation.verification import ClaimVerifier

SESSION = "session-closed-loop"
INVESTIGATION = "investigation-closed-loop"
TASK = "task-closed-loop"
ATTEMPT = "attempt-closed-loop"
RUN = "run-closed-loop"


def _evidence(statement: ClaimStatement, source: object, evidence_id: str) -> Evidence:
    provenance = EvidenceProvenance(
        session_id=SESSION,
        investigation_id=INVESTIGATION,
        task_id=TASK,
        attempt_id=ATTEMPT,
        run_id=RUN,
        tool_call_id=f"tool-{evidence_id}",
        source_id=source.source.id,
        document_id=source.documents[0].id,
        chunk_id=source.chunks[0].id,
        source_reference=source.source.reference,
    )
    return Evidence(
        investigation_id=INVESTIGATION,
        source=source.source.reference,
        claim=statement,
        provenance=provenance,
        confidence=0.95,
        source_quality=0.9,
        excerpt=source.chunks[0].text,
        payload={},
        evidence_id=evidence_id,
    )


def _learn(engine, source_a: object, source_b: object) -> KnowledgeUpdateIntegrator:
    statement = ClaimStatement(
        text="Atlas Corp is headquartered in London",
        subject="Atlas",
        predicate="headquartered_in",
        object="London",
        claim_id="claim-closed-loop-atlas",
    )
    evidence = EvidenceSet(
        session_id=SESSION,
        evidence=(
            _evidence(statement, source_a, "evidence-closed-loop-a"),
            _evidence(statement, source_b, "evidence-closed-loop-b"),
        ),
    )
    report = ClaimVerifier().verify(EvidenceEvaluator().evaluate(evidence))
    return KnowledgeUpdateIntegrator(engine), report, evidence


def _assert_verified(report) -> None:
    assert [(item.claim.claim_id, item.status.value) for item in report.decisions] == [
        ("claim-closed-loop-atlas", "confirmed")
    ]
    assert report.eligible_claims[0].claim.claim_id == "claim-closed-loop-atlas"


def test_closed_loop_a_learns_and_engine_confirms(ingested_engine) -> None:
    source_a = ingested_engine.ingest(
        Source("cl-a", SourceKind.TEXT, "closed-loop://a"),
        "Atlas Corp is headquartered in London, according to its registry filings.",
    )
    source_b = ingested_engine.ingest(
        Source("cl-b", SourceKind.TEXT, "closed-loop://b"),
        "The registry shows Atlas Corp headquartered in London for tax purposes.",
    )
    integrator, report, evidence = _learn(ingested_engine, source_a, source_b)
    _assert_verified(report)

    result = integrator.apply(integrator.prepare(report, evidence))

    assert result.committed_claim_ids == ("claim-closed-loop-atlas",)
    assert result.fully_applied
    assert result.rejected_records == 0
    assert result.verification_states["claim-closed-loop-atlas"] == "verified"


def test_closed_loop_claim_is_persisted_in_the_knowledge_store(ingested_engine) -> None:
    source_a = ingested_engine.ingest(
        Source("cl-a2", SourceKind.TEXT, "closed-loop://a2"),
        "Atlas Corp is headquartered in London, per official filings.",
    )
    source_b = ingested_engine.ingest(
        Source("cl-b2", SourceKind.TEXT, "closed-loop://b2"),
        "Official filings list Atlas Corp as headquartered in London.",
    )
    integrator, report, evidence = _learn(ingested_engine, source_a, source_b)
    integrator.apply(integrator.prepare(report, evidence))

    stored = ingested_engine.repository.claims.get("claim-closed-loop-atlas")

    assert stored is not None
    assert stored.subject == "Atlas"
    assert stored.predicate == "headquartered_in"
    assert stored.object == "London"
    assert ingested_engine.repository.claims.by_subject("Atlas")
    assert "claim-closed-loop-atlas" in {
        item.id for item in ingested_engine.repository.claims.by_subject("Atlas")
    }
    assert len(ingested_engine.repository.evidence.by_claim("claim-closed-loop-atlas")) == 2


def test_closed_loop_learned_claim_resolves_provenance(ingested_engine) -> None:
    source_a = ingested_engine.ingest(
        Source("cl-a3", SourceKind.TEXT, "closed-loop://a3"),
        "Atlas Corp is headquartered in London, per the registry.",
    )
    source_b = ingested_engine.ingest(
        Source("cl-b3", SourceKind.TEXT, "closed-loop://b3"),
        "The registry confirms Atlas Corp is headquartered in London.",
    )
    integrator, report, evidence = _learn(ingested_engine, source_a, source_b)
    integrator.apply(integrator.prepare(report, evidence))

    provenance = ingested_engine.provenance("claim-closed-loop-atlas")

    assert provenance.claim_text == "Atlas Corp is headquartered in London"
    assert {source_a.source.id, source_b.source.id} <= set(provenance.provenance.source_ids)
    assert {source_a.chunks[0].id, source_b.chunks[0].id} <= set(provenance.provenance.chunk_ids)
    assert len(provenance.evidence) == 2


def test_closed_loop_b_retrieves_the_grounding_corpus(ingested_engine) -> None:
    source_a = ingested_engine.ingest(
        Source("cl-a4", SourceKind.TEXT, "closed-loop://a4"),
        "Atlas Corp is headquartered in London, per corporate records.",
    )
    source_b = ingested_engine.ingest(
        Source("cl-b4", SourceKind.TEXT, "closed-loop://b4"),
        "Corporate records place Atlas Corp headquarters in London.",
    )
    integrator, report, evidence = _learn(ingested_engine, source_a, source_b)
    integrator.apply(integrator.prepare(report, evidence))

    retrieved = ingested_engine.retrieve("Atlas Corp headquartered in London", top_k=8)
    retrieved_chunk_ids = {candidate.chunk_id for candidate in retrieved.candidates}

    assert {source_a.chunks[0].id, source_b.chunks[0].id} <= retrieved_chunk_ids


def test_closed_loop_relearn_is_idempotent(ingested_engine) -> None:
    source_a = ingested_engine.ingest(
        Source("cl-a5", SourceKind.TEXT, "closed-loop://a5"),
        "Atlas Corp is headquartered in London, per the filings.",
    )
    source_b = ingested_engine.ingest(
        Source("cl-b5", SourceKind.TEXT, "closed-loop://b5"),
        "The filings record Atlas Corp headquarters in London.",
    )
    integrator, report, evidence = _learn(ingested_engine, source_a, source_b)
    submission = integrator.prepare(report, evidence)
    count_before = ingested_engine.repository.claims.count()

    first = integrator.apply(submission)
    second = integrator.apply(submission.from_dict(submission.to_dict()))

    assert first.committed_claim_ids == ("claim-closed-loop-atlas",)
    assert second.committed_claim_ids == first.committed_claim_ids
    assert ingested_engine.repository.claims.count() == count_before + 1
    assert len(ingested_engine.repository.evidence.by_claim("claim-closed-loop-atlas")) == 2


def test_closed_loop_learned_knowledge_serves_later_sessions(ingested_engine) -> None:
    source_a = ingested_engine.ingest(
        Source("cl-a6", SourceKind.TEXT, "closed-loop://a6"),
        "Atlas Corp is headquartered in London, per the records.",
    )
    source_b = ingested_engine.ingest(
        Source("cl-b6", SourceKind.TEXT, "closed-loop://b6"),
        "The records say Atlas Corp is headquartered in London.",
    )
    integrator, report, evidence = _learn(ingested_engine, source_a, source_b)
    integrator.apply(integrator.prepare(report, evidence))
    count_before = ingested_engine.repository.claims.count()

    assessment = ingested_engine.verify_claim("claim-closed-loop-atlas")

    assert assessment.verification_state.value == "verified"
    assert ingested_engine.repository.claims.count() == count_before
