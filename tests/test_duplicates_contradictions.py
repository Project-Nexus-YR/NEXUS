"""Section 7: duplicates and contradictions must not corrupt corroboration.

Duplicates of the same evidence are classified, never promoted to independent
support.  Contradicting claims are surfaced as conflicts and block trusted
verification instead of being silently merged or resolved by confidence.
"""

from __future__ import annotations

from nexus_runtime.investigation.evaluation import EvidenceEvaluator
from nexus_runtime.investigation.evidence import (
    ClaimStatement,
    Evidence,
    EvidenceRole,
    EvidenceSet,
)
from nexus_runtime.investigation.fusion import EvidenceFusion
from nexus_runtime.investigation.provenance import EvidenceProvenance
from nexus_runtime.investigation.verification import (
    ClaimVerifier,
    EpistemicStatus,
)


def _provenance(
    *,
    source_id: str = "source-a",
    source_reference: str = "https://example.test/a",
    document_id: str = "document-a",
    chunk_id: str = "chunk-a",
) -> EvidenceProvenance:
    return EvidenceProvenance(
        session_id="session-d1",
        investigation_id="investigation-d1",
        task_id="task-d1",
        attempt_id="attempt-d1",
        run_id="run-d1",
        tool_call_id=f"tool-{source_id}",
        source_id=source_id,
        document_id=document_id,
        chunk_id=chunk_id,
        source_reference=source_reference,
    )


def _evidence(
    *,
    claim: ClaimStatement | None = None,
    source_id: str = "source-a",
    source_reference: str = "https://example.test/a",
    excerpt: str = "The primary record states London.",
    confidence: float = 0.95,
    source_quality: float = 0.9,
    role: EvidenceRole = EvidenceRole.SUPPORTING,
    evidence_id: str | None = None,
    document_id: str = "document-a",
    chunk_id: str = "chunk-a",
) -> Evidence:
    kwargs = {}
    if evidence_id is not None:
        kwargs["evidence_id"] = evidence_id
    return Evidence(
        investigation_id="investigation-d1",
        source=source_reference,
        claim=claim
        or ClaimStatement(
            text="Atlas is headquartered in London",
            subject="Atlas",
            predicate="headquartered_in",
            object="London",
        ),
        provenance=_provenance(
            source_id=source_id,
            source_reference=source_reference,
            document_id=document_id,
            chunk_id=chunk_id,
        ),
        confidence=confidence,
        excerpt=excerpt,
        source_quality=source_quality,
        role=role,
        **kwargs,
    )


def _london() -> ClaimStatement:
    return ClaimStatement(
        text="Atlas is headquartered in London",
        subject="Atlas",
        predicate="headquartered_in",
        object="London",
    )


def _paris() -> ClaimStatement:
    return ClaimStatement(
        text="Atlas is headquartered in Paris",
        subject="Atlas",
        predicate="headquartered_in",
        object="Paris",
    )


def test_duplicate_evidence_does_not_inflate_independent_source_count() -> None:
    claim_statement = _london()
    original = _evidence(claim=claim_statement, evidence_id="evidence-original")
    duplicate = _evidence(claim=claim_statement, evidence_id="evidence-duplicate")

    evaluation = EvidenceEvaluator().evaluate(
        EvidenceSet(session_id="session-d1", evidence=(original, duplicate))
    )

    assert evaluation.duplicate_evidence_ids == ("evidence-duplicate",)
    assert evaluation.claims[0].independent_source_count == 1
    assert evaluation.accepted_evidence_count == 1


def test_two_copies_of_one_source_do_not_verify() -> None:
    claim_statement = _london()
    evidence = (
        _evidence(claim=claim_statement, evidence_id="evidence-a"),
        _evidence(claim=claim_statement, evidence_id="evidence-b"),
    )

    report = ClaimVerifier().verify(
        EvidenceEvaluator().evaluate(EvidenceSet(session_id="session-d1", evidence=evidence))
    )

    assert report.decisions[0].status == EpistemicStatus.INSUFFICIENT_EVIDENCE
    assert not report.decisions[0].eligible_for_update


def test_same_source_different_excerpts_count_as_one_independent_source() -> None:
    claim_statement = _london()
    evidence = (
        _evidence(
            claim=claim_statement,
            excerpt="Primary record states London.",
            evidence_id="evidence-a",
        ),
        _evidence(
            claim=claim_statement,
            excerpt="A second passage in the same document states London.",
            source_reference="https://example.test/a",
            evidence_id="evidence-b",
        ),
    )

    evaluation = EvidenceEvaluator().evaluate(
        EvidenceSet(session_id="session-d1", evidence=evidence)
    )

    assert evaluation.duplicate_evidence_ids == ()
    assert evaluation.claims[0].independent_source_count == 1


def test_two_independent_sources_verify_when_quality_sufficient() -> None:
    claim_statement = _london()
    evidence = (
        _evidence(claim=claim_statement, evidence_id="evidence-a"),
        _evidence(
            claim=claim_statement,
            source_id="source-b",
            source_reference="https://example.test/b",
            excerpt="Independent registry record.",
            evidence_id="evidence-b",
            document_id="document-b",
            chunk_id="chunk-b",
        ),
    )

    report = ClaimVerifier().verify(
        EvidenceEvaluator().evaluate(EvidenceSet(session_id="session-d1", evidence=evidence))
    )

    assert report.decisions[0].status == EpistemicStatus.CONFIRMED
    assert report.decisions[0].eligible_for_update


def test_duplicate_original_still_supports_the_claim() -> None:
    claim_statement = _london()
    original = _evidence(claim=claim_statement, evidence_id="evidence-original")
    duplicate = _evidence(claim=claim_statement, evidence_id="evidence-duplicate")

    evaluation = EvidenceEvaluator().evaluate(
        EvidenceSet(session_id="session-d1", evidence=(original, duplicate))
    )

    claim_evaluation = evaluation.claims[0]
    assert claim_evaluation.duplicate_evidence_ids == ("evidence-duplicate",)
    assert {item.evidence_id for item in claim_evaluation.supporting} == {"evidence-original"}


def test_contradicting_claims_block_verification_on_both_sides() -> None:
    london = _london()
    paris = _paris()
    evidence = (
        _evidence(claim=london, evidence_id="evidence-london"),
        _evidence(
            claim=paris,
            source_id="source-b",
            source_reference="https://example.test/b",
            excerpt="A second registry lists Paris.",
            evidence_id="evidence-paris",
            document_id="document-b",
            chunk_id="chunk-b",
        ),
    )

    evaluation = EvidenceEvaluator().evaluate(
        EvidenceSet(session_id="session-d1", evidence=evidence)
    )
    report = ClaimVerifier().verify(evaluation)

    assert len(evaluation.conflict_ids) == 1
    assert all(decision.status == EpistemicStatus.CONTRADICTED for decision in report.decisions)
    assert all(not decision.eligible_for_update for decision in report.decisions)


def test_explicit_contradicting_role_evidence_blocks_verification() -> None:
    claim_statement = _london()
    evidence = (
        _evidence(claim=claim_statement, evidence_id="evidence-a"),
        _evidence(
            claim=claim_statement,
            source_id="source-b",
            source_reference="https://example.test/b",
            excerpt="A rival report contradicts this.",
            role=EvidenceRole.CONTRADICTING,
            evidence_id="evidence-b",
            document_id="document-b",
            chunk_id="chunk-b",
        ),
    )

    evaluation = EvidenceEvaluator().evaluate(
        EvidenceSet(session_id="session-d1", evidence=evidence)
    )
    report = ClaimVerifier().verify(evaluation)

    assert evaluation.claims[0].unresolved_contradiction
    assert report.decisions[0].status == EpistemicStatus.CONTRADICTED
    assert not report.decisions[0].eligible_for_update


def test_conflict_id_is_deterministic_across_fusions() -> None:
    london = _london()
    paris = _paris()
    evidence = (
        _evidence(claim=london, evidence_id="evidence-london"),
        _evidence(
            claim=paris,
            source_id="source-b",
            source_reference="https://example.test/b",
            excerpt="A second registry lists Paris.",
            evidence_id="evidence-paris",
            document_id="document-b",
            chunk_id="chunk-b",
        ),
    )
    first = EvidenceFusion().fuse(EvidenceSet(session_id="session-d1", evidence=evidence))
    second = EvidenceFusion().fuse(EvidenceSet(session_id="session-d1", evidence=evidence))

    assert first.conflicts[0].conflict_id == second.conflicts[0].conflict_id
    assert first.conflicts[0].conflict_id


def test_opposite_claims_are_never_merged_into_corroboration() -> None:
    london = _london()
    paris = _paris()
    evidence = (
        _evidence(claim=london, evidence_id="evidence-london"),
        _evidence(
            claim=paris,
            source_id="source-b",
            source_reference="https://example.test/b",
            excerpt="A second registry lists Paris.",
            evidence_id="evidence-paris",
            document_id="document-b",
            chunk_id="chunk-b",
        ),
    )

    fused = EvidenceFusion().fuse(EvidenceSet(session_id="session-d1", evidence=evidence))

    assert len(fused.claims) == 2
    assert {claim.claim.object for claim in fused.claims} == {"London", "Paris"}
    assert len(fused.conflicts) == 1
    assert fused.conflicts[0].evidence_a_ids and fused.conflicts[0].evidence_b_ids
