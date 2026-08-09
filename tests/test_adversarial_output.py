"""Section 12: adversarial agent output must not bypass the trust boundary.

Hallucinated observation references, tool-status leakage into claims, forged
source ids, cross-run provenance, and injected contradictions are all
rejected, surfaced, or neutralized deterministically.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from nexus_runtime.investigation.candidate_claims import CandidateClaimExtractor
from nexus_runtime.investigation.evaluation import EvidenceEvaluator
from nexus_runtime.investigation.evidence import (
    AgentConclusion,
    ClaimStatement,
    EvidenceSet,
    InvestigationResult,
    InvestigationResultState,
    ToolObservation,
)
from nexus_runtime.investigation.verification import ClaimVerifier, EpistemicStatus

SESSION = "session-adv1"
INVESTIGATION = "investigation-adv1"
TASK = "task-adv1"
ATTEMPT = "attempt-adv1"
RUN = "run-adv1"


def observation(
    observation_id: str,
    source: str,
    status: str = "SUCCEEDED",
    excerpt: str = "report from source",
) -> ToolObservation:
    return ToolObservation(
        observation_id=observation_id,
        tool_name="search",
        status=status,
        input={"source": source},
        output={"excerpt": excerpt},
        source_reference=f"source://{source}",
        metadata={
            "source_id": f"source-{source}",
            "document_id": f"document-{source}",
            "chunk_id": f"chunk-{source}",
            "source_reference": f"source://{source}",
            "source_quality": 0.9,
        },
    )


def claim_statement(
    claim_id: str,
    subject: str,
    object_value: str,
) -> ClaimStatement:
    return ClaimStatement(
        text=f"{subject} is {object_value}",
        subject=subject,
        predicate="status",
        object=object_value,
        claim_id=claim_id,
    )


def conclusion(
    statement: ClaimStatement,
    observation_ids: tuple[str, ...],
    conclusion_id: str,
    confidence: float = 0.8,
) -> AgentConclusion:
    return AgentConclusion(
        claim=statement,
        supporting_observation_ids=observation_ids,
        confidence=confidence,
        conclusion_id=conclusion_id,
    )


def extraction_for(
    observations: tuple[ToolObservation, ...],
    conclusions: tuple[AgentConclusion, ...],
):
    investigation_result = InvestigationResult(
        session_id=SESSION,
        investigation_id=INVESTIGATION,
        task_id=TASK,
        attempt_id=ATTEMPT,
        run_id=RUN,
        state=InvestigationResultState.COMPLETED,
        evidence_set=EvidenceSet(session_id=SESSION, evidence=()),
        conclusions=conclusions,
        observations=observations,
        final_answer="ok",
    )
    return CandidateClaimExtractor().extract(investigation_result)


def test_hallucinated_observation_reference_never_creates_evidence() -> None:
    statement = claim_statement("claim-a", "Acme", "active")
    extraction = extraction_for(
        (observation("observation-real", "real"),),
        (conclusion(statement, ("observation-real", "observation-hallucinated"), "conclusion-a"),),
    )

    assert not extraction.candidates
    assert not extraction.evidence_set.evidence
    assert any(
        diagnostic.code == "unknown_observation_reference"
        and "observation-hallucinated" in diagnostic.message
        for diagnostic in extraction.diagnostics
    )


def test_pure_hallucination_yields_no_claims() -> None:
    statement = claim_statement("claim-b", "Beta", "active")
    extraction = extraction_for(
        (),
        (conclusion(statement, ("observation-ghost-1", "observation-ghost-2"), "conclusion-b"),),
    )

    assert not extraction.candidates
    assert all(
        diagnostic.code == "unknown_observation_reference"
        for diagnostic in extraction.diagnostics
    )


def test_tool_status_never_becomes_claim_text() -> None:
    failed = observation("observation-failed", "unreachable", status="FAILED", excerpt="")
    statement = claim_statement("claim-c", "Gamma", "active")
    extraction = extraction_for(
        (failed,),
        (conclusion(statement, ("observation-failed",), "conclusion-c"),),
    )

    candidate = extraction.candidates[0]
    assert candidate.claim.text == "Gamma is active"
    assert candidate.claim.object == "active"
    assert candidate.claim.subject == "Gamma"
    assert "FAILED" not in candidate.claim.text
    assert "FAILED" not in candidate.evidence[0].excerpt


def test_cross_run_evidence_injection_is_rejected_at_result_boundary() -> None:
    statement = claim_statement("claim-d", "Delta", "active")
    extraction = extraction_for(
        (observation("observation-a", "a"),),
        (conclusion(statement, ("observation-a",), "conclusion-d"),),
    )
    forged = replace(
        extraction.evidence_set.evidence[0],
        provenance=replace(
            extraction.evidence_set.evidence[0].provenance,
            run_id="run-other",
        ),
    )

    with pytest.raises(ValueError, match="mismatched result lineage"):
        InvestigationResult(
            session_id=SESSION,
            investigation_id=INVESTIGATION,
            task_id=TASK,
            attempt_id=ATTEMPT,
            run_id=RUN,
            state=InvestigationResultState.COMPLETED,
            evidence_set=EvidenceSet(session_id=SESSION, evidence=(forged,)),
            observations=(observation("observation-a", "a"),),
            conclusions=(
                conclusion(
                    claim_statement("claim-d", "Delta", "active"),
                    ("observation-a",),
                    "conclusion-d",
                ),
            ),
        )


def test_cross_session_evidence_injection_is_rejected_in_evidence_set() -> None:
    statement = claim_statement("claim-e", "Epsilon", "active")
    extraction = extraction_for(
        (observation("observation-a", "a"),),
        (conclusion(statement, ("observation-a",), "conclusion-e"),),
    )
    forged = replace(
        extraction.evidence_set.evidence[0],
        provenance=replace(
            extraction.evidence_set.evidence[0].provenance,
            session_id="session-other",
        ),
    )

    with pytest.raises(ValueError, match="evidence belongs to another session"):
        EvidenceSet(session_id=SESSION, evidence=(forged,))


def test_duplicate_source_inflation_never_raises_independent_support() -> None:
    statement = claim_statement("claim-f", "Zeta", "active")
    copies = tuple(
        observation(f"observation-copy-{index}", "single-source", excerpt=f"copy {index}")
        for index in range(5)
    )
    extraction = extraction_for(
        copies,
        (conclusion(statement, tuple(item.observation_id for item in copies), "conclusion-f"),),
    )

    evaluation = EvidenceEvaluator().evaluate(extraction.evidence_set)
    report = ClaimVerifier().verify(evaluation)

    assert evaluation.claims[0].independent_source_count == 1
    assert report.decisions[0].status == EpistemicStatus.INSUFFICIENT_EVIDENCE
    assert not report.decisions[0].eligible_for_update


def test_forged_source_id_reuse_cannot_inflate_corroboration() -> None:
    statement = claim_statement("claim-g", "Eta", "active")
    first = observation("observation-1", "shared-source", excerpt="first passage")
    second = observation("observation-2", "shared-source", excerpt="second passage")
    extraction = extraction_for(
        (first, second),
        (conclusion(statement, ("observation-1", "observation-2"), "conclusion-g"),),
    )

    evaluation = EvidenceEvaluator().evaluate(extraction.evidence_set)
    report = ClaimVerifier().verify(evaluation)

    assert evaluation.claims[0].independent_source_count == 1
    assert report.decisions[0].status == EpistemicStatus.INSUFFICIENT_EVIDENCE
    assert not report.decisions[0].eligible_for_update


def test_injected_contradiction_is_surfaced_not_merged() -> None:
    active = claim_statement("claim-h-active", "Theta", "active")
    inactive = claim_statement("claim-h-inactive", "Theta", "inactive")
    extraction = extraction_for(
        (observation("observation-h1", "theta-one"), observation("observation-h2", "theta-two")),
        (
            conclusion(active, ("observation-h1",), "conclusion-h-active"),
            conclusion(inactive, ("observation-h2",), "conclusion-h-inactive"),
        ),
    )

    evaluation = EvidenceEvaluator().evaluate(extraction.evidence_set)
    report = ClaimVerifier().verify(evaluation)

    assert len(evaluation.conflict_ids) == 1
    assert {claim.claim.object for claim in evaluation.claims} == {"active", "inactive"}
    assert all(decision.status == EpistemicStatus.CONTRADICTED for decision in report.decisions)
    assert all(not decision.eligible_for_update for decision in report.decisions)


def test_malformed_conclusion_text_is_never_coerced_into_a_claim() -> None:
    investigation_result = InvestigationResult(
        session_id=SESSION,
        investigation_id=INVESTIGATION,
        task_id=TASK,
        attempt_id=ATTEMPT,
        run_id=RUN,
        state=InvestigationResultState.COMPLETED,
        evidence_set=EvidenceSet(session_id=SESSION, evidence=()),
        observations=(observation("observation-a", "a"),),
        conclusions=(),
        final_answer="ok",
        metadata={
            "malformed_conclusions": [
                {"conclusion_id": "conclusion-broken", "reason": "claim not structured"}
            ]
        },
    )

    extraction = CandidateClaimExtractor().extract(investigation_result)

    assert not extraction.candidates
    assert any(
        diagnostic.code == "malformed_conclusion"
        and diagnostic.conclusion_id == "conclusion-broken"
        for diagnostic in extraction.diagnostics
    )


def test_observation_cited_twice_does_not_double_count_evidence() -> None:
    statement = claim_statement("claim-i", "Iota", "active")
    same_observation = observation("observation-single", "iota-one")
    extraction = extraction_for(
        (same_observation,),
        (
            conclusion(statement, ("observation-single",), "conclusion-i-one"),
            conclusion(statement, ("observation-single",), "conclusion-i-two"),
        ),
    )

    assert len(extraction.candidates) == 1
    assert len(extraction.evidence_set.evidence) == 1
