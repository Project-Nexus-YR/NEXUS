"""Sections 9-11: explicit acquisition lifecycle outcomes.

Verified claims advance to knowledge; deferred claims are preserved with their
evidence and reason (never committed); rejected claims never become knowledge.
A single session can mix all three outcomes without cross-contamination.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from nexus_runtime.investigation.acquisition import (
    AcquisitionReport,
    CandidateStatus,
    ClaimAcquisitionService,
)
from nexus_runtime.investigation.candidate_claims import (
    CandidateClaimExtractor,
)
from nexus_runtime.investigation.evaluation import EvidenceEvaluator
from nexus_runtime.investigation.evidence import (
    AgentConclusion,
    ClaimStatement,
    EvidenceSet,
    InvestigationResult,
    InvestigationResultState,
    ToolObservation,
)
from nexus_runtime.investigation.knowledge_update import KnowledgeUpdateIntegrator
from nexus_runtime.investigation.verification import (
    ClaimVerifier,
    EpistemicStatus,
    VerificationPolicy,
)

SESSION = "session-acq1"
INVESTIGATION = "investigation-acq1"
TASK = "task-acq1"
ATTEMPT = "attempt-acq1"
RUN = "run-acq1"


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


def observation(observation_id: str, source: str) -> ToolObservation:
    return ToolObservation(
        observation_id=observation_id,
        tool_name="search",
        status="SUCCEEDED",
        input={"source": source},
        output={"excerpt": f"report from {source}"},
        source_reference=f"source://{source}",
        metadata={
            "source_id": f"source-{source}",
            "document_id": f"document-{source}",
            "chunk_id": f"chunk-{source}",
            "source_reference": f"source://{source}",
            "source_quality": 0.9,
        },
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


def verification_for(extraction):
    evaluation = EvidenceEvaluator().evaluate(extraction.evidence_set)
    return ClaimVerifier().verify(evaluation)


class _AcceptingKnowledge:
    def validate_evidence_provenance(
        self,
        source_id: str,
        document_id: str,
        chunk_id: str,
        source_reference: str,
    ) -> bool:
        return all((source_id, document_id, chunk_id, source_reference))


# ---------------------------------------------------------------------------
# Section 9: deferred claims preserve evidence and reason, and never commit.
# ---------------------------------------------------------------------------


def test_single_source_candidate_is_deferred_with_reason_and_evidence() -> None:
    statement = claim_statement("claim-beta", "Beta", "active")
    extraction = extraction_for(
        (observation("observation-b", "beta"),),
        (conclusion(statement, ("observation-b",), "conclusion-beta"),),
    )

    report = ClaimAcquisitionService().acquire(extraction, verification_for(extraction))

    acquisition = report.deferred[0]
    assert report.verified == ()
    assert report.rejected == ()
    assert acquisition.status == CandidateStatus.DEFERRED
    assert acquisition.decision is not None
    assert acquisition.decision.status == EpistemicStatus.INSUFFICIENT_EVIDENCE
    assert "independent source" in acquisition.reason
    assert acquisition.candidate.evidence_ids == extraction.candidates[0].evidence_ids
    assert acquisition.candidate.claim.claim_id == "claim-beta"


def test_probable_candidate_is_deferred_when_updates_disallowed() -> None:
    policy = VerificationPolicy(
        min_independent_sources=1,
        confidence_threshold=0.9,
        probable_threshold=0.5,
    )
    statement = claim_statement("claim-probable", "Delta", "active")
    extraction = extraction_for(
        (observation("observation-d1", "delta-one"),),
        (conclusion(statement, ("observation-d1",), "conclusion-probable", confidence=0.8),),
    )

    evaluation = EvidenceEvaluator().evaluate(extraction.evidence_set)
    report = ClaimAcquisitionService().acquire(extraction, ClaimVerifier(policy).verify(evaluation))

    acquisition = report.acquisitions[0]
    assert acquisition.status == CandidateStatus.DEFERRED
    assert acquisition.decision.status == EpistemicStatus.PROBABLE
    assert "probable" in acquisition.reason


def test_deferred_acquisition_round_trips_through_serialization() -> None:
    statement = claim_statement("claim-beta", "Beta", "active")
    extraction = extraction_for(
        (observation("observation-b", "beta"),),
        (conclusion(statement, ("observation-b",), "conclusion-beta"),),
    )
    report = ClaimAcquisitionService().acquire(extraction, verification_for(extraction))

    restored = AcquisitionReport.from_dict(report.to_dict())

    assert restored.session_id == SESSION
    assert len(restored.deferred) == 1
    deferred = restored.deferred[0]
    assert deferred.status == CandidateStatus.DEFERRED
    assert deferred.reason == report.deferred[0].reason
    assert deferred.candidate.evidence_ids == report.deferred[0].candidate.evidence_ids
    assert deferred.decision.to_dict() == report.deferred[0].decision.to_dict()


def test_deferred_claim_never_reaches_knowledge_submission() -> None:
    statement = claim_statement("claim-beta", "Beta", "active")
    extraction = extraction_for(
        (observation("observation-b", "beta"),),
        (conclusion(statement, ("observation-b",), "conclusion-beta"),),
    )
    report = ClaimAcquisitionService().acquire(extraction, verification_for(extraction))

    submission = KnowledgeUpdateIntegrator(_AcceptingKnowledge()).prepare(
        verification_for(extraction), extraction.evidence_set
    )

    assert report.deferred
    assert submission.claims == ()


# ---------------------------------------------------------------------------
# Section 10: rejected claims never become knowledge.
# ---------------------------------------------------------------------------


def test_contradicted_candidate_is_rejected_with_reason() -> None:
    active = claim_statement("claim-gamma-active", "Gamma", "active")
    inactive = claim_statement("claim-gamma-inactive", "Gamma", "inactive")
    extraction = extraction_for(
        (observation("observation-g1", "gamma-one"), observation("observation-g2", "gamma-two")),
        (
            conclusion(active, ("observation-g1",), "conclusion-gamma-active"),
            conclusion(inactive, ("observation-g2",), "conclusion-gamma-inactive"),
        ),
    )
    verification = verification_for(extraction)

    report = ClaimAcquisitionService().acquire(extraction, verification)

    assert report.verified == ()
    assert len(report.rejected) == 2
    for acquisition in report.rejected:
        assert acquisition.status == CandidateStatus.REJECTED
        assert acquisition.decision.status == EpistemicStatus.CONTRADICTED
        assert "contradict" in acquisition.reason
    assert report.deferred == ()


def test_candidate_without_a_verification_decision_is_rejected() -> None:
    acme = claim_statement("claim-acme", "Acme", "active")
    orphan = claim_statement("claim-orphan", "Orphan", "active")
    extraction = extraction_for(
        (
            observation("observation-a1", "acme-one"),
            observation("observation-a2", "acme-two"),
            observation("observation-o", "orphan"),
        ),
        (
            conclusion(acme, ("observation-a1", "observation-a2"), "conclusion-acme"),
            conclusion(orphan, ("observation-o",), "conclusion-orphan"),
        ),
    )
    acme_evidence = tuple(
        item for item in extraction.evidence_set.evidence if item.claim.claim_id == "claim-acme"
    )
    partial_evidence = EvidenceSet(session_id=SESSION, evidence=acme_evidence)
    verification = ClaimVerifier().verify(EvidenceEvaluator().evaluate(partial_evidence))

    report = ClaimAcquisitionService().acquire(extraction, verification)

    orphan_acquisition = next(
        item for item in report.acquisitions if item.candidate.claim.claim_id == "claim-orphan"
    )
    assert orphan_acquisition.status == CandidateStatus.REJECTED
    assert orphan_acquisition.decision is None
    assert "no verification decision" in orphan_acquisition.reason


def test_rejected_candidate_keeps_its_evidence_in_the_report() -> None:
    active = claim_statement("claim-gamma-active", "Gamma", "active")
    inactive = claim_statement("claim-gamma-inactive", "Gamma", "inactive")
    extraction = extraction_for(
        (observation("observation-g1", "gamma-one"), observation("observation-g2", "gamma-two")),
        (
            conclusion(active, ("observation-g1",), "conclusion-gamma-active"),
            conclusion(inactive, ("observation-g2",), "conclusion-gamma-inactive"),
        ),
    )

    report = ClaimAcquisitionService().acquire(extraction, verification_for(extraction))

    for acquisition in report.rejected:
        assert acquisition.candidate.evidence_ids
        assert acquisition.candidate.evidence[0].provenance.source_id.startswith("source-gamma")


# ---------------------------------------------------------------------------
# Section 11: a mixed session keeps verified, deferred, and rejected apart.
# ---------------------------------------------------------------------------


def test_mixed_session_reports_all_lifecycle_outcomes() -> None:
    acme = claim_statement("claim-acme", "Acme", "active")
    beta = claim_statement("claim-beta", "Beta", "active")
    gamma_active = claim_statement("claim-gamma-active", "Gamma", "active")
    gamma_inactive = claim_statement("claim-gamma-inactive", "Gamma", "inactive")
    extraction = extraction_for(
        (
            observation("observation-a1", "acme-one"),
            observation("observation-a2", "acme-two"),
            observation("observation-b", "beta"),
            observation("observation-g1", "gamma-one"),
            observation("observation-g2", "gamma-two"),
        ),
        (
            conclusion(acme, ("observation-a1", "observation-a2"), "conclusion-acme"),
            conclusion(beta, ("observation-b",), "conclusion-beta"),
            conclusion(gamma_active, ("observation-g1",), "conclusion-gamma-active"),
            conclusion(gamma_inactive, ("observation-g2",), "conclusion-gamma-inactive"),
        ),
    )

    report = ClaimAcquisitionService().acquire(extraction, verification_for(extraction))

    assert [item.candidate.claim.claim_id for item in report.verified] == ["claim-acme"]
    assert [item.candidate.claim.claim_id for item in report.deferred] == ["claim-beta"]
    assert {item.candidate.claim.claim_id for item in report.rejected} == {
        "claim-gamma-active",
        "claim-gamma-inactive",
    }


def test_mixed_session_submission_contains_only_verified_claims() -> None:
    acme = claim_statement("claim-acme", "Acme", "active")
    beta = claim_statement("claim-beta", "Beta", "active")
    extraction = extraction_for(
        (
            observation("observation-a1", "acme-one"),
            observation("observation-a2", "acme-two"),
            observation("observation-b", "beta"),
        ),
        (
            conclusion(acme, ("observation-a1", "observation-a2"), "conclusion-acme"),
            conclusion(beta, ("observation-b",), "conclusion-beta"),
        ),
    )
    verification = verification_for(extraction)

    report = ClaimAcquisitionService().acquire(extraction, verification)
    submission = KnowledgeUpdateIntegrator(_AcceptingKnowledge()).prepare(
        verification, extraction.evidence_set
    )

    assert [item.candidate.claim.claim_id for item in report.verified] == ["claim-acme"]
    assert {item.decision.claim.claim_id for item in submission.claims} == {"claim-acme"}


def test_acquisition_requires_same_session_for_extraction_and_verification() -> None:
    statement = claim_statement("claim-beta", "Beta", "active")
    extraction = extraction_for(
        (observation("observation-b", "beta"),),
        (conclusion(statement, ("observation-b",), "conclusion-beta"),),
    )
    foreign_verification = replace(
        verification_for(extraction),
        session_id="session-other",
    )

    with pytest.raises(ValueError, match="same session"):
        ClaimAcquisitionService().acquire(extraction, foreign_verification)


def test_candidate_status_enum_is_complete() -> None:
    assert CandidateStatus.CANDIDATE.value == "candidate"
    assert CandidateStatus.EVALUATED.value == "evaluated"
    assert CandidateStatus.VERIFIED.value == "verified"
    assert CandidateStatus.REJECTED.value == "rejected"
    assert CandidateStatus.DEFERRED.value == "deferred"
