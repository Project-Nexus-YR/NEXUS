"""Section 13: crash-replay safety across the acquisition pipeline.

Every persisted boundary round-trips deterministically: re-running from any
crash point produces identical claim ids, evidence ids, and lifecycle
outcomes, never duplicates, and never loses a deferred or contradicted claim.
"""

from __future__ import annotations

from nexus_knowledge.domain.source import Source, SourceKind
from nexus_runtime.investigation.acquisition import ClaimAcquisitionService
from nexus_runtime.investigation.candidate_claims import CandidateClaimExtractor
from nexus_runtime.investigation.evaluation import EvidenceEvaluator
from nexus_runtime.investigation.evidence import (
    AgentConclusion,
    ClaimStatement,
    Evidence,
    EvidenceSet,
    InvestigationResult,
    InvestigationResultState,
    ToolObservation,
)
from nexus_runtime.investigation.knowledge_update import KnowledgeUpdateIntegrator
from nexus_runtime.investigation.provenance import EvidenceProvenance
from nexus_runtime.investigation.verification import (
    ClaimVerifier,
)

SESSION = "session-replay1"
INVESTIGATION = "investigation-replay1"
TASK = "task-replay1"
ATTEMPT = "attempt-replay1"
RUN = "run-replay1"


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


def claim_statement(claim_id: str, subject: str, object_value: str) -> ClaimStatement:
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
) -> AgentConclusion:
    return AgentConclusion(
        claim=statement,
        supporting_observation_ids=observation_ids,
        confidence=0.8,
        conclusion_id=conclusion_id,
    )


def persisted_result(
    observations: tuple[ToolObservation, ...],
    conclusions: tuple[AgentConclusion, ...],
) -> InvestigationResult:
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
    return InvestigationResult.from_dict(investigation_result.to_dict())


def run_pipeline(
    observations: tuple[ToolObservation, ...],
    conclusions: tuple[AgentConclusion, ...],
):
    investigation_result = persisted_result(observations, conclusions)
    extraction = CandidateClaimExtractor().extract(investigation_result)
    evaluation = EvidenceEvaluator().evaluate(extraction.evidence_set)
    verification = ClaimVerifier().verify(evaluation)
    acquisition = ClaimAcquisitionService().acquire(extraction, verification)
    return extraction, evaluation, verification, acquisition

def ids(extraction) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    return (
        tuple(item.claim.claim_id for item in extraction.candidates),
        tuple(item.evidence_id for item in extraction.evidence_set.evidence),
        tuple(item.candidate_id for item in extraction.candidates),
    )


def verified_pair():
    return (
        (observation("observation-a1", "acme-one"), observation("observation-a2", "acme-two")),
        (
            conclusion(
                claim_statement("claim-acme", "Acme", "active"),
                ("observation-a1", "observation-a2"),
                "conclusion-acme",
            ),
        ),
    )


# Crash point A: persisted extraction, crash before evaluation.
def test_replay_after_extraction_reproduces_identical_evaluation() -> None:
    observations, conclusions = verified_pair()
    first = run_pipeline(observations, conclusions)

    restored_extraction = first[0].from_dict(first[0].to_dict())
    replayed_evaluation = EvidenceEvaluator().evaluate(restored_extraction.evidence_set)
    fresh_evaluation = EvidenceEvaluator().evaluate(first[0].evidence_set)

    assert ids(restored_extraction) == ids(first[0])
    assert {claim.claim.claim_id for claim in replayed_evaluation.claims} == {
        claim.claim.claim_id for claim in fresh_evaluation.claims
    }
    assert replayed_evaluation.conflict_ids == fresh_evaluation.conflict_ids
    assert replayed_evaluation.duplicate_evidence_ids == fresh_evaluation.duplicate_evidence_ids


# Crash point B: persisted evidence set, crash before verification.
def test_replay_after_evaluation_reproduces_identical_decisions() -> None:
    observations, conclusions = verified_pair()
    first = run_pipeline(observations, conclusions)

    restored_evidence = EvidenceSet.from_dict(first[0].evidence_set.to_dict())
    replayed_report = ClaimVerifier().verify(EvidenceEvaluator().evaluate(restored_evidence))

    assert [(item.claim.claim_id, item.status.value) for item in replayed_report.decisions] == [
        (item.claim.claim_id, item.status.value) for item in first[2].decisions
    ]


# Crash point C: persisted verification report, crash before knowledge update.
def test_replay_after_verification_reproduces_identical_submission() -> None:
    observations, conclusions = verified_pair()
    first = run_pipeline(observations, conclusions)

    restored_report = first[2].from_dict(first[2].to_dict())
    integrator = KnowledgeUpdateIntegrator(_AcceptingKnowledge())
    submission = integrator.prepare(restored_report, first[0].evidence_set)
    expected_evidence_ids = tuple(item.evidence_id for item in first[0].evidence_set.evidence)

    assert {item.decision.claim.claim_id for item in submission.claims} == {"claim-acme"}
    assert submission.evidence_ids == expected_evidence_ids


# Crash point D: committed knowledge, crash before ack. Re-submission is idempotent.
def test_replay_after_knowledge_commit_does_not_duplicate_claims(ingested_engine) -> None:
    first = ingested_engine.ingest(
        Source("replay-a", SourceKind.TEXT, "replay://a"),
        "Acme has a registered office in London.",
    )
    second = ingested_engine.ingest(
        Source("replay-b", SourceKind.TEXT, "replay://b"),
        "Acme maintains its headquarters in London.",
    )
    statement = claim_statement("claim-replay-acme", "Acme", "active")
    evidence = (
        _engine_evidence(statement, first, "evidence-replay-a"),
        _engine_evidence(statement, second, "evidence-replay-b"),
    )
    evidence_set = EvidenceSet(session_id=SESSION, evidence=evidence)
    report = ClaimVerifier().verify(EvidenceEvaluator().evaluate(evidence_set))
    integrator = KnowledgeUpdateIntegrator(ingested_engine)
    submission = integrator.prepare(report, evidence_set)

    first_apply = integrator.apply(submission)
    count_after_first = ingested_engine.repository.claims.count()
    replayed_submission = submission.from_dict(submission.to_dict())
    second_apply = integrator.apply(replayed_submission)

    assert first_apply.committed_claim_ids == ("claim-replay-acme",)
    assert second_apply.committed_claim_ids == first_apply.committed_claim_ids
    assert ingested_engine.repository.claims.count() == count_after_first


# Crash point E: replay of collection re-extracts identical evidence, no dupes.
def test_replay_after_collection_reproduces_identical_evidence_without_duplicates() -> None:
    observations, conclusions = verified_pair()

    first_extraction = CandidateClaimExtractor().extract(
        persisted_result(observations, conclusions)
    )
    second_extraction = CandidateClaimExtractor().extract(
        persisted_result(observations, conclusions)
    )

    assert ids(second_extraction) == ids(first_extraction)
    assert len(second_extraction.evidence_set.evidence) == len(
        first_extraction.evidence_set.evidence
    )
    assert len({item.evidence_id for item in second_extraction.evidence_set.evidence}) == len(
        second_extraction.evidence_set.evidence
    )


def test_full_pipeline_replay_is_identical_on_deterministic_semantics() -> None:
    observations, conclusions = verified_pair()

    first = run_pipeline(observations, conclusions)
    second = run_pipeline(observations, conclusions)

    assert ids(second[0]) == ids(first[0])
    assert _decision_semantics(second[2]) == _decision_semantics(first[2])
    assert _acquisition_semantics(second[3]) == _acquisition_semantics(first[3])


def _decision_semantics(report) -> tuple:
    return tuple(
        (
            decision.claim.claim_id,
            decision.status.value,
            decision.eligible_for_update,
            decision.supporting_evidence_ids,
            decision.contradicting_evidence_ids,
            decision.reasons,
        )
        for decision in report.decisions
    )


def _acquisition_semantics(report) -> tuple:
    return tuple(
        (
            acquisition.candidate.claim.claim_id,
            acquisition.status.value,
            acquisition.reason,
            acquisition.candidate.evidence_ids,
        )
        for acquisition in report.acquisitions
    )


def test_replay_preserves_deferred_claims_with_their_reason() -> None:
    observations = (observation("observation-b", "beta"),)
    conclusions = (
        conclusion(
            claim_statement("claim-beta", "Beta", "active"),
            ("observation-b",),
            "conclusion-beta",
        ),
    )

    first = run_pipeline(observations, conclusions)
    second = run_pipeline(observations, conclusions)

    first_deferred = first[3].deferred[0]
    second_deferred = second[3].deferred[0]
    assert first_deferred.status.value == "deferred"
    assert second_deferred.status.value == "deferred"
    assert second_deferred.reason == first_deferred.reason
    assert second_deferred.candidate.evidence_ids == first_deferred.candidate.evidence_ids
    assert second_deferred.candidate.claim.claim_id == "claim-beta"


def test_replay_preserves_contradictions_with_identical_conflict_ids() -> None:
    observations = (
        observation("observation-t1", "theta-one"),
        observation("observation-t2", "theta-two"),
    )
    conclusions = (
        conclusion(
            claim_statement("claim-theta-active", "Theta", "active"),
            ("observation-t1",),
            "conclusion-theta-active",
        ),
        conclusion(
            claim_statement("claim-theta-inactive", "Theta", "inactive"),
            ("observation-t2",),
            "conclusion-theta-inactive",
        ),
    )

    first = run_pipeline(observations, conclusions)
    second = run_pipeline(observations, conclusions)

    assert first[1].conflict_ids == second[1].conflict_ids
    assert len(first[1].conflict_ids) == 1
    assert {item.status.value for item in first[3].rejected} == {"rejected"}
    assert {item.status.value for item in second[3].rejected} == {"rejected"}


class _AcceptingKnowledge:
    def validate_evidence_provenance(
        self,
        source_id: str,
        document_id: str,
        chunk_id: str,
        source_reference: str,
    ) -> bool:
        return all((source_id, document_id, chunk_id, source_reference))


def _engine_evidence(statement: ClaimStatement, ingested, evidence_id: str) -> Evidence:
    provenance = EvidenceProvenance(
        session_id=SESSION,
        investigation_id=INVESTIGATION,
        task_id=TASK,
        attempt_id=ATTEMPT,
        run_id=RUN,
        tool_call_id=f"tool-{evidence_id}",
        source_id=ingested.source.id,
        document_id=ingested.documents[0].id,
        chunk_id=ingested.chunks[0].id,
        source_reference=ingested.source.reference,
    )
    return Evidence(
        investigation_id=INVESTIGATION,
        source=ingested.source.reference,
        claim=statement,
        provenance=provenance,
        confidence=0.95,
        source_quality=0.9,
        excerpt=ingested.chunks[0].text,
        payload={},
        evidence_id=evidence_id,
    )
