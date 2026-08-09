"""Section 18: pipeline performance sanity at scale.

The acquisition pipeline must not lose or duplicate evidence as the corpus
grows, must stay linear-ish (not quadratic/cubic), and must complete a
large synthetic session comfortably within a wall-clock budget.
"""

from __future__ import annotations

import time

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
from nexus_runtime.investigation.verification import ClaimVerifier

SESSION = "session-perf"
INVESTIGATION = "investigation-perf"
TASK = "task-perf"
ATTEMPT = "attempt-perf"
RUN = "run-perf"

_WALL_CLOCK_BUDGET_SECONDS = 10.0


def _observation(observation_id: str) -> ToolObservation:
    return ToolObservation(
        observation_id=observation_id,
        tool_name="search",
        status="SUCCEEDED",
        input={"query": observation_id},
        output={"excerpt": f"independent record {observation_id}"},
        source_reference=f"source://{observation_id}",
        metadata={
            "source_id": f"source-{observation_id}",
            "document_id": f"document-{observation_id}",
            "chunk_id": f"chunk-{observation_id}",
            "source_reference": f"source://{observation_id}",
            "source_quality": 0.9,
        },
    )


def _claim(index: int) -> ClaimStatement:
    return ClaimStatement(
        text=f"Entity{index} is active",
        subject=f"Entity{index}",
        predicate="status",
        object="active",
        claim_id=f"claim-perf-{index}",
    )


def _result(claim_count: int) -> InvestigationResult:
    observations = tuple(
        _observation(f"observation-{index}-a") for index in range(claim_count)
    ) + tuple(_observation(f"observation-{index}-b") for index in range(claim_count))
    conclusions = tuple(
        AgentConclusion(
            claim=_claim(index),
            supporting_observation_ids=(f"observation-{index}-a", f"observation-{index}-b"),
            confidence=0.9,
            conclusion_id=f"conclusion-perf-{index}",
        )
        for index in range(claim_count)
    )
    return InvestigationResult(
        session_id=SESSION,
        investigation_id=INVESTIGATION,
        task_id=TASK,
        attempt_id=ATTEMPT,
        run_id=RUN,
        state=InvestigationResultState.COMPLETED,
        evidence_set=EvidenceSet(session_id=SESSION, evidence=()),
        conclusions=conclusions,
        observations=observations,
    )


def _run(claim_count: int):
    started = time.perf_counter()
    extraction = CandidateClaimExtractor().extract(_result(claim_count))
    report = ClaimVerifier().verify(EvidenceEvaluator().evaluate(extraction.evidence_set))
    elapsed = time.perf_counter() - started
    return extraction, report, elapsed


def test_scale_handles_two_hundred_claims_without_loss_or_duplication() -> None:
    extraction, report, elapsed = _run(200)

    assert elapsed < _WALL_CLOCK_BUDGET_SECONDS
    assert len(extraction.candidates) == 200
    assert len(extraction.evidence_set.evidence) == 400
    assert len({item.evidence_id for item in extraction.evidence_set.evidence}) == 400
    assert len(report.decisions) == 200
    assert {item.status.value for item in report.decisions} == {"confirmed"}
    assert len(report.eligible_claims) == 200


def test_scale_does_not_blow_up_superlinearly() -> None:
    _, _, small_elapsed = _run(50)
    _, _, large_elapsed = _run(400)

    assert large_elapsed < small_elapsed * 12.0


def test_scale_keeps_evidence_ids_deterministic() -> None:
    first = _run(150)[0]
    second = _run(150)[0]

    assert [item.evidence_id for item in first.evidence_set.evidence] == [
        item.evidence_id for item in second.evidence_set.evidence
    ]
    assert len(first.evidence_set.evidence) == 300
