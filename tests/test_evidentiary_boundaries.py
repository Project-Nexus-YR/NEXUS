"""Section 15: evidentiary-strength boundary and clamping validation.

A single inflated signal cannot carry an item: the geometric mean collapses to
zero when either component is zero, out-of-range inputs are rejected or
clamped, and the policy gate honors its thresholds exactly.
"""

from __future__ import annotations

import pytest

from nexus_runtime.investigation.candidate_claims import CandidateClaimExtractor
from nexus_runtime.investigation.evaluation import EvidenceEvaluator, EvidenceQualityPolicy
from nexus_runtime.investigation.evidence import (
    ClaimStatement,
    Evidence,
    EvidenceGrade,
    EvidenceSet,
    InvestigationResult,
    InvestigationResultState,
    ToolObservation,
    grade_for_strength,
)
from nexus_runtime.investigation.provenance import EvidenceProvenance


def _evidence(
    confidence: float,
    source_quality: float,
    *,
    evidence_id: str = "evidence-boundary",
    source_id: str = "source-b1",
    excerpt: str = "The primary record states London.",
) -> Evidence:
    provenance = EvidenceProvenance(
        session_id="session-b1",
        investigation_id="investigation-b1",
        task_id="task-b1",
        attempt_id="attempt-b1",
        run_id="run-b1",
        tool_call_id=f"tool-{source_id}",
        source_id=source_id,
        document_id=f"document-{source_id}",
        chunk_id=f"chunk-{source_id}",
        source_reference=f"https://example.test/{source_id}",
    )
    claim = ClaimStatement(
        text="Atlas is headquartered in London",
        subject="Atlas",
        predicate="headquartered_in",
        object="London",
    )
    return Evidence(
        investigation_id="investigation-b1",
        source=f"https://example.test/{source_id}",
        claim=claim,
        provenance=provenance,
        confidence=confidence,
        source_quality=source_quality,
        excerpt=excerpt,
        evidence_id=evidence_id,
    )


def test_zero_source_quality_collapses_strength_despite_max_confidence() -> None:
    evidence = _evidence(confidence=1.0, source_quality=0.0)

    assert evidence.evidentiary_strength == 0.0
    assert evidence.grade == EvidenceGrade.WEAK


def test_zero_confidence_collapses_strength_despite_max_quality() -> None:
    evidence = _evidence(confidence=0.0, source_quality=1.0)

    assert evidence.evidentiary_strength == 0.0
    assert evidence.grade == EvidenceGrade.WEAK


def test_single_inflated_signal_never_carries_an_item() -> None:
    inflated_confidence = _evidence(confidence=1.0, source_quality=0.25, evidence_id="e-c")
    inflated_quality = _evidence(confidence=0.25, source_quality=1.0, evidence_id="e-q")

    assert inflated_confidence.evidentiary_strength == pytest.approx(0.5)
    assert inflated_quality.evidentiary_strength == pytest.approx(0.5)
    assert inflated_confidence.grade == EvidenceGrade.MODERATE


def test_strength_is_bounded_to_the_unit_interval() -> None:
    assert 0.0 <= _evidence(confidence=0.0, source_quality=0.0).evidentiary_strength <= 1.0
    assert 0.0 <= _evidence(confidence=1.0, source_quality=1.0).evidentiary_strength <= 1.0


def test_grade_thresholds_are_inclusive_at_the_boundary() -> None:
    assert grade_for_strength(0.7) == EvidenceGrade.STRONG
    assert grade_for_strength(0.4) == EvidenceGrade.MODERATE
    assert grade_for_strength(0.0) == EvidenceGrade.WEAK


def test_policy_gate_honors_min_strength_exactly() -> None:
    at_threshold = _evidence(confidence=0.9, source_quality=0.9, evidence_id="e-at")
    below_threshold = _evidence(
        confidence=0.88,
        source_quality=0.88,
        evidence_id="e-below",
        source_id="source-b2",
        excerpt="A secondary registry also states London.",
    )
    policy = EvidenceQualityPolicy(min_evidentiary_strength=0.89)

    evaluation = EvidenceEvaluator(policy).evaluate(
        EvidenceSet(session_id="session-b1", evidence=(at_threshold, below_threshold))
    )

    assert evaluation.low_quality_evidence_ids == ("e-below",)
    assert evaluation.accepted_evidence_count == 1


def test_out_of_range_confidence_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        _evidence(confidence=1.5, source_quality=0.5)


def test_out_of_range_source_quality_is_rejected_at_construction() -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        _evidence(confidence=0.5, source_quality=-0.1)


def test_metadata_source_quality_is_clamped_by_the_extractor() -> None:
    claim = ClaimStatement(
        text="Atlas is headquartered in London",
        subject="Atlas",
        predicate="headquartered_in",
        object="London",
    )
    observation = ToolObservation(
        observation_id="observation-clamp",
        tool_name="search",
        status="SUCCEEDED",
        input={},
        output={"text": "registry"},
        source_reference="https://example.test/clamp",
        metadata={"source_quality": 7.5},
    )
    conclusion = _conclusion(claim, ("observation-clamp",))
    investigation_result = InvestigationResult(
        session_id="session-b1",
        investigation_id="investigation-b1",
        task_id="task-b1",
        attempt_id="attempt-b1",
        run_id="run-b1",
        state=InvestigationResultState.COMPLETED,
        evidence_set=EvidenceSet(session_id="session-b1", evidence=()),
        conclusions=(conclusion,),
        observations=(observation,),
    )

    evidence = CandidateClaimExtractor().extract(investigation_result).evidence_set.evidence[0]

    assert evidence.source_quality == 1.0


def test_metadata_source_quality_clamps_low_values_to_zero() -> None:
    claim = ClaimStatement(
        text="Atlas is headquartered in London",
        subject="Atlas",
        predicate="headquartered_in",
        object="London",
    )
    observation = ToolObservation(
        observation_id="observation-clamp-low",
        tool_name="search",
        status="SUCCEEDED",
        input={},
        output={"text": "registry"},
        source_reference="https://example.test/clamp-low",
        metadata={"source_quality": -3.0},
    )
    investigation_result = InvestigationResult(
        session_id="session-b1",
        investigation_id="investigation-b1",
        task_id="task-b1",
        attempt_id="attempt-b1",
        run_id="run-b1",
        state=InvestigationResultState.COMPLETED,
        evidence_set=EvidenceSet(session_id="session-b1", evidence=()),
        conclusions=(_conclusion(claim, ("observation-clamp-low",)),),
        observations=(observation,),
    )

    evidence = CandidateClaimExtractor().extract(investigation_result).evidence_set.evidence[0]

    assert evidence.source_quality == 0.0


def _conclusion(claim: ClaimStatement, observation_ids: tuple[str, ...]):
    from nexus_runtime.investigation.evidence import AgentConclusion

    return AgentConclusion(
        claim=claim,
        supporting_observation_ids=observation_ids,
        confidence=0.8,
    )
