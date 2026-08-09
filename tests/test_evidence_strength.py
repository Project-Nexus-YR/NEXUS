"""Evidentiary-strength composite, grades, and set-level summary (Section 3)."""

from __future__ import annotations

import pytest

from nexus_runtime.investigation.evaluation import (
    EvidenceEvaluator,
    EvidenceQualityPolicy,
)
from nexus_runtime.investigation.evidence import (
    ClaimStatement,
    Evidence,
    EvidenceGrade,
    EvidenceSet,
    grade_for_strength,
)
from nexus_runtime.investigation.provenance import EvidenceProvenance


def _evidence(
    *,
    confidence: float,
    source_quality: float,
    evidence_id: str,
    source_id: str = "source-a",
    excerpt: str = "Registry record.",
) -> Evidence:
    provenance = EvidenceProvenance(
        session_id="session-1",
        investigation_id="investigation-a",
        task_id="task-1",
        attempt_id="attempt-1",
        run_id="run-1",
        tool_call_id=f"tool-{source_id}",
        source_id=source_id,
        document_id="document-a",
        chunk_id="chunk-a",
        source_reference=f"https://example.test/{source_id}",
    )
    return Evidence(
        investigation_id="investigation-a",
        source=provenance.source_reference,
        claim=ClaimStatement("Atlas in London", "Atlas", "located_in", "London"),
        provenance=provenance,
        confidence=confidence,
        source_quality=source_quality,
        excerpt=excerpt,
        evidence_id=evidence_id,
    )


class TestEvidentiaryStrength:
    def test_strength_is_geometric_mean(self):
        evidence = _evidence(confidence=0.9, source_quality=0.4, evidence_id="e1")
        assert evidence.evidentiary_strength == pytest.approx((0.9 * 0.4) ** 0.5)

    def test_strength_is_conservative_under_imbalance(self):
        confident = _evidence(confidence=0.99, source_quality=0.1, evidence_id="e-confident")
        quality = _evidence(confidence=0.1, source_quality=0.99, evidence_id="e-quality")
        assert confident.evidentiary_strength < 0.5
        assert quality.evidentiary_strength < 0.5

    def test_strength_clamped_to_unit_interval(self):
        low = _evidence(confidence=0.0, source_quality=0.0, evidence_id="e-low")
        high = _evidence(confidence=1.0, source_quality=1.0, evidence_id="e-high")
        assert 0.0 <= low.evidentiary_strength
        assert high.evidentiary_strength == 1.0


class TestEvidenceGrade:
    def test_grade_for_strength_boundaries(self):
        assert grade_for_strength(0.7) == EvidenceGrade.STRONG
        assert grade_for_strength(0.699999) == EvidenceGrade.MODERATE
        assert grade_for_strength(0.4) == EvidenceGrade.MODERATE
        assert grade_for_strength(0.399999) == EvidenceGrade.WEAK

    def test_evidence_grade_matches_strength(self):
        strong = _evidence(confidence=0.9, source_quality=0.9, evidence_id="e-strong")
        assert strong.grade == EvidenceGrade.STRONG
        moderate = _evidence(confidence=0.6, source_quality=0.5, evidence_id="e-moderate")
        assert moderate.grade == EvidenceGrade.MODERATE
        weak = _evidence(confidence=0.2, source_quality=0.2, evidence_id="e-weak")
        assert weak.grade == EvidenceGrade.WEAK

    def test_serialization_preserves_derived_fields(self):
        evidence = _evidence(confidence=0.9, source_quality=0.9, evidence_id="e-roundtrip")
        payload = evidence.to_dict()
        assert payload["evidentiary_strength"] == pytest.approx(0.9)
        assert payload["grade"] == EvidenceGrade.STRONG.value
        restored = Evidence.from_dict(payload)
        assert restored.evidentiary_strength == evidence.evidentiary_strength
        assert restored.grade == evidence.grade


class TestEvidenceSetSummary:
    def test_grade_counts_and_mean_strength(self):
        evidence_set = EvidenceSet(
            session_id="session-1",
            evidence=(
                _evidence(confidence=0.9, source_quality=0.9, evidence_id="e-strong"),
                _evidence(confidence=0.6, source_quality=0.5, evidence_id="e-moderate"),
                _evidence(confidence=0.2, source_quality=0.2, evidence_id="e-weak"),
            ),
        )
        assert evidence_set.grade_counts == {
            "strong": 1,
            "moderate": 1,
            "weak": 1,
        }
        expected = (0.9 + (0.6 * 0.5) ** 0.5 + 0.2) / 3
        assert evidence_set.mean_evidentiary_strength == pytest.approx(expected)

    def test_empty_set_has_zero_mean(self):
        evidence_set = EvidenceSet(session_id="session-1", evidence=())
        assert evidence_set.mean_evidentiary_strength == 0.0
        assert evidence_set.grade_counts == {"strong": 0, "moderate": 0, "weak": 0}


class TestEvaluationPolicyUsesStrength:
    def test_min_evidentiary_strength_rejects_imbalanced_evidence(self):
        evidence = (
            _evidence(confidence=0.9, source_quality=0.9, evidence_id="e-strong"),
            _evidence(
                confidence=0.99,
                source_quality=0.3,
                evidence_id="e-imbalanced",
                source_id="source-b",
                excerpt="A secondary note mentions London.",
            ),
        )
        policy = EvidenceQualityPolicy(min_evidentiary_strength=0.7)
        evaluation = EvidenceEvaluator(policy).evaluate(EvidenceSet("session-1", evidence))
        assert evaluation.low_quality_evidence_ids == ("e-imbalanced",)
        assert evaluation.accepted_evidence_count == 1

    def test_default_policy_accepts_by_component_thresholds(self):
        evaluation = EvidenceEvaluator().evaluate(
            EvidenceSet(
                "session-1",
                (_evidence(confidence=0.8, source_quality=0.8, evidence_id="e-ok"),),
            )
        )
        assert evaluation.low_quality_evidence_ids == ()
