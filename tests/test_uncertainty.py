"""Uncertainty model tests."""

import pytest

from nexus_knowledge.domain.claim import Claim
from nexus_knowledge.domain.common import Confidence, VerificationState
from nexus_knowledge.domain.source import Source, SourceKind
from nexus_knowledge.knowledge.uncertainty import (
    UncertaintyModel,
)
from nexus_knowledge.persistence.memory import InMemoryKnowledgeRepository


def _repo_with_sources(qualities: list[float] | None = None) -> InMemoryKnowledgeRepository:
    repo = InMemoryKnowledgeRepository()
    for i, quality in enumerate(qualities or []):
        repo.sources.save(
            Source(
                title=f"s{i}",
                kind=SourceKind.TEXT,
                reference=f"ref{i}",
                id=f"src{i}",
                metadata={"quality": quality},
            )
        )
    return repo


class TestConfidenceAggregation:
    def test_support_raises_confidence(self):
        repo = _repo_with_sources([0.8])
        claim = Claim(
            text="x",
            confidence=Confidence(0.5),
            supporting_evidence=["e1", "e2"],
            source_ids=["src0"],
        )
        assessment = UncertaintyModel().evaluate(claim, repo.evidence, repo.sources)
        assert assessment.confidence > 0.5

    def test_contradiction_lowers_confidence(self):
        repo = _repo_with_sources([0.8])
        claim = Claim(
            text="x",
            confidence=Confidence(0.9),
            supporting_evidence=["e1"],
            contradicting_evidence=["e2"],
            source_ids=["src0"],
        )
        assessment = UncertaintyModel().evaluate(claim, repo.evidence, repo.sources)
        assert assessment.confidence < 0.9

    def test_clamped_to_unit_interval(self):
        repo = InMemoryKnowledgeRepository()
        claim = Claim(
            text="x",
            confidence=Confidence(1.0),
            supporting_evidence=["e1", "e2", "e3", "e4", "e5", "e6"],
            source_ids=["src0"],
        )
        assessment = UncertaintyModel().evaluate(claim, repo.evidence, repo.sources)
        assert 0.0 <= assessment.confidence <= 1.0

    def test_no_sources_neutral(self):
        repo = InMemoryKnowledgeRepository()
        claim = Claim(text="x", confidence=Confidence(0.5))
        assessment = UncertaintyModel().evaluate(claim, repo.evidence, repo.sources)
        assert assessment.source_quality == 0.5
        assert assessment.source_diversity == 0.0


class TestVerificationStates:
    def test_verified_with_strong_support(self):
        repo = _repo_with_sources([0.9])
        claim = Claim(
            text="x",
            confidence=Confidence(0.8),
            supporting_evidence=["e1", "e2"],
            source_ids=["src0"],
        )
        assessment = UncertaintyModel().evaluate(claim, repo.evidence, repo.sources)
        assert assessment.verification_state == VerificationState.VERIFIED

    def test_supported_but_not_confident(self):
        repo = _repo_with_sources([0.5])
        claim = Claim(text="x", confidence=Confidence(0.5), supporting_evidence=["e1"])
        assessment = UncertaintyModel().evaluate(claim, repo.evidence, repo.sources)
        assert assessment.verification_state == VerificationState.SUPPORTED

    def test_uncertain_when_low_confidence_no_evidence(self):
        repo = InMemoryKnowledgeRepository()
        claim = Claim(text="x", confidence=Confidence(0.3))
        assessment = UncertaintyModel().evaluate(claim, repo.evidence, repo.sources)
        assert assessment.verification_state == VerificationState.UNCERTAIN

    def test_refuted_no_support(self):
        repo = _repo_with_sources([0.9])
        claim = Claim(
            text="x",
            confidence=Confidence(0.5),
            contradicting_evidence=["e1"],
        )
        assessment = UncertaintyModel().evaluate(claim, repo.evidence, repo.sources)
        assert assessment.verification_state == VerificationState.REFUTED

    def test_contradicted_with_support(self):
        repo = _repo_with_sources([0.9])
        claim = Claim(
            text="x",
            confidence=Confidence(0.8),
            supporting_evidence=["e1"],
            contradicting_evidence=["e2"],
        )
        assessment = UncertaintyModel().evaluate(claim, repo.evidence, repo.sources)
        assert assessment.verification_state == VerificationState.CONTRADICTED


class TestSourceSignals:
    def test_quality_is_mean(self):
        repo = _repo_with_sources([0.8, 0.6])
        claim = Claim(text="x", source_ids=["src0", "src1"])
        assessment = UncertaintyModel().evaluate(claim, repo.evidence, repo.sources)
        assert assessment.source_quality == pytest.approx(0.7)

    def test_diversity_capped_at_target(self):
        repo = _repo_with_sources([0.9, 0.9, 0.9, 0.9])
        claim = Claim(text="x", source_ids=["src0", "src1", "src2", "src3"])
        assessment = UncertaintyModel().evaluate(claim, repo.evidence, repo.sources)
        assert assessment.source_diversity == 1.0


class TestRecency:
    def test_fresh_claim_is_recent(self):
        repo = InMemoryKnowledgeRepository()
        claim = Claim(text="x", observed_at="2999-01-01T00:00:00Z")
        assessment = UncertaintyModel().evaluate(claim, repo.evidence, repo.sources)
        assert assessment.recency > 0.9

    def test_old_claim_decays(self):
        repo = InMemoryKnowledgeRepository()
        claim = Claim(text="x", observed_at="2000-01-01T00:00:00Z")
        assessment = UncertaintyModel().evaluate(claim, repo.evidence, repo.sources)
        assert assessment.recency < 0.1

    def test_no_timestamp_neutral(self):
        repo = InMemoryKnowledgeRepository()
        claim = Claim(text="x", observed_at="", created_at="", updated_at="")
        assessment = UncertaintyModel().evaluate(claim, repo.evidence, repo.sources)
        assert assessment.recency == 0.5


class TestAssessmentDict:
    def test_to_dict(self):
        repo = InMemoryKnowledgeRepository()
        claim = Claim(text="x")
        payload = UncertaintyModel().evaluate(claim, repo.evidence, repo.sources).to_dict()
        assert set(payload) == {
            "claim_id",
            "confidence",
            "uncertainty",
            "verification_state",
            "supporting_evidence_count",
            "contradicting_evidence_count",
            "source_quality",
            "source_diversity",
            "recency",
            "components",
        }
