"""Explicit, deterministic uncertainty model.

Claim confidence is aggregated from measurable evidence properties:

* base confidence of the claim
* supporting / contradicting evidence counts
* source quality
* source diversity
* recency

The aggregation is interpretable (every component contributes a
documented amount) and is open to a learned confidence estimator later.
Uncertainty is never represented purely as an LLM-generated sentence.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime

from ..domain.claim import Claim
from ..domain.common import VerificationState
from ..port.repository import EvidenceRepository, SourceRepository

__all__ = ["UncertaintyWeights", "UncertaintyAssessment", "UncertaintyModel"]

_ISO_FORMATS = ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d")


@dataclass(frozen=True, slots=True)
class UncertaintyWeights:
    base: float = 1.0  # multiplier applied to the claim's own confidence
    support_gain: float = 0.15  # per supporting evidence
    contra_penalty: float = 0.35  # per contradicting evidence
    source_quality_weight: float = 0.15
    diversity_weight: float = 0.10
    recency_weight: float = 0.10
    evidence_cap: int = 5  # diminishing returns beyond this many items
    recency_half_life_days: float = 365.0
    stale_age_days: float = 2.0 * 365.0
    diversity_target: int = 3  # ideal number of independent sources


@dataclass(frozen=True, slots=True)
class UncertaintyAssessment:
    claim_id: str
    confidence: float
    uncertainty: float
    verification_state: VerificationState
    supporting_evidence_count: int
    contradicting_evidence_count: int
    source_quality: float
    source_diversity: float
    recency: float
    components: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "confidence": round(self.confidence, 4),
            "uncertainty": round(self.uncertainty, 4),
            "verification_state": self.verification_state.value,
            "supporting_evidence_count": self.supporting_evidence_count,
            "contradicting_evidence_count": self.contradicting_evidence_count,
            "source_quality": round(self.source_quality, 4),
            "source_diversity": round(self.source_diversity, 4),
            "recency": round(self.recency, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
        }


class UncertaintyModel:
    """Aggregates evidence into a confidence and verification state."""

    def __init__(self, weights: UncertaintyWeights | None = None) -> None:
        self._weights = weights or UncertaintyWeights()

    def evaluate(
        self,
        claim: Claim,
        evidence: EvidenceRepository,
        sources: SourceRepository,
    ) -> UncertaintyAssessment:
        w = self._weights
        support = claim.supporting_evidence
        contra = claim.contradicting_evidence
        support_count = len(support)
        contra_count = len(contra)

        quality, diversity = self._source_signals(claim, sources)
        recency = self._recency(claim)

        cap = max(w.evidence_cap, 1)
        components: dict[str, float] = {
            "base": float(claim.confidence),
            "support_gain": w.support_gain * min(support_count, cap),
            "contra_penalty": -w.contra_penalty * min(contra_count, cap),
            "source_quality": w.source_quality_weight * (2.0 * quality - 1.0),
            "diversity": w.diversity_weight * diversity,
            "recency": w.recency_weight * (recency - 0.5),
        }
        raw = (
            w.base * float(claim.confidence)
            + components["support_gain"]
            + components["contra_penalty"]
            + components["source_quality"]
            + components["diversity"]
            + components["recency"]
        )
        confidence = min(1.0, max(0.0, raw))
        state = self._verification_state(claim, support_count, contra_count, confidence, recency)
        return UncertaintyAssessment(
            claim_id=claim.id,
            confidence=confidence,
            uncertainty=1.0 - confidence,
            verification_state=state,
            supporting_evidence_count=support_count,
            contradicting_evidence_count=contra_count,
            source_quality=quality,
            source_diversity=diversity,
            recency=recency,
            components=components,
        )

    def _source_signals(self, claim: Claim, sources: SourceRepository) -> tuple[float, float]:
        """Return ``(mean_source_quality, normalized_diversity)``."""
        source_ids = set(claim.source_ids)
        if not source_ids:
            return 0.5, 0.0
        qualities: list[float] = []
        for source_id in source_ids:
            source = sources.get(source_id)
            quality = 0.5
            if source is not None:
                quality = float(source.metadata.get("quality", 0.5))
            qualities.append(min(1.0, max(0.0, quality)))
        quality = sum(qualities) / len(qualities)
        diversity = min(1.0, len(source_ids) / self._weights.diversity_target)
        return quality, diversity

    def _recency(self, claim: Claim) -> float:
        """0..1 recency signal based on the most recent observation time."""
        w = self._weights
        latest = self._latest_timestamp(claim)
        if latest is None:
            return 0.5  # neutral when timing is unknown
        age_days = max(0.0, (datetime.now(UTC) - latest).total_seconds() / 86400.0)
        return float(math.exp(-age_days / w.recency_half_life_days))

    @staticmethod
    def _latest_timestamp(claim: Claim) -> datetime | None:
        candidates = [claim.observed_at, claim.updated_at, claim.created_at]
        for text in candidates:
            parsed = UncertaintyModel._parse_iso(text)
            if parsed is not None:
                return parsed
        return None

    @staticmethod
    def _parse_iso(text: str) -> datetime | None:
        if not text:
            return None
        for fmt in _ISO_FORMATS:
            try:
                value = datetime.strptime(text, fmt)
            except ValueError:
                continue
            return value.replace(tzinfo=UTC) if value.tzinfo is None else value
        return None

    def _verification_state(
        self,
        claim: Claim,
        support_count: int,
        contra_count: int,
        confidence: float,
        recency: float,
    ) -> VerificationState:
        if contra_count > 0 and contra_count >= support_count:
            return (
                VerificationState.REFUTED if support_count == 0 else VerificationState.CONTRADICTED
            )
        if support_count > 0 and confidence >= 0.7:
            return VerificationState.VERIFIED
        if support_count > 0:
            return VerificationState.SUPPORTED
        if confidence < 0.4:
            return VerificationState.UNCERTAIN
        return VerificationState.UNVERIFIED
