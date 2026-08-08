"""Deterministic reranker.

A transparent feature-weighted reranker used as the default. An LLM or
learned reranker can be plugged in later through the ``Reranker`` port.
"""

from __future__ import annotations

from ..port.reranker import RerankCandidate

__all__ = ["DeterministicReranker"]


class DeterministicReranker:
    """Scores candidates as a weighted sum of interpretable features.

    The default weights favour fused retrieval agreement (via the
    per-method contributions) and penalize hallucination by rewarding
    verified evidence and entity centrality.
    """

    WEIGHTS: dict[str, float] = {
        "fused_score": 1.0,
        "evidence_count": 0.4,
        "claim_count": 0.2,
        "entity_centrality": 0.3,
        "source_diversity": 0.1,
    }

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self._weights = dict(weights or self.WEIGHTS)

    def rerank(self, query: str, candidates: list[RerankCandidate]) -> list[RerankCandidate]:
        for candidate in candidates:
            features = candidate.features
            score = 0.0
            for key, weight in self._weights.items():
                if key not in features:
                    continue
                raw = features[key]
                score += weight * self._normalize(key, raw)
            candidate.score = score
        return sorted(candidates, key=lambda c: c.score, reverse=True)

    @staticmethod
    def _normalize(key: str, value: float) -> float:
        if key == "fused_score":
            # RRF scores are small; rescale to 0..1 using a saturating factor
            return min(1.0, value * 60.0)
        if key == "evidence_count" or key == "claim_count":
            return min(1.0, value / 10.0)
        return min(1.0, max(0.0, value))
