"""Investigation scoring for the planner-facing API.

The baseline scorer estimates the value of a candidate investigation as

    value = expected_information_gain
          x importance
          x uncertainty_reduction
          / cost

The estimator is replaceable: the same API supports random, centrality
and uncertainty-only selection for experiments, and later a learned
policy.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol

from ..domain.knowledge_gap import GapKind, Investigation, KnowledgeGap

__all__ = [
    "GainEstimator",
    "BaselineGainEstimator",
    "InvestigationScorer",
    "ScoredInvestigation",
    "RandomInvestigationScorer",
    "CentralityInvestigationScorer",
]


class GainEstimator(Protocol):
    def expected_information_gain(self, gap: KnowledgeGap) -> float: ...

    def uncertainty_reduction(self, gap: KnowledgeGap) -> float: ...


class BaselineGainEstimator:
    """Deterministic, interpretable gain heuristics per gap kind."""

    def expected_information_gain(self, gap: KnowledgeGap) -> float:
        kind = gap.kind
        importance = gap.importance
        uncertainty = gap.uncertainty
        gains = {
            GapKind.MISSING_RELATION: 0.3 + 0.7 * importance,
            GapKind.LOW_CONFIDENCE: 0.2 + 0.8 * uncertainty,
            GapKind.UNSUPPORTED_CLAIM: 0.4 * uncertainty + 0.4 * importance + 0.2,
            GapKind.CONTRADICTION: 0.6 * uncertainty,
            GapKind.STALE: 0.5,
            GapKind.DISCONNECTED_ENTITY: 0.5,
            GapKind.MISSING_EVIDENCE: 0.3 + 0.7 * uncertainty,
            GapKind.LOW_DIVERSITY: 0.25 + 0.75 * uncertainty,
        }
        return min(1.0, max(0.0, gains.get(kind, 0.5)))

    def uncertainty_reduction(self, gap: KnowledgeGap) -> float:
        return min(1.0, max(0.0, gap.uncertainty))


@dataclass(frozen=True, slots=True)
class ScoredInvestigation:
    investigation: Investigation
    score: float
    components: dict[str, float]

    def to_dict(self) -> dict[str, object]:
        return {
            "investigation_id": self.investigation.id,
            "gap_id": self.investigation.gap_id,
            "description": self.investigation.description,
            "score": round(self.score, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
        }


class InvestigationScorer:
    """Scores investigations with the information-gain formula."""

    def __init__(self, gain_estimator: GainEstimator | None = None) -> None:
        self._gain = gain_estimator or BaselineGainEstimator()

    def score(self, investigation: Investigation, gap: KnowledgeGap) -> ScoredInvestigation:
        gain = self._gain.expected_information_gain(gap)
        uncertainty_reduction = self._gain.uncertainty_reduction(gap)
        importance = min(1.0, max(0.0, gap.importance))
        cost = max(investigation.estimated_cost, 1e-6)
        score = (gain * importance * uncertainty_reduction) / cost
        return ScoredInvestigation(
            investigation=investigation,
            score=score,
            components={
                "expected_information_gain": gain,
                "importance": importance,
                "uncertainty_reduction": uncertainty_reduction,
                "estimated_cost": cost,
            },
        )

    def score_gaps(self, gaps: list[KnowledgeGap]) -> list[ScoredInvestigation]:
        scored: list[ScoredInvestigation] = []
        for gap in gaps:
            for investigation in gap.candidate_investigations:
                scored.append(self.score(investigation, gap))
        scored.sort(key=lambda s: s.score, reverse=True)
        for result in scored:
            self._commit(result)
        return scored

    def _commit(self, scored: ScoredInvestigation) -> None:
        investigation = scored.investigation
        investigation.expected_information_gain = scored.components["expected_information_gain"]
        investigation.uncertainty_reduction = scored.components["uncertainty_reduction"]
        investigation.importance = scored.components["importance"]
        investigation.score = scored.score


class RandomInvestigationScorer:
    """Deterministic pseudo-random selection for experiment comparison."""

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed

    def score(self, investigation: Investigation, gap: KnowledgeGap) -> ScoredInvestigation:
        digest = hashlib.sha256(
            f"{self._seed}:{investigation.id}".encode("utf-8")
        ).hexdigest()
        value = int(digest[:8], 16) / 0xFFFFFFFF
        return ScoredInvestigation(
            investigation=investigation,
            score=value,
            components={
                "expected_information_gain": 0.0,
                "importance": 0.0,
                "uncertainty_reduction": 0.0,
                "estimated_cost": investigation.estimated_cost,
            },
        )

    def score_gaps(self, gaps: list[KnowledgeGap]) -> list[ScoredInvestigation]:
        scored = [self.score(inv, gap) for gap in gaps for inv in gap.candidate_investigations]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored


class CentralityInvestigationScorer:
    """Selection driven purely by graph centrality (importance / cost)."""

    def score(self, investigation: Investigation, gap: KnowledgeGap) -> ScoredInvestigation:
        score = gap.importance / max(investigation.estimated_cost, 1e-6)
        return ScoredInvestigation(
            investigation=investigation,
            score=score,
            components={
                "expected_information_gain": 0.0,
                "importance": gap.importance,
                "uncertainty_reduction": 0.0,
                "estimated_cost": investigation.estimated_cost,
            },
        )

    def score_gaps(self, gaps: list[KnowledgeGap]) -> list[ScoredInvestigation]:
        scored = [self.score(inv, gap) for gap in gaps for inv in gap.candidate_investigations]
        scored.sort(key=lambda s: s.score, reverse=True)
        return scored
