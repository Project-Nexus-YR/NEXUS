"""Cost-aware, explainable investigation scoring built on knowledge scoring."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import import_module
from typing import Any, Protocol, cast

from nexus_runtime.models import DomainError

from .generator import CandidateInvestigation, KnowledgeGapLike


class KnowledgeScoreLike(Protocol):
    score: float


class KnowledgeScorerLike(Protocol):
    def score(self, investigation: object, gap: object) -> KnowledgeScoreLike: ...


@dataclass(frozen=True, slots=True)
class ScoringWeights:
    information_gain: float = 0.25
    gap_importance: float = 0.15
    uncertainty_reduction: float = 0.15
    evidence_availability: float = 0.10
    priority: float = 0.10
    knowledge_score: float = 0.10
    cost_penalty: float = 0.05
    time_penalty: float = 0.03
    risk_penalty: float = 0.04
    redundancy_penalty: float = 0.08

    def __post_init__(self) -> None:
        values = (
            self.information_gain,
            self.gap_importance,
            self.uncertainty_reduction,
            self.evidence_availability,
            self.priority,
            self.knowledge_score,
            self.cost_penalty,
            self.time_penalty,
            self.risk_penalty,
            self.redundancy_penalty,
        )
        if any(value < 0 for value in values):
            raise DomainError("scoring weights cannot be negative")


@dataclass(frozen=True, slots=True)
class InvestigationScore:
    candidate: CandidateInvestigation
    score: float
    components: dict[str, float]
    rationale: str

    def to_dict(self) -> dict[str, object]:
        return {
            "investigation_id": self.candidate.investigation_id,
            "gap_id": self.candidate.gap_id,
            "score": self.score,
            "components": dict(self.components),
            "rationale": self.rationale,
        }


@dataclass(frozen=True, slots=True)
class InformationGainForecast:
    expected_information_gain: float
    expected_uncertainty_reduction: float
    estimated_cost: float
    information_gain_per_cost: float

    def to_dict(self) -> dict[str, float]:
        return {
            "expected_information_gain": self.expected_information_gain,
            "expected_uncertainty_reduction": self.expected_uncertainty_reduction,
            "estimated_cost": self.estimated_cost,
            "information_gain_per_cost": self.information_gain_per_cost,
        }


class InvestigationScoringModel:
    """Extend the existing gain/cost scorer with bounded planning trade-offs."""

    def __init__(
        self,
        knowledge_scorer: KnowledgeScorerLike | None = None,
        weights: ScoringWeights | None = None,
    ) -> None:
        if knowledge_scorer is None:
            scorer_type = cast(
                "Any",
                import_module("nexus_knowledge.knowledge.scorer").InvestigationScorer,
            )
            knowledge_scorer = cast(KnowledgeScorerLike, scorer_type())
        self._knowledge_scorer = knowledge_scorer
        self._investigation_type = cast(
            "Any",
            import_module("nexus_knowledge.domain.knowledge_gap").Investigation,
        )
        self._weights = weights or ScoringWeights()

    def score(
        self,
        candidate: CandidateInvestigation,
        gap: KnowledgeGapLike,
        *,
        redundancy: float = 0.0,
    ) -> InvestigationScore:
        if candidate.gap_id != gap.id:
            raise DomainError("candidate and gap identifiers do not match")
        if not 0.0 <= redundancy <= 1.0:
            raise DomainError("redundancy must be between zero and one")
        existing = self._existing_score(candidate, gap)
        evidence_availability = self._unit_metadata(candidate, "evidence_availability", 0.5)
        cost = candidate.estimated_cost / (1.0 + candidate.estimated_cost)
        duration = candidate.estimated_duration_seconds / (
            candidate.estimated_duration_seconds + 300.0
        )
        knowledge_signal = existing / (1.0 + max(0.0, existing))
        components = {
            "information_gain": candidate.expected_information_gain,
            "gap_importance": min(1.0, max(0.0, gap.importance)),
            "uncertainty_reduction": candidate.uncertainty_reduction,
            "evidence_availability": evidence_availability,
            "priority": candidate.priority,
            "knowledge_score": knowledge_signal,
            "cost_penalty": cost,
            "time_penalty": duration,
            "risk_penalty": candidate.risk,
            "redundancy_penalty": redundancy,
        }
        weights = self._weights
        benefit = (
            weights.information_gain * components["information_gain"]
            + weights.gap_importance * components["gap_importance"]
            + weights.uncertainty_reduction * components["uncertainty_reduction"]
            + weights.evidence_availability * components["evidence_availability"]
            + weights.priority * components["priority"]
            + weights.knowledge_score * components["knowledge_score"]
        )
        penalty = (
            weights.cost_penalty * components["cost_penalty"]
            + weights.time_penalty * components["time_penalty"]
            + weights.risk_penalty * components["risk_penalty"]
            + weights.redundancy_penalty * components["redundancy_penalty"]
        )
        total = min(1.0, max(0.0, benefit - penalty))
        rationale = (
            f"gain={components['information_gain']:.3f}, "
            f"importance={components['gap_importance']:.3f}, "
            f"cost_penalty={components['cost_penalty']:.3f}, "
            f"risk_penalty={components['risk_penalty']:.3f}, score={total:.3f}"
        )
        return InvestigationScore(candidate, total, components, rationale)

    def score_all(
        self,
        candidates: Sequence[CandidateInvestigation],
        gaps: Sequence[KnowledgeGapLike],
    ) -> tuple[InvestigationScore, ...]:
        gaps_by_id = {gap.id: gap for gap in gaps}
        counts = Counter(candidate.redundancy_key for candidate in candidates)
        scored: list[InvestigationScore] = []
        for candidate in candidates:
            gap = gaps_by_id.get(candidate.gap_id)
            if gap is None:
                raise DomainError(f"unknown knowledge gap: {candidate.gap_id}")
            count = counts[candidate.redundancy_key]
            redundancy = 0.0 if count == 1 else (count - 1) / count
            scored.append(self.score(candidate, gap, redundancy=redundancy))
        scored.sort(key=lambda item: (-item.score, item.candidate.investigation_id))
        return tuple(scored)

    def forecast(self, scored: Sequence[InvestigationScore]) -> InformationGainForecast:
        if not scored:
            return InformationGainForecast(0.0, 0.0, 0.0, 0.0)
        expected_gain = sum(
            item.candidate.expected_information_gain * item.score for item in scored
        )
        uncertainty_reduction = sum(
            item.candidate.uncertainty_reduction * item.score for item in scored
        )
        cost = sum(item.candidate.estimated_cost for item in scored)
        return InformationGainForecast(
            expected_information_gain=expected_gain,
            expected_uncertainty_reduction=uncertainty_reduction,
            estimated_cost=cost,
            information_gain_per_cost=expected_gain / cost if cost else expected_gain,
        )

    def _existing_score(self, candidate: CandidateInvestigation, gap: KnowledgeGapLike) -> float:
        legacy = self._investigation_type(
            id=candidate.investigation_id,
            gap_id=gap.id,
            description=candidate.question,
            target_entities=list(candidate.target_entities),
            estimated_cost=candidate.estimated_cost,
        )
        return self._knowledge_scorer.score(legacy, gap).score

    @staticmethod
    def _unit_metadata(candidate: CandidateInvestigation, key: str, default: float) -> float:
        value = candidate.metadata.get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return default
        return min(1.0, max(0.0, float(value)))
