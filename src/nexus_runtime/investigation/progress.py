"""Measure epistemic progress between investigation iterations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from math import isfinite

from nexus_runtime.models import new_id, utcnow


@dataclass(frozen=True, slots=True)
class GapState:
    gap_id: str
    uncertainty: float

    def __post_init__(self) -> None:
        if not self.gap_id.strip():
            raise ValueError("gap_id must be a non-empty string")
        if not 0.0 <= self.uncertainty <= 1.0:
            raise ValueError("gap uncertainty must be between zero and one")


@dataclass(frozen=True, slots=True)
class ProgressReport:
    session_id: str
    iteration: int
    gaps_before: int
    gaps_after: int
    resolved_gap_ids: tuple[str, ...]
    new_gap_ids: tuple[str, ...]
    remaining_gap_ids: tuple[str, ...]
    uncertainty_reduced: float
    contradictions_introduced: tuple[str, ...]
    contradictions_resolved: tuple[str, ...]
    evidence_collected: int
    knowledge_updates: int
    information_gain: float
    cost: float
    cost_per_resolved_gap: float | None
    report_id: str = field(default_factory=lambda: new_id("progress"))
    measured_at: datetime = field(default_factory=utcnow)


class ProgressMeasurer:
    """Compare explicit before/after gap and contradiction snapshots."""

    def measure(
        self,
        *,
        session_id: str,
        iteration: int,
        before_gaps: Iterable[GapState],
        after_gaps: Iterable[GapState],
        before_contradiction_ids: Iterable[str] = (),
        after_contradiction_ids: Iterable[str] = (),
        evidence_collected: int = 0,
        knowledge_updates: int = 0,
        cost: float = 0.0,
    ) -> ProgressReport:
        if not session_id.strip():
            raise ValueError("session_id must be a non-empty string")
        if iteration < 1:
            raise ValueError("iteration must be positive")
        if evidence_collected < 0 or knowledge_updates < 0:
            raise ValueError("progress counts cannot be negative")
        if cost < 0.0 or not isfinite(cost):
            raise ValueError("cost must be finite and non-negative")

        before = self._index(before_gaps)
        after = self._index(after_gaps)
        resolved = tuple(sorted(before.keys() - after.keys()))
        discovered = tuple(sorted(after.keys() - before.keys()))
        remaining = tuple(sorted(after))

        uncertainty_reduced = sum(before[gap_id].uncertainty for gap_id in resolved)
        uncertainty_reduced += sum(
            max(0.0, before[gap_id].uncertainty - after[gap_id].uncertainty)
            for gap_id in before.keys() & after.keys()
        )
        uncertainty_added = sum(after[gap_id].uncertainty for gap_id in discovered)
        baseline_uncertainty = max(1.0, sum(gap.uncertainty for gap in before.values()))

        before_conflicts = {item for item in before_contradiction_ids if item}
        after_conflicts = {item for item in after_contradiction_ids if item}
        introduced = tuple(sorted(after_conflicts - before_conflicts))
        contradictions_resolved = tuple(sorted(before_conflicts - after_conflicts))
        conflict_penalty = len(introduced) / max(1, len(before_conflicts) + len(introduced))
        gain = (uncertainty_reduced - uncertainty_added) / baseline_uncertainty
        gain -= 0.25 * conflict_penalty
        information_gain = min(1.0, max(-1.0, gain))
        cost_per_resolved = cost / len(resolved) if resolved else None

        return ProgressReport(
            session_id=session_id,
            iteration=iteration,
            gaps_before=len(before),
            gaps_after=len(after),
            resolved_gap_ids=resolved,
            new_gap_ids=discovered,
            remaining_gap_ids=remaining,
            uncertainty_reduced=uncertainty_reduced,
            contradictions_introduced=introduced,
            contradictions_resolved=contradictions_resolved,
            evidence_collected=evidence_collected,
            knowledge_updates=knowledge_updates,
            information_gain=information_gain,
            cost=cost,
            cost_per_resolved_gap=cost_per_resolved,
        )

    @staticmethod
    def _index(gaps: Iterable[GapState]) -> dict[str, GapState]:
        result: dict[str, GapState] = {}
        for gap in gaps:
            if gap.gap_id in result:
                raise ValueError(f"duplicate gap id: {gap.gap_id}")
            result[gap.gap_id] = gap
        return result
