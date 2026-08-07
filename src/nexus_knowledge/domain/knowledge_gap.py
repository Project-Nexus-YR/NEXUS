"""Knowledge gaps and candidate investigations.

A :class:`KnowledgeGap` is derived from *measurable* graph/evidence
properties, never from arbitrary generative suggestions. It feeds the
planner-facing investigation scorer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import now_iso
from .ids import new_id

__all__ = ["KnowledgeGap", "GapKind", "Investigation"]


class GapKind:
    """Well-known classes of knowledge gaps."""

    MISSING_RELATION = "missing_relation"
    LOW_CONFIDENCE = "low_confidence"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    CONTRADICTION = "contradiction"
    STALE = "stale"
    DISCONNECTED_ENTITY = "disconnected_entity"
    MISSING_EVIDENCE = "missing_evidence"
    ANOMALOUS_REGION = "anomalous_region"
    LOW_DIVERSITY = "low_diversity"


@dataclass(slots=True)
class KnowledgeGap:
    """A derived, measurable deficiency in the knowledge graph."""

    kind: str
    description: str
    reason: str
    affected_entities: list[str] = field(default_factory=list)
    affected_relations: list[str] = field(default_factory=list)
    affected_claims: list[str] = field(default_factory=list)
    uncertainty: float = 0.0  # 0..1, how uncertain the region is
    importance: float = 0.0  # 0..1
    estimated_cost: float = 0.0  # relative cost units
    candidate_investigations: list["Investigation"] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("gap"))
    created_at: str = field(default_factory=now_iso)

    @property
    def priority(self) -> float:
        """Aggregate priority used for default ordering."""
        return self.importance * (1.0 - self.uncertainty) * 0.5 + self.uncertainty * 0.5


@dataclass(slots=True)
class Investigation:
    """A candidate investigation addressing a knowledge gap.

    Scored by an ``InvestigationScorer`` which estimates expected
    information gain, uncertainty reduction, importance and cost.
    """

    gap_id: str
    description: str
    target_entities: list[str] = field(default_factory=list)
    expected_information_gain: float = 0.0
    uncertainty_reduction: float = 0.0
    importance: float = 0.0
    estimated_cost: float = 0.0
    score: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("inv"))
    created_at: str = field(default_factory=now_iso)
