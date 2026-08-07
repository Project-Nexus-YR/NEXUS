"""Contradiction objects.

Contradictions are first-class records. Contradictory information is
never deleted automatically: both sides are preserved along with their
evidence, and a :class:`Contradiction` object documents the conflict.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import now_iso
from .ids import new_id

__all__ = ["Contradiction", "ContradictionKind"]


class ContradictionKind:
    CONFLICTING_CLAIMS = "conflicting_claims"
    MUTUALLY_EXCLUSIVE_RELATIONS = "mutually_exclusive_relations"
    STALE_CLAIM = "stale_claim"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    SOURCE_DISAGREEMENT = "source_disagreement"


@dataclass(slots=True)
class Contradiction:
    """A detected conflict between two knowledge records.

    ``claim_a``/``claim_b`` are claim or relation ids; both are
    preserved, and the evidence supporting each side is recorded.
    """

    kind: str
    claim_a_id: str
    claim_b_id: str
    description: str
    evidence_a: list[str] = field(default_factory=list)
    evidence_b: list[str] = field(default_factory=list)
    strength: float = 0.5
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("contra"))
    detected_at: str = field(default_factory=now_iso)
