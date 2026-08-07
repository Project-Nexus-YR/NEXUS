"""Entities and relations in the knowledge graph."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import Confidence, VerificationState, now_iso
from .ids import new_id

__all__ = ["Entity", "Relation"]


@dataclass(slots=True)
class Entity:
    """A node in the knowledge graph.

    Entities are deduplicated by ``stable_id`` produced from the
    canonical name (and optionally type), so the same real-world
    object across documents converges to a single node.
    """

    name: str
    entity_type: str = "unknown"
    canonical_name: str | None = None
    aliases: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("ent"))

    @property
    def canonical(self) -> str:
        return self.canonical_name or self.name

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Entity) and other.id == self.id


@dataclass(slots=True)
class Relation:
    """A typed, directed edge between two entities.

    Relations carry their own confidence, provenance and verification
    state; a relation without provenance is treated as unverified and is
    never promoted to trusted knowledge automatically.
    """

    subject_id: str
    predicate: str
    object_id: str
    confidence: Confidence = Confidence(0.5)
    provenance: list[str] = field(default_factory=list)  # chunk ids
    source_ids: list[str] = field(default_factory=list)  # source ids
    supporting_evidence: list[str] = field(default_factory=list)  # evidence ids
    contradicting_evidence: list[str] = field(default_factory=list)
    verification_state: VerificationState = VerificationState.UNVERIFIED
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("rel"))
    created_at: str = field(default_factory=now_iso)
    updated_at: str = field(default_factory=now_iso)
    observed_at: str = ""

    @property
    def tuple(self) -> tuple[str, str, str]:
        return (self.subject_id, self.predicate, self.object_id)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Relation) and other.id == self.id
