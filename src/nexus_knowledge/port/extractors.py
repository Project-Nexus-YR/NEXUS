"""Entity and relation extraction ports.

Extraction is provider-independent: deterministic (pattern-based) and
LLM-based implementations are interchangeable adapters. Extracted
objects carry source spans so evidence remains verifiable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from ..domain.common import Confidence
from ..domain.document import Chunk, Document, Span

__all__ = ["ExtractedEntity", "ExtractedRelation", "EntityExtractor", "RelationExtractor"]


@dataclass(frozen=True, slots=True)
class ExtractedEntity:
    name: str
    entity_type: str = "unknown"
    span: Span | None = None
    confidence: Confidence = Confidence(1.0)
    aliases: list[str] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class ExtractedRelation:
    subject: str  # entity name
    predicate: str
    object: str  # entity name
    span: Span | None = None
    confidence: Confidence = Confidence(1.0)


class EntityExtractor(Protocol):
    """Extracts entities from a chunk."""

    def extract(self, chunk: Chunk, document: Document) -> list[ExtractedEntity]: ...


class RelationExtractor(Protocol):
    """Extracts relations between entities in a chunk."""

    def extract(
        self,
        chunk: Chunk,
        document: Document,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedRelation]: ...
