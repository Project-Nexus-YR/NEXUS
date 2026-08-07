"""LLM-backed extraction adapters (dependency-injected).

These adapters call a user-supplied ``extract_fn`` callable and are
transport-agnostic: the same adapter works with any LLM provider via a
thin provider adapter. The domain layer only sees the extractor ports.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from ..domain.common import Confidence
from ..domain.document import Chunk, Document
from ..port.extractors import ExtractedEntity, ExtractedRelation

__all__ = ["CallbackEntityExtractor", "CallbackRelationExtractor"]

_EntityFn = Callable[[str], list[dict[str, object]]]
_RelationFn = Callable[[str, list[dict[str, object]]], list[dict[str, object]]]


@dataclass(slots=True)
class CallbackEntityExtractor:
    """Entity extractor driven by an external function.

    The callback receives the chunk text and returns a list of dicts
    with keys ``name``, ``entity_type``, and optionally ``confidence``.
    """

    extract_fn: _EntityFn

    def extract(self, chunk: Chunk, document: Document) -> list[ExtractedEntity]:
        results: list[ExtractedEntity] = []
        for raw in self.extract_fn(chunk.text):
            name = str(raw.get("name", "")).strip()
            if not name:
                continue
            results.append(
                ExtractedEntity(
                    name=name,
                    entity_type=str(raw.get("entity_type", "unknown")),
                    confidence=Confidence(float(raw.get("confidence", 1.0))),
                )
            )
        return results


@dataclass(slots=True)
class CallbackRelationExtractor:
    """Relation extractor driven by an external function.

    The callback receives the chunk text and the entity list (as dicts)
    and returns dicts with keys ``subject``, ``predicate``, ``object``,
    and optionally ``confidence``.
    """

    extract_fn: _RelationFn

    def extract(
        self,
        chunk: Chunk,
        document: Document,
        entities: list[ExtractedEntity],
    ) -> list[ExtractedRelation]:
        entity_dicts = [
            {
                "name": e.name,
                "entity_type": e.entity_type,
                "confidence": float(e.confidence),
            }
            for e in entities
        ]
        results: list[ExtractedRelation] = []
        for raw in self.extract_fn(chunk.text, entity_dicts):
            subject = str(raw.get("subject", "")).strip()
            predicate = str(raw.get("predicate", "")).strip()
            object_ = str(raw.get("object", "")).strip()
            if not subject or not predicate or not object_:
                continue
            results.append(
                ExtractedRelation(
                    subject=subject,
                    predicate=predicate,
                    object=object_,
                    confidence=Confidence(float(raw.get("confidence", 1.0))),
                )
            )
        return results
