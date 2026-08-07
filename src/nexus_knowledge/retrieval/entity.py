"""Entity index for query-side entity linking.

Builds a gazetteer from the ingested entities and matches query text
against it, returning the canonical entity records.
"""

from __future__ import annotations

from ..domain.document import Chunk, Document
from ..domain.entity import Entity
from ..extraction.deterministic import GazetteerEntityExtractor
from ..port.repository import EntityRepository

__all__ = ["EntityIndex"]


class EntityIndex:
    """Maps entity names/aliases to canonical :class:`Entity` records."""

    def __init__(self, repository: EntityRepository) -> None:
        self._extractor = GazetteerEntityExtractor()
        self._by_lower: dict[str, Entity] = {}
        self._entities: dict[str, Entity] = {}
        for entity in repository.all():
            self._entities[entity.id] = entity
            self._register(entity)

    def _register(self, entity: Entity) -> None:
        self._extractor.add_term(entity.canonical, entity.entity_type)
        self._by_lower.setdefault(entity.canonical.lower(), entity)
        for alias in entity.aliases:
            if not alias:
                continue
            self._extractor.add_term(alias, entity.entity_type)
            self._by_lower.setdefault(alias.lower(), entity)

    def get(self, entity_id: str) -> Entity | None:
        return self._entities.get(entity_id)

    def match(self, text: str) -> list[Entity]:
        """Return entities mentioned in ``text`` in order of mention."""
        document = Document(source_id="", title="", content_type="query", text=text)
        chunk = Chunk(document_id=document.id, index=0, text=text)
        extracted = self._extractor.extract(chunk, document)
        matched: list[Entity] = []
        seen: set[str] = set()
        for item in extracted:
            entity = self._by_lower.get(item.name.lower())
            if entity is not None and entity.id not in seen:
                seen.add(entity.id)
                matched.append(entity)
        return matched

    def size(self) -> int:
        return len(self._entities)
