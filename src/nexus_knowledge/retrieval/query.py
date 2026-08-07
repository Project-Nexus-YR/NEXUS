"""Query analysis for retrieval.

Splits a query into tokens, identifies entity mentions against the
ingested knowledge base, and carries optional metadata filters. Used by
the hybrid retrieval pipeline as its first stage.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..embedding.hashing import tokenize

__all__ = ["QueryAnalysis", "analyze_query"]


@dataclass(frozen=True, slots=True)
class QueryAnalysis:
    text: str
    tokens: tuple[str, ...]
    entity_ids: tuple[str, ...] = ()
    entity_names: tuple[str, ...] = ()
    filters: dict[str, object] = field(default_factory=dict)

    def has_entities(self) -> bool:
        return bool(self.entity_ids)


def analyze_query(
    text: str,
    entity_ids: list[str] | None = None,
    entity_names: list[str] | None = None,
    filters: dict[str, object] | None = None,
) -> QueryAnalysis:
    """Build a :class:`QueryAnalysis` from a raw query string."""
    return QueryAnalysis(
        text=text.strip(),
        tokens=tuple(tokenize(text)),
        entity_ids=tuple(entity_ids or []),
        entity_names=tuple(entity_names or []),
        filters=dict(filters or {}),
    )
