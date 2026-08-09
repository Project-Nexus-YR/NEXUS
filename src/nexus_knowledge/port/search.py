"""External search provider port.

The other subsystem (autonomous research runtime / existing search
system) plugs in here through an adapter. This subsystem works against
a mock provider and never depends on a concrete search implementation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

__all__ = ["SearchResult", "SearchProvider"]


@dataclass(frozen=True, slots=True)
class SearchResult:
    reference: str  # stable reference usable with get_document()
    title: str
    snippet: str
    score: float
    metadata: dict[str, Any] = field(default_factory=dict)


class SearchProvider(Protocol):
    """Queries an external document/search system."""

    def search(self, query: str, limit: int = 10) -> list[SearchResult]: ...

    def search_batch(self, queries: Sequence[str], limit: int = 10) -> list[list[SearchResult]]: ...

    def get_document(self, reference: str) -> str | None:
        """Return the raw document text for a reference, if available."""
        ...
