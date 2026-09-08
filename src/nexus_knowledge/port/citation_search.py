"""Public contract for provenance-preserving citation candidate search."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "CitationCandidate",
    "CitationCandidatePort",
    "CitationSearchFilters",
    "CitationSearchPort",
    "CitationSearchRepository",
]


class _CitationRecordRepository(Protocol):
    def all(self) -> list[Any]: ...


class _CitationChunkRepository(Protocol):
    def all(self) -> list[Any]: ...


class CitationSearchRepository(Protocol):
    """Minimal persistence surface required by citation candidate search."""

    sources: _CitationRecordRepository
    documents: _CitationRecordRepository
    chunks: _CitationChunkRepository


@dataclass(frozen=True, slots=True)
class CitationCandidate:
    """A scored document segment with citation-ready source provenance."""

    source_id: str
    source_title: str
    source_reference: str
    source_kind: str
    source_metadata: Mapping[str, Any]
    document_id: str
    document_title: str
    document_content_type: str
    chunk_id: str
    segment_index: int
    text: str
    content_hash: str
    page: int | None
    chapter: int | None
    section: str | None
    line_start: int
    line_end: int
    char_start: int
    char_end: int
    segment_metadata: Mapping[str, Any]
    score: float

    @property
    def segment_id(self) -> str:
        """Alias the persisted chunk identifier for segment-oriented consumers."""

        return self.chunk_id


@dataclass(frozen=True, slots=True)
class CitationSearchFilters:
    """Basic identity filters for citation-ingested sources and documents."""

    source_id: str | None = None
    document_id: str | None = None
    source_kind: str | None = None


class CitationSearchPort(Protocol):
    """Search citation-ingested chunks without altering other retrieval APIs."""

    def search(
        self,
        query: str,
        limit: int = 20,
        filters: CitationSearchFilters | Mapping[str, str] | None = None,
    ) -> list[CitationCandidate]: ...


@runtime_checkable
class CitationCandidatePort(Protocol):
    """Enumerate every eligible persisted chunk without a lexical query gate.

    Implementations apply the same aggregate validation and identity filters as
    citation search. Returned scores are zero: ranking is derived downstream.
    This additive port does not require existing search-only ports to change.
    """

    def candidates(
        self,
        filters: CitationSearchFilters | Mapping[str, str] | None = None,
    ) -> list[CitationCandidate]: ...
