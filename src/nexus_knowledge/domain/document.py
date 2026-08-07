"""Documents and chunks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .common import now_iso
from .ids import new_id, stable_id

__all__ = ["Document", "Chunk", "Span"]


@dataclass(frozen=True, slots=True)
class Span:
    """Character span into a document's normalized text (half-open)."""

    start: int
    end: int

    def __post_init__(self) -> None:
        if self.start < 0 or self.end < self.start:
            raise ValueError(f"invalid span {self.start}:{self.end}")

    def slice(self, text: str) -> str:
        return text[self.start : self.end]


@dataclass(slots=True)
class Document:
    """A normalized unit of ingested information.

    ``raw`` preserves the original bytes (or source text) while
    ``text`` holds the normalized form used for chunking and
    extraction.
    """

    source_id: str
    title: str
    content_type: str
    text: str
    raw: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    id: str = field(default_factory=lambda: new_id("doc"))
    ingested_at: str = field(default_factory=now_iso)


@dataclass(slots=True)
class Chunk:
    """A slice of a document used for embedding, extraction and evidence.

    ``span`` is expressed in the *document's* normalized text so that
    source spans remain verifiable against the original artifact.
    """

    document_id: str
    index: int
    text: str
    span: Span | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        """Deterministic chunk id derived from the document and index."""
        return stable_id("chunk", self.document_id, self.index)

    def __hash__(self) -> int:
        return hash(self.id)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Chunk) and other.id == self.id
