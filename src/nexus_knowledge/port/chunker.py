"""Chunking port used by the ingestion pipeline."""

from __future__ import annotations

from typing import Protocol

from ..domain.document import Chunk, Document

__all__ = ["Chunker"]


class Chunker(Protocol):
    def chunk(self, document: Document) -> list[Chunk]: ...
