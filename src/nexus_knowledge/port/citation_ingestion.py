"""Public boundary for citation-ready local document ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

__all__ = ["CITATION_INGESTION_VERSION", "CitationIngestionPort"]


CITATION_INGESTION_VERSION = "citation-ingestion-v1"


class CitationIngestionPort(Protocol):
    """Ingest a local document and return its persisted document identifier."""

    def ingest_document(self, path: str | Path) -> str: ...
