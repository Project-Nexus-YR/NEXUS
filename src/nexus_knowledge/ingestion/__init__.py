"""Ingestion pipeline and source adapters."""

from .adapters import (
    JsonAdapter,
    MarkdownAdapter,
    RawDocument,
    RepositoryAdapter,
    SourceAdapter,
    TextAdapter,
)
from .citation import CitationIngestionLimits, CitationIngestionService
from .normalization import RecursiveChunker, normalize_text
from .pipeline import IngestionPipeline, IngestionResult

__all__ = [
    "IngestionPipeline",
    "IngestionResult",
    "CitationIngestionLimits",
    "CitationIngestionService",
    "JsonAdapter",
    "MarkdownAdapter",
    "RawDocument",
    "RecursiveChunker",
    "RepositoryAdapter",
    "SourceAdapter",
    "TextAdapter",
    "normalize_text",
]
