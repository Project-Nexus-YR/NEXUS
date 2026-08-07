"""Ingestion pipeline and source adapters."""

from .adapters import (
    JsonAdapter,
    MarkdownAdapter,
    RawDocument,
    RepositoryAdapter,
    SourceAdapter,
    TextAdapter,
)
from .normalization import RecursiveChunker, normalize_text
from .pipeline import IngestionPipeline, IngestionResult

__all__ = [
    "IngestionPipeline",
    "IngestionResult",
    "JsonAdapter",
    "MarkdownAdapter",
    "RawDocument",
    "RecursiveChunker",
    "RepositoryAdapter",
    "SourceAdapter",
    "TextAdapter",
    "normalize_text",
]
