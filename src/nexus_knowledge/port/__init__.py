"""Typed ports for external dependencies, resolved lazily by public name."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "CITATION_INGESTION_VERSION",
    "Chunker",
    "CitationCandidate",
    "CitationCandidatePort",
    "CitationIngestionPort",
    "CitationSearchFilters",
    "CitationSearchPort",
    "CitationSearchRepository",
    "ClaimRepository",
    "DocumentRepository",
    "Embedding",
    "EmbeddingProvider",
    "EntityExtractor",
    "EntityRepository",
    "EvidenceRepository",
    "ExtractedEntity",
    "ExtractedRelation",
    "KnowledgeRepository",
    "RelationExtractor",
    "RelationRepository",
    "RerankCandidate",
    "Reranker",
    "SearchProvider",
    "SearchResult",
    "SourceRepository",
    "VectorHit",
    "VectorStore",
]

_LAZY_EXPORTS = {
    "CITATION_INGESTION_VERSION": (".citation_ingestion", "CITATION_INGESTION_VERSION"),
    "Chunker": (".chunker", "Chunker"),
    "CitationCandidate": (".citation_search", "CitationCandidate"),
    "CitationCandidatePort": (".citation_search", "CitationCandidatePort"),
    "CitationIngestionPort": (".citation_ingestion", "CitationIngestionPort"),
    "CitationSearchFilters": (".citation_search", "CitationSearchFilters"),
    "CitationSearchPort": (".citation_search", "CitationSearchPort"),
    "CitationSearchRepository": (".citation_search", "CitationSearchRepository"),
    "ClaimRepository": (".repository", "ClaimRepository"),
    "DocumentRepository": (".repository", "DocumentRepository"),
    "Embedding": (".embeddings", "Embedding"),
    "EmbeddingProvider": (".embeddings", "EmbeddingProvider"),
    "EntityExtractor": (".extractors", "EntityExtractor"),
    "EntityRepository": (".repository", "EntityRepository"),
    "EvidenceRepository": (".repository", "EvidenceRepository"),
    "ExtractedEntity": (".extractors", "ExtractedEntity"),
    "ExtractedRelation": (".extractors", "ExtractedRelation"),
    "KnowledgeRepository": (".repository", "KnowledgeRepository"),
    "RelationExtractor": (".extractors", "RelationExtractor"),
    "RelationRepository": (".repository", "RelationRepository"),
    "RerankCandidate": (".reranker", "RerankCandidate"),
    "Reranker": (".reranker", "Reranker"),
    "SearchProvider": (".search", "SearchProvider"),
    "SearchResult": (".search", "SearchResult"),
    "SourceRepository": (".repository", "SourceRepository"),
    "VectorHit": (".vector_store", "VectorHit"),
    "VectorStore": (".vector_store", "VectorStore"),
}


def __getattr__(name: str) -> Any:
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
