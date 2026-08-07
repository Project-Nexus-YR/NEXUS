"""Typed ports (interfaces) for all external dependencies.

The domain and application layers depend only on these abstractions.
Concrete providers (embedding models, vector stores, graph databases,
search systems) are adapters injected from the composition root.
"""

from __future__ import annotations

from .chunker import Chunker
from .embeddings import Embedding, EmbeddingProvider
from .extractors import (
    EntityExtractor,
    ExtractedEntity,
    ExtractedRelation,
    RelationExtractor,
)
from .reranker import RerankCandidate, Reranker
from .repository import (
    ClaimRepository,
    DocumentRepository,
    EvidenceRepository,
    EntityRepository,
    KnowledgeRepository,
    RelationRepository,
    SourceRepository,
)
from .search import SearchProvider, SearchResult
from .vector_store import VectorHit, VectorStore

__all__ = [
    "Chunker",
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
