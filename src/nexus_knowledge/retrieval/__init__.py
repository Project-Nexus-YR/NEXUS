"""Hybrid retrieval and GraphRAG with dependency-isolated lazy exports."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .citation import CitationLexicalSearch

__all__ = [
    "DeterministicReranker",
    "CitationLexicalSearch",
    "EntityIndex",
    "EntityRetriever",
    "EvidenceGraph",
    "FeatureExtractor",
    "GraphRAGEngine",
    "GraphRetriever",
    "HybridRetriever",
    "LexicalRetriever",
    "MethodLatency",
    "QueryAnalysis",
    "RankedCandidate",
    "ReciprocalRankFusion",
    "RetrievalHit",
    "RetrievalResult",
    "RetrievalTrace",
    "VectorRetriever",
    "analyze_query",
]

_LAZY_EXPORTS = {
    "DeterministicReranker": (".rerank", "DeterministicReranker"),
    "EntityIndex": (".entity", "EntityIndex"),
    "EntityRetriever": (".entity_retrieval", "EntityRetriever"),
    "EvidenceGraph": (".graphrag", "EvidenceGraph"),
    "FeatureExtractor": (".features", "FeatureExtractor"),
    "GraphRAGEngine": (".graphrag", "GraphRAGEngine"),
    "GraphRetriever": (".vector_graph", "GraphRetriever"),
    "HybridRetriever": (".hybrid", "HybridRetriever"),
    "LexicalRetriever": (".lexical", "LexicalRetriever"),
    "MethodLatency": (".observability", "MethodLatency"),
    "QueryAnalysis": (".query", "QueryAnalysis"),
    "RankedCandidate": (".hybrid", "RankedCandidate"),
    "ReciprocalRankFusion": (".fusion", "ReciprocalRankFusion"),
    "RetrievalHit": (".lexical", "RetrievalHit"),
    "RetrievalResult": (".hybrid", "RetrievalResult"),
    "RetrievalTrace": (".observability", "RetrievalTrace"),
    "VectorRetriever": (".vector_graph", "VectorRetriever"),
    "analyze_query": (".query", "analyze_query"),
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
