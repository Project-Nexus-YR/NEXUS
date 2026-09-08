"""NEXUS Knowledge Intelligence Engine.

The knowledge/ML subsystem of the NEXUS autonomous knowledge-discovery
platform. Responsible for ingestion, knowledge-graph construction,
hybrid retrieval, GraphRAG, uncertainty modelling, contradiction
detection, knowledge-gap analysis and investigation scoring.

The subsystem is transport- and provider-independent. Every external
dependency (embedding models, vector stores, graph databases, search
systems) is accessed exclusively through typed ports (interfaces)
defined in :mod:`nexus_knowledge.port`.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.7.0"

__all__ = [
    "CitationCandidate",
    "CitationCandidatePort",
    "CitationIngestionPort",
    "CitationIngestionService",
    "CitationLexicalSearch",
    "CitationSearchFilters",
    "CitationSearchPort",
    "__version__",
]

_LAZY_EXPORTS = {
    "CitationCandidate": (".port.citation_search", "CitationCandidate"),
    "CitationCandidatePort": (".port.citation_search", "CitationCandidatePort"),
    "CitationIngestionPort": (".port.citation_ingestion", "CitationIngestionPort"),
    "CitationIngestionService": (".ingestion.citation", "CitationIngestionService"),
    "CitationLexicalSearch": (".retrieval.citation", "CitationLexicalSearch"),
    "CitationSearchFilters": (".port.citation_search", "CitationSearchFilters"),
    "CitationSearchPort": (".port.citation_search", "CitationSearchPort"),
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
