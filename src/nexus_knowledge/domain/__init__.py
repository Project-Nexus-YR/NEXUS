"""Domain objects for the knowledge intelligence engine."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "Claim",
    "Chunk",
    "Confidence",
    "Contradiction",
    "ContradictionKind",
    "Document",
    "Entity",
    "Evidence",
    "EvidenceRole",
    "Experiment",
    "GapKind",
    "Hypothesis",
    "Investigation",
    "KnowledgeGap",
    "Observation",
    "Provenance",
    "Relation",
    "Result",
    "Source",
    "SourceKind",
    "Span",
    "VerificationState",
    "new_id",
    "stable_id",
]

_LAZY_EXPORTS = {
    "Claim": (".claim", "Claim"),
    "Chunk": (".document", "Chunk"),
    "Confidence": (".common", "Confidence"),
    "Contradiction": (".contradiction", "Contradiction"),
    "ContradictionKind": (".contradiction", "ContradictionKind"),
    "Document": (".document", "Document"),
    "Entity": (".entity", "Entity"),
    "Evidence": (".claim", "Evidence"),
    "EvidenceRole": (".claim", "EvidenceRole"),
    "Experiment": (".hypothesis", "Experiment"),
    "GapKind": (".knowledge_gap", "GapKind"),
    "Hypothesis": (".hypothesis", "Hypothesis"),
    "Investigation": (".knowledge_gap", "Investigation"),
    "KnowledgeGap": (".knowledge_gap", "KnowledgeGap"),
    "Observation": (".hypothesis", "Observation"),
    "Provenance": (".claim", "Provenance"),
    "Relation": (".entity", "Relation"),
    "Result": (".hypothesis", "Result"),
    "Source": (".source", "Source"),
    "SourceKind": (".source", "SourceKind"),
    "Span": (".document", "Span"),
    "VerificationState": (".common", "VerificationState"),
    "new_id": (".ids", "new_id"),
    "stable_id": (".ids", "stable_id"),
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
