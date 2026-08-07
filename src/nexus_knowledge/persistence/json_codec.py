"""JSON codec for domain objects.

Supports snapshot persistence of the full knowledge base for offline
use, fixtures and deterministic evaluation. Only standard library and
domain types are handled; embeddings are stored via :class:`Embedding`
in the vector store and are not part of the domain snapshot.
"""

from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

from ..domain.claim import Claim, Evidence
from ..domain.common import Confidence, VerificationState
from ..domain.contradiction import Contradiction
from ..domain.document import Chunk, Document, Span
from ..domain.entity import Entity, Relation
from ..domain.hypothesis import Experiment, Hypothesis, Observation, Result
from ..domain.knowledge_gap import Investigation, KnowledgeGap
from ..domain.source import Source
from .memory import InMemoryKnowledgeRepository

__all__ = [
    "dumps",
    "loads",
    "save_snapshot",
    "load_snapshot",
    "to_plain",
    "from_plain",
]

_TYPE_KEY = "$type"
_IMPORTANT_NAMES = {
    "confidence": Confidence,
    "span": Span,
}


def to_plain(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Confidence):
        return {_TYPE_KEY: "confidence", "value": value.value}
    if isinstance(value, Span):
        return {_TYPE_KEY: "span", "start": value.start, "end": value.end}
    if isinstance(value, tuple):
        return [to_plain(x) for x in value]
    if isinstance(value, list):
        return [to_plain(x) for x in value]
    if isinstance(value, dict):
        return {k: to_plain(v) for k, v in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if is_dataclass(value):
        result: dict[str, Any] = {_TYPE_KEY: value.__class__.__name__.lower()}
        for field in value.__dataclass_fields__:
            result[field] = to_plain(getattr(value, field))
        return result
    raise TypeError(f"cannot serialize {type(value).__name__}: {value!r}")


def _resolve(value: str) -> type[Any]:
    return {
        "claim": Claim,
        "chunk": Chunk,
        "confidence": Confidence,
        "contradiction": Contradiction,
        "document": Document,
        "entity": Entity,
        "evidence": Evidence,
        "experiment": Experiment,
        "hypothesis": Hypothesis,
        "investigation": Investigation,
        "knowledgegap": KnowledgeGap,
        "observation": Observation,
        "relation": Relation,
        "result": Result,
        "source": Source,
        "span": Span,
    }[value]


def from_plain(value: Any) -> Any:
    if isinstance(value, list):
        return [from_plain(x) for x in value]
    if isinstance(value, dict):
        if _TYPE_KEY in value:
            type_name = value[_TYPE_KEY]
            data = {k: from_plain(v) for k, v in value.items() if k != _TYPE_KEY}
            if type_name == "confidence":
                return Confidence(data["value"])
            if type_name == "span":
                return Span(data["start"], data["end"])
            cls = _resolve(type_name)
            if type_name == "chunk":
                return Chunk(
                    document_id=data["document_id"],
                    index=data["index"],
                    text=data["text"],
                    span=data.get("span"),
                    metadata=data.get("metadata", {}),
                )
            return cls(**data)
        return {k: from_plain(v) for k, v in value.items()}
    return value


def dumps(obj: Any) -> str:
    return json.dumps(to_plain(obj), indent=2, sort_keys=True)


def loads(text: str) -> Any:
    return from_plain(json.loads(text))


def save_snapshot(repo: InMemoryKnowledgeRepository, path: str | Path) -> None:
    """Persist the entire knowledge base to a JSON snapshot file."""
    payload = {
        "version": 1,
        "sources": [to_plain(s) for s in repo.sources.all()],
        "documents": [to_plain(d) for d in repo.documents.all()],
        "chunks": [to_plain(c) for c in repo.chunks.all()],
        "entities": [to_plain(e) for e in repo.entities.all()],
        "relations": [to_plain(r) for r in repo.relations.all()],
        "claims": [to_plain(c) for c in repo.claims.all()],
        "evidence": [to_plain(e) for e in repo.evidence.all()],
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def load_snapshot(path: str | Path) -> InMemoryKnowledgeRepository:
    """Load a knowledge base from a JSON snapshot file."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    repo = InMemoryKnowledgeRepository()
    for item in payload.get("sources", []):
        repo.sources.save(from_plain(item))
    for item in payload.get("documents", []):
        repo.documents.save(from_plain(item))
    for item in payload.get("chunks", []):
        repo.chunks.save(from_plain(item))
    for item in payload.get("entities", []):
        repo.entities.save(from_plain(item))
    for item in payload.get("relations", []):
        repo.relations.save(from_plain(item))
    for item in payload.get("claims", []):
        repo.claims.save(from_plain(item))
    for item in payload.get("evidence", []):
        repo.evidence.save(from_plain(item))
    return repo
