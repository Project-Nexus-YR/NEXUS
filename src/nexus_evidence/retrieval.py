"""Evidence retrieval ports and adapters for NEXUS knowledge and search providers."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from .model import EvidenceSource, EvidenceSpan, stable_id
from .planning import SearchQuery
from .security import UnsafeSourceError, normalize_untrusted_text, validate_public_reference


@dataclass(frozen=True, slots=True)
class RetrievedEvidence:
    source: EvidenceSource
    span: EvidenceSpan
    score: float


class EvidenceRetriever(Protocol):
    def search(self, query: SearchQuery, *, limit: int = 5) -> tuple[RetrievedEvidence, ...]: ...


class KnowledgeEvidencePort(Protocol):
    def retrieve_evidence(self, query: str, top_k: int = 5) -> list[dict[str, Any]]: ...


class SearchProviderPort(Protocol):
    def search(self, query: str) -> list[dict[str, Any]]: ...


class KnowledgeEvidenceRetriever:
    """Adapt the public KnowledgeEngine provenance response to evidence candidates."""

    def __init__(self, knowledge: KnowledgeEvidencePort) -> None:
        self._knowledge = knowledge

    def search(self, query: SearchQuery, *, limit: int = 5) -> tuple[RetrievedEvidence, ...]:
        records = self._knowledge.retrieve_evidence(query.text, top_k=limit)
        return tuple(self._record(query, record, rank) for rank, record in enumerate(records, 1))

    @staticmethod
    def _record(query: SearchQuery, record: Mapping[str, Any], rank: int) -> RetrievedEvidence:
        text = normalize_untrusted_text(str(record["text"]))
        reference = str(record["source_reference"])
        source = EvidenceSource(
            source_id=str(record["source_id"]),
            title=str(record.get("source_title", reference)),
            reference=reference,
            source_type=str(record.get("source_type", "knowledge")),
            publisher=str(record.get("publisher", "")),
            published_at=str(record.get("published_at", "")),
            content_hash=str(record.get("content_hash", hashlib.sha256(text.encode()).hexdigest())),
            quality=float(record.get("source_quality", 0.7)),
            metadata=dict(record.get("source_metadata", {})),
        )
        span = EvidenceSpan(
            source_id=source.source_id,
            text=text,
            document_id=str(record["document_id"]),
            chunk_id=str(record["chunk_id"]),
            search_query=query.text,
            retrieval_strategy=query.strategy,
            rank=rank,
            page=_optional_int(record.get("page")),
            section=str(record.get("section", "")),
            paragraph=_optional_int(record.get("paragraph")),
            char_start=int(record.get("char_start", 0)),
            char_end=int(record.get("char_end", len(text))),
            metadata={"request_id": str(record.get("request_id", ""))},
        )
        return RetrievedEvidence(source, span, float(record.get("score", 0.0)))


class WebSearchEvidenceRetriever:
    """Normalize provider-neutral search results without executing source instructions."""

    def __init__(self, search_provider: SearchProviderPort) -> None:
        self._search = search_provider

    def search(self, query: SearchQuery, *, limit: int = 5) -> tuple[RetrievedEvidence, ...]:
        output: list[RetrievedEvidence] = []
        for raw in self._search.search(query.text)[:limit]:
            try:
                reference = validate_public_reference(str(raw.get("url", "")))
            except UnsafeSourceError:
                continue
            text = normalize_untrusted_text(str(raw.get("excerpt") or raw.get("text") or ""))
            if not text.strip():
                continue
            rank = len(output) + 1
            source_id = str(raw.get("source_id") or stable_id("web_source", reference))
            source = EvidenceSource(
                source_id=source_id,
                title=str(raw.get("title") or reference),
                reference=reference,
                source_type="web",
                publisher=str(raw.get("publisher", "")),
                published_at=str(raw.get("published_at", "")),
                content_hash=hashlib.sha256(text.encode()).hexdigest(),
                quality=float(raw.get("source_quality", 0.5)),
            )
            document_id = str(raw.get("document_id") or stable_id("web_document", reference))
            span = EvidenceSpan(
                source_id=source_id,
                text=text,
                document_id=document_id,
                chunk_id=str(raw.get("chunk_id") or stable_id("web_chunk", document_id, text)),
                search_query=query.text,
                retrieval_strategy=query.strategy,
                rank=rank,
                page=_optional_int(raw.get("page")),
                section=str(raw.get("section", "")),
                paragraph=_optional_int(raw.get("paragraph")),
                char_start=int(raw.get("char_start", 0)),
                char_end=int(raw.get("char_end", len(text))),
            )
            output.append(RetrievedEvidence(source, span, float(raw.get("score", 0.0))))
        return tuple(output)


class CompositeEvidenceRetriever:
    def __init__(self, *retrievers: EvidenceRetriever) -> None:
        self._retrievers = retrievers

    def search(self, query: SearchQuery, *, limit: int = 5) -> tuple[RetrievedEvidence, ...]:
        candidates = [
            item for retriever in self._retrievers for item in retriever.search(query, limit=limit)
        ]
        unique: dict[str, RetrievedEvidence] = {}
        for item in candidates:
            previous = unique.get(item.span.evidence_id)
            if previous is None or item.score > previous.score:
                unique[item.span.evidence_id] = item
        return tuple(
            sorted(unique.values(), key=lambda item: (-item.score, item.span.evidence_id))[:limit]
        )


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool) or not isinstance(value, (str, int, float)):
        raise ValueError("optional locator must be numeric")
    return int(value)
