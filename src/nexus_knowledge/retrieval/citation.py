"""Deterministic lexical search over citation-ingested document segments."""

from __future__ import annotations

import hashlib
import math
import re
import unicodedata
from bisect import bisect_left, bisect_right
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .._citation_epub import (
    EPUB_PARSE_LIMITS_KEY,
    EpubParseLimits,
    decode_epub_raw,
    epub_locator_hash,
    epub_parse_limits_from_metadata,
    parse_epub,
)
from ..domain.document import Chunk, Document, Span
from ..domain.ids import stable_id
from ..domain.source import Source
from ..port.citation_ingestion import CITATION_INGESTION_VERSION
from ..port.citation_search import (
    CitationCandidate,
    CitationSearchFilters,
    CitationSearchRepository,
)

__all__ = ["CitationLexicalSearch"]

_FILTER_KEYS = frozenset(("source_id", "document_id", "source_kind"))
_CITATION_KINDS = frozenset(("text", "markdown", "pdf", "epub"))
_LINE_BREAK = re.compile(r"\r\n|\r|\n")
_MARKDOWN_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$")
_EPUB_PROVENANCE_LIMITS = EpubParseLimits()


@dataclass(frozen=True, slots=True)
class _ValidatedAggregate:
    source: Source
    document: Document
    chunks: tuple[Chunk, ...]
    content_hash: str


@lru_cache(maxsize=1)
def _lexical_retriever_type() -> type[Any]:
    from .lexical import LexicalRetriever

    return LexicalRetriever


def _citation_tokenize(text: str) -> list[str]:
    """Return NFKC-casefolded runs of Unicode alphanumeric characters."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens: list[str] = []
    current: list[str] = []
    for character in normalized:
        if character.isalnum():
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current.clear()
    if current:
        tokens.append("".join(current))
    return tokens


class CitationLexicalSearch:
    """Search only complete, internally consistent citation aggregates."""

    def __init__(self, repository: CitationSearchRepository) -> None:
        self._repository = repository

    def candidates(
        self,
        filters: CitationSearchFilters | Mapping[str, str] | None = None,
    ) -> list[CitationCandidate]:
        """Return every validated eligible chunk, independent of any query.

        Enumeration deliberately reuses the lexical aggregate validator. It
        never reads a semantic index and cannot promote an index's text or IDs
        into authoritative provenance.
        """

        normalized_filters = self._validate_filters(filters)
        candidates = [
            self._candidate((aggregate, chunk), 0.0)
            for aggregate in self._current_aggregates(normalized_filters)
            for chunk in aggregate.chunks
        ]
        candidates.sort(
            key=lambda candidate: (
                candidate.source_id,
                candidate.document_id,
                candidate.segment_index,
                candidate.chunk_id,
            )
        )
        return candidates

    def search(
        self,
        query: str,
        limit: int = 20,
        filters: CitationSearchFilters | Mapping[str, str] | None = None,
    ) -> list[CitationCandidate]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")
        if isinstance(limit, bool) or not isinstance(limit, int):
            raise TypeError("limit must be an integer")
        if limit <= 0:
            raise ValueError("limit must be positive")
        normalized_filters = self._validate_filters(filters)
        query_tokens = list(dict.fromkeys(_citation_tokenize(query)))
        if not query_tokens:
            return []

        aggregates = self._current_aggregates(normalized_filters)
        chunk_context: dict[str, tuple[_ValidatedAggregate, Chunk]] = {}
        for aggregate in aggregates:
            for chunk in aggregate.chunks:
                chunk_context[chunk.id] = (aggregate, chunk)
        if not chunk_context:
            return []

        retriever = _lexical_retriever_type()(tokenizer=_citation_tokenize)
        retriever.add_chunks([context[1] for context in chunk_context.values()])
        hits = retriever.search(query_tokens, top_k=len(chunk_context))
        candidates = [
            self._candidate(chunk_context[hit.object_id], hit.score)
            for hit in hits
            if hit.object_id in chunk_context and math.isfinite(hit.score) and hit.score > 0.0
        ]
        candidates.sort(
            key=lambda candidate: (
                -candidate.score,
                candidate.source_id,
                candidate.document_id,
                candidate.segment_index,
                candidate.chunk_id,
            )
        )
        return candidates[:limit]

    @staticmethod
    def _validate_filters(
        filters: CitationSearchFilters | Mapping[str, str] | None,
    ) -> CitationSearchFilters:
        if filters is None:
            return CitationSearchFilters()
        if isinstance(filters, CitationSearchFilters):
            values = {
                "source_id": filters.source_id,
                "document_id": filters.document_id,
                "source_kind": filters.source_kind,
            }
            allow_none = True
        elif isinstance(filters, Mapping):
            unknown = set(filters) - _FILTER_KEYS
            if unknown:
                raise ValueError(f"unknown citation search filters: {sorted(unknown, key=str)}")
            values = dict(filters)
            allow_none = False
        else:
            raise TypeError("filters must be CitationSearchFilters, a mapping, or None")

        validated: dict[str, str | None] = {}
        for key in _FILTER_KEYS:
            if key not in values:
                validated[key] = None
                continue
            value = values[key]
            if value is None and allow_none:
                validated[key] = None
                continue
            if not isinstance(value, str):
                raise TypeError(f"citation search filter {key} must be a string")
            if not value.strip():
                raise ValueError(f"citation search filter {key} must not be empty")
            validated[key] = value
        return CitationSearchFilters(**validated)

    def _current_aggregates(
        self,
        filters: CitationSearchFilters,
    ) -> list[_ValidatedAggregate]:
        sources: dict[str, Source] = {}
        for source in self._repository.sources.all():
            try:
                is_valid = isinstance(source, Source) and self._valid_source(source)
            except Exception:
                is_valid = False
            if not is_valid:
                continue
            if filters.source_id is not None and source.id != filters.source_id:
                continue
            if filters.source_kind is not None and source.kind != filters.source_kind:
                continue
            sources[source.id] = source

        chunks_by_document: dict[str, list[Chunk]] = {}
        for chunk in self._repository.chunks.all():
            if not isinstance(chunk, Chunk) or not isinstance(chunk.document_id, str):
                continue
            chunks_by_document.setdefault(chunk.document_id, []).append(chunk)

        aggregates: list[_ValidatedAggregate] = []
        for document in self._repository.documents.all():
            if not isinstance(document, Document):
                continue
            try:
                document_id = document.id
                source_id = document.source_id
                if not isinstance(document_id, str) or not isinstance(source_id, str):
                    continue
                if filters.document_id is not None and document_id != filters.document_id:
                    continue
                source = sources.get(source_id)
            except Exception:
                continue
            if source is None:
                continue
            try:
                aggregate = self._validate_aggregate(
                    source,
                    document,
                    chunks_by_document.get(document_id, []),
                )
            except Exception:
                aggregate = None
            if aggregate is not None:
                aggregates.append(aggregate)
        return aggregates

    @classmethod
    def _validate_aggregate(
        cls,
        source: Source,
        document: Document,
        chunks: list[Chunk],
    ) -> _ValidatedAggregate | None:
        if not cls._valid_source(source):
            return None
        content_hash = source.metadata["content_hash"]
        if not cls._valid_document(source, document, content_hash):
            return None
        segment_count = document.metadata["segment_count"]
        if len(chunks) != segment_count:
            return None
        if any(
            not isinstance(chunk, Chunk)
            or isinstance(chunk.index, bool)
            or not isinstance(chunk.index, int)
            or chunk.index < 0
            for chunk in chunks
        ):
            return None
        ordered = sorted(chunks, key=lambda chunk: chunk.index)
        if [chunk.index for chunk in ordered] != list(range(segment_count)):
            return None

        cursor = 0
        break_ends = [match.end() for match in _LINE_BREAK.finditer(document.text)]
        for expected_index, chunk in enumerate(ordered):
            if not cls._valid_chunk(
                source,
                document,
                chunk,
                content_hash,
                expected_index,
                cursor,
                break_ends,
            ):
                return None
            assert isinstance(chunk.span, Span)
            cursor = chunk.span.end
        if cursor != len(document.text):
            return None
        if source.kind == "pdf" and not cls._valid_pdf_locators(ordered):
            return None
        if source.kind == "markdown" and not cls._valid_markdown_sections(
            source,
            document,
            ordered,
        ):
            return None
        if source.kind == "epub" and not cls._valid_epub_provenance(
            document,
            content_hash,
            ordered,
        ):
            return None
        deepcopy(source.metadata)
        for chunk in ordered:
            deepcopy(chunk.metadata)
        return _ValidatedAggregate(source, document, tuple(ordered), content_hash)

    @staticmethod
    def _valid_source(source: Source) -> bool:
        if not isinstance(source, Source):
            return False
        if not all(
            isinstance(value, str) and bool(value)
            for value in (
                source.id,
                source.title,
                source.kind,
                source.reference,
                source.ingested_at,
            )
        ):
            return False
        if source.kind not in _CITATION_KINDS or not isinstance(source.metadata, dict):
            return False
        metadata = source.metadata
        content_hash = metadata.get("content_hash")
        if not (
            isinstance(content_hash, str)
            and len(content_hash) == 64
            and all(character in "0123456789abcdef" for character in content_hash)
        ):
            return False
        if metadata.get("ingestion_version") != CITATION_INGESTION_VERSION:
            return False
        if metadata.get("source_reference") != source.reference:
            return False
        return source.id == stable_id(
            "src",
            CITATION_INGESTION_VERSION,
            source.reference,
            content_hash,
        )

    @staticmethod
    def _valid_document(source: Source, document: Document, content_hash: str) -> bool:
        if not isinstance(document, Document):
            return False
        if not all(
            isinstance(value, str) and bool(value)
            for value in (
                document.id,
                document.source_id,
                document.title,
                document.content_type,
                document.ingested_at,
            )
        ):
            return False
        if not isinstance(document.text, str) or not isinstance(document.raw, str):
            return False
        if not document.text:
            return False
        if source.kind != "epub" and document.raw != document.text:
            return False
        if source.kind in {"text", "markdown"}:
            # Strict UTF-8 ingestion preserves BOMs and every original line ending.
            if hashlib.sha256(document.raw.encode("utf-8")).hexdigest() != content_hash:
                return False
        if not isinstance(document.metadata, dict):
            return False
        metadata = document.metadata
        segment_count = metadata.get("segment_count")
        if (
            isinstance(segment_count, bool)
            or not isinstance(segment_count, int)
            or segment_count <= 0
        ):
            return False
        if document.source_id != source.id or document.title != source.title:
            return False
        if document.content_type != source.kind or document.ingested_at != source.ingested_at:
            return False
        if metadata.get("ingestion_version") != CITATION_INGESTION_VERSION:
            return False
        if metadata.get("content_hash") != content_hash:
            return False
        if metadata.get("source_reference") != source.reference:
            return False
        return document.id == stable_id(
            "doc",
            CITATION_INGESTION_VERSION,
            source.reference,
            content_hash,
            source.kind,
        )

    @classmethod
    def _valid_chunk(
        cls,
        source: Source,
        document: Document,
        chunk: Chunk,
        content_hash: str,
        expected_index: int,
        cursor: int,
        break_ends: list[int],
    ) -> bool:
        if chunk.document_id != document.id or chunk.index != expected_index:
            return False
        if not isinstance(chunk.text, str) or not chunk.text:
            return False
        span = chunk.span
        if not isinstance(span, Span):
            return False
        if any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (span.start, span.end)
        ):
            return False
        if span.start != cursor or span.end <= cursor:
            return False
        if span.end > len(document.text):
            return False
        if chunk.text != document.text[span.start : span.end]:
            return False
        if not isinstance(chunk.metadata, dict):
            return False
        metadata = chunk.metadata
        required_matches = {
            "source_id": source.id,
            "source_reference": source.reference,
            "content_hash": content_hash,
            "ingestion_version": CITATION_INGESTION_VERSION,
            "source_kind": source.kind,
            "ingested_at": source.ingested_at,
            "char_start": span.start,
            "char_end": span.end,
        }
        if any(metadata.get(key) != value for key, value in required_matches.items()):
            return False
        if any(
            isinstance(metadata.get(key), bool) or not isinstance(metadata.get(key), int)
            for key in ("char_start", "char_end", "line_start", "line_end")
        ):
            return False
        line_start = metadata["line_start"]
        line_end = metadata["line_end"]
        expected_line_start = bisect_right(break_ends, span.start) + 1
        expected_line_end = max(
            expected_line_start,
            bisect_left(break_ends, span.end) + 1,
        )
        if (
            line_start <= 0
            or line_end < line_start
            or line_start != expected_line_start
            or line_end != expected_line_end
        ):
            return False
        return cls._valid_format_metadata(source.kind, expected_index, metadata)

    @classmethod
    def _valid_format_metadata(
        cls,
        kind: str,
        segment_index: int,
        metadata: dict[str, Any],
    ) -> bool:
        if not cls._valid_optional_locator(metadata, "page", int, positive=True):
            return False
        if not cls._valid_optional_locator(metadata, "chapter", int, positive=True):
            return False
        if not cls._valid_optional_locator(metadata, "section", str, positive=False):
            return False

        if kind == "pdf":
            page = metadata.get("page")
            page_index = metadata.get("page_index")
            page_count = metadata.get("page_count")
            return (
                cls._positive_int(page)
                and cls._nonnegative_int(page_index)
                and cls._positive_int(page_count)
                and page_index == page - 1
                and page_count >= page
                and "chapter" not in metadata
                and "section" not in metadata
            )
        if kind == "epub":
            section_derivation = metadata.get("section_derivation")
            locator_hash = metadata.get("locator_hash")
            return (
                metadata.get("chapter") == segment_index + 1
                and isinstance(metadata.get("chapter_id"), str)
                and bool(metadata["chapter_id"])
                and isinstance(metadata.get("chapter_path"), str)
                and bool(metadata["chapter_path"])
                and isinstance(metadata.get("section"), str)
                and bool(metadata["section"])
                and section_derivation in {"heading", "chapter_path_stem"}
                and isinstance(locator_hash, str)
                and len(locator_hash) == 64
                and all(character in "0123456789abcdef" for character in locator_hash)
                and (
                    cls._positive_int(metadata.get("section_line"))
                    if section_derivation == "heading"
                    else "section_line" not in metadata
                )
                and "page" not in metadata
            )
        if kind == "markdown":
            return (
                isinstance(metadata.get("section"), str)
                and bool(metadata["section"])
                and "page" not in metadata
                and "chapter" not in metadata
            )
        return all(key not in metadata for key in ("page", "chapter", "section"))

    @classmethod
    def _valid_pdf_locators(cls, chunks: list[Chunk]) -> bool:
        page_count: int | None = None
        previous_page = 0
        previous_page_index = -1
        for chunk in chunks:
            metadata = chunk.metadata
            page = metadata.get("page")
            page_index = metadata.get("page_index")
            current_page_count = metadata.get("page_count")
            if not (
                cls._positive_int(page)
                and cls._nonnegative_int(page_index)
                and cls._positive_int(current_page_count)
                and page_index == page - 1
                and page <= current_page_count
            ):
                return False
            if page <= previous_page or page_index <= previous_page_index:
                return False
            if page_count is None:
                page_count = current_page_count
            elif current_page_count != page_count:
                return False
            previous_page = page
            previous_page_index = page_index
        return page_count is not None

    @staticmethod
    def _valid_markdown_sections(
        source: Source,
        document: Document,
        chunks: list[Chunk],
    ) -> bool:
        canonical_title = Path(source.reference).stem
        if source.title != canonical_title or document.title != canonical_title:
            return False

        headings: list[tuple[int, str]] = []
        offset = 0
        for line in document.text.splitlines(keepends=True):
            logical_line = line.rstrip("\r\n")
            match = _MARKDOWN_HEADING.fullmatch(logical_line)
            if match is not None:
                headings.append((offset, match.group(2).strip() or canonical_title))
            offset += len(line)

        expected: list[tuple[int, int, str]] = []
        if not headings:
            expected.append((0, len(document.text), canonical_title))
        else:
            if headings[0][0] > 0:
                expected.append((0, headings[0][0], canonical_title))
            for index, (start, section) in enumerate(headings):
                end = headings[index + 1][0] if index + 1 < len(headings) else len(document.text)
                if end > start:
                    expected.append((start, end, section))

        if len(chunks) != len(expected):
            return False
        for chunk, (start, end, section) in zip(chunks, expected, strict=True):
            if not isinstance(chunk.span, Span):
                return False
            if (chunk.span.start, chunk.span.end) != (start, end):
                return False
            if chunk.metadata.get("section") != section:
                return False
        return True

    @staticmethod
    def _valid_epub_provenance(
        document: Document,
        content_hash: str,
        chunks: list[Chunk],
    ) -> bool:
        limits = epub_parse_limits_from_metadata(document.metadata.get(EPUB_PARSE_LIMITS_KEY))
        payload = decode_epub_raw(
            document.raw,
            max_file_bytes=limits.max_file_bytes,
        )
        if hashlib.sha256(payload).hexdigest() != content_hash:
            return False
        parsed = parse_epub(payload, limits=limits)
        if len(parsed) != len(chunks):
            return False

        expected_text_parts: list[str] = []
        cursor = 0
        for index, (chunk, segment) in enumerate(zip(chunks, parsed, strict=True)):
            metadata = chunk.metadata
            expected_text = segment.text + ("\n" if index + 1 < len(parsed) else "")
            expected_end = cursor + len(expected_text)
            if not isinstance(chunk.span, Span):
                return False
            if chunk.text != expected_text:
                return False
            if (chunk.span.start, chunk.span.end) != (cursor, expected_end):
                return False
            expected_metadata = segment.metadata()
            if any(metadata.get(key) != value for key, value in expected_metadata.items()):
                return False
            if segment.section_line is None and "section_line" in metadata:
                return False

            expected_hash = epub_locator_hash(
                content_hash=content_hash,
                document_id=document.id,
                chunk_id=chunk.id,
                segment_index=chunk.index,
                chapter=segment.chapter,
                chapter_id=segment.chapter_id,
                chapter_path=segment.chapter_path,
                section=segment.section,
                section_derivation=segment.section_derivation,
                section_line=segment.section_line,
            )
            if metadata.get("locator_hash") != expected_hash:
                return False
            expected_text_parts.append(expected_text)
            cursor = expected_end
        return "".join(expected_text_parts) == document.text

    @staticmethod
    def _valid_optional_locator(
        metadata: dict[str, Any],
        key: str,
        expected_type: type[int] | type[str],
        *,
        positive: bool,
    ) -> bool:
        if key not in metadata:
            return True
        value = metadata[key]
        if expected_type is int:
            return CitationLexicalSearch._positive_int(value) if positive else False
        return isinstance(value, str) and bool(value)

    @staticmethod
    def _positive_int(value: Any) -> bool:
        return not isinstance(value, bool) and isinstance(value, int) and value > 0

    @staticmethod
    def _nonnegative_int(value: Any) -> bool:
        return not isinstance(value, bool) and isinstance(value, int) and value >= 0

    @staticmethod
    def _candidate(
        context: tuple[_ValidatedAggregate, Chunk],
        score: float,
    ) -> CitationCandidate:
        aggregate, chunk = context
        assert isinstance(chunk.span, Span)
        metadata = chunk.metadata
        return CitationCandidate(
            source_id=aggregate.source.id,
            source_title=aggregate.source.title,
            source_reference=aggregate.source.reference,
            source_kind=aggregate.source.kind,
            source_metadata=deepcopy(aggregate.source.metadata),
            document_id=aggregate.document.id,
            document_title=aggregate.document.title,
            document_content_type=aggregate.document.content_type,
            chunk_id=chunk.id,
            segment_index=chunk.index,
            text=chunk.text,
            content_hash=aggregate.content_hash,
            page=metadata.get("page"),
            chapter=metadata.get("chapter"),
            section=metadata.get("section"),
            line_start=metadata["line_start"],
            line_end=metadata["line_end"],
            char_start=chunk.span.start,
            char_end=chunk.span.end,
            segment_metadata=deepcopy(metadata),
            score=float(score),
        )
