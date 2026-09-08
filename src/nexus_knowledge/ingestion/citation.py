"""Deterministic, bounded, citation-ready ingestion for local documents.

The service stops at the source/document/chunk boundary. It preserves exact
decoded text, assigns stable content-version identifiers, and never invokes
extraction, embeddings, retrieval, or graph updates.
"""

from __future__ import annotations

import hashlib
import math
import re
from bisect import bisect_left, bisect_right
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from io import BytesIO
from itertools import chain
from pathlib import Path
from threading import RLock
from typing import Any

from .._citation_epub import (
    EPUB_MAX_SEGMENTS,
    EPUB_MAX_TEXT_CHARS,
    EPUB_PARSE_LIMITS_KEY,
    EpubParseLimits,
    encode_epub_raw,
    epub_locator_hash,
    epub_parse_limits_metadata,
    parse_epub,
)
from ..domain.common import now_iso
from ..domain.document import Chunk, Document, Span
from ..domain.ids import stable_id
from ..domain.source import Source, SourceKind
from ..port.citation_ingestion import CITATION_INGESTION_VERSION
from ..port.repository import KnowledgeRepository

__all__ = ["CitationIngestionLimits", "CitationIngestionService"]

_INGESTION_VERSION = CITATION_INGESTION_VERSION
_SUPPORTED_SUFFIXES = {".txt", ".md", ".pdf", ".epub"}
_MARKDOWN_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*$")
_LINE_BREAK = re.compile(r"\r\n|\r|\n")
_MAX_FINAL_SEGMENTS = 10_000
_MAX_FINAL_TEXT_CHARS = 20_000_000
_PDF_FILTER_OUTPUT_LIMITS = (
    "ZLIB_MAX_OUTPUT_LENGTH",
    "LZW_MAX_OUTPUT_LENGTH",
    "RUN_LENGTH_MAX_OUTPUT_LENGTH",
    "MAX_ARRAY_BASED_STREAM_OUTPUT_LENGTH",
)
_PDF_CONTENT_FILTERS = {
    "/ASCIIHexDecode",
    "/AHx",
    "/ASCII85Decode",
    "/A85",
    "/LZWDecode",
    "/LZW",
    "/FlateDecode",
    "/Fl",
    "/RunLengthDecode",
    "/RL",
}
_PDF_ASCII_WHITESPACE = frozenset((0, 9, 10, 12, 13, 32))
_PDF_LIMIT_LOCK = RLock()


@dataclass(frozen=True, slots=True)
class CitationIngestionLimits:
    """Resource ceilings applied before persistence or unbounded expansion."""

    max_file_bytes: int = 64 * 1024 * 1024
    max_pdf_pages: int = 2_000
    max_pdf_text_chars: int = 20_000_000
    max_pdf_decoded_stream_bytes: int = 8 * 1024 * 1024
    max_epub_members: int = 10_000
    max_epub_expanded_bytes: int = 128 * 1024 * 1024
    max_epub_compression_ratio: float = 200.0

    def __post_init__(self) -> None:
        integer_limits = {
            "max_file_bytes": self.max_file_bytes,
            "max_pdf_pages": self.max_pdf_pages,
            "max_pdf_text_chars": self.max_pdf_text_chars,
            "max_pdf_decoded_stream_bytes": self.max_pdf_decoded_stream_bytes,
            "max_epub_members": self.max_epub_members,
            "max_epub_expanded_bytes": self.max_epub_expanded_bytes,
        }
        for name, value in integer_limits.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        ratio = self.max_epub_compression_ratio
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise ValueError("max_epub_compression_ratio must be a positive number")
        if not math.isfinite(float(ratio)) or ratio <= 0:
            raise ValueError("max_epub_compression_ratio must be a positive finite number")


@dataclass(frozen=True, slots=True)
class _Segment:
    text: str
    metadata: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _Aggregate:
    source: Source
    document: Document
    chunks: tuple[Chunk, ...]


@dataclass(frozen=True, slots=True)
class _RepositorySnapshot:
    source: Source | None
    document: Document | None
    chunks: dict[str, Chunk | None]


@dataclass(slots=True)
class _PdfDecodeBudget:
    limit: int
    decoded_bytes: int = 0
    streams: set[int] = field(default_factory=set)
    arrays: set[int] = field(default_factory=set)
    resources: set[int] = field(default_factory=set)


class CitationIngestionService:
    """Persist an exact and deterministic citation aggregate for one local file."""

    def __init__(
        self,
        repository: KnowledgeRepository,
        limits: CitationIngestionLimits | None = None,
    ) -> None:
        self._repository = repository
        self._limits = limits or CitationIngestionLimits()

    def ingest_document(self, path: str | Path) -> str:
        """Parse and atomically persist a supported citation document."""
        source_path = Path(path).resolve()
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        if not source_path.is_file():
            raise ValueError(f"unsupported citation document path: {source_path}")

        suffix = source_path.suffix.lower()
        if suffix not in _SUPPORTED_SUFFIXES:
            raise ValueError(f"unsupported citation document format: {suffix or '<none>'}")
        file_size = source_path.stat().st_size
        self._check_limit(file_size, self._limits.max_file_bytes, "compressed file bytes")

        payload = self._read_bounded(source_path)
        aggregate = self._build_aggregate(source_path, suffix, payload)
        if self._aggregate_matches(aggregate):
            return aggregate.document.id
        self._commit(aggregate)
        return aggregate.document.id

    def _read_bounded(self, path: Path) -> bytes:
        """Read at most the configured limit plus one overflow sentinel byte."""
        with path.open("rb") as handle:
            payload = handle.read(self._limits.max_file_bytes + 1)
        self._check_limit(
            len(payload),
            self._limits.max_file_bytes,
            "compressed file bytes",
        )
        return payload

    def _existing_timestamp(
        self,
        *,
        source_id: str,
        document_id: str,
        reference: str,
        content_hash: str,
        kind: str,
    ) -> str:
        """Require at least one agreeing complete witness before repairing records."""
        source = self._repository.sources.get(source_id)
        document = self._repository.documents.get(document_id)
        chunks = sorted(
            self._repository.chunks.by_document(document_id),
            key=lambda chunk: (chunk.index, chunk.id),
        )
        surviving_records = int(source is not None) + int(document is not None) + len(chunks)
        timestamps: set[str] = set()

        if source is not None:
            state = self._source_witness_state(source, reference, content_hash, kind)
            if state == "conflict":
                self._identity_collision(f"source identity witness disagrees: {source.id}")
            if state == "match":
                timestamps.add(source.ingested_at)
        if document is not None:
            state = self._document_witness_state(
                document,
                source_id,
                reference,
                content_hash,
                kind,
            )
            if state == "conflict":
                self._identity_collision(f"document identity witness disagrees: {document.id}")
            if state == "match":
                timestamps.add(document.ingested_at)
        for chunk in chunks:
            state = self._chunk_witness_state(
                chunk,
                source_id=source_id,
                document_id=document_id,
                reference=reference,
                content_hash=content_hash,
                kind=kind,
            )
            if state == "conflict":
                self._identity_collision(f"chunk identity witness disagrees: {chunk.id}")
            if state == "match":
                timestamps.add(str(chunk.metadata["ingested_at"]))

        if not surviving_records:
            return now_iso()
        if not timestamps:
            self._identity_collision("no surviving complete identity witness agrees")
        if len(timestamps) != 1:
            self._identity_collision("aggregate timestamp witnesses disagree")
        return timestamps.pop()

    @staticmethod
    def _source_witness_state(
        source: Source,
        reference: str,
        content_hash: str,
        kind: str,
    ) -> str:
        metadata = source.metadata
        required = {"source_reference", "content_hash", "ingestion_version"}
        if not required.issubset(metadata):
            return "incomplete"
        matches = (
            source.reference == reference
            and source.kind == kind
            and metadata.get("source_reference") == reference
            and metadata.get("content_hash") == content_hash
            and metadata.get("ingestion_version") == _INGESTION_VERSION
        )
        return "match" if matches else "conflict"

    @staticmethod
    def _document_witness_state(
        document: Document,
        source_id: str,
        reference: str,
        content_hash: str,
        kind: str,
    ) -> str:
        metadata = document.metadata
        required = {"source_reference", "content_hash", "ingestion_version"}
        if not required.issubset(metadata):
            return "incomplete"
        matches = (
            document.source_id == source_id
            and document.content_type == kind
            and metadata.get("source_reference") == reference
            and metadata.get("content_hash") == content_hash
            and metadata.get("ingestion_version") == _INGESTION_VERSION
        )
        return "match" if matches else "conflict"

    @staticmethod
    def _chunk_witness_state(
        chunk: Chunk,
        *,
        source_id: str,
        document_id: str,
        reference: str,
        content_hash: str,
        kind: str,
    ) -> str:
        metadata = chunk.metadata
        required = {
            "source_id",
            "source_reference",
            "content_hash",
            "ingestion_version",
            "source_kind",
            "ingested_at",
        }
        if not required.issubset(metadata):
            return "incomplete"
        timestamp = metadata.get("ingested_at")
        matches = (
            chunk.document_id == document_id
            and metadata.get("source_id") == source_id
            and metadata.get("source_reference") == reference
            and metadata.get("content_hash") == content_hash
            and metadata.get("ingestion_version") == _INGESTION_VERSION
            and metadata.get("source_kind") == kind
            and isinstance(timestamp, str)
            and bool(timestamp)
        )
        return "match" if matches else "conflict"

    @staticmethod
    def _identity_collision(detail: str) -> None:
        raise ValueError(f"citation identity collision; existing aggregate preserved ({detail})")

    def _build_aggregate(self, path: Path, suffix: str, payload: bytes) -> _Aggregate:
        content_hash = hashlib.sha256(payload).hexdigest()
        reference = str(path)
        kind = self._kind_for_suffix(suffix)
        source_id = stable_id("src", _INGESTION_VERSION, reference, content_hash)
        document_id = stable_id("doc", _INGESTION_VERSION, reference, content_hash, kind)
        aggregate_timestamp = self._existing_timestamp(
            source_id=source_id,
            document_id=document_id,
            reference=reference,
            content_hash=content_hash,
            kind=kind,
        )

        segments = self._parse(path, suffix, payload)
        if not segments:
            raise ValueError(f"empty extraction from citation document: {path}")
        self._check_limit(len(segments), _MAX_FINAL_SEGMENTS, "final segment count")
        separator = "" if suffix in {".txt", ".md"} else "\n"
        text, segments = self._join_segments(segments, separator=separator)
        if not text or not text.strip():
            raise ValueError(f"empty extraction from citation document: {path}")
        if suffix == ".pdf":
            self._check_limit(
                len(text),
                self._limits.max_pdf_text_chars,
                "PDF canonical text characters",
            )
        self._check_limit(len(text), _MAX_FINAL_TEXT_CHARS, "final text characters")

        common_metadata: dict[str, Any] = {
            "content_hash": content_hash,
            "ingestion_version": _INGESTION_VERSION,
            "source_reference": reference,
        }
        document_metadata = {**common_metadata, "segment_count": len(segments)}
        if kind == SourceKind.EPUB:
            document_metadata[EPUB_PARSE_LIMITS_KEY] = epub_parse_limits_metadata(
                self._epub_parse_limits()
            )
        source = Source(
            title=path.stem,
            kind=kind,
            reference=reference,
            metadata=dict(common_metadata),
            id=source_id,
            ingested_at=aggregate_timestamp,
        )
        document = Document(
            source_id=source_id,
            title=path.stem,
            content_type=kind,
            text=text,
            raw=encode_epub_raw(payload) if kind == SourceKind.EPUB else text,
            metadata=document_metadata,
            id=document_id,
            ingested_at=aggregate_timestamp,
        )
        chunks = tuple(
            self._chunks(
                document_id=document_id,
                source_id=source_id,
                reference=reference,
                content_hash=content_hash,
                kind=kind,
                ingested_at=aggregate_timestamp,
                text=text,
                segments=segments,
            )
        )
        return _Aggregate(source, document, chunks)

    @staticmethod
    def _kind_for_suffix(suffix: str) -> str:
        return {
            ".txt": SourceKind.TEXT,
            ".md": SourceKind.MARKDOWN,
            ".pdf": SourceKind.PDF,
            ".epub": SourceKind.EPUB,
        }[suffix]

    def _parse(self, path: Path, suffix: str, payload: bytes) -> list[_Segment]:
        if suffix == ".txt":
            text = self._decode_utf8(payload, "text")
            return [_Segment(text, {})] if text.strip() else []
        if suffix == ".md":
            text = self._decode_utf8(payload, "Markdown")
            return self._markdown_segments(path, text) if text.strip() else []
        if suffix == ".pdf":
            return self._pdf_segments(payload)
        return self._epub_segments(payload)

    @staticmethod
    def _decode_utf8(payload: bytes, label: str) -> str:
        try:
            return payload.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError(f"invalid UTF-8 {label} document") from exc

    @staticmethod
    def _markdown_segments(path: Path, text: str) -> list[_Segment]:
        headings: list[tuple[int, str]] = []
        offset = 0
        for line in text.splitlines(keepends=True):
            logical_line = line.rstrip("\r\n")
            match = _MARKDOWN_HEADING.fullmatch(logical_line)
            if match is not None:
                headings.append((offset, match.group(2).strip() or path.stem))
            offset += len(line)

        if not headings:
            return [_Segment(text, {"section": path.stem})]
        segments: list[_Segment] = []
        if headings[0][0] > 0:
            segments.append(_Segment(text[: headings[0][0]], {"section": path.stem}))
        for index, (start, heading) in enumerate(headings):
            end = headings[index + 1][0] if index + 1 < len(headings) else len(text)
            if end > start:
                segments.append(_Segment(text[start:end], {"section": heading}))
        return segments

    def _pdf_segments(self, payload: bytes) -> list[_Segment]:
        try:
            from pypdf import PdfReader
            from pypdf import filters as pdf_filters
            from pypdf import generic as pdf_generic
            from pypdf.errors import LimitReachedError
        except ImportError as exc:  # pragma: no cover - installation-profile dependent
            raise RuntimeError("PDF ingestion requires the optional 'pypdf' dependency") from exc

        try:
            with self._bounded_pypdf_decoding(
                pdf_filters,
                self._limits.max_pdf_decoded_stream_bytes,
            ):
                reader = PdfReader(BytesIO(payload), strict=False)
                if reader.is_encrypted:
                    raise ValueError("encrypted PDF documents are unsupported")
                page_count = len(reader.pages)
                self._check_limit(page_count, self._limits.max_pdf_pages, "PDF page count")
                segments: list[_Segment] = []
                extracted_chars = 0
                canonical_chars = 0
                for page_index, page in enumerate(reader.pages):
                    self._preflight_pdf_page(
                        page,
                        pdf_filters,
                        pdf_generic,
                        LimitReachedError,
                    )
                    visitor_chars = 0
                    separator_chars = 1 if segments else 0
                    used_before_page = max(extracted_chars, canonical_chars)

                    def visitor_text(
                        fragment: str,
                        _cm: Any,
                        _tm: Any,
                        _font: Any,
                        _font_size: Any,
                        _used_before: int = used_before_page,
                        _separator: int = separator_chars,
                    ) -> None:
                        nonlocal visitor_chars
                        visitor_chars += len(fragment)
                        self._check_limit(
                            _used_before + _separator + visitor_chars,
                            self._limits.max_pdf_text_chars,
                            "PDF extracted characters",
                        )

                    try:
                        with self._bounded_pypdf_decoding(
                            pdf_filters,
                            self._limits.max_pdf_decoded_stream_bytes,
                        ):
                            page_text = page.extract_text(visitor_text=visitor_text) or ""
                    except LimitReachedError as exc:
                        raise ValueError("PDF decoded content stream byte limit exceeded") from exc
                    except NotImplementedError as exc:
                        raise ValueError("unsupported PDF content stream filter") from exc
                    extracted_chars += max(visitor_chars, len(page_text))
                    self._check_limit(
                        extracted_chars,
                        self._limits.max_pdf_text_chars,
                        "PDF extracted characters",
                    )
                    if page_text.strip():
                        canonical_chars += separator_chars + len(page_text)
                        self._check_limit(
                            canonical_chars,
                            self._limits.max_pdf_text_chars,
                            "PDF canonical text characters",
                        )
                        segments.append(
                            _Segment(
                                page_text,
                                {
                                    "page": page_index + 1,
                                    "page_count": page_count,
                                    "page_index": page_index,
                                },
                            )
                        )
                return segments
        except LimitReachedError as exc:
            raise ValueError("PDF decoded structural stream byte limit exceeded") from exc
        except ValueError:
            raise
        except NotImplementedError as exc:
            raise ValueError("unsupported PDF structural stream filter") from exc
        except Exception as exc:
            raise ValueError("invalid or corrupt PDF document") from exc

    def _preflight_pdf_page(
        self,
        page: Any,
        pdf_filters: Any,
        pdf_generic: Any,
        limit_error: type[Exception],
    ) -> None:
        """Bound every page and nested Form content stream before extraction."""
        get_value = getattr(page, "get", None)
        if not callable(get_value):
            # Real pypdf PageObjects implement the public mapping API. This path
            # keeps callback-only reader test doubles usable without weakening
            # production decoding.
            return

        budget = _PdfDecodeBudget(self._limits.max_pdf_decoded_stream_bytes)
        self._preflight_pdf_content_object(
            get_value("/Contents"),
            budget,
            pdf_filters,
            pdf_generic,
            limit_error,
        )
        self._preflight_pdf_resources(
            get_value("/Resources"),
            budget,
            pdf_filters,
            pdf_generic,
            limit_error,
        )

    def _preflight_pdf_content_object(
        self,
        value: Any,
        budget: _PdfDecodeBudget,
        pdf_filters: Any,
        pdf_generic: Any,
        limit_error: type[Exception],
    ) -> None:
        resolved = self._resolve_pdf_object(value)
        if resolved is None or isinstance(resolved, pdf_generic.NullObject):
            return
        if isinstance(resolved, pdf_generic.ArrayObject):
            object_key = id(resolved)
            if object_key in budget.arrays:
                raise ValueError("invalid PDF content stream: cyclic array")
            budget.arrays.add(object_key)
            try:
                for item in resolved:
                    self._preflight_pdf_content_object(
                        item,
                        budget,
                        pdf_filters,
                        pdf_generic,
                        limit_error,
                    )
            finally:
                budget.arrays.remove(object_key)
            return
        if not isinstance(resolved, pdf_generic.StreamObject):
            raise ValueError("invalid PDF content stream object")
        self._preflight_pdf_stream(
            resolved,
            budget,
            pdf_filters,
            pdf_generic,
            limit_error,
        )

    def _preflight_pdf_stream(
        self,
        stream: Any,
        budget: _PdfDecodeBudget,
        pdf_filters: Any,
        pdf_generic: Any,
        limit_error: type[Exception],
    ) -> None:
        stream_key = id(stream)
        if stream_key in budget.streams:
            return
        budget.streams.add(stream_key)
        self._validate_pdf_filters(stream, pdf_generic)
        remaining = budget.limit - budget.decoded_bytes
        try:
            with self._bounded_pypdf_decoding(pdf_filters, max(1, remaining)):
                decoded = stream.get_data()
        except limit_error as exc:
            raise ValueError("PDF decoded content stream byte limit exceeded") from exc
        except NotImplementedError as exc:
            raise ValueError("unsupported PDF content stream filter") from exc
        budget.decoded_bytes += len(decoded)
        self._check_limit(
            budget.decoded_bytes,
            budget.limit,
            "PDF decoded content stream bytes",
        )

    def _preflight_pdf_resources(
        self,
        value: Any,
        budget: _PdfDecodeBudget,
        pdf_filters: Any,
        pdf_generic: Any,
        limit_error: type[Exception],
    ) -> None:
        resources = self._resolve_pdf_object(value)
        if resources is None or isinstance(resources, pdf_generic.NullObject):
            return
        if not isinstance(resources, pdf_generic.DictionaryObject):
            raise ValueError("invalid PDF resource dictionary")
        resource_key = id(resources)
        if resource_key in budget.resources:
            return
        budget.resources.add(resource_key)

        xobjects = self._resolve_pdf_object(resources.get("/XObject"))
        if xobjects is None or isinstance(xobjects, pdf_generic.NullObject):
            return
        if not isinstance(xobjects, pdf_generic.DictionaryObject):
            raise ValueError("invalid PDF XObject resource dictionary")
        for reference in xobjects.values():
            xobject = self._resolve_pdf_object(reference)
            if not isinstance(xobject, pdf_generic.StreamObject):
                raise ValueError("invalid PDF XObject stream")
            subtype = self._resolve_pdf_object(xobject.get("/Subtype"))
            if str(subtype) != "/Form":
                continue
            self._preflight_pdf_stream(
                xobject,
                budget,
                pdf_filters,
                pdf_generic,
                limit_error,
            )
            self._preflight_pdf_resources(
                xobject.get("/Resources"),
                budget,
                pdf_filters,
                pdf_generic,
                limit_error,
            )

    @staticmethod
    def _resolve_pdf_object(value: Any) -> Any:
        resolver = getattr(value, "get_object", None)
        return resolver() if callable(resolver) else value

    def _validate_pdf_filters(self, stream: Any, pdf_generic: Any) -> None:
        filters = self._resolve_pdf_object(stream.get("/Filter"))
        if filters is None or isinstance(filters, pdf_generic.NullObject):
            return
        if isinstance(filters, pdf_generic.ArrayObject):
            filter_values = filters
        else:
            filter_values = (filters,)
        for filter_value in filter_values:
            name = str(self._resolve_pdf_object(filter_value))
            if name not in _PDF_CONTENT_FILTERS:
                raise ValueError(f"unsupported PDF content stream filter: {name}")

    @staticmethod
    @contextmanager
    def _bounded_pypdf_decoding(pdf_filters: Any, byte_limit: int) -> Iterator[None]:
        """Temporarily apply pypdf's public decompressor output ceilings."""
        from pypdf.errors import LimitReachedError

        with _PDF_LIMIT_LOCK:
            missing = [name for name in _PDF_FILTER_OUTPUT_LIMITS if not hasattr(pdf_filters, name)]
            if missing:
                raise ValueError(
                    "installed pypdf cannot safely bound PDF content stream decoding; "
                    f"missing public limit {missing[0]}"
                )
            originals = {name: getattr(pdf_filters, name) for name in _PDF_FILTER_OUTPUT_LIMITS}
            decoder_classes = {
                "ASCIIHexDecode": getattr(pdf_filters, "ASCIIHexDecode", None),
                "ASCII85Decode": getattr(pdf_filters, "ASCII85Decode", None),
            }
            if any(decoder is None for decoder in decoder_classes.values()):
                raise ValueError("installed pypdf cannot safely bound ASCII PDF stream decoding")
            original_decoders = {name: decoder.decode for name, decoder in decoder_classes.items()}

            def bounded_ascii_hex(
                data: bytes | str,
                decode_parms: Any = None,
                **kwargs: Any,
            ) -> bytes:
                CitationIngestionService._check_ascii_hex_decoded_size(
                    data,
                    byte_limit,
                    LimitReachedError,
                )
                return original_decoders["ASCIIHexDecode"](
                    data,
                    decode_parms,
                    **kwargs,
                )

            def bounded_ascii85(
                data: bytes | str,
                decode_parms: Any = None,
                **kwargs: Any,
            ) -> bytes:
                CitationIngestionService._check_ascii85_decoded_size(
                    data,
                    byte_limit,
                    LimitReachedError,
                )
                return original_decoders["ASCII85Decode"](
                    data,
                    decode_parms,
                    **kwargs,
                )

            try:
                for name, original in originals.items():
                    bounded = byte_limit if original == 0 else min(original, byte_limit)
                    setattr(pdf_filters, name, bounded)
                decoder_classes["ASCIIHexDecode"].decode = staticmethod(bounded_ascii_hex)
                decoder_classes["ASCII85Decode"].decode = staticmethod(bounded_ascii85)
                yield
            finally:
                for name, decoder in decoder_classes.items():
                    decoder.decode = staticmethod(original_decoders[name])
                for name, original in originals.items():
                    setattr(pdf_filters, name, original)

    @staticmethod
    def _pdf_ascii_values(data: bytes | str) -> Iterator[int]:
        if isinstance(data, str):
            yield from map(ord, data)
            return
        yield from data

    @staticmethod
    def _check_ascii_hex_decoded_size(
        data: bytes | str,
        byte_limit: int,
        limit_error: type[Exception],
    ) -> None:
        digits = 0
        for value in CitationIngestionService._pdf_ascii_values(data):
            if value == ord(">"):
                break
            if value not in _PDF_ASCII_WHITESPACE:
                digits += 1
                if (digits + 1) // 2 > byte_limit:
                    raise limit_error("PDF ASCIIHex decoded stream byte limit exceeded")

    @staticmethod
    def _check_ascii85_decoded_size(
        data: bytes | str,
        byte_limit: int,
        limit_error: type[Exception],
    ) -> None:
        values = (
            value
            for value in CitationIngestionService._pdf_ascii_values(data)
            if value not in _PDF_ASCII_WHITESPACE
        )
        first = next(values, None)
        second = next(values, None)
        pending = [value for value in (first, second) if value is not None]
        if pending[:2] == [ord("<"), ord("~")]:
            pending.clear()

        decoded_bytes = 0
        group_size = 0

        def consume(value: int) -> None:
            nonlocal decoded_bytes, group_size
            if value == ord("z"):
                decoded_bytes += 4
            else:
                group_size += 1
                if group_size == 5:
                    decoded_bytes += 4
                    group_size = 0
            CitationIngestionService._raise_ascii85_if_over_limit(
                decoded_bytes,
                group_size,
                byte_limit,
                limit_error,
            )

        iterator = iter(chain(pending, values))
        for value in iterator:
            if value != ord("~"):
                consume(value)
                continue
            following = next(iterator, None)
            if following == ord(">"):
                break
            consume(value)
            if following is not None:
                consume(following)

    @staticmethod
    def _raise_ascii85_if_over_limit(
        decoded_bytes: int,
        group_size: int,
        byte_limit: int,
        limit_error: type[Exception],
    ) -> None:
        partial_bytes = max(0, group_size - 1)
        if decoded_bytes + partial_bytes > byte_limit:
            raise limit_error("PDF ASCII85 decoded stream byte limit exceeded")

    def _epub_segments(self, payload: bytes) -> list[_Segment]:
        parsed = parse_epub(
            payload,
            limits=self._epub_parse_limits(),
        )
        return [_Segment(segment.text, segment.metadata()) for segment in parsed]

    def _epub_parse_limits(self) -> EpubParseLimits:
        return EpubParseLimits(
            max_file_bytes=self._limits.max_file_bytes,
            max_members=self._limits.max_epub_members,
            max_expanded_bytes=self._limits.max_epub_expanded_bytes,
            max_compression_ratio=self._limits.max_epub_compression_ratio,
            max_segments=EPUB_MAX_SEGMENTS,
            max_text_chars=EPUB_MAX_TEXT_CHARS,
        )

    @staticmethod
    def _join_segments(segments: list[_Segment], *, separator: str) -> tuple[str, list[_Segment]]:
        """Join extracted units while assigning separators to the preceding unit."""
        joined: list[_Segment] = []
        for index, segment in enumerate(segments):
            suffix = separator if index + 1 < len(segments) else ""
            joined.append(_Segment(segment.text + suffix, dict(segment.metadata)))
        return "".join(segment.text for segment in joined), joined

    @staticmethod
    def _chunks(
        *,
        document_id: str,
        source_id: str,
        reference: str,
        content_hash: str,
        kind: str,
        ingested_at: str,
        text: str,
        segments: list[_Segment],
    ) -> list[Chunk]:
        chunks: list[Chunk] = []
        cursor = 0
        break_ends = [match.end() for match in _LINE_BREAK.finditer(text)]
        for index, segment in enumerate(segments):
            end = cursor + len(segment.text)
            if end <= cursor:
                continue
            line_start = bisect_right(break_ends, cursor) + 1
            metadata = {
                "source_id": source_id,
                "source_reference": reference,
                "content_hash": content_hash,
                "ingestion_version": _INGESTION_VERSION,
                "source_kind": kind,
                "ingested_at": ingested_at,
                "char_start": cursor,
                "char_end": end,
                "line_start": line_start,
                "line_end": max(line_start, bisect_left(break_ends, end) + 1),
                **segment.metadata,
            }
            if kind == SourceKind.EPUB:
                chunk_id = stable_id("chunk", document_id, index)
                metadata["locator_hash"] = epub_locator_hash(
                    content_hash=content_hash,
                    document_id=document_id,
                    chunk_id=chunk_id,
                    segment_index=index,
                    chapter=metadata["chapter"],
                    chapter_id=metadata["chapter_id"],
                    chapter_path=metadata["chapter_path"],
                    section=metadata["section"],
                    section_derivation=metadata["section_derivation"],
                    section_line=metadata.get("section_line"),
                )
            chunks.append(
                Chunk(
                    document_id=document_id,
                    index=index,
                    text=text[cursor:end],
                    span=Span(cursor, end),
                    metadata=metadata,
                )
            )
            cursor = end
        if cursor != len(text):
            raise ValueError("citation segments do not exactly cover the document")
        return chunks

    def _aggregate_matches(self, desired: _Aggregate) -> bool:
        source = self._repository.sources.get(desired.source.id)
        document = self._repository.documents.get(desired.document.id)
        chunks = sorted(
            self._repository.chunks.by_document(desired.document.id),
            key=lambda chunk: (chunk.index, chunk.id),
        )
        return (
            self._source_matches(source, desired.source)
            and self._document_matches(document, desired.document)
            and len(chunks) == len(desired.chunks)
            and all(
                self._chunk_matches(actual, expected)
                for actual, expected in zip(chunks, desired.chunks, strict=True)
            )
        )

    @staticmethod
    def _source_matches(actual: Source | None, expected: Source) -> bool:
        return actual is not None and (
            actual.id,
            actual.title,
            actual.kind,
            actual.reference,
            actual.metadata,
            actual.ingested_at,
        ) == (
            expected.id,
            expected.title,
            expected.kind,
            expected.reference,
            expected.metadata,
            expected.ingested_at,
        )

    @staticmethod
    def _document_matches(actual: Document | None, expected: Document) -> bool:
        return actual is not None and (
            actual.id,
            actual.source_id,
            actual.title,
            actual.content_type,
            actual.text,
            actual.raw,
            actual.metadata,
            actual.ingested_at,
        ) == (
            expected.id,
            expected.source_id,
            expected.title,
            expected.content_type,
            expected.text,
            expected.raw,
            expected.metadata,
            expected.ingested_at,
        )

    @staticmethod
    def _chunk_matches(actual: Chunk, expected: Chunk) -> bool:
        return (
            actual.id,
            actual.document_id,
            actual.index,
            actual.text,
            actual.span,
            actual.metadata,
        ) == (
            expected.id,
            expected.document_id,
            expected.index,
            expected.text,
            expected.span,
            expected.metadata,
        )

    def _commit(self, desired: _Aggregate) -> None:
        existing_document_chunks = self._repository.chunks.by_document(desired.document.id)
        affected_chunk_ids = {chunk.id for chunk in existing_document_chunks}
        affected_chunk_ids.update(chunk.id for chunk in desired.chunks)
        snapshot = _RepositorySnapshot(
            source=self._repository.sources.get(desired.source.id),
            document=self._repository.documents.get(desired.document.id),
            chunks={
                chunk_id: self._repository.chunks.get(chunk_id) for chunk_id in affected_chunk_ids
            },
        )
        desired_chunk_ids = {chunk.id for chunk in desired.chunks}
        try:
            if not self._source_matches(snapshot.source, desired.source):
                self._repository.sources.save(desired.source)
            if not self._document_matches(snapshot.document, desired.document):
                self._repository.documents.save(desired.document)
            for chunk in desired.chunks:
                actual = snapshot.chunks.get(chunk.id)
                if actual is None or not self._chunk_matches(actual, chunk):
                    self._repository.chunks.save(chunk)
            for chunk_id in sorted(affected_chunk_ids - desired_chunk_ids):
                self._repository.chunks.delete(chunk_id)
        except Exception as commit_error:
            rollback_errors = self._restore(desired, snapshot)
            if rollback_errors:
                raise RuntimeError(
                    "citation ingestion commit failed and rollback was incomplete: "
                    + "; ".join(rollback_errors)
                ) from commit_error
            raise

    def _restore(self, desired: _Aggregate, snapshot: _RepositorySnapshot) -> list[str]:
        errors: list[str] = []
        for chunk_id, previous in snapshot.chunks.items():
            try:
                if previous is None:
                    self._repository.chunks.delete(chunk_id)
                else:
                    self._repository.chunks.save(previous)
            except Exception as exc:  # pragma: no cover - pathological store fails twice
                errors.append(f"chunk {chunk_id}: {exc}")
        for repository, item_id, previous, label in (
            (self._repository.documents, desired.document.id, snapshot.document, "document"),
            (self._repository.sources, desired.source.id, snapshot.source, "source"),
        ):
            try:
                if previous is None:
                    repository.delete(item_id)
                else:
                    repository.save(previous)
            except Exception as exc:  # pragma: no cover - pathological store fails twice
                errors.append(f"{label} {item_id}: {exc}")
        return errors

    @staticmethod
    def _check_limit(value: int, limit: int, label: str) -> None:
        if value > limit:
            raise ValueError(f"{label} limit exceeded: {value} > {limit}")
