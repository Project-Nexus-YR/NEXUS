"""Source adapters.

Each adapter converts a raw source payload into one or more
:class:`RawDocument` records, retaining source metadata on every
artifact. Additional source types are added by implementing the
:class:`SourceAdapter` protocol.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from ..domain.source import Source
from .normalization import normalize_text

__all__ = [
    "RawDocument",
    "SourceAdapter",
    "TextAdapter",
    "MarkdownAdapter",
    "JsonAdapter",
    "RepositoryAdapter",
]

_SUPPORTED_EXTENSIONS = {".txt", ".md", ".json"}


@dataclass(frozen=True, slots=True)
class RawDocument:
    title: str
    content_type: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


class SourceAdapter(Protocol):
    def read(self, source: Source, payload: Any) -> list[RawDocument]: ...


class TextAdapter:
    """Plain-text source: a single normalized document."""

    def read(self, source: Source, payload: Any) -> list[RawDocument]:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)
        return [
            RawDocument(
                title=source.title or "untitled",
                content_type="text",
                text=normalize_text(text),
                metadata={"source_reference": source.reference},
            )
        ]


class MarkdownAdapter:
    """Markdown source: split into documents at top-level headings."""

    def read(self, source: Source, payload: Any) -> list[RawDocument]:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)
        sections = self._split_headings(text)
        if not sections:
            return [
                RawDocument(
                    title=source.title,
                    content_type="markdown",
                    text=normalize_text(text),
                    metadata={"source_reference": source.reference},
                )
            ]
        return [
            RawDocument(
                title=heading or f"{source.title} #{i + 1}",
                content_type="markdown",
                text=normalize_text(body),
                metadata={"source_reference": source.reference, "heading": heading},
            )
            for i, (heading, body) in enumerate(sections)
        ]

    def _split_headings(self, text: str) -> list[tuple[str, str]]:
        sections: list[tuple[str, str]] = []
        current_heading = ""
        current_body: list[str] = []
        for line in text.splitlines():
            if line.startswith("# "):
                if current_body or current_heading:
                    sections.append((current_heading, "\n".join(current_body)))
                current_heading = line[2:].strip()
                current_body = []
            else:
                current_body.append(line)
        if current_body or current_heading:
            sections.append((current_heading, "\n".join(current_body)))
        return sections


class JsonAdapter:
    """JSON source: flatten to text, preserve structure in metadata."""

    def read(self, source: Source, payload: Any) -> list[RawDocument]:
        text = payload.decode("utf-8") if isinstance(payload, bytes) else str(payload)
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return TextAdapter().read(source, payload)
        if isinstance(data, dict):
            records: list[tuple[str, dict[str, Any]]] = [(source.title or "document", data)]
        elif isinstance(data, list):
            records = [
                (f"{source.title} #{i + 1}", item)
                for i, item in enumerate(data)
                if isinstance(item, dict)
            ]
        else:
            records = [(source.title or "document", {"value": data})]
        documents: list[RawDocument] = []
        for i, (title, record) in enumerate(records):
            documents.append(
                RawDocument(
                    title=title,
                    content_type="json",
                    text=normalize_text(self._flatten(record)),
                    metadata={
                        "source_reference": source.reference,
                        "record_index": i,
                        "record": record,
                    },
                )
            )
        return documents

    def _flatten(self, record: dict[str, Any], prefix: str = "") -> str:
        parts: list[str] = []
        for key, value in record.items():
            label = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                parts.append(self._flatten(value, label))
            elif isinstance(value, list):
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        parts.append(self._flatten(item, f"{label}[{i}]"))
                    else:
                        parts.append(f"{label}: {item}")
            else:
                parts.append(f"{label}: {value}")
        return "\n".join(parts)


class RepositoryAdapter:
    """Repository/directory source: one document per supported file."""

    def read(self, source: Source, payload: Any) -> list[RawDocument]:
        root = Path(payload)
        if not root.is_dir():
            return TextAdapter().read(source, root.read_text(encoding="utf-8"))
        documents: list[RawDocument] = []
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in _SUPPORTED_EXTENSIONS:
                continue
            try:
                content = path.read_text(encoding="utf-8")
            except (UnicodeDecodeError, OSError):
                continue
            relative = str(path.relative_to(root)).replace("\\", "/")
            adapter: SourceAdapter
            if path.suffix.lower() == ".md":
                adapter = MarkdownAdapter()
            elif path.suffix.lower() == ".json":
                adapter = JsonAdapter()
            else:
                adapter = TextAdapter()
            for doc in adapter.read(source, content):
                doc.metadata.setdefault("path", relative)
                documents.append(doc)
        return documents
