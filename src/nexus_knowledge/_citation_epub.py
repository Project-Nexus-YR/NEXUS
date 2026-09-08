"""Neutral, bounded EPUB provenance parsing for citation ingestion and search."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
import posixpath
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit
from zipfile import ZIP_DEFLATED, ZIP_STORED, BadZipFile, ZipFile, ZipInfo

EPUB_RAW_PREFIX = "citation-epub-raw-v1:"
EPUB_PARSE_LIMITS_KEY = "epub_parse_limits"
EPUB_MAX_SEGMENTS = 10_000
EPUB_MAX_TEXT_CHARS = 20_000_000

_CONTAINER_NAMESPACE = "urn:oasis:names:tc:opendocument:xmlns:container"
_OPF_NAMESPACE = "http://www.idpf.org/2007/opf"
_XHTML_NAMESPACE = "http://www.w3.org/1999/xhtml"
_ROOTFILE_MEDIA_TYPE = "application/oebps-package+xml"
_EPUB_MIMETYPE = b"application/epub+zip"
_MAX_EPUB_MEMBER_BYTES = 16 * 1024 * 1024
_BLOCK_TAGS = {
    "address",
    "article",
    "aside",
    "blockquote",
    "br",
    "dd",
    "div",
    "dl",
    "dt",
    "figcaption",
    "figure",
    "footer",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "header",
    "hr",
    "li",
    "main",
    "nav",
    "ol",
    "p",
    "pre",
    "section",
    "table",
    "tbody",
    "td",
    "tfoot",
    "th",
    "thead",
    "tr",
    "ul",
}
_IGNORED_TAGS = {"script", "style"}


@dataclass(frozen=True, slots=True)
class EpubParseLimits:
    max_file_bytes: int = 64 * 1024 * 1024
    max_members: int = 10_000
    max_expanded_bytes: int = 128 * 1024 * 1024
    max_compression_ratio: float = 200.0
    max_segments: int = EPUB_MAX_SEGMENTS
    max_text_chars: int = EPUB_MAX_TEXT_CHARS

    def __post_init__(self) -> None:
        integer_limits = {
            "max_file_bytes": self.max_file_bytes,
            "max_members": self.max_members,
            "max_expanded_bytes": self.max_expanded_bytes,
            "max_segments": self.max_segments,
            "max_text_chars": self.max_text_chars,
        }
        for name, value in integer_limits.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        ratio = self.max_compression_ratio
        if isinstance(ratio, bool) or not isinstance(ratio, (int, float)):
            raise ValueError("max_compression_ratio must be a positive number")
        if not math.isfinite(float(ratio)) or ratio <= 0:
            raise ValueError("max_compression_ratio must be a positive finite number")


@dataclass(frozen=True, slots=True)
class EpubSegment:
    text: str
    chapter: int
    chapter_id: str
    chapter_path: str
    section: str
    section_derivation: str
    section_line: int | None

    def metadata(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "chapter": self.chapter,
            "chapter_id": self.chapter_id,
            "chapter_path": self.chapter_path,
            "section": self.section,
            "section_derivation": self.section_derivation,
        }
        if self.section_line is not None:
            result["section_line"] = self.section_line
        return result


@dataclass(slots=True)
class _Budget:
    expanded_bytes_read: int = 0
    extracted_chars: int = 0
    spine_items: int = 0


def encode_epub_raw(payload: bytes) -> str:
    return EPUB_RAW_PREFIX + base64.b64encode(payload).decode("ascii")


def epub_parse_limits_metadata(limits: EpubParseLimits) -> dict[str, int | float]:
    return {
        "version": 1,
        "max_file_bytes": limits.max_file_bytes,
        "max_members": limits.max_members,
        "max_expanded_bytes": limits.max_expanded_bytes,
        "max_compression_ratio": limits.max_compression_ratio,
        "max_segments": limits.max_segments,
        "max_text_chars": limits.max_text_chars,
    }


def epub_parse_limits_from_metadata(value: Any) -> EpubParseLimits:
    expected_keys = {
        "version",
        "max_file_bytes",
        "max_members",
        "max_expanded_bytes",
        "max_compression_ratio",
        "max_segments",
        "max_text_chars",
    }
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("invalid persisted EPUB parse limits shape")
    version = value["version"]
    if isinstance(version, bool) or not isinstance(version, int) or version != 1:
        raise ValueError("unsupported persisted EPUB parse limits version")
    limits = EpubParseLimits(
        max_file_bytes=value["max_file_bytes"],
        max_members=value["max_members"],
        max_expanded_bytes=value["max_expanded_bytes"],
        max_compression_ratio=value["max_compression_ratio"],
        max_segments=value["max_segments"],
        max_text_chars=value["max_text_chars"],
    )
    if limits.max_segments != EPUB_MAX_SEGMENTS:
        raise ValueError("persisted EPUB max_segments does not match ingestion policy")
    if limits.max_text_chars != EPUB_MAX_TEXT_CHARS:
        raise ValueError("persisted EPUB max_text_chars does not match ingestion policy")
    return limits


def decode_epub_raw(raw: str, *, max_file_bytes: int) -> bytes:
    if not isinstance(raw, str) or not raw.startswith(EPUB_RAW_PREFIX):
        raise ValueError("invalid EPUB raw provenance envelope")
    encoded = raw.removeprefix(EPUB_RAW_PREFIX)
    max_encoded_chars = 4 * ((max_file_bytes + 2) // 3)
    _check_limit(len(encoded), max_encoded_chars, "EPUB raw provenance base64 characters")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("invalid EPUB raw provenance base64") from exc
    _check_limit(len(payload), max_file_bytes, "EPUB raw provenance bytes")
    if base64.b64encode(payload).decode("ascii") != encoded:
        raise ValueError("non-canonical EPUB raw provenance base64")
    return payload


def epub_locator_hash(
    *,
    content_hash: str,
    document_id: str,
    chunk_id: str,
    segment_index: int,
    chapter: int,
    chapter_id: str,
    chapter_path: str,
    section: str,
    section_derivation: str,
    section_line: int | None,
) -> str:
    witness = {
        "chapter": chapter,
        "chapter_id": chapter_id,
        "chapter_path": chapter_path,
        "chunk_id": chunk_id,
        "content_hash": content_hash,
        "document_id": document_id,
        "section": section,
        "section_derivation": section_derivation,
        "section_line": section_line,
        "segment_index": segment_index,
    }
    canonical = json.dumps(
        witness,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def parse_epub(payload: bytes, *, limits: EpubParseLimits) -> tuple[EpubSegment, ...]:
    _check_limit(len(payload), limits.max_file_bytes, "compressed file bytes")
    try:
        with ZipFile(BytesIO(payload)) as archive:
            budget = _Budget()
            infos = _validate_archive(archive, budget, limits)
            container_bytes = _read_member(
                archive,
                infos,
                "META-INF/container.xml",
                budget,
                limits,
            )
            container = _parse_xml(container_bytes, "EPUB container")
            if container.tag != f"{{{_CONTAINER_NAMESPACE}}}container":
                raise ValueError("invalid EPUB container namespace")
            rootfile = container.find(f".//{{{_CONTAINER_NAMESPACE}}}rootfile")
            if rootfile is None or not rootfile.get("full-path"):
                raise ValueError("invalid EPUB container: missing package path")
            if rootfile.get("media-type") != _ROOTFILE_MEDIA_TYPE:
                raise ValueError("invalid EPUB container rootfile media type")
            package_path = _safe_member("", rootfile.get("full-path", ""))
            package_bytes = _read_member(archive, infos, package_path, budget, limits)
            package = _parse_xml(package_bytes, "EPUB package")
            if package.tag != f"{{{_OPF_NAMESPACE}}}package":
                raise ValueError("invalid EPUB package namespace")
            return _spine_segments(
                archive,
                infos,
                package_path,
                package,
                budget,
                limits,
            )
    except ValueError:
        raise
    except (BadZipFile, KeyError, UnicodeError, OSError) as exc:
        raise ValueError("invalid or corrupt EPUB document") from exc


def _validate_archive(
    archive: ZipFile,
    budget: _Budget,
    limits: EpubParseLimits,
) -> dict[str, ZipInfo]:
    info_list = archive.infolist()
    _check_limit(len(info_list), limits.max_members, "EPUB member count")
    if not info_list:
        raise ValueError("invalid EPUB archive: no members")
    first = info_list[0]
    if first.filename != "mimetype" or first.compress_type != ZIP_STORED:
        raise ValueError("invalid EPUB archive: first member must be stored mimetype")
    names = [info.filename for info in info_list]
    if len(set(names)) != len(names):
        raise ValueError("invalid EPUB archive: duplicate member names")

    total_expanded = 0
    per_member_limit = min(_MAX_EPUB_MEMBER_BYTES, limits.max_expanded_bytes)
    for info in info_list:
        if info.flag_bits & 0x1:
            raise ValueError(f"invalid EPUB archive: encrypted member {info.filename}")
        if info.compress_type not in {ZIP_STORED, ZIP_DEFLATED}:
            raise ValueError(
                f"invalid EPUB archive: unsupported compression method for {info.filename}"
            )
        _check_limit(
            info.file_size,
            per_member_limit,
            f"EPUB expanded member bytes ({info.filename})",
        )
        total_expanded += info.file_size
        _check_limit(total_expanded, limits.max_expanded_bytes, "EPUB total expanded bytes")
        ratio = _compression_ratio(info)
        if ratio > limits.max_compression_ratio:
            raise ValueError(
                f"EPUB compression ratio limit exceeded for {info.filename}: "
                f"{ratio:.2f} > {limits.max_compression_ratio:.2f}"
            )
    infos = {info.filename: info for info in info_list}
    mimetype = _read_member(archive, infos, "mimetype", budget, limits)
    if mimetype != _EPUB_MIMETYPE:
        raise ValueError("invalid EPUB archive: incorrect mimetype content")
    return infos


def _compression_ratio(info: ZipInfo) -> float:
    if info.file_size == 0:
        return 0.0
    if info.compress_size == 0:
        return math.inf
    return info.file_size / info.compress_size


def _read_member(
    archive: ZipFile,
    infos: dict[str, ZipInfo],
    name: str,
    budget: _Budget,
    limits: EpubParseLimits,
) -> bytes:
    info = infos.get(name)
    if info is None:
        raise ValueError(f"invalid EPUB archive: missing member {name}")
    projected_bytes = budget.expanded_bytes_read + info.file_size
    _check_limit(
        projected_bytes,
        limits.max_expanded_bytes,
        "EPUB cumulative expanded bytes read",
    )
    try:
        data = archive.read(info)
    except (RuntimeError, NotImplementedError) as exc:
        raise ValueError(f"invalid or unsupported EPUB member: {name}") from exc
    if len(data) != info.file_size:
        raise ValueError(f"invalid EPUB archive: truncated member {name}")
    budget.expanded_bytes_read = projected_bytes
    return data


def _parse_xml(payload: bytes, label: str) -> Any:
    try:
        from defusedxml import ElementTree as DefusedElementTree
        from defusedxml.common import DefusedXmlException
    except ImportError as exc:  # pragma: no cover - installation-profile dependent
        raise RuntimeError(
            "EPUB ingestion and verification require the optional 'defusedxml' dependency"
        ) from exc
    try:
        return DefusedElementTree.fromstring(
            _xml_input(payload),
            forbid_dtd=True,
            forbid_entities=True,
            forbid_external=True,
        )
    except DefusedXmlException as exc:
        raise ValueError(f"invalid {label}: DTD and entity declarations are forbidden") from exc
    except DefusedElementTree.ParseError as exc:
        raise ValueError(f"invalid {label} XML") from exc


def _xml_input(payload: bytes) -> bytes | str:
    signatures = (
        (b"\xff\xfe\x00\x00", "utf-32"),
        (b"\x00\x00\xfe\xff", "utf-32"),
        (b"\x3c\x00\x00\x00", "utf-32-le"),
        (b"\x00\x00\x00\x3c", "utf-32-be"),
        (b"\xff\xfe", "utf-16"),
        (b"\xfe\xff", "utf-16"),
        (b"\x3c\x00\x3f\x00", "utf-16-le"),
        (b"\x00\x3c\x00\x3f", "utf-16-be"),
    )
    for signature, encoding in signatures:
        if payload.startswith(signature):
            return payload.decode(encoding)
    return payload


def _spine_segments(
    archive: ZipFile,
    infos: dict[str, ZipInfo],
    package_path: str,
    package: Any,
    budget: _Budget,
    limits: EpubParseLimits,
) -> tuple[EpubSegment, ...]:
    manifest: dict[str, Any] = {}
    manifest_path = f".//{{{_OPF_NAMESPACE}}}manifest/{{{_OPF_NAMESPACE}}}item"
    for item in package.findall(manifest_path):
        item_id = item.get("id")
        if not item_id:
            raise ValueError("invalid EPUB package: manifest item without id")
        if item_id in manifest:
            raise ValueError("invalid EPUB package: duplicate manifest id")
        manifest[item_id] = item

    segments: list[EpubSegment] = []
    spine_path = f".//{{{_OPF_NAMESPACE}}}spine/{{{_OPF_NAMESPACE}}}itemref"
    for itemref in package.findall(spine_path):
        budget.spine_items += 1
        _check_limit(budget.spine_items, limits.max_members, "EPUB spine item count")
        if itemref.get("linear", "yes").lower() == "no":
            continue
        item_id = itemref.get("idref", "")
        item = manifest.get(item_id)
        if item is None:
            raise ValueError(f"invalid EPUB spine reference: {item_id or '<empty>'}")
        if item.get("media-type") != "application/xhtml+xml":
            continue
        if "nav" in item.get("properties", "").split():
            continue
        member = _safe_member(package_path, item.get("href", ""))
        chapter_bytes = _read_member(archive, infos, member, budget, limits)
        chapter_root = _parse_xml(chapter_bytes, f"EPUB chapter {member}")
        if chapter_root.tag != f"{{{_XHTML_NAMESPACE}}}html":
            raise ValueError(f"invalid EPUB XHTML namespace: {member}")
        body = chapter_root.find(f"{{{_XHTML_NAMESPACE}}}body")
        if body is None:
            raise ValueError(f"invalid EPUB XHTML body: {member}")
        heading_element = _heading_element(body)
        heading = (
            " ".join("".join(heading_element.itertext()).split())
            if heading_element is not None
            else ""
        )
        chapter_text, section_line = _xhtml_text_and_heading_line(
            body,
            heading_element if heading else None,
        )
        if not chapter_text.strip():
            continue
        separator_chars = 1 if segments else 0
        projected_chars = budget.extracted_chars + separator_chars + len(chapter_text)
        _check_limit(
            projected_chars,
            limits.max_text_chars,
            "EPUB cumulative extracted characters",
        )
        _check_limit(len(segments) + 1, limits.max_segments, "final segment count")
        if heading and section_line is not None:
            section = heading
            derivation = "heading"
        else:
            section = PurePosixPath(member).stem
            derivation = "chapter_path_stem"
        segments.append(
            EpubSegment(
                text=chapter_text,
                chapter=len(segments) + 1,
                chapter_id=item_id,
                chapter_path=member,
                section=section,
                section_derivation=derivation,
                section_line=section_line,
            )
        )
        budget.extracted_chars = projected_chars
    return tuple(segments)


def _xhtml_text_and_heading_line(
    body: Any,
    heading_element: Any | None,
) -> tuple[str, int | None]:
    parts: list[str] = []
    heading_part_index: int | None = None

    def add_break() -> None:
        if parts and not parts[-1].endswith("\n"):
            parts.append("\n")

    def walk(element: Any) -> None:
        nonlocal heading_part_index
        namespace, tag = _qualified_name(element.tag)
        if namespace != _XHTML_NAMESPACE:
            raise ValueError("invalid EPUB XHTML child namespace")
        if tag in _IGNORED_TAGS:
            return
        if tag in _BLOCK_TAGS:
            add_break()
        if element is heading_element:
            heading_part_index = len(parts)
        if element.text:
            parts.append(element.text)
        for child in element:
            walk(child)
            if child.tail:
                parts.append(child.tail)
        if tag in _BLOCK_TAGS:
            add_break()

    if body.text:
        parts.append(body.text)
    for child in body:
        walk(child)
        if child.tail:
            parts.append(child.tail)
    canonical_lines = [
        normalized for line in "".join(parts).splitlines() if (normalized := " ".join(line.split()))
    ]
    text = "\n".join(canonical_lines)
    if heading_part_index is None:
        return text, None

    prefix = "".join(parts[:heading_part_index])
    preceding_lines = sum(bool(" ".join(line.split())) for line in prefix.splitlines())
    section_line = preceding_lines + 1
    heading = " ".join("".join(heading_element.itertext()).split())
    if section_line > len(canonical_lines) or canonical_lines[section_line - 1] != heading:
        # A heading may span canonical lines. Keep its text and use the explicit
        # chapter-path section fallback instead of inventing a one-line locator.
        return text, None
    return text, section_line


def _heading_element(body: Any) -> Any | None:
    heading = None
    for element in body.iter():
        namespace, tag = _qualified_name(element.tag)
        if namespace != _XHTML_NAMESPACE:
            raise ValueError("invalid EPUB XHTML child namespace")
        if heading is None and tag in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            heading = element
    return heading


def _qualified_name(tag: str) -> tuple[str, str]:
    if not tag.startswith("{") or "}" not in tag:
        return "", tag
    namespace, local = tag[1:].split("}", 1)
    return namespace, local


def _safe_member(base: str, href: str) -> str:
    path = unquote(urlsplit(href).path)
    if not path or path.startswith("/") or "\\" in path or "\x00" in path:
        raise ValueError("invalid EPUB package member path")
    parent = posixpath.dirname(base)
    normalized = posixpath.normpath(posixpath.join(parent, path))
    if normalized == ".." or normalized.startswith("../"):
        raise ValueError("invalid EPUB package member path")
    return normalized


def _check_limit(value: int, limit: int, label: str) -> None:
    if value > limit:
        raise ValueError(f"{label} limit exceeded: {value} > {limit}")
