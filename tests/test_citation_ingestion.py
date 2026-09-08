"""Acceptance tests for deterministic, citation-ready local document ingestion."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import struct
import tomllib
import zlib
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import pytest

from nexus_knowledge.domain.claim import Claim, Evidence
from nexus_knowledge.domain.contradiction import Contradiction
from nexus_knowledge.domain.document import Chunk, Document, Span
from nexus_knowledge.domain.entity import Entity, Relation
from nexus_knowledge.domain.hypothesis import Experiment, Hypothesis, Observation, Result
from nexus_knowledge.domain.ids import stable_id
from nexus_knowledge.domain.knowledge_gap import Investigation, KnowledgeGap
from nexus_knowledge.domain.source import Source, SourceKind
from nexus_knowledge.persistence.json_codec import load_snapshot, save_snapshot, to_plain
from nexus_knowledge.persistence.memory import InMemoryKnowledgeRepository

TXT_TEXT = "Alpha  evidence.\r\n\r\nBeta café — 東京.\r\n"
MARKDOWN_TEXT = (
    "# Overview\r\n\r\n"
    "Alpha  evidence stays exact.\r\n\r\n"
    "## Details\r\n\r\n"
    "Unicode café — 東京 remains source-aligned.\r\n"
)
PDF_PAGE_TEXTS = (
    "First PDF page contains alpha evidence.",
    "Second PDF page contains beta citation text.",
)
PDF_SEGMENT_TEXTS = (f"{PDF_PAGE_TEXTS[0]}\n", PDF_PAGE_TEXTS[1])
PDF_CANONICAL_TEXT = "".join(PDF_SEGMENT_TEXTS)
PDF_BYTES = base64.b64decode(
    "JVBERi0xLjMKJeLjz9MKMSAwIG9iago8PAovUHJvZHVjZXIgKHB5cGRmKQo+PgplbmRvYmoK"
    "MiAwIG9iago8PAovVHlwZSAvUGFnZXMKL0NvdW50IDIKL0tpZHMgWyA1IDAgUiA3IDAgUiBd"
    "Cj4+CmVuZG9iagozIDAgb2JqCjw8Ci9UeXBlIC9DYXRhbG9nCi9QYWdlcyAyIDAgUgo+Pgpl"
    "bmRvYmoKNCAwIG9iago8PAovVHlwZSAvRm9udAovU3VidHlwZSAvVHlwZTEKL0Jhc2VGb250"
    "IC9IZWx2ZXRpY2EKPj4KZW5kb2JqCjUgMCBvYmoKPDwKL1R5cGUgL1BhZ2UKL1Jlc291cmNl"
    "cyA8PAovRm9udCA8PAovRjEgNCAwIFIKPj4KPj4KL01lZGlhQm94IFsgMC4wIDAuMCA2MTIg"
    "NzkyIF0KL1BhcmVudCAyIDAgUgovQ29udGVudHMgNiAwIFIKPj4KZW5kb2JqCjYgMCBvYmoK"
    "PDwKL0xlbmd0aCA3MAo+PgpzdHJlYW0KQlQgL0YxIDEyIFRmIDcyIDcyMCBUZCAoRmlyc3Qg"
    "UERGIHBhZ2UgY29udGFpbnMgYWxwaGEgZXZpZGVuY2UuKSBUaiBFVAplbmRzdHJlYW0KZW5k"
    "b2JqCjcgMCBvYmoKPDwKL1R5cGUgL1BhZ2UKL1Jlc291cmNlcyA8PAovRm9udCA8PAovRjEg"
    "NCAwIFIKPj4KPj4KL01lZGlhQm94IFsgMC4wIDAuMCA2MTIgNzkyIF0KL1BhcmVudCAyIDAg"
    "UgovQ29udGVudHMgOCAwIFIKPj4KZW5kb2JqCjggMCBvYmoKPDwKL0xlbmd0aCA3NQo+Pgpz"
    "dHJlYW0KQlQgL0YxIDEyIFRmIDcyIDcyMCBUZCAoU2Vjb25kIFBERiBwYWdlIGNvbnRhaW5z"
    "IGJldGEgY2l0YXRpb24gdGV4dC4pIFRqIEVUCmVuZHN0cmVhbQplbmRvYmoKeHJlZgowIDkK"
    "MDAwMDAwMDAwMCA2NTUzNSBmIAowMDAwMDAwMDE1IDAwMDAwIG4gCjAwMDAwMDAwNTQgMDAw"
    "MDAgbiAKMDAwMDAwMDExOSAwMDAwMCBuIAowMDAwMDAwMTY4IDAwMDAwIG4gCjAwMDAwMDAy"
    "MzggMDAwMDAgbiAKMDAwMDAwMDM3MCAwMDAwMCBuIAowMDAwMDAwNDkwIDAwMDAwIG4gCjAw"
    "MDAwMDA2MjIgMDAwMDAgbiAKdHJhaWxlcgo8PAovU2l6ZSA5Ci9Sb290IDMgMCBSCi9JbmZv"
    "IDEgMCBSCj4+CnN0YXJ0eHJlZgo3NDcKJSVFT0YK"
)
EPUB_CHAPTERS = (
    ("Opening", "The opening chapter contains alpha evidence."),
    ("Findings", "The findings chapter contains beta citation text."),
)
EPUB_SEGMENT_TEXTS = (
    f"{EPUB_CHAPTERS[0][0]}\n{EPUB_CHAPTERS[0][1]}\n",
    f"{EPUB_CHAPTERS[1][0]}\n{EPUB_CHAPTERS[1][1]}",
)
EPUB_CANONICAL_TEXT = "".join(EPUB_SEGMENT_TEXTS)
FORMAT_FIXTURES = ("txt_path", "markdown_path", "pdf_path", "epub_path")
ALL_REPOSITORIES = (
    "sources",
    "documents",
    "chunks",
    "entities",
    "relations",
    "claims",
    "evidence",
    "contradictions",
    "hypotheses",
    "experiments",
    "results",
    "observations",
    "gaps",
    "investigations",
)
UNRELATED_REPOSITORIES = ALL_REPOSITORIES[3:]
PYPDF_DECODE_LIMIT_NAMES = (
    "ZLIB_MAX_OUTPUT_LENGTH",
    "LZW_MAX_OUTPUT_LENGTH",
    "RUN_LENGTH_MAX_OUTPUT_LENGTH",
    "MAX_ARRAY_BASED_STREAM_OUTPUT_LENGTH",
)


def _citation_symbols() -> tuple[type[Any], type[Any], type[Any]]:
    """Load through public package exports so missing exports are acceptance failures."""
    from nexus_knowledge.ingestion import CitationIngestionLimits, CitationIngestionService
    from nexus_knowledge.port import CitationIngestionPort

    return CitationIngestionPort, CitationIngestionLimits, CitationIngestionService


def _new_service(repository: Any, *, limits: Any = None) -> Any:
    _, _, service_type = _citation_symbols()
    if limits is None:
        return service_type(repository=repository)
    return service_type(repository=repository, limits=limits)


def _limits(**overrides: Any) -> Any:
    _, limits_type, _ = _citation_symbols()
    return limits_type(**overrides)


def _stored_records(
    repository: InMemoryKnowledgeRepository, document_id: str
) -> tuple[Any, Any, list[Any]]:
    document = repository.documents.get(document_id)
    assert document is not None
    source = repository.sources.get(document.source_id)
    assert source is not None
    chunks = repository.chunks.by_document(document_id)
    assert chunks
    return source, document, chunks


def _counts(repository: InMemoryKnowledgeRepository) -> tuple[int, int, int]:
    return (
        repository.sources.count(),
        repository.documents.count(),
        repository.chunks.count(),
    )


def _assert_source_lineage(
    repository: InMemoryKnowledgeRepository,
    document_id: str,
    path: Path,
) -> tuple[Any, Any, list[Any]]:
    source, document, chunks = _stored_records(repository, document_id)
    content_hash = hashlib.sha256(path.read_bytes()).hexdigest()

    assert source.reference == str(path.resolve())
    assert source.metadata["content_hash"] == content_hash
    assert document.source_id == source.id
    assert document.metadata["content_hash"] == content_hash
    assert document.metadata["source_reference"] == source.reference

    for chunk in chunks:
        assert chunk.document_id == document.id
        assert chunk.id
        assert chunk.metadata["source_id"] == source.id
        assert chunk.metadata["source_reference"] == source.reference

    return source, document, chunks


def _assert_exact_reconstruction(repository: InMemoryKnowledgeRepository, document_id: str) -> str:
    _, document, chunks = _stored_records(repository, document_id)
    cursor = 0
    texts: list[str] = []

    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    for chunk in chunks:
        assert chunk.span is not None
        assert chunk.span.start == cursor
        assert chunk.span.end > chunk.span.start
        assert chunk.text == document.text[chunk.span.start : chunk.span.end]
        assert chunk.metadata["char_start"] == chunk.span.start
        assert chunk.metadata["char_end"] == chunk.span.end
        texts.append(chunk.text)
        cursor = chunk.span.end

    assert cursor == len(document.text)
    reconstructed = "".join(texts)
    assert reconstructed == document.text
    return reconstructed


def _repository_projection(
    repository: InMemoryKnowledgeRepository, document_id: str
) -> dict[str, Any]:
    source, document, chunks = _stored_records(repository, document_id)
    return {
        "source": to_plain(source),
        "document": to_plain(document),
        "chunks": [to_plain(chunk) for chunk in chunks],
        "chunk_ids": [chunk.id for chunk in chunks],
    }


def _aggregate_projection(
    repository: Any,
    *,
    source_id: str,
    document_id: str,
) -> dict[str, Any]:
    """Capture an aggregate even when one of its records is missing or corrupt."""
    source = repository.sources.get(source_id)
    document = repository.documents.get(document_id)
    return {
        "source": None if source is None else to_plain(source),
        "document": None if document is None else to_plain(document),
        "chunks": [to_plain(chunk) for chunk in repository.chunks.by_document(document_id)],
    }


def _full_repository_projection(
    repository: InMemoryKnowledgeRepository,
    names: tuple[str, ...] = ALL_REPOSITORIES,
) -> dict[str, list[Any]]:
    return {name: [to_plain(item) for item in getattr(repository, name).all()] for name in names}


def _full_repository_projection_bytes(
    repository: InMemoryKnowledgeRepository,
    names: tuple[str, ...] = ALL_REPOSITORIES,
) -> bytes:
    """Serialize a repository projection for exact before/after isolation checks."""
    return json.dumps(
        _full_repository_projection(repository, names),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _seed_full_repository(repository: InMemoryKnowledgeRepository) -> None:
    source = Source(
        title="sentinel source",
        kind=SourceKind.TEXT,
        reference="sentinel://source",
        id="src_sentinel",
        ingested_at="2026-01-01T00:00:00+00:00",
    )
    document = Document(
        source_id=source.id,
        title="sentinel document",
        content_type="text",
        text="sentinel evidence",
        raw="sentinel evidence",
        id="doc_sentinel",
        ingested_at="2026-01-01T00:00:00+00:00",
    )
    chunk = Chunk(
        document_id=document.id,
        index=0,
        text=document.text,
        span=Span(0, len(document.text)),
    )
    entity = Entity(name="Sentinel", id="entity_sentinel")
    relation = Relation(
        subject_id=entity.id,
        predicate="self",
        object_id=entity.id,
        id="relation_sentinel",
    )
    claim = Claim(text="Sentinel exists", id="claim_sentinel")
    evidence = Evidence(
        claim_id=claim.id,
        chunk_id=chunk.id,
        document_id=document.id,
        text=document.text,
        id="evidence_sentinel",
    )
    contradiction = Contradiction(
        kind="sentinel",
        claim_a_id=claim.id,
        claim_b_id=claim.id,
        description="sentinel contradiction",
        id="contradiction_sentinel",
    )
    hypothesis = Hypothesis(statement="Sentinel hypothesis", id="hypothesis_sentinel")
    experiment = Experiment(
        hypothesis_id=hypothesis.id,
        design="Sentinel experiment",
        id="experiment_sentinel",
    )
    observation = Observation(
        source_id=source.id,
        kind="sentinel",
        payload={"value": "sentinel"},
        id="observation_sentinel",
    )
    result = Result(
        experiment_id=experiment.id,
        outcome="sentinel result",
        observation_ids=[observation.id],
        id="result_sentinel",
    )
    gap = KnowledgeGap(
        kind="sentinel",
        description="sentinel gap",
        reason="sentinel reason",
        id="gap_sentinel",
    )
    investigation = Investigation(
        gap_id=gap.id,
        description="sentinel investigation",
        id="investigation_sentinel",
    )

    for name, item in (
        ("sources", source),
        ("documents", document),
        ("chunks", chunk),
        ("entities", entity),
        ("relations", relation),
        ("claims", claim),
        ("evidence", evidence),
        ("contradictions", contradiction),
        ("hypotheses", hypothesis),
        ("experiments", experiment),
        ("results", result),
        ("observations", observation),
        ("gaps", gap),
        ("investigations", investigation),
    ):
        getattr(repository, name).save(item)


def _assert_rejected_without_mutation(
    path: Path,
    *,
    limits: Any = None,
    match: str,
) -> None:
    repository = InMemoryKnowledgeRepository()
    _seed_full_repository(repository)
    expected = _full_repository_projection(repository)

    with pytest.raises(ValueError, match=match):
        _new_service(repository, limits=limits).ingest_document(path)

    assert _full_repository_projection(repository) == expected


def _zip_info(name: str, compression: int = ZIP_DEFLATED) -> ZipInfo:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = compression
    info.external_attr = 0o600 << 16
    return info


def _write_epub(
    path: Path,
    *,
    mimetype: str = "application/epub+zip",
    rootfile_media_type: str = "application/oebps-package+xml",
    container: str | bytes | None = None,
    package: str | bytes | None = None,
    opening: str | bytes | None = None,
    findings: str | bytes | None = None,
) -> None:
    container = (
        container
        or f"""<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles>
    <rootfile full-path="OEBPS/content.opf" media-type="{rootfile_media_type}"/>
  </rootfiles>
</container>
"""
    )
    package = (
        package
        or """<?xml version="1.0" encoding="UTF-8"?>
<package version="3.0" unique-identifier="book-id"
         xmlns="http://www.idpf.org/2007/opf">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="book-id">nexus-citation-fixture</dc:identifier>
    <dc:title>Citation Fixture</dc:title>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="opening" href="z-opening.xhtml" media-type="application/xhtml+xml"/>
    <item id="findings" href="a-findings.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine>
    <itemref idref="opening"/>
    <itemref idref="findings"/>
  </spine>
</package>
"""
    )
    opening = (
        opening
        or """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><body>
<h1>Opening</h1><p>The opening chapter contains alpha evidence.</p>
</body></html>
"""
    )
    findings = (
        findings
        or """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><body>
<h1>Findings</h1><p>The findings chapter contains beta citation text.</p>
</body></html>
"""
    )

    with ZipFile(path, "w") as archive:
        archive.writestr(_zip_info("mimetype", ZIP_STORED), mimetype)
        archive.writestr(_zip_info("META-INF/container.xml"), container)
        archive.writestr(_zip_info("OEBPS/content.opf"), package)
        archive.writestr(_zip_info("OEBPS/z-opening.xhtml"), opening)
        archive.writestr(_zip_info("OEBPS/a-findings.xhtml"), findings)


def _epub_package(manifest: str, spine: str) -> str:
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<package version="3.0" unique-identifier="book-id"
         xmlns="http://www.idpf.org/2007/opf">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="book-id">nexus-hostile-fixture</dc:identifier>
    <dc:title>Hostile Fixture</dc:title>
    <dc:language>en</dc:language>
  </metadata>
  <manifest>{manifest}</manifest>
  <spine>{spine}</spine>
</package>
"""


def _patch_zip_member_headers(
    payload: bytes,
    member_name: str,
    *,
    compression: int | None = None,
    encrypted: bool = False,
) -> bytes:
    """Patch public ZIP header fields to make deterministic hostile fixtures."""
    patched = bytearray(payload)
    with ZipFile(BytesIO(payload)) as archive:
        local_offset = archive.getinfo(member_name).header_offset
    assert patched[local_offset : local_offset + 4] == b"PK\x03\x04"
    if compression is not None:
        struct.pack_into("<H", patched, local_offset + 8, compression)
    if encrypted:
        flags = struct.unpack_from("<H", patched, local_offset + 6)[0]
        struct.pack_into("<H", patched, local_offset + 6, flags | 1)

    cursor = 0
    found_central = False
    while True:
        cursor = patched.find(b"PK\x01\x02", cursor)
        if cursor < 0:
            break
        name_length = struct.unpack_from("<H", patched, cursor + 28)[0]
        extra_length = struct.unpack_from("<H", patched, cursor + 30)[0]
        comment_length = struct.unpack_from("<H", patched, cursor + 32)[0]
        name = bytes(patched[cursor + 46 : cursor + 46 + name_length]).decode("utf-8")
        if name == member_name:
            found_central = True
            if compression is not None:
                struct.pack_into("<H", patched, cursor + 10, compression)
            if encrypted:
                flags = struct.unpack_from("<H", patched, cursor + 8)[0]
                struct.pack_into("<H", patched, cursor + 8, flags | 1)
            break
        cursor += 46 + name_length + extra_length + comment_length
    assert found_central
    return bytes(patched)


def _write_pdf(path: Path) -> None:
    path.write_bytes(PDF_BYTES)


def _write_classic_pdf(path: Path, objects: tuple[bytes, ...]) -> None:
    """Write numbered indirect objects with a deterministic classic XRef table."""
    payload = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{object_number} 0 obj\n".encode())
        payload.extend(body)
        payload.extend(b"\nendobj\n")
    xref_offset = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode())
    payload.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    path.write_bytes(payload)


def _flate_stream(decoded: bytes, *, extra_dictionary: bytes = b"") -> bytes:
    compressed = zlib.compress(decoded, level=9)
    return (
        b"<< /Length "
        + str(len(compressed)).encode()
        + b" /Filter /FlateDecode "
        + extra_dictionary
        + b">>\nstream\n"
        + compressed
        + b"\nendstream"
    )


def _write_flate_pdf(path: Path, decoded_stream: bytes) -> None:
    """Write a minimal one-page PDF with a Flate-compressed content stream."""
    compressed = zlib.compress(decoded_stream, level=9)
    objects = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << >> /Contents 4 0 R >>"
        ),
        (
            f"<< /Length {len(compressed)} /Filter /FlateDecode >>\nstream\n".encode()
            + compressed
            + b"\nendstream"
        ),
    )
    payload = bytearray(b"%PDF-1.7\n%\xe2\xe3\xcf\xd3\n")
    offsets = [0]
    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{object_number} 0 obj\n".encode())
        payload.extend(body)
        payload.extend(b"\nendobj\n")
    xref_offset = len(payload)
    payload.extend(f"xref\n0 {len(objects) + 1}\n".encode())
    payload.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        payload.extend(f"{offset:010d} 00000 n \n".encode())
    payload.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_offset}\n%%EOF\n"
        ).encode()
    )
    path.write_bytes(payload)


def _write_flate_xref_pdf(path: Path, decoded_padding: bytes) -> None:
    """Write a valid PDF 1.5 whose structural XRef stream has compressible padding."""
    payload = bytearray(b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n")
    objects = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << >> /Contents 4 0 R >>"
        ),
        b"<< /Length 0 >>\nstream\n\nendstream",
    )
    offsets = [0]
    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{object_number} 0 obj\n".encode())
        payload.extend(body)
        payload.extend(b"\nendobj\n")

    xref_offset = len(payload)
    offsets.append(xref_offset)
    entries = bytearray(b"\x00" + (0).to_bytes(4, "big") + (65535).to_bytes(2, "big"))
    for offset in offsets[1:]:
        entries.extend(b"\x01" + offset.to_bytes(4, "big") + (0).to_bytes(2, "big"))
    compressed = zlib.compress(bytes(entries) + decoded_padding, level=9)
    payload.extend(b"5 0 obj\n")
    payload.extend(
        (
            f"<< /Type /XRef /Length {len(compressed)} /Filter /FlateDecode "
            "/Size 6 /Root 1 0 R /W [1 4 2] /Index [0 6] >>\nstream\n"
        ).encode()
    )
    payload.extend(compressed)
    payload.extend(b"\nendstream\nendobj\n")
    payload.extend(f"startxref\n{xref_offset}\n%%EOF\n".encode())
    path.write_bytes(payload)


def _write_encoded_xref_pdf(
    path: Path,
    decoded_padding: bytes,
    *,
    filter_expression: bytes,
    encode: Any,
) -> None:
    """Write a PDF 1.5 XRef stream encoded by one or more named filters."""
    payload = bytearray(b"%PDF-1.5\n%\xe2\xe3\xcf\xd3\n")
    objects = (
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
        (
            b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            b"/Resources << >> /Contents 4 0 R >>"
        ),
        b"<< /Length 0 >>\nstream\n\nendstream",
    )
    offsets = [0]
    for object_number, body in enumerate(objects, start=1):
        offsets.append(len(payload))
        payload.extend(f"{object_number} 0 obj\n".encode())
        payload.extend(body)
        payload.extend(b"\nendobj\n")

    xref_offset = len(payload)
    offsets.append(xref_offset)
    entries = bytearray(b"\x00" + (0).to_bytes(4, "big") + (65535).to_bytes(2, "big"))
    for offset in offsets[1:]:
        entries.extend(b"\x01" + offset.to_bytes(4, "big") + (0).to_bytes(2, "big"))
    encoded = encode(bytes(entries) + decoded_padding)
    payload.extend(b"5 0 obj\n")
    payload.extend(
        b"<< /Type /XRef /Length "
        + str(len(encoded)).encode()
        + b" /Filter "
        + filter_expression
        + b" /Size 6 /Root 1 0 R /W [1 4 2] /Index [0 6] >>\nstream\n"
    )
    payload.extend(encoded)
    payload.extend(b"\nendstream\nendobj\n")
    payload.extend(f"startxref\n{xref_offset}\n%%EOF\n".encode())
    path.write_bytes(payload)


def _write_contents_array_pdf(path: Path, decoded_streams: tuple[bytes, ...]) -> None:
    references = " ".join(f"{index} 0 R" for index in range(4, 4 + len(decoded_streams)))
    _write_classic_pdf(
        path,
        (
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
            (
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                + f"/Resources << >> /Contents [{references}] >>".encode()
            ),
            *(_flate_stream(decoded) for decoded in decoded_streams),
        ),
    )


def _write_nested_form_pdf(path: Path, decoded_nested_form: bytes) -> None:
    _write_classic_pdf(
        path,
        (
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
            (
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                b"/Contents 4 0 R /Resources << /XObject << /Outer 5 0 R >> >> >>"
            ),
            b"<< /Length 0 >>\nstream\n\nendstream",
            _flate_stream(
                b"",
                extra_dictionary=(
                    b"/Type /XObject /Subtype /Form /BBox [0 0 1 1] "
                    b"/Resources << /XObject << /Nested 6 0 R >> >> "
                ),
            ),
            _flate_stream(
                decoded_nested_form,
                extra_dictionary=(
                    b"/Type /XObject /Subtype /Form /BBox [0 0 1 1] /Resources << >> "
                ),
            ),
        ),
    )


def _write_chained_filter_pdf(path: Path, decoded_stream: bytes) -> None:
    compressed = zlib.compress(decoded_stream, level=9)
    encoded = compressed.hex().encode("ascii") + b">"
    stream = (
        f"<< /Length {len(encoded)} /Filter [/ASCIIHexDecode /FlateDecode] >>\nstream\n".encode()
        + encoded
        + b"\nendstream"
    )
    _write_classic_pdf(
        path,
        (
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Count 1 /Kids [3 0 R] >>",
            (
                b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                b"/Resources << >> /Contents 4 0 R >>"
            ),
            stream,
        ),
    )


@pytest.fixture
def txt_path(tmp_path: Path) -> Path:
    path = tmp_path / "exact.txt"
    path.write_bytes(TXT_TEXT.encode("utf-8"))
    return path


@pytest.fixture
def markdown_path(tmp_path: Path) -> Path:
    path = tmp_path / "exact.md"
    path.write_bytes(MARKDOWN_TEXT.encode("utf-8"))
    return path


@pytest.fixture
def repeated_markdown_path(tmp_path: Path) -> Path:
    path = tmp_path / "repeated.md"
    path.write_text("# Same\nBody\n# Same\nBody\n", encoding="utf-8")
    return path


@pytest.fixture
def pdf_path(tmp_path: Path) -> Path:
    path = tmp_path / "pages.pdf"
    _write_pdf(path)
    return path


@pytest.fixture
def epub_path(tmp_path: Path) -> Path:
    path = tmp_path / "chapters.epub"
    _write_epub(path)
    return path


class TestPublicContract:
    def test_public_citation_ingestion_exports(self, txt_path: Path) -> None:
        from nexus_knowledge.ingestion import CitationIngestionLimits, CitationIngestionService
        from nexus_knowledge.ingestion.citation import (
            CitationIngestionLimits as ConcreteCitationIngestionLimits,
        )
        from nexus_knowledge.ingestion.citation import (
            CitationIngestionService as ConcreteCitationIngestionService,
        )
        from nexus_knowledge.port import CitationIngestionPort
        from nexus_knowledge.port.citation_ingestion import (
            CitationIngestionPort as ModuleCitationIngestionPort,
        )

        assert CitationIngestionService is ConcreteCitationIngestionService
        assert CitationIngestionLimits is ConcreteCitationIngestionLimits
        assert CitationIngestionPort is ModuleCitationIngestionPort

        repository = InMemoryKnowledgeRepository()
        service: CitationIngestionPort = CitationIngestionService(repository=repository)
        assert callable(service.ingest_document)
        assert service.ingest_document(txt_path)

    def test_ingest_document_accepts_path_and_string(self, txt_path: Path) -> None:
        repository = InMemoryKnowledgeRepository()
        service = _new_service(repository)

        from_path = service.ingest_document(txt_path)
        from_string = service.ingest_document(str(txt_path))

        assert isinstance(from_path, str) and from_path
        assert from_string == from_path

    def test_citation_dependency_profiles_share_supported_parser_floors(self) -> None:
        project = tomllib.loads(
            (Path(__file__).parents[1] / "pyproject.toml").read_text(encoding="utf-8")
        )
        optional = project["project"]["optional-dependencies"]
        profiles = ("citation", "test", "dev")
        pypdf_requirements: list[str] = []
        defusedxml_requirements: list[str] = []

        for profile in profiles:
            requirements = optional[profile]
            pypdf = [item for item in requirements if item.lower().startswith("pypdf")]
            defusedxml = [item for item in requirements if item.lower().startswith("defusedxml")]
            assert len(pypdf) == 1
            assert len(defusedxml) == 1
            assert ">=6.16.2" in pypdf[0] and "<7" in pypdf[0]
            assert ">=0.7" in defusedxml[0] and "<1" in defusedxml[0]
            pypdf_requirements.extend(pypdf)
            defusedxml_requirements.extend(defusedxml)

        assert len(set(pypdf_requirements)) == 1
        assert len(set(defusedxml_requirements)) == 1

    def test_installed_pypdf_exposes_supported_public_decoder_surface(self) -> None:
        import pypdf.filters as pdf_filters
        from pypdf.errors import LimitReachedError

        assert issubclass(LimitReachedError, Exception)
        for name in PYPDF_DECODE_LIMIT_NAMES:
            assert isinstance(getattr(pdf_filters, name), int)
        for name in (
            "ASCIIHexDecode",
            "ASCII85Decode",
            "FlateDecode",
            "LZWDecode",
            "RunLengthDecode",
        ):
            assert callable(getattr(pdf_filters, name).decode)

    def test_citation_module_has_no_forbidden_pipeline_dependencies(self) -> None:
        import nexus_knowledge.ingestion.citation as citation_module

        source_path = Path(citation_module.__file__ or "")
        tree = ast.parse(source_path.read_text(encoding="utf-8"), filename=str(source_path))
        imported_modules = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        }
        imported_modules.update(
            node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        )
        forbidden_parts = {
            "claim",
            "claims",
            "entities",
            "entity",
            "retrieval",
            "graph",
            "graphrag",
            "relation",
            "relations",
            "runtime",
            "evidence",
            "embedding",
            "embeddings",
            "vector",
            "vector_store",
        }
        assert not {
            module
            for module in imported_modules
            if forbidden_parts.intersection(module.lower().split("."))
        }
        assert not {
            (node.func.id if isinstance(node.func, ast.Name) else f"attribute:{node.func.attr}")
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and (
                (
                    isinstance(node.func, ast.Name)
                    and node.func.id in {"__import__", "import_module"}
                )
                or (isinstance(node.func, ast.Attribute) and node.func.attr == "import_module")
            )
        }

        pypdf_aliases: set[str] = set()
        private_pypdf_imports: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "pypdf" or alias.name.startswith("pypdf."):
                        pypdf_aliases.add(alias.asname or alias.name.split(".")[0])
                        if any(part.startswith("_") for part in alias.name.split(".")):
                            private_pypdf_imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("pypdf"):
                module = node.module or ""
                if any(part.startswith("_") for part in module.split(".")):
                    private_pypdf_imports.append(module)
                for alias in node.names:
                    pypdf_aliases.add(alias.asname or alias.name)
                    if alias.name.startswith("_"):
                        private_pypdf_imports.append(f"{module}.{alias.name}")

        def attribute_root(node: ast.expr) -> str | None:
            while isinstance(node, ast.Attribute):
                node = node.value
            return node.id if isinstance(node, ast.Name) else None

        private_pypdf_attributes = {
            (node.lineno, node.attr)
            for node in ast.walk(tree)
            if isinstance(node, ast.Attribute)
            and node.attr.startswith("_")
            and attribute_root(node.value) in pypdf_aliases
        }
        assert not private_pypdf_imports
        assert not private_pypdf_attributes


class TestFormatsAndReconstruction:
    def test_ingests_txt_without_losing_source_text(self, txt_path: Path) -> None:
        repository = InMemoryKnowledgeRepository()
        document_id = _new_service(repository).ingest_document(txt_path)
        source, document, chunks = _assert_source_lineage(repository, document_id, txt_path)

        assert source.kind == "text"
        assert document.content_type == "text"
        assert document.raw == TXT_TEXT
        assert document.text == TXT_TEXT
        assert len(chunks) == 1
        assert (chunks[0].metadata["line_start"], chunks[0].metadata["line_end"]) == (1, 3)
        assert _assert_exact_reconstruction(repository, document_id) == TXT_TEXT

    def test_ingests_markdown_without_losing_headings_or_body(self, markdown_path: Path) -> None:
        repository = InMemoryKnowledgeRepository()
        document_id = _new_service(repository).ingest_document(markdown_path)
        source, document, chunks = _assert_source_lineage(repository, document_id, markdown_path)

        assert source.kind == "markdown"
        assert document.content_type == "markdown"
        assert document.raw == MARKDOWN_TEXT
        assert document.text == MARKDOWN_TEXT
        assert _assert_exact_reconstruction(repository, document_id) == MARKDOWN_TEXT
        expected_texts = [
            "# Overview\r\n\r\nAlpha  evidence stays exact.\r\n\r\n",
            "## Details\r\n\r\nUnicode café — 東京 remains source-aligned.\r\n",
        ]
        split = len(expected_texts[0])
        assert len(chunks) == 2
        assert [chunk.text for chunk in chunks] == expected_texts
        assert [
            (
                chunk.index,
                chunk.span.start,
                chunk.span.end,
                chunk.metadata["char_start"],
                chunk.metadata["char_end"],
                chunk.metadata["line_start"],
                chunk.metadata["line_end"],
                chunk.metadata["section"],
            )
            for chunk in chunks
        ] == [
            (0, 0, split, 0, split, 1, 4, "Overview"),
            (1, split, len(MARKDOWN_TEXT), split, len(MARKDOWN_TEXT), 5, 7, "Details"),
        ]

    def test_ingests_pdf_pages_in_source_order(self, pdf_path: Path) -> None:
        repository = InMemoryKnowledgeRepository()
        document_id = _new_service(repository).ingest_document(pdf_path)
        source, document, chunks = _assert_source_lineage(repository, document_id, pdf_path)

        assert source.kind == "pdf"
        assert document.content_type == "pdf"
        assert document.text == PDF_CANONICAL_TEXT
        assert [chunk.metadata["page"] for chunk in chunks] == [1, 2]
        assert [chunk.text for chunk in chunks] == list(PDF_SEGMENT_TEXTS)
        assert [(chunk.span.start, chunk.span.end) for chunk in chunks] == [
            (0, len(PDF_SEGMENT_TEXTS[0])),
            (len(PDF_SEGMENT_TEXTS[0]), len(PDF_CANONICAL_TEXT)),
        ]
        assert [
            (
                chunk.metadata["page"],
                chunk.metadata["page_index"],
                chunk.metadata["page_count"],
                chunk.metadata["line_start"],
                chunk.metadata["line_end"],
            )
            for chunk in chunks
        ] == [(1, 0, 2, 1, 1), (2, 1, 2, 2, 2)]
        assert _assert_exact_reconstruction(repository, document_id) == PDF_CANONICAL_TEXT

    def test_ingests_epub_chapters_in_spine_order(self, epub_path: Path) -> None:
        repository = InMemoryKnowledgeRepository()
        document_id = _new_service(repository).ingest_document(epub_path)
        source, document, chunks = _assert_source_lineage(repository, document_id, epub_path)

        assert source.kind == "epub"
        assert document.content_type == "epub"
        assert document.text == EPUB_CANONICAL_TEXT
        assert [chunk.metadata["chapter"] for chunk in chunks] == [1, 2]
        assert [chunk.metadata["section"] for chunk in chunks] == ["Opening", "Findings"]
        assert [chunk.text for chunk in chunks] == list(EPUB_SEGMENT_TEXTS)
        assert [(chunk.span.start, chunk.span.end) for chunk in chunks] == [
            (0, len(EPUB_SEGMENT_TEXTS[0])),
            (len(EPUB_SEGMENT_TEXTS[0]), len(EPUB_CANONICAL_TEXT)),
        ]
        assert [
            (
                chunk.metadata["chapter"],
                chunk.metadata["chapter_id"],
                chunk.metadata["chapter_path"],
                chunk.metadata["line_start"],
                chunk.metadata["line_end"],
            )
            for chunk in chunks
        ] == [
            (1, "opening", "OEBPS/z-opening.xhtml", 1, 2),
            (2, "findings", "OEBPS/a-findings.xhtml", 3, 4),
        ]
        assert _assert_exact_reconstruction(repository, document_id) == EPUB_CANONICAL_TEXT

    def test_markdown_bare_carriage_returns_have_exact_line_locators(self, tmp_path: Path) -> None:
        text = "# First\rOne\r# Second\rTwo\r"
        path = tmp_path / "bare-cr.md"
        path.write_bytes(text.encode("utf-8"))
        repository = InMemoryKnowledgeRepository()

        document_id = _new_service(repository).ingest_document(path)
        _, document, chunks = _stored_records(repository, document_id)

        assert document.text == text
        assert [chunk.text for chunk in chunks] == ["# First\rOne\r", "# Second\rTwo\r"]
        assert [chunk.metadata["section"] for chunk in chunks] == ["First", "Second"]
        assert [(chunk.metadata["line_start"], chunk.metadata["line_end"]) for chunk in chunks] == [
            (1, 2),
            (3, 4),
        ]
        assert _assert_exact_reconstruction(repository, document_id) == text


class TestIdentityAndPersistence:
    @pytest.mark.parametrize("fixture_name", FORMAT_FIXTURES)
    def test_document_source_and_segment_ids_are_stable_across_fresh_repositories(
        self, fixture_name: str, request: pytest.FixtureRequest
    ) -> None:
        path = request.getfixturevalue(fixture_name)
        first_repository = InMemoryKnowledgeRepository()
        second_repository = InMemoryKnowledgeRepository()

        first_id = _new_service(first_repository).ingest_document(path)
        second_id = _new_service(second_repository).ingest_document(path)
        first_source, _, first_chunks = _stored_records(first_repository, first_id)
        second_source, _, second_chunks = _stored_records(second_repository, second_id)

        assert second_id == first_id
        assert second_source.id == first_source.id
        assert [chunk.id for chunk in second_chunks] == [chunk.id for chunk in first_chunks]

    def test_repeated_identical_segments_have_unique_ordered_stable_ids(
        self, repeated_markdown_path: Path
    ) -> None:
        first_repository = InMemoryKnowledgeRepository()
        second_repository = InMemoryKnowledgeRepository()

        first_id = _new_service(first_repository).ingest_document(repeated_markdown_path)
        second_id = _new_service(second_repository).ingest_document(repeated_markdown_path)
        _, _, first_chunks = _stored_records(first_repository, first_id)
        _, _, second_chunks = _stored_records(second_repository, second_id)

        assert first_id == second_id
        assert [chunk.text for chunk in first_chunks] == ["# Same\nBody\n", "# Same\nBody\n"]
        assert [chunk.index for chunk in first_chunks] == [0, 1]
        assert [chunk.id for chunk in first_chunks] == [
            stable_id("chunk", first_id, 0),
            stable_id("chunk", first_id, 1),
        ]
        assert len({chunk.id for chunk in first_chunks}) == 2
        assert [chunk.id for chunk in second_chunks] == [chunk.id for chunk in first_chunks]

    def test_distinct_paths_with_identical_content_have_distinct_aggregate_ids(
        self, tmp_path: Path
    ) -> None:
        first_path = tmp_path / "first.txt"
        second_path = tmp_path / "second.txt"
        first_path.write_bytes(TXT_TEXT.encode("utf-8"))
        second_path.write_bytes(TXT_TEXT.encode("utf-8"))
        repository = InMemoryKnowledgeRepository()
        service = _new_service(repository)

        first_id = service.ingest_document(first_path)
        second_id = service.ingest_document(second_path)
        first_source, _, first_chunks = _stored_records(repository, first_id)
        second_source, _, second_chunks = _stored_records(repository, second_id)

        assert second_id != first_id
        assert second_source.id != first_source.id
        assert {chunk.id for chunk in first_chunks}.isdisjoint(
            {chunk.id for chunk in second_chunks}
        )
        assert _counts(repository) == (2, 2, 2)

    def test_changed_content_at_same_path_creates_a_distinct_version(self, tmp_path: Path) -> None:
        path = tmp_path / "versioned.txt"
        path.write_text("version one", encoding="utf-8")
        repository = InMemoryKnowledgeRepository()
        service = _new_service(repository)

        first_id = service.ingest_document(path)
        first_source, _, first_chunks = _stored_records(repository, first_id)
        path.write_text("version two", encoding="utf-8")
        second_id = service.ingest_document(path)
        second_source, _, second_chunks = _stored_records(repository, second_id)

        assert second_id != first_id
        assert second_source.id != first_source.id
        assert first_source.reference == second_source.reference == str(path.resolve())
        assert [chunk.id for chunk in first_chunks] != [chunk.id for chunk in second_chunks]
        assert _counts(repository) == (2, 2, 2)

    def test_forced_aggregate_id_collision_is_atomic(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import nexus_knowledge.ingestion.citation as citation_module

        first_path = tmp_path / "first.txt"
        second_path = tmp_path / "second.txt"
        first_path.write_text("first collision candidate", encoding="utf-8")
        second_path.write_text("different second candidate", encoding="utf-8")

        def colliding_id(prefix: str, *parts: object) -> str:
            del parts
            return f"{prefix}_forced_collision"

        monkeypatch.setattr(citation_module, "stable_id", colliding_id)
        repository = InMemoryKnowledgeRepository()
        service = _new_service(repository)
        first_id = service.ingest_document(first_path)
        before = _full_repository_projection_bytes(repository)

        with pytest.raises(ValueError, match="collision|conflict|already exists|identifier"):
            service.ingest_document(second_path)

        assert _full_repository_projection_bytes(repository) == before
        assert _assert_exact_reconstruction(repository, first_id) == "first collision candidate"

    @pytest.mark.parametrize("fixture_name", FORMAT_FIXTURES)
    def test_duplicate_ingestion_is_a_repository_no_op(
        self, fixture_name: str, request: pytest.FixtureRequest
    ) -> None:
        path = request.getfixturevalue(fixture_name)
        repository = InMemoryKnowledgeRepository()
        service = _new_service(repository)

        first_id = service.ingest_document(path)
        first_counts = _counts(repository)
        first_projection = _repository_projection(repository, first_id)
        second_id = service.ingest_document(path)

        assert second_id == first_id
        assert _counts(repository) == first_counts
        assert _repository_projection(repository, second_id) == first_projection

    @pytest.mark.parametrize("fixture_name", FORMAT_FIXTURES)
    def test_snapshot_round_trip_preserves_citation_records_and_reconstruction(
        self, fixture_name: str, request: pytest.FixtureRequest, tmp_path: Path
    ) -> None:
        path = request.getfixturevalue(fixture_name)
        repository = InMemoryKnowledgeRepository()
        document_id = _new_service(repository).ingest_document(path)
        expected = _repository_projection(repository, document_id)
        snapshot = tmp_path / "citation-snapshot.json"

        save_snapshot(repository, snapshot)
        restored = load_snapshot(snapshot)

        assert _repository_projection(restored, document_id) == expected
        _assert_exact_reconstruction(restored, document_id)

    @pytest.mark.parametrize("fixture_name", FORMAT_FIXTURES)
    def test_reingestion_after_snapshot_restore_is_a_no_op(
        self, fixture_name: str, request: pytest.FixtureRequest, tmp_path: Path
    ) -> None:
        path = request.getfixturevalue(fixture_name)
        repository = InMemoryKnowledgeRepository()
        document_id = _new_service(repository).ingest_document(path)
        snapshot = tmp_path / "citation-snapshot.json"
        save_snapshot(repository, snapshot)
        restored = load_snapshot(snapshot)
        expected_counts = _counts(restored)
        expected_projection = _repository_projection(restored, document_id)

        restored_id = _new_service(restored).ingest_document(path)

        assert restored_id == document_id
        assert _counts(restored) == expected_counts
        assert _repository_projection(restored, restored_id) == expected_projection

    @pytest.mark.parametrize("fixture_name", FORMAT_FIXTURES)
    def test_successful_ingestion_does_not_touch_unrelated_repositories(
        self, fixture_name: str, request: pytest.FixtureRequest
    ) -> None:
        path = request.getfixturevalue(fixture_name)
        repository = InMemoryKnowledgeRepository()
        _seed_full_repository(repository)
        expected = _full_repository_projection_bytes(repository, UNRELATED_REPOSITORIES)

        _new_service(repository).ingest_document(path)

        assert _full_repository_projection_bytes(repository, UNRELATED_REPOSITORIES) == expected

    @pytest.mark.parametrize("fixture_name", FORMAT_FIXTURES)
    def test_successful_ingestion_accepts_a_narrow_repository_without_other_stores(
        self, fixture_name: str, request: pytest.FixtureRequest
    ) -> None:
        path = request.getfixturevalue(fixture_name)
        backing = InMemoryKnowledgeRepository()
        _seed_full_repository(backing)
        expected = _full_repository_projection_bytes(backing, UNRELATED_REPOSITORIES)

        class NarrowRepository:
            sources = backing.sources
            documents = backing.documents
            chunks = backing.chunks

            def __getattr__(self, name: str) -> Any:
                if name in UNRELATED_REPOSITORIES:
                    raise AssertionError(f"citation ingestion accessed unrelated store: {name}")
                raise AttributeError(name)

        repository = NarrowRepository()
        document_id = _new_service(repository).ingest_document(path)

        assert backing.documents.get(document_id) is not None
        _assert_exact_reconstruction(backing, document_id)
        assert _full_repository_projection_bytes(backing, UNRELATED_REPOSITORIES) == expected


class TestTransactionalPersistence:
    def test_document_save_failure_rolls_back_the_entire_aggregate_and_retry_succeeds(
        self, txt_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repository = InMemoryKnowledgeRepository()
        _seed_full_repository(repository)
        expected = _full_repository_projection(repository)

        with monkeypatch.context() as patch:

            def fail_document_save(item: Any) -> Any:
                del item
                raise RuntimeError("injected document save failure")

            patch.setattr(repository.documents, "save", fail_document_save)
            with pytest.raises(RuntimeError, match="injected document save failure"):
                _new_service(repository).ingest_document(txt_path)

        assert _full_repository_projection(repository) == expected
        document_id = _new_service(repository).ingest_document(txt_path)
        assert repository.documents.get(document_id) is not None
        _assert_exact_reconstruction(repository, document_id)

    def test_intermediate_chunk_save_failure_rolls_back_and_retry_succeeds(
        self, markdown_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repository = InMemoryKnowledgeRepository()
        _seed_full_repository(repository)
        expected = _full_repository_projection(repository)
        healthy_save = repository.chunks.save
        calls = 0

        with monkeypatch.context() as patch:

            def fail_second_chunk(item: Any) -> Any:
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise RuntimeError("injected intermediate chunk save failure")
                return healthy_save(item)

            patch.setattr(repository.chunks, "save", fail_second_chunk)
            with pytest.raises(RuntimeError, match="injected intermediate chunk save failure"):
                _new_service(repository).ingest_document(markdown_path)

        assert calls == 2
        assert _full_repository_projection(repository) == expected
        document_id = _new_service(repository).ingest_document(markdown_path)
        assert len(repository.chunks.by_document(document_id)) == 2
        _assert_exact_reconstruction(repository, document_id)


class TestAggregateVerificationAndRepair:
    def test_duplicate_ingestion_repairs_a_deleted_chunk(self, markdown_path: Path) -> None:
        repository = InMemoryKnowledgeRepository()
        service = _new_service(repository)
        document_id = service.ingest_document(markdown_path)
        expected = _repository_projection(repository, document_id)
        chunks = repository.chunks.by_document(document_id)
        assert len(chunks) == 2
        assert repository.chunks.delete(chunks[1].id)

        replayed_id = service.ingest_document(markdown_path)

        assert replayed_id == document_id
        assert _repository_projection(repository, document_id) == expected
        _assert_exact_reconstruction(repository, document_id)

    def test_duplicate_ingestion_repairs_a_corrupt_chunk(self, markdown_path: Path) -> None:
        repository = InMemoryKnowledgeRepository()
        service = _new_service(repository)
        document_id = service.ingest_document(markdown_path)
        expected = _repository_projection(repository, document_id)
        original = repository.chunks.by_document(document_id)[0]
        repository.chunks.save(
            Chunk(
                document_id=document_id,
                index=original.index,
                text="corrupt",
                span=Span(0, len("corrupt")),
                metadata={"corrupt": True},
            )
        )

        replayed_id = service.ingest_document(markdown_path)

        assert replayed_id == document_id
        assert _repository_projection(repository, document_id) == expected
        _assert_exact_reconstruction(repository, document_id)

    def test_repair_later_chunk_save_failure_restores_corrupt_precall_state(
        self, markdown_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repository = InMemoryKnowledgeRepository()
        service = _new_service(repository)
        document_id = service.ingest_document(markdown_path)
        source, _, chunks = _stored_records(repository, document_id)
        assert len(chunks) == 2
        corrupt = Chunk(
            document_id=document_id,
            index=chunks[0].index,
            text="corrupt first segment",
            span=Span(0, len("corrupt first segment")),
            metadata={"corrupt": True},
        )
        repository.chunks.save(corrupt)
        assert repository.chunks.delete(chunks[1].id)
        before = _aggregate_projection(repository, source_id=source.id, document_id=document_id)
        healthy_save = repository.chunks.save
        save_calls = 0

        with monkeypatch.context() as patch:

            def fail_second_repair_save(item: Any) -> Any:
                nonlocal save_calls
                save_calls += 1
                if save_calls == 2:
                    raise RuntimeError("injected later repair chunk failure")
                return healthy_save(item)

            patch.setattr(repository.chunks, "save", fail_second_repair_save)
            with pytest.raises(RuntimeError, match="injected later repair chunk failure"):
                service.ingest_document(markdown_path)

        assert save_calls >= 3
        assert (
            _aggregate_projection(repository, source_id=source.id, document_id=document_id)
            == before
        )
        assert service.ingest_document(markdown_path) == document_id
        _assert_exact_reconstruction(repository, document_id)

    def test_missing_source_and_document_recover_timestamp_from_surviving_chunks(
        self, markdown_path: Path
    ) -> None:
        repository = InMemoryKnowledgeRepository()
        service = _new_service(repository)
        document_id = service.ingest_document(markdown_path)
        source, document, chunks = _stored_records(repository, document_id)
        expected = _repository_projection(repository, document_id)
        original_timestamp = source.ingested_at
        assert document.ingested_at == original_timestamp
        assert chunks

        assert repository.sources.delete(source.id)
        assert repository.documents.delete(document_id)
        assert repository.chunks.by_document(document_id) == chunks

        assert service.ingest_document(markdown_path) == document_id

        repaired_source, repaired_document, _ = _stored_records(repository, document_id)
        assert repaired_source.ingested_at == original_timestamp
        assert repaired_document.ingested_at == original_timestamp
        assert _repository_projection(repository, document_id) == expected

    @pytest.mark.parametrize("mode", ["missing", "corrupt"])
    def test_duplicate_ingestion_repairs_source_and_preserves_original_timestamp(
        self, mode: str, txt_path: Path
    ) -> None:
        repository = InMemoryKnowledgeRepository()
        service = _new_service(repository)
        document_id = service.ingest_document(txt_path)
        original_source, original_document, _ = _stored_records(repository, document_id)
        expected = _repository_projection(repository, document_id)

        if mode == "missing":
            assert repository.sources.delete(original_source.id)
        else:
            repository.sources.save(
                Source(
                    title="corrupt title",
                    kind=SourceKind.OTHER,
                    reference="corrupt://reference",
                    metadata={"corrupt": True},
                    id=original_source.id,
                    ingested_at=original_source.ingested_at,
                )
            )

        assert service.ingest_document(txt_path) == document_id

        repaired_source, repaired_document, _ = _stored_records(repository, document_id)
        assert _repository_projection(repository, document_id) == expected
        assert repaired_source.ingested_at == original_source.ingested_at
        assert repaired_document.ingested_at == original_document.ingested_at

    @pytest.mark.parametrize("mode", ["missing", "corrupt"])
    def test_duplicate_ingestion_repairs_document_and_preserves_original_timestamp(
        self, mode: str, txt_path: Path
    ) -> None:
        repository = InMemoryKnowledgeRepository()
        service = _new_service(repository)
        document_id = service.ingest_document(txt_path)
        original_source, original_document, _ = _stored_records(repository, document_id)
        expected = _repository_projection(repository, document_id)

        if mode == "missing":
            assert repository.documents.delete(document_id)
        else:
            repository.documents.save(
                Document(
                    source_id=original_source.id,
                    title="corrupt title",
                    content_type="corrupt",
                    text="corrupt",
                    raw="corrupt",
                    metadata={"corrupt": True},
                    id=document_id,
                    ingested_at=original_document.ingested_at,
                )
            )

        assert service.ingest_document(txt_path) == document_id

        repaired_source, repaired_document, _ = _stored_records(repository, document_id)
        assert _repository_projection(repository, document_id) == expected
        assert repaired_source.ingested_at == original_source.ingested_at
        assert repaired_document.ingested_at == original_document.ingested_at

    def test_repair_save_failure_restores_exact_corrupt_precall_aggregate(
        self, txt_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repository = InMemoryKnowledgeRepository()
        service = _new_service(repository)
        document_id = service.ingest_document(txt_path)
        source, document, _ = _stored_records(repository, document_id)
        repository.sources.save(
            Source(
                title="corrupt source",
                kind=SourceKind.OTHER,
                reference="corrupt://source",
                id=source.id,
                ingested_at=source.ingested_at,
            )
        )
        repository.documents.save(
            Document(
                source_id=source.id,
                title="corrupt document",
                content_type="corrupt",
                text="corrupt",
                raw="corrupt",
                id=document.id,
                ingested_at=document.ingested_at,
            )
        )
        before = _aggregate_projection(repository, source_id=source.id, document_id=document_id)
        healthy_document_save = repository.documents.save
        fault_count = 0

        with monkeypatch.context() as patch:

            def fail_document_save(item: Any) -> Any:
                nonlocal fault_count
                if fault_count == 0:
                    fault_count += 1
                    raise RuntimeError("injected repair save failure")
                return healthy_document_save(item)

            patch.setattr(repository.documents, "save", fail_document_save)
            with pytest.raises(RuntimeError, match="injected repair save failure"):
                service.ingest_document(txt_path)

        assert fault_count == 1
        assert (
            _aggregate_projection(repository, source_id=source.id, document_id=document_id)
            == before
        )
        assert service.ingest_document(txt_path) == document_id
        _assert_exact_reconstruction(repository, document_id)

    def test_extra_chunk_delete_failure_restores_exact_precall_aggregate(
        self, txt_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        repository = InMemoryKnowledgeRepository()
        service = _new_service(repository)
        document_id = service.ingest_document(txt_path)
        source, _, _ = _stored_records(repository, document_id)
        extra = Chunk(
            document_id=document_id,
            index=99,
            text="unexpected extra segment",
            span=Span(0, len("unexpected extra segment")),
            metadata={"extra": True},
        )
        repository.chunks.save(extra)
        before = _aggregate_projection(repository, source_id=source.id, document_id=document_id)
        healthy_delete = repository.chunks.delete
        fault_count = 0

        with monkeypatch.context() as patch:

            def fail_extra_delete(item_id: str) -> bool:
                nonlocal fault_count
                if item_id == extra.id and fault_count == 0:
                    fault_count += 1
                    raise RuntimeError("injected extra chunk delete failure")
                return healthy_delete(item_id)

            patch.setattr(repository.chunks, "delete", fail_extra_delete)
            with pytest.raises(RuntimeError, match="injected extra chunk delete failure"):
                service.ingest_document(txt_path)

        assert fault_count == 1
        assert (
            _aggregate_projection(repository, source_id=source.id, document_id=document_id)
            == before
        )
        assert service.ingest_document(txt_path) == document_id
        assert repository.chunks.get(extra.id) is None
        _assert_exact_reconstruction(repository, document_id)


class TestResourceLimits:
    def test_oversized_file_is_rejected_before_persistence(self, tmp_path: Path) -> None:
        path = tmp_path / "oversized.txt"
        path.write_bytes(b"x" * 11)
        _assert_rejected_without_mutation(
            path,
            limits=_limits(max_file_bytes=10),
            match="file|size|limit|large",
        )

    def test_stat_read_race_uses_a_bounded_open_handle_read(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "raced-size.txt"
        path.write_bytes(b"small")
        resolved = path.resolve()
        read_sizes: list[int] = []

        class RacedFile(BytesIO):
            def read(self, size: int = -1) -> bytes:
                read_sizes.append(size)
                if size < 0:
                    raise AssertionError("citation ingestion performed an unbounded file read")
                return super().read(size)

        original_open = Path.open

        def raced_open(file_path: Path, mode: str = "r", *args: Any, **kwargs: Any) -> Any:
            if file_path == resolved and mode == "rb":
                return RacedFile(b"x" * 12)
            return original_open(file_path, mode, *args, **kwargs)

        repository = InMemoryKnowledgeRepository()
        _seed_full_repository(repository)
        expected = _full_repository_projection_bytes(repository)
        monkeypatch.setattr(Path, "open", raced_open)

        with pytest.raises(ValueError, match="file|size|limit|large"):
            _new_service(repository, limits=_limits(max_file_bytes=10)).ingest_document(path)

        assert read_sizes == [11]
        assert _full_repository_projection_bytes(repository) == expected

    @pytest.mark.parametrize("invalid_limit", [0, -1, True, 1.5])
    def test_pdf_decoded_stream_limit_requires_a_positive_integer(self, invalid_limit: Any) -> None:
        with pytest.raises(ValueError, match="max_pdf_decoded_stream_bytes|positive integer"):
            _limits(max_pdf_decoded_stream_bytes=invalid_limit)

    def test_flate_pdf_bomb_is_rejected_under_a_bounded_decode_limit(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pypdf.filters as pdf_filters

        path = tmp_path / "flate-whitespace-bomb.pdf"
        decoded_size = 12 * 1024 * 1024
        decoded_limit = 1024 * 1024
        _write_flate_pdf(path, b" " * decoded_size)
        assert path.stat().st_size < 32 * 1024

        repository = InMemoryKnowledgeRepository()
        _seed_full_repository(repository)
        expected = _full_repository_projection_bytes(repository)
        observed_library_limits: list[int] = []
        original_decode = pdf_filters.FlateDecode.decode
        original_library_limits = {
            name: getattr(pdf_filters, name) for name in PYPDF_DECODE_LIMIT_NAMES
        }

        def guarded_decode(data: bytes, decode_parms: Any = None, **kwargs: Any) -> bytes:
            observed_library_limits.append(pdf_filters.ZLIB_MAX_OUTPUT_LENGTH)
            if not 0 < pdf_filters.ZLIB_MAX_OUTPUT_LENGTH <= decoded_limit:
                raise AssertionError("PDF Flate decoding began without the configured bound")
            return original_decode(data, decode_parms, **kwargs)

        monkeypatch.setattr(
            pdf_filters.FlateDecode,
            "decode",
            staticmethod(guarded_decode),
        )

        with pytest.raises(ValueError, match="PDF|decoded|stream|limit|expanded"):
            _new_service(
                repository,
                limits=_limits(max_pdf_decoded_stream_bytes=decoded_limit),
            ).ingest_document(path)

        assert not observed_library_limits or all(
            0 < limit <= decoded_limit for limit in observed_library_limits
        )
        assert {
            name: getattr(pdf_filters, name) for name in PYPDF_DECODE_LIMIT_NAMES
        } == original_library_limits
        assert _full_repository_projection_bytes(repository) == expected

    def test_flate_xref_bomb_is_bounded_during_reader_construction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pypdf
        import pypdf.filters as pdf_filters

        path = tmp_path / "flate-xref-bomb.pdf"
        decoded_limit = 1024 * 1024
        _write_flate_xref_pdf(path, b"\x00" * (12 * 1024 * 1024))
        assert path.stat().st_size < 32 * 1024

        repository = InMemoryKnowledgeRepository()
        _seed_full_repository(repository)
        expected = _full_repository_projection_bytes(repository)
        reader_entry_limits: list[dict[str, int]] = []
        structural_decode_limits: list[int] = []
        original_reader = pypdf.PdfReader
        original_decode = pdf_filters.FlateDecode.decode
        original_library_limits = {
            name: getattr(pdf_filters, name) for name in PYPDF_DECODE_LIMIT_NAMES
        }

        def guarded_reader(*args: Any, **kwargs: Any) -> Any:
            reader_entry_limits.append(
                {name: getattr(pdf_filters, name) for name in PYPDF_DECODE_LIMIT_NAMES}
            )
            return original_reader(*args, **kwargs)

        def guarded_decode(data: bytes, decode_parms: Any = None, **kwargs: Any) -> bytes:
            structural_decode_limits.append(pdf_filters.ZLIB_MAX_OUTPUT_LENGTH)
            return original_decode(data, decode_parms, **kwargs)

        monkeypatch.setattr(pypdf, "PdfReader", guarded_reader)
        monkeypatch.setattr(
            pdf_filters.FlateDecode,
            "decode",
            staticmethod(guarded_decode),
        )

        with pytest.raises(ValueError, match="PDF|decoded|stream|limit|expanded"):
            _new_service(
                repository,
                limits=_limits(max_pdf_decoded_stream_bytes=decoded_limit),
            ).ingest_document(path)

        assert reader_entry_limits and all(
            all(0 < limit <= decoded_limit for limit in snapshot.values())
            for snapshot in reader_entry_limits
        )
        assert structural_decode_limits and all(
            0 < limit <= decoded_limit for limit in structural_decode_limits
        )
        assert {
            name: getattr(pdf_filters, name) for name in PYPDF_DECODE_LIMIT_NAMES
        } == original_library_limits
        assert _full_repository_projection_bytes(repository) == expected

    @pytest.mark.parametrize(
        ("filter_name", "decoder_name", "encoder"),
        [
            (
                b"/ASCIIHexDecode",
                "ASCIIHexDecode",
                lambda data: data.hex().encode("ascii") + b">",
            ),
            (
                b"/ASCII85Decode",
                "ASCII85Decode",
                lambda data: base64.a85encode(data, adobe=True),
            ),
        ],
        ids=["asciihex", "ascii85"],
    )
    def test_ascii_structural_xref_bomb_is_guarded_and_decoder_restored(
        self,
        filter_name: bytes,
        decoder_name: str,
        encoder: Any,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import pypdf
        import pypdf.filters as pdf_filters

        path = tmp_path / f"{decoder_name.lower()}-xref-bomb.pdf"
        decoded_limit = 1024 * 1024
        _write_encoded_xref_pdf(
            path,
            b"\x00" * (2 * decoded_limit),
            filter_expression=filter_name,
            encode=encoder,
        )
        assert path.stat().st_size < 5 * 1024 * 1024

        repository = InMemoryKnowledgeRepository()
        _seed_full_repository(repository)
        expected = _full_repository_projection_bytes(repository)
        decoder_type = getattr(pdf_filters, decoder_name)
        original_decode = decoder_type.decode
        original_reader = pypdf.PdfReader
        unsafe_decode_calls = 0
        bounded_reader_entries = 0

        def watched_decode(data: bytes, decode_parms: Any = None, **kwargs: Any) -> bytes:
            nonlocal unsafe_decode_calls
            unsafe_decode_calls += 1
            return original_decode(data, decode_parms, **kwargs)

        monkeypatch.setattr(decoder_type, "decode", staticmethod(watched_decode))

        def guarded_reader(*args: Any, **kwargs: Any) -> Any:
            nonlocal bounded_reader_entries
            bounded_reader_entries += 1
            assert decoder_type.decode is not watched_decode
            return original_reader(*args, **kwargs)

        monkeypatch.setattr(pypdf, "PdfReader", guarded_reader)

        with pytest.raises(ValueError, match="PDF|ASCII|decoded|stream|limit"):
            _new_service(
                repository,
                limits=_limits(max_pdf_decoded_stream_bytes=decoded_limit),
            ).ingest_document(path)

        assert bounded_reader_entries == 1
        assert unsafe_decode_calls == 0
        assert decoder_type.decode is watched_decode
        assert _full_repository_projection_bytes(repository) == expected

    def test_structural_asciihex_flate_chain_preserves_filter_order_and_bounds(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pypdf
        import pypdf.filters as pdf_filters

        path = tmp_path / "asciihex-flate-xref-bomb.pdf"
        decoded_limit = 1024 * 1024
        _write_encoded_xref_pdf(
            path,
            b"\x00" * (12 * 1024 * 1024),
            filter_expression=b"[/ASCIIHexDecode /FlateDecode]",
            encode=lambda data: zlib.compress(data, level=9).hex().encode("ascii") + b">",
        )
        assert path.stat().st_size < 32 * 1024

        repository = InMemoryKnowledgeRepository()
        _seed_full_repository(repository)
        expected = _full_repository_projection_bytes(repository)
        original_reader = pypdf.PdfReader
        original_asciihex_decode = pdf_filters.ASCIIHexDecode.decode
        original_library_limits = {
            name: getattr(pdf_filters, name) for name in PYPDF_DECODE_LIMIT_NAMES
        }
        asciihex_calls = 0
        bounded_reader_entries = 0

        def watched_asciihex(data: bytes, decode_parms: Any = None, **kwargs: Any) -> bytes:
            nonlocal asciihex_calls
            asciihex_calls += 1
            return original_asciihex_decode(data, decode_parms, **kwargs)

        monkeypatch.setattr(
            pdf_filters.ASCIIHexDecode,
            "decode",
            staticmethod(watched_asciihex),
        )

        def guarded_reader(*args: Any, **kwargs: Any) -> Any:
            nonlocal bounded_reader_entries
            bounded_reader_entries += 1
            assert pdf_filters.ASCIIHexDecode.decode is not watched_asciihex
            assert all(
                0 < getattr(pdf_filters, name) <= decoded_limit for name in PYPDF_DECODE_LIMIT_NAMES
            )
            return original_reader(*args, **kwargs)

        monkeypatch.setattr(pypdf, "PdfReader", guarded_reader)

        with pytest.raises(ValueError, match="PDF|decoded|stream|limit"):
            _new_service(
                repository,
                limits=_limits(max_pdf_decoded_stream_bytes=decoded_limit),
            ).ingest_document(path)

        assert bounded_reader_entries == 1
        assert asciihex_calls >= 1
        assert pdf_filters.ASCIIHexDecode.decode is watched_asciihex
        assert {
            name: getattr(pdf_filters, name) for name in PYPDF_DECODE_LIMIT_NAMES
        } == original_library_limits
        assert _full_repository_projection_bytes(repository) == expected

    def test_contents_array_uses_one_cumulative_decoded_stream_budget(self, tmp_path: Path) -> None:
        path = tmp_path / "contents-array-budget.pdf"
        decoded_limit = 1024 * 1024
        _write_contents_array_pdf(path, (b" " * (700 * 1024), b" " * (700 * 1024)))
        assert path.stat().st_size < 16 * 1024

        _assert_rejected_without_mutation(
            path,
            limits=_limits(max_pdf_decoded_stream_bytes=decoded_limit),
            match="PDF|decoded|content|stream|limit",
        )

    def test_nested_form_xobject_is_included_in_decoded_stream_budget(self, tmp_path: Path) -> None:
        path = tmp_path / "nested-form-budget.pdf"
        decoded_limit = 1024 * 1024
        _write_nested_form_pdf(path, b" " * (2 * decoded_limit))
        assert path.stat().st_size < 16 * 1024

        _assert_rejected_without_mutation(
            path,
            limits=_limits(max_pdf_decoded_stream_bytes=decoded_limit),
            match="PDF|decoded|content|stream|limit",
        )

    def test_chained_content_filters_obey_decoded_stream_budget(self, tmp_path: Path) -> None:
        path = tmp_path / "chained-filter-budget.pdf"
        decoded_limit = 1024 * 1024
        _write_chained_filter_pdf(path, b" " * (12 * 1024 * 1024))
        assert path.stat().st_size < 32 * 1024

        _assert_rejected_without_mutation(
            path,
            limits=_limits(max_pdf_decoded_stream_bytes=decoded_limit),
            match="PDF|decoded|content|stream|limit",
        )

    def test_pdf_page_limit_is_enforced(self, pdf_path: Path) -> None:
        _assert_rejected_without_mutation(
            pdf_path,
            limits=_limits(max_pdf_pages=1),
            match="PDF|page|limit",
        )

    def test_pdf_text_limit_is_enforced(self, pdf_path: Path) -> None:
        _assert_rejected_without_mutation(
            pdf_path,
            limits=_limits(max_pdf_text_chars=sum(map(len, PDF_PAGE_TEXTS)) - 1),
            match="PDF|text|character|limit",
        )

    def test_pdf_text_limit_aborts_during_public_visitor_extraction(
        self, pdf_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        import pypdf

        emitted: list[str] = []
        completed_full_result = False

        class FragmentingPage:
            def extract_text(self, *, visitor_text: Any) -> str:
                nonlocal completed_full_result
                for fragment in ("1234", "5678", "must-not-be-emitted"):
                    emitted.append(fragment)
                    visitor_text(fragment, None, None, None, None)
                completed_full_result = True
                return "".join(emitted)

        class FragmentingReader:
            is_encrypted = False
            pages = [FragmentingPage()]

            def __init__(self, stream: Any, *, strict: bool) -> None:
                assert stream.read(4) == b"%PDF"
                assert strict is False

        repository = InMemoryKnowledgeRepository()
        _seed_full_repository(repository)
        before = _full_repository_projection(repository)
        monkeypatch.setattr(pypdf, "PdfReader", FragmentingReader)

        with pytest.raises(ValueError, match="PDF|text|character|limit"):
            _new_service(repository, limits=_limits(max_pdf_text_chars=7)).ingest_document(pdf_path)

        assert emitted == ["1234", "5678"]
        assert completed_full_result is False
        assert _full_repository_projection(repository) == before

    def test_epub_member_count_limit_is_enforced(self, epub_path: Path) -> None:
        _assert_rejected_without_mutation(
            epub_path,
            limits=_limits(max_epub_members=4),
            match="EPUB|member|limit",
        )

    def test_epub_expanded_size_limit_is_enforced(self, epub_path: Path) -> None:
        with ZipFile(epub_path) as archive:
            expanded_size = sum(member.file_size for member in archive.infolist())
        _assert_rejected_without_mutation(
            epub_path,
            limits=_limits(max_epub_expanded_bytes=expanded_size - 1),
            match="EPUB|expanded|size|limit",
        )

    def test_epub_compression_ratio_limit_is_enforced(self, epub_path: Path) -> None:
        with ZipFile(epub_path) as archive:
            ratios = [
                member.file_size / member.compress_size
                for member in archive.infolist()
                if member.compress_size
            ]
        assert max(ratios) > 1.0
        _assert_rejected_without_mutation(
            epub_path,
            limits=_limits(max_epub_compression_ratio=1.0),
            match="EPUB|compression|ratio|limit",
        )

    @pytest.mark.parametrize("aliased", [False, True], ids=["repeated-idref", "aliased-idrefs"])
    def test_epub_spine_item_budget_counts_repeated_and_aliased_references(
        self, aliased: bool, tmp_path: Path
    ) -> None:
        path = tmp_path / "spine-count-bypass.epub"
        if aliased:
            manifest = "".join(
                f'<item id="alias-{index}" href="z-opening.xhtml" '
                'media-type="application/xhtml+xml"/>'
                for index in range(6)
            )
            spine = "".join(f'<itemref idref="alias-{index}"/>' for index in range(6))
        else:
            manifest = (
                '<item id="opening" href="z-opening.xhtml" media-type="application/xhtml+xml"/>'
            )
            spine = '<itemref idref="opening"/>' * 6
        _write_epub(path, package=_epub_package(manifest, spine))

        # The archive itself has five physical members, so this must reach the
        # logical-spine check rather than failing archive member validation.
        with ZipFile(path) as archive:
            assert len(archive.infolist()) == 5
        _assert_rejected_without_mutation(
            path,
            limits=_limits(max_epub_members=5),
            match="EPUB|spine|segment|member|limit",
        )

    def test_epub_cumulative_read_budget_counts_repeated_member_reads(self, tmp_path: Path) -> None:
        path = tmp_path / "repeated-read-bypass.epub"
        manifest = '<item id="opening" href="z-opening.xhtml" media-type="application/xhtml+xml"/>'
        spine = '<itemref idref="opening"/>' * 4
        _write_epub(path, package=_epub_package(manifest, spine))
        with ZipFile(path) as archive:
            physical_expanded_size = sum(item.file_size for item in archive.infolist())

        _assert_rejected_without_mutation(
            path,
            limits=_limits(max_epub_expanded_bytes=physical_expanded_size),
            match="EPUB|cumulative|expanded|read|limit",
        )

    def test_epub_cumulative_text_budget_counts_aliased_spine_text(self, tmp_path: Path) -> None:
        path = tmp_path / "aliased-text-bypass.epub"
        manifest = """
<item id="first" href="z-opening.xhtml" media-type="application/xhtml+xml"/>
<item id="alias" href="z-opening.xhtml" media-type="application/xhtml+xml"/>
"""
        spine = '<itemref idref="first"/><itemref idref="alias"/>'
        chapter = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<html xmlns="http://www.w3.org/1999/xhtml"><body><h1>Large</h1><p>'
            + ("x" * 10_000_100)
            + "</p></body></html>"
        )
        _write_epub(
            path,
            package=_epub_package(manifest, spine),
            opening=chapter,
        )

        _assert_rejected_without_mutation(
            path,
            match="EPUB|cumulative|text|character|limit",
        )


class TestEpubValidation:
    def test_rejects_invalid_epub_mimetype(self, tmp_path: Path) -> None:
        path = tmp_path / "wrong-mimetype.epub"
        _write_epub(path, mimetype="application/zip")
        _assert_rejected_without_mutation(path, match="EPUB|mimetype|media type")

    def test_rejects_invalid_rootfile_media_type(self, tmp_path: Path) -> None:
        path = tmp_path / "wrong-rootfile.epub"
        _write_epub(path, rootfile_media_type="text/xml")
        _assert_rejected_without_mutation(path, match="EPUB|rootfile|media type")

    @pytest.mark.parametrize("declaration", ["<!DOCTYPE html>", "<!ENTITY xxe 'boom'>"])
    def test_rejects_dtd_and_entity_declarations(self, declaration: str, tmp_path: Path) -> None:
        path = tmp_path / "declaration.epub"
        chapter = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f"{declaration}\n"
            '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
            "<h1>Opening</h1><p>Unsafe declaration.</p></body></html>"
        )
        _write_epub(path, opening=chapter)
        _assert_rejected_without_mutation(path, match="EPUB|DTD|entity|declaration")

    def test_rejects_malformed_xhtml(self, tmp_path: Path) -> None:
        path = tmp_path / "malformed.epub"
        malformed = (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
            "<h1>Opening</h1><p>Unclosed paragraph</body></html>"
        )
        _write_epub(path, opening=malformed)
        _assert_rejected_without_mutation(path, match="EPUB|XHTML|malformed|invalid")

    @pytest.mark.parametrize("encoding", ["utf-16", "utf-32"])
    @pytest.mark.parametrize("target", ["container", "package", "xhtml"])
    def test_rejects_encoded_dtd_internal_entities_without_expansion(
        self, encoding: str, target: str, tmp_path: Path
    ) -> None:
        path = tmp_path / f"encoded-{target}-{encoding}.epub"
        encoding_label = encoding.upper()
        entity = '<!DOCTYPE root [<!ENTITY injected "EXPANSION-MARKER">]>'
        if target == "container":
            container = f"""<?xml version="1.0" encoding="{encoding_label}"?>
{entity}
<container version="1.0"
 xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
 <rootfiles><rootfile full-path="&injected;"
  media-type="application/oebps-package+xml"/></rootfiles>
</container>""".encode(encoding)
            _write_epub(path, container=container)
        elif target == "package":
            package = f"""<?xml version="1.0" encoding="{encoding_label}"?>
{entity}
<package version="3.0" unique-identifier="book-id"
 xmlns="http://www.idpf.org/2007/opf">
 <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
  <dc:identifier id="book-id">fixture</dc:identifier>
  <dc:title>&injected;</dc:title>
 </metadata>
 <manifest><item id="opening" href="z-opening.xhtml"
  media-type="application/xhtml+xml"/></manifest>
 <spine><itemref idref="opening"/></spine>
</package>""".encode(encoding)
            _write_epub(path, package=package)
        else:
            chapter = f"""<?xml version="1.0" encoding="{encoding_label}"?>
{entity}
<html xmlns="http://www.w3.org/1999/xhtml"><body>
<h1>Unsafe</h1><p>&injected;</p></body></html>""".encode(encoding)
            _write_epub(path, opening=chapter)

        _assert_rejected_without_mutation(
            path,
            match="EPUB|DTD|entity|declaration|forbidden",
        )

    @pytest.mark.parametrize("hostility", ["unsupported-compression", "encrypted"])
    def test_hostile_zip_member_errors_are_normalized_and_atomic(
        self, hostility: str, tmp_path: Path
    ) -> None:
        valid_path = tmp_path / "valid.epub"
        hostile_path = tmp_path / f"{hostility}.epub"
        _write_epub(valid_path)
        payload = valid_path.read_bytes()
        if hostility == "unsupported-compression":
            payload = _patch_zip_member_headers(
                payload,
                "OEBPS/z-opening.xhtml",
                compression=99,
            )
            match = "EPUB|unsupported|compression|invalid"
        else:
            payload = _patch_zip_member_headers(
                payload,
                "OEBPS/z-opening.xhtml",
                encrypted=True,
            )
            match = "EPUB|encrypted|unsupported|invalid"
        hostile_path.write_bytes(payload)

        _assert_rejected_without_mutation(hostile_path, match=match)


class TestAtomicFailures:
    def test_missing_file_does_not_mutate_repository(self, tmp_path: Path) -> None:
        repository = InMemoryKnowledgeRepository()
        service = _new_service(repository)

        with pytest.raises(FileNotFoundError):
            service.ingest_document(tmp_path / "missing.txt")

        assert _counts(repository) == (0, 0, 0)

    def test_unsupported_format_does_not_mutate_repository(self, tmp_path: Path) -> None:
        path = tmp_path / "unsupported.docx"
        path.write_bytes(b"not a supported citation document")
        repository = InMemoryKnowledgeRepository()
        service = _new_service(repository)

        with pytest.raises(ValueError, match="unsupported"):
            service.ingest_document(path)

        assert _counts(repository) == (0, 0, 0)

    @pytest.mark.parametrize(
        ("suffix", "payload"),
        [
            (".pdf", b"%PDF-1.7\nthis is not a complete PDF"),
            (".epub", b"this is not a ZIP archive"),
        ],
    )
    def test_corrupt_binary_document_does_not_mutate_repository(
        self, suffix: str, payload: bytes, tmp_path: Path
    ) -> None:
        path = tmp_path / f"corrupt{suffix}"
        path.write_bytes(payload)
        repository = InMemoryKnowledgeRepository()
        service = _new_service(repository)

        with pytest.raises(ValueError, match="corrupt|invalid|parse"):
            service.ingest_document(path)

        assert _counts(repository) == (0, 0, 0)
