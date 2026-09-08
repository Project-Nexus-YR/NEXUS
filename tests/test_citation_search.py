"""Acceptance tests for citation-only lexical candidate retrieval."""

from __future__ import annotations

import ast
import base64
import hashlib
import inspect
import json
import math
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, get_args, get_origin, get_type_hints
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile, ZipInfo

import pytest

from nexus_knowledge.domain.document import Chunk, Document, Span
from nexus_knowledge.domain.source import Source, SourceKind
from nexus_knowledge.persistence.json_codec import (
    dumps,
    load_snapshot,
    loads,
    save_snapshot,
    to_plain,
)
from nexus_knowledge.persistence.memory import InMemoryKnowledgeRepository

TXT_TEXT = "Alpha witness evidence.\r\n\r\nBeta café — 東京.\r\n"
MARKDOWN_TEXT = (
    "# Overview\r\n\r\n"
    "Alpha witness evidence stays exact.\r\n\r\n"
    "## Details\r\n\r\n"
    "Unicode café — 東京 remains source-aligned.\r\n"
)
PDF_PAGE_TEXTS = (
    "First PDF page contains alpha witness evidence.",
    "Second PDF page contains beta citation text.",
)
EPUB_CHAPTERS = (
    ("Opening", "The opening chapter contains alpha witness evidence."),
    ("Findings", "The findings chapter contains beta citation text."),
)
FORMAT_CASES = (
    ("txt_path", "Beta", None, None, None),
    ("markdown_path", "source aligned", None, None, "Details"),
    ("pdf_path", "beta citation", 2, None, None),
    ("epub_path", "opening chapter", None, 1, "Opening"),
)
UNRELATED_REPOSITORIES = (
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

# A deterministic two-page PDF generated for the citation-ingestion acceptance suite.
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


def _symbols() -> tuple[type[Any], type[Any], type[Any], type[Any]]:
    """Resolve only through public exports so missing exports fail acceptance."""
    from nexus_knowledge.port import CitationCandidate, CitationSearchFilters, CitationSearchPort
    from nexus_knowledge.retrieval import CitationLexicalSearch

    return CitationCandidate, CitationSearchFilters, CitationSearchPort, CitationLexicalSearch


def _new_search(repository: Any) -> Any:
    return _symbols()[3](repository=repository)


def _new_filters(**values: str) -> Any:
    return _symbols()[1](**values)


def _ingest(repository: Any, path: Path) -> str:
    from nexus_knowledge.ingestion import CitationIngestionService

    return CitationIngestionService(repository=repository).ingest_document(path)


def _zip_info(name: str, compression: int = ZIP_DEFLATED) -> ZipInfo:
    info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = compression
    info.external_attr = 0o600 << 16
    return info


def _write_epub(path: Path, *, opening_xhtml: str | None = None) -> None:
    container = """<?xml version="1.0" encoding="UTF-8"?>
<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">
  <rootfiles><rootfile full-path="OEBPS/content.opf"
    media-type="application/oebps-package+xml"/></rootfiles>
</container>
"""
    package = """<?xml version="1.0" encoding="UTF-8"?>
<package version="3.0" unique-identifier="book-id"
         xmlns="http://www.idpf.org/2007/opf">
  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">
    <dc:identifier id="book-id">citation-search-fixture</dc:identifier>
    <dc:title>Citation Search Fixture</dc:title><dc:language>en</dc:language>
  </metadata>
  <manifest>
    <item id="opening" href="z-opening.xhtml" media-type="application/xhtml+xml"/>
    <item id="findings" href="a-findings.xhtml" media-type="application/xhtml+xml"/>
  </manifest>
  <spine><itemref idref="opening"/><itemref idref="findings"/></spine>
</package>
"""
    opening = (
        opening_xhtml
        or f"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><body>
<h1>{EPUB_CHAPTERS[0][0]}</h1><p>{EPUB_CHAPTERS[0][1]}</p>
</body></html>
"""
    )
    findings = f"""<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><body>
<h1>{EPUB_CHAPTERS[1][0]}</h1><p>{EPUB_CHAPTERS[1][1]}</p>
</body></html>
"""
    with ZipFile(path, "w") as archive:
        archive.writestr(_zip_info("mimetype", ZIP_STORED), "application/epub+zip")
        archive.writestr(_zip_info("META-INF/container.xml"), container)
        archive.writestr(_zip_info("OEBPS/content.opf"), package)
        archive.writestr(_zip_info("OEBPS/z-opening.xhtml"), opening)
        archive.writestr(_zip_info("OEBPS/a-findings.xhtml"), findings)


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
def pdf_path(tmp_path: Path) -> Path:
    path = tmp_path / "pages.pdf"
    path.write_bytes(PDF_BYTES)
    return path


@pytest.fixture
def epub_path(tmp_path: Path) -> Path:
    path = tmp_path / "chapters.epub"
    _write_epub(path)
    return path


def _stored(repository: Any, document_id: str) -> tuple[Any, Any, list[Any]]:
    document = repository.documents.get(document_id)
    assert document is not None
    source = repository.sources.get(document.source_id)
    assert source is not None
    chunks = repository.chunks.by_document(document_id)
    assert chunks
    return source, document, chunks


def _projection(repository: Any) -> bytes:
    payload = {
        name: [to_plain(item) for item in getattr(repository, name).all()]
        for name in ("sources", "documents", "chunks")
    }
    return json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")


def _candidate_ids(candidates: list[Any]) -> list[str]:
    return [candidate.chunk_id for candidate in candidates]


def _assert_locator_hash(metadata: dict[str, Any]) -> str:
    locator_hash = metadata["locator_hash"]
    assert isinstance(locator_hash, str)
    assert len(locator_hash) == 64
    assert all(character in "0123456789abcdef" for character in locator_hash)
    return locator_hash


def _recomputed_epub_locator_hash(
    *,
    content_hash: str,
    document_id: str,
    chunk: Chunk,
) -> str:
    """Recompute the public-data locator witness as a capable attacker could."""
    metadata = chunk.metadata
    witness = {
        "chapter": metadata["chapter"],
        "chapter_id": metadata["chapter_id"],
        "chapter_path": metadata["chapter_path"],
        "chunk_id": chunk.id,
        "content_hash": content_hash,
        "document_id": document_id,
        "section": metadata["section"],
        "section_derivation": metadata["section_derivation"],
        "section_line": metadata.get("section_line"),
        "segment_index": chunk.index,
    }
    canonical = json.dumps(
        witness,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def _decode_epub_raw_envelope(raw: str) -> bytes:
    prefix = "citation-epub-raw-v1:"
    assert raw.startswith(prefix)
    return base64.b64decode(raw.removeprefix(prefix), validate=True)


def _expected_epub_parse_limits(limits: Any) -> dict[str, int | float]:
    return {
        "version": 1,
        "max_file_bytes": limits.max_file_bytes,
        "max_members": limits.max_epub_members,
        "max_expanded_bytes": limits.max_epub_expanded_bytes,
        "max_compression_ratio": limits.max_epub_compression_ratio,
        "max_segments": 10_000,
        "max_text_chars": 20_000_000,
    }


def _run_fresh_python(script: str, **extra_environment: str) -> subprocess.CompletedProcess[str]:
    """Run an import probe without inheriting this pytest process's module cache."""
    environment = dict(os.environ)
    source_root = str(Path(__file__).parents[1] / "src")
    inherited_pythonpath = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root if not inherited_pythonpath else source_root + os.pathsep + inherited_pythonpath
    )
    environment.update(extra_environment)
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )


class TestPublicContract:
    def test_public_exports_and_exact_search_signature(self) -> None:
        candidate, filters, port, concrete = _symbols()
        from nexus_knowledge.port.citation_search import (
            CitationCandidate as ModuleCitationCandidate,
        )
        from nexus_knowledge.port.citation_search import (
            CitationSearchFilters as ModuleCitationSearchFilters,
        )
        from nexus_knowledge.port.citation_search import (
            CitationSearchPort as ModuleCitationSearchPort,
        )

        assert candidate is ModuleCitationCandidate
        assert filters is ModuleCitationSearchFilters
        assert port is ModuleCitationSearchPort
        for owner in (port, concrete):
            signature = inspect.signature(owner.search)
            parameters = list(signature.parameters.values())
            assert [item.name for item in parameters] == ["self", "query", "limit", "filters"]
            assert parameters[2].default == 20
            assert parameters[3].default is None

    def test_concrete_implements_port_shape(self, txt_path: Path) -> None:
        _, _, port, concrete = _symbols()
        repository = InMemoryKnowledgeRepository()
        _ingest(repository, txt_path)
        search: port = concrete(repository=repository)

        assert callable(search.search)
        assert search.search("Beta")

    def test_default_limit_is_twenty(self, tmp_path: Path) -> None:
        repository = InMemoryKnowledgeRepository()
        for index in range(25):
            path = tmp_path / f"document-{index:02d}.txt"
            path.write_text("shared citation token", encoding="utf-8")
            _ingest(repository, path)

        assert len(_new_search(repository).search("shared")) == 20

    def test_public_runtime_type_hints_resolve_to_meaningful_contracts(self) -> None:
        candidate, filters, port, concrete = _symbols()
        from nexus_knowledge.domain.document import Chunk
        from nexus_knowledge.port import CitationSearchRepository
        from nexus_knowledge.retrieval.lexical import LexicalRetriever

        constructor_hints = get_type_hints(concrete.__init__)
        concrete_search_hints = get_type_hints(concrete.search)
        port_search_hints = get_type_hints(port.search)
        lexical_add_hints = get_type_hints(LexicalRetriever.add_chunks)

        assert constructor_hints == {
            "repository": CitationSearchRepository,
            "return": type(None),
        }
        for hints in (concrete_search_hints, port_search_hints):
            assert hints["query"] is str
            assert hints["limit"] is int
            assert filters in get_args(hints["filters"])
            assert type(None) in get_args(hints["filters"])
            assert get_origin(hints["return"]) is list
            assert get_args(hints["return"]) == (candidate,)
        assert get_origin(lexical_add_hints["chunks"]) is list
        assert get_args(lexical_add_hints["chunks"]) == (Chunk,)
        assert lexical_add_hints["return"] is type(None)

    def test_public_narrow_repository_protocol_exposes_only_citation_stores(self) -> None:
        from nexus_knowledge.port import CitationSearchRepository
        from nexus_knowledge.port.citation_search import (
            CitationSearchRepository as ModuleCitationSearchRepository,
        )

        assert CitationSearchRepository is ModuleCitationSearchRepository
        assert getattr(CitationSearchRepository, "_is_protocol", False)
        repository_hints = get_type_hints(CitationSearchRepository)
        assert set(repository_hints) == {"sources", "documents", "chunks"}
        assert all(repository_hints[name] is not Any for name in repository_hints)


class TestExactProvenance:
    @pytest.mark.parametrize(("fixture_name", "query", "page", "chapter", "section"), FORMAT_CASES)
    def test_known_passage_reconstructs_exact_candidate(
        self,
        fixture_name: str,
        query: str,
        page: int | None,
        chapter: int | None,
        section: str | None,
        request: pytest.FixtureRequest,
    ) -> None:
        path = request.getfixturevalue(fixture_name)
        repository = InMemoryKnowledgeRepository()
        document_id = _ingest(repository, path)
        source, document, chunks = _stored(repository, document_id)

        candidates = _new_search(repository).search(query)

        assert candidates
        candidate = candidates[0]
        chunk = repository.chunks.get(candidate.chunk_id)
        assert chunk is not None and chunk in chunks
        assert candidate.source_id == source.id
        assert candidate.source_title == source.title
        assert candidate.source_reference == source.reference
        assert candidate.source_kind == source.kind
        assert candidate.source_metadata == source.metadata
        assert candidate.source_metadata is not source.metadata
        assert candidate.document_id == document.id
        assert candidate.document_title == document.title
        assert candidate.document_content_type == document.content_type
        assert candidate.chunk_id == chunk.id
        assert candidate.segment_id == chunk.id
        assert candidate.segment_index == chunk.index
        assert candidate.text == chunk.text
        assert candidate.char_start == chunk.span.start
        assert candidate.char_end == chunk.span.end
        assert candidate.text == document.text[candidate.char_start : candidate.char_end]
        assert candidate.content_hash == source.metadata["content_hash"]
        assert candidate.page == page
        assert candidate.chapter == chapter
        assert candidate.section == section
        assert candidate.line_start == chunk.metadata["line_start"]
        assert candidate.line_end == chunk.metadata["line_end"]
        assert candidate.segment_metadata == chunk.metadata
        assert candidate.segment_metadata is not chunk.metadata
        assert isinstance(candidate.score, float)
        assert math.isfinite(candidate.score) and candidate.score > 0.0

    @pytest.mark.parametrize("query", ["café", "東京"])
    def test_unicode_passages_are_lexically_retrievable(self, query: str, txt_path: Path) -> None:
        repository = InMemoryKnowledgeRepository()
        _ingest(repository, txt_path)

        candidates = _new_search(repository).search(query)

        assert candidates
        assert query.casefold() in candidates[0].text.casefold()

    def test_returned_metadata_is_detached_from_repository(self, markdown_path: Path) -> None:
        repository = InMemoryKnowledgeRepository()
        document_id = _ingest(repository, markdown_path)
        source, _, chunks = _stored(repository, document_id)
        candidate = _new_search(repository).search("source aligned")[0]
        source_before = dict(source.metadata)
        chunk_before = dict(chunks[1].metadata)

        candidate.source_metadata["mutated"] = True
        candidate.segment_metadata["mutated"] = True

        assert source.metadata == source_before
        assert chunks[1].metadata == chunk_before


class TestCitationBoundary:
    def test_only_full_consistent_citation_witness_is_searchable(self, tmp_path: Path) -> None:
        repository = InMemoryKnowledgeRepository()
        genuine_path = tmp_path / "genuine.txt"
        genuine_path.write_text("genuine boundary witness", encoding="utf-8")
        genuine_id = _ingest(repository, genuine_path)
        genuine_source, genuine_document, genuine_chunks = _stored(repository, genuine_id)
        genuine = genuine_chunks[0]

        ordinary_source = Source(
            title="ordinary", kind=SourceKind.TEXT, reference="ordinary://source", id="src_ordinary"
        )
        ordinary_document = Document(
            source_id=ordinary_source.id,
            title="ordinary",
            content_type="text",
            text="ordinary boundary witness",
            raw="ordinary boundary witness",
            id="doc_ordinary",
        )
        ordinary_chunk = Chunk(
            document_id=ordinary_document.id,
            index=0,
            text=ordinary_document.text,
            span=Span(0, len(ordinary_document.text)),
        )
        repository.sources.save(ordinary_source)
        repository.documents.save(ordinary_document)
        repository.chunks.save(ordinary_chunk)

        marker_source = Source(
            title="marker", kind=SourceKind.TEXT, reference="marker://source", id="src_marker"
        )
        marker_document = Document(
            source_id=marker_source.id,
            title="marker",
            content_type="text",
            text="marker boundary witness",
            raw="marker boundary witness",
            metadata={"ingestion_version": "citation-ingestion-v1"},
            id="doc_marker",
        )
        marker_chunk = Chunk(
            document_id=marker_document.id,
            index=0,
            text=marker_document.text,
            span=Span(0, len(marker_document.text)),
            metadata={"ingestion_version": "citation-ingestion-v1"},
        )
        repository.sources.save(marker_source)
        repository.documents.save(marker_document)
        repository.chunks.save(marker_chunk)

        forged_source = Source(
            title="forged",
            kind=genuine_source.kind,
            reference="forged://source",
            metadata=dict(genuine_source.metadata),
            id="src_forged",
        )
        forged_document = Document(
            source_id=forged_source.id,
            title="forged",
            content_type=genuine_document.content_type,
            text="forged boundary witness",
            raw="forged boundary witness",
            metadata=dict(genuine_document.metadata),
            id="doc_forged",
        )
        forged_chunk = Chunk(
            document_id=forged_document.id,
            index=0,
            text=forged_document.text,
            span=Span(0, len(forged_document.text)),
            metadata=dict(genuine.metadata),
        )
        repository.sources.save(forged_source)
        repository.documents.save(forged_document)
        repository.chunks.save(forged_chunk)

        orphan = Chunk(
            document_id="doc_missing",
            index=0,
            text="orphan boundary witness",
            span=Span(0, 23),
            metadata=dict(genuine.metadata),
        )
        repository.chunks.save(orphan)
        missing_source_document = Document(
            source_id="src_missing",
            title="missing source",
            content_type="text",
            text="missing source boundary witness",
            raw="missing source boundary witness",
            metadata=dict(genuine_document.metadata),
            id="doc_missing_source",
        )
        missing_source_chunk = Chunk(
            document_id=missing_source_document.id,
            index=0,
            text=missing_source_document.text,
            span=Span(0, len(missing_source_document.text)),
            metadata=dict(genuine.metadata),
        )
        repository.documents.save(missing_source_document)
        repository.chunks.save(missing_source_chunk)

        candidates = _new_search(repository).search("boundary witness", limit=20)

        assert _candidate_ids(candidates) == [genuine.id]

    def test_corrupt_witness_is_skipped_without_hiding_valid_candidate(
        self, tmp_path: Path
    ) -> None:
        repository = InMemoryKnowledgeRepository()
        first = tmp_path / "first.txt"
        second = tmp_path / "second.txt"
        first.write_text("shared valid citation", encoding="utf-8")
        second.write_text("shared corrupt citation", encoding="utf-8")
        valid_id = _ingest(repository, first)
        corrupt_id = _ingest(repository, second)
        valid = repository.chunks.by_document(valid_id)[0]
        corrupt = repository.chunks.by_document(corrupt_id)[0]
        corrupt.metadata["char_end"] += 1

        candidates = _new_search(repository).search("shared citation")

        assert _candidate_ids(candidates) == [valid.id]

    def test_source_kind_is_verified_not_trusted_from_chunk_metadata(self, txt_path: Path) -> None:
        repository = InMemoryKnowledgeRepository()
        document_id = _ingest(repository, txt_path)
        chunk = repository.chunks.by_document(document_id)[0]
        chunk.metadata["source_kind"] = SourceKind.PDF

        assert _new_search(repository).search("Beta") == []

    @pytest.mark.parametrize(
        ("attribute", "malformed"),
        [
            pytest.param("span", "not-a-span", id="string-span"),
            pytest.param("span", object(), id="object-span"),
            pytest.param("metadata", "not-metadata", id="string-metadata"),
            pytest.param("metadata", object(), id="object-metadata"),
            pytest.param("index", "0", id="string-index"),
            pytest.param("index", True, id="boolean-index"),
        ],
    )
    def test_malformed_chunk_shape_is_skipped_without_hiding_valid_results(
        self, attribute: str, malformed: Any, tmp_path: Path
    ) -> None:
        repository = InMemoryKnowledgeRepository()
        valid_path = tmp_path / "valid.txt"
        corrupt_path = tmp_path / "corrupt.txt"
        valid_path.write_text("shared parserguard valid", encoding="utf-8")
        corrupt_path.write_text("shared parserguard corrupt", encoding="utf-8")
        valid_document_id = _ingest(repository, valid_path)
        corrupt_document_id = _ingest(repository, corrupt_path)
        valid_chunk = repository.chunks.by_document(valid_document_id)[0]
        corrupt_chunk = repository.chunks.by_document(corrupt_document_id)[0]
        setattr(corrupt_chunk, attribute, malformed)

        candidates = _new_search(repository).search("shared parserguard")

        assert _candidate_ids(candidates) == [valid_chunk.id]

    @pytest.mark.parametrize(
        "mutations",
        [
            pytest.param(
                {1: {"page": 1, "page_index": 0}},
                id="duplicate-page-locator",
            ),
            pytest.param(
                {1: {"page_count": 3}},
                id="inconsistent-page-count",
            ),
            pytest.param(
                {
                    0: {"page": 2, "page_index": 1},
                    1: {"page": 1, "page_index": 0},
                },
                id="decreasing-page-locators",
            ),
        ],
    )
    def test_pdf_page_sequence_corruption_rejects_entire_aggregate(
        self, mutations: dict[int, dict[str, int]], pdf_path: Path
    ) -> None:
        repository = InMemoryKnowledgeRepository()
        document_id = _ingest(repository, pdf_path)
        chunks = repository.chunks.by_document(document_id)
        assert len(chunks) == 2
        assert "beta citation text" in chunks[1].text.lower()
        for chunk_index, changes in mutations.items():
            chunks[chunk_index].metadata.update(changes)

        search = _new_search(repository)

        assert search.search("beta citation text") == []
        assert search.search("alpha evidence") == []

    @pytest.mark.parametrize(
        (
            "filename",
            "content",
            "chunk_index",
            "expected_section",
            "unique_query",
        ),
        [
            pytest.param(
                "heading-only.md",
                "# Authentic Heading\nshared sectionguard headingunique\n",
                0,
                "Authentic Heading",
                "headingunique",
                id="atx-heading",
            ),
            pytest.param(
                "no-heading.md",
                "shared sectionguard noheadingunique\n",
                0,
                "no-heading",
                "noheadingunique",
                id="filename-stem-without-heading",
            ),
            pytest.param(
                "preamble-source.md",
                "shared sectionguard preambleunique\n# Later\nlater body\n",
                0,
                "preamble-source",
                "preambleunique",
                id="filename-stem-preamble",
            ),
        ],
    )
    def test_forged_markdown_section_rejects_only_corrupt_aggregate(
        self,
        filename: str,
        content: str,
        chunk_index: int,
        expected_section: str,
        unique_query: str,
        tmp_path: Path,
    ) -> None:
        repository = InMemoryKnowledgeRepository()
        corrupt_path = tmp_path / filename
        corrupt_path.write_text(content, encoding="utf-8")
        corrupt_document_id = _ingest(repository, corrupt_path)
        corrupt_chunks = repository.chunks.by_document(corrupt_document_id)
        corrupt_chunk = corrupt_chunks[chunk_index]
        assert corrupt_chunk.metadata["section"] == expected_section
        assert corrupt_chunk.id in _candidate_ids(_new_search(repository).search(unique_query))

        valid_path = tmp_path / "independent-valid.txt"
        valid_path.write_text("shared sectionguard independentvalid", encoding="utf-8")
        valid_document_id = _ingest(repository, valid_path)
        valid_chunk = repository.chunks.by_document(valid_document_id)[0]
        corrupt_chunk.metadata["section"] = "forged section label"

        search = _new_search(repository)

        assert search.search(unique_query) == []
        assert _candidate_ids(search.search("shared sectionguard")) == [valid_chunk.id]

    def test_forged_epub_heading_section_rejects_only_corrupt_aggregate(
        self, epub_path: Path, tmp_path: Path
    ) -> None:
        repository = InMemoryKnowledgeRepository()
        corrupt_document_id = _ingest(repository, epub_path)
        corrupt_chunks = repository.chunks.by_document(corrupt_document_id)
        opening = corrupt_chunks[0]
        assert opening.metadata["section"] == "Opening"
        assert opening.text.splitlines()[0] == "Opening"
        assert opening.id in _candidate_ids(_new_search(repository).search("opening chapter"))

        valid_path = tmp_path / "independent-valid.txt"
        valid_path.write_text("alpha witness evidence independentvalid", encoding="utf-8")
        valid_document_id = _ingest(repository, valid_path)
        valid_chunk = repository.chunks.by_document(valid_document_id)[0]
        opening.metadata["section"] = "Forged EPUB Section"

        search = _new_search(repository)

        assert search.search("opening chapter") == []
        assert _candidate_ids(search.search("alpha witness evidence")) == [valid_chunk.id]

    def test_epub_section_cannot_be_a_cross_line_flattened_substring(self, epub_path: Path) -> None:
        repository = InMemoryKnowledgeRepository()
        document_id = _ingest(repository, epub_path)
        chunks = repository.chunks.by_document(document_id)
        opening = chunks[0]
        canonical_lines = {line.strip() for line in opening.text.splitlines() if line.strip()}
        flattened = " ".join(opening.text.split())
        forged_section = "Opening The opening chapter"
        chapter_stem = Path(opening.metadata["chapter_path"]).stem
        assert forged_section in flattened
        assert forged_section not in canonical_lines
        assert forged_section != chapter_stem
        opening.metadata["section"] = forged_section

        assert _new_search(repository).search("opening chapter") == []

    def test_epub_heading_section_witness_is_exact_idempotent_and_id_stable(
        self, epub_path: Path
    ) -> None:
        repository = InMemoryKnowledgeRepository()
        document_id = _ingest(repository, epub_path)
        chunks = repository.chunks.by_document(document_id)
        opening = chunks[0]
        metadata_before = json.loads(json.dumps(opening.metadata, sort_keys=True))
        chunk_ids_before = [chunk.id for chunk in chunks]
        source_id_before = repository.documents.get(document_id).source_id
        section_line = opening.metadata["section_line"]

        assert opening.metadata["section_derivation"] == "heading"
        locator_hash_before = _assert_locator_hash(opening.metadata)
        assert isinstance(section_line, int) and not isinstance(section_line, bool)
        assert section_line > 0
        assert opening.text.splitlines()[section_line - 1].strip() == opening.metadata["section"]

        replayed_id = _ingest(repository, epub_path)
        replayed_chunks = repository.chunks.by_document(replayed_id)

        assert replayed_id == document_id
        assert repository.documents.get(replayed_id).source_id == source_id_before
        assert [chunk.id for chunk in replayed_chunks] == chunk_ids_before
        assert replayed_chunks[0].metadata == metadata_before
        assert _assert_locator_hash(replayed_chunks[0].metadata) == locator_hash_before

    @pytest.mark.parametrize(
        "corruption",
        [
            "body-line-with-original-witness",
            "remove-section-line",
            "move-section-line-to-body",
            "remove-derivation",
            "wrong-derivation",
            "remove-locator-hash",
            "replace-locator-hash",
        ],
    )
    def test_epub_heading_section_witness_corruption_rejects_entire_aggregate(
        self, corruption: str, epub_path: Path
    ) -> None:
        repository = InMemoryKnowledgeRepository()
        document_id = _ingest(repository, epub_path)
        opening = repository.chunks.by_document(document_id)[0]
        assert opening.metadata["section_derivation"] == "heading"
        assert opening.metadata["section_line"] == 1
        _assert_locator_hash(opening.metadata)
        body_line = opening.text.splitlines()[1].strip()
        assert body_line == EPUB_CHAPTERS[0][1]

        if corruption == "body-line-with-original-witness":
            opening.metadata["section"] = body_line
        elif corruption == "remove-section-line":
            opening.metadata.pop("section_line")
        elif corruption == "move-section-line-to-body":
            opening.metadata["section_line"] = 2
        elif corruption == "remove-derivation":
            opening.metadata.pop("section_derivation")
        elif corruption == "wrong-derivation":
            opening.metadata["section_derivation"] = "chapter_path_stem"
        elif corruption == "remove-locator-hash":
            opening.metadata.pop("locator_hash")
        else:
            opening.metadata["locator_hash"] = "0" * 64

        search = _new_search(repository)

        assert search.search("opening chapter") == []
        assert search.search("findings chapter") == []

    def test_epub_path_stem_section_witness_is_explicit_idempotent_and_id_stable(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "fallback.epub"
        opening_xhtml = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><body>
<p>Fallback opening contains fallbackunique evidence.</p>
</body></html>
"""
        _write_epub(path, opening_xhtml=opening_xhtml)
        repository = InMemoryKnowledgeRepository()
        document_id = _ingest(repository, path)
        chunks = repository.chunks.by_document(document_id)
        opening = chunks[0]
        metadata_before = json.loads(json.dumps(opening.metadata, sort_keys=True))
        chunk_ids_before = [chunk.id for chunk in chunks]
        source_id_before = repository.documents.get(document_id).source_id

        assert opening.metadata["section"] == "z-opening"
        assert opening.metadata["section_derivation"] == "chapter_path_stem"
        assert "section_line" not in opening.metadata
        locator_hash_before = _assert_locator_hash(opening.metadata)

        replayed_id = _ingest(repository, path)
        replayed_chunks = repository.chunks.by_document(replayed_id)

        assert replayed_id == document_id
        assert repository.documents.get(replayed_id).source_id == source_id_before
        assert [chunk.id for chunk in replayed_chunks] == chunk_ids_before
        assert replayed_chunks[0].metadata == metadata_before
        assert _assert_locator_hash(replayed_chunks[0].metadata) == locator_hash_before

    @pytest.mark.parametrize(
        "corruption",
        [
            "body-line-with-original-witness",
            "remove-derivation",
            "wrong-derivation",
            "invent-section-line",
            "remove-locator-hash",
            "replace-locator-hash",
        ],
    )
    def test_epub_path_stem_section_witness_corruption_rejects_entire_aggregate(
        self, corruption: str, tmp_path: Path
    ) -> None:
        path = tmp_path / "fallback.epub"
        opening_xhtml = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><body>
<p>Fallback opening contains fallbackunique evidence.</p>
</body></html>
"""
        _write_epub(path, opening_xhtml=opening_xhtml)
        repository = InMemoryKnowledgeRepository()
        document_id = _ingest(repository, path)
        opening = repository.chunks.by_document(document_id)[0]
        assert opening.metadata["section"] == "z-opening"
        assert opening.metadata["section_derivation"] == "chapter_path_stem"
        _assert_locator_hash(opening.metadata)

        if corruption == "body-line-with-original-witness":
            opening.metadata["section"] = opening.text.splitlines()[0].strip()
        elif corruption == "remove-derivation":
            opening.metadata.pop("section_derivation")
        elif corruption == "wrong-derivation":
            opening.metadata["section_derivation"] = "heading"
        elif corruption == "invent-section-line":
            opening.metadata["section_line"] = 1
        elif corruption == "remove-locator-hash":
            opening.metadata.pop("locator_hash")
        else:
            opening.metadata["locator_hash"] = "f" * 64

        assert _new_search(repository).search("fallbackunique") == []

    @pytest.mark.parametrize(
        "corruption",
        [
            "coordinated-path-and-section",
            "chapter-id",
            "chapter-number",
        ],
    )
    def test_epub_locator_tuple_mutation_is_rejected_by_unchanged_hash(
        self, corruption: str, tmp_path: Path
    ) -> None:
        path = tmp_path / "fallback.epub"
        opening_xhtml = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><body>
<p>Fallback opening contains fallbackunique evidence.</p>
</body></html>
"""
        _write_epub(path, opening_xhtml=opening_xhtml)
        repository = InMemoryKnowledgeRepository()
        document_id = _ingest(repository, path)
        opening = repository.chunks.by_document(document_id)[0]
        locator_hash = _assert_locator_hash(opening.metadata)
        assert opening.metadata["section_derivation"] == "chapter_path_stem"

        if corruption == "coordinated-path-and-section":
            opening.metadata["chapter_path"] = "OEBPS/forged.xhtml"
            opening.metadata["section"] = "forged"
            assert Path(opening.metadata["chapter_path"]).stem == opening.metadata["section"]
        elif corruption == "chapter-id":
            opening.metadata["chapter_id"] = "forged-opening"
        else:
            opening.metadata["chapter"] = 2
        assert opening.metadata["locator_hash"] == locator_hash

        search = _new_search(repository)

        assert search.search("fallbackunique") == []
        assert search.search("findings chapter") == []

    def test_epub_raw_envelope_round_trips_exact_package_and_stable_identity(
        self, epub_path: Path, tmp_path: Path
    ) -> None:
        original_bytes = epub_path.read_bytes()
        repository = InMemoryKnowledgeRepository()
        document_id = _ingest(repository, epub_path)
        source, document, chunks = _stored(repository, document_id)
        source_id = source.id
        chunk_ids = [chunk.id for chunk in chunks]
        raw_before = document.raw

        assert _decode_epub_raw_envelope(document.raw) == original_bytes
        assert hashlib.sha256(original_bytes).hexdigest() == source.metadata["content_hash"]
        assert document.metadata["epub_parse_limits"]["max_segments"] == 10_000
        assert document.metadata["epub_parse_limits"]["max_text_chars"] == 20_000_000
        for chunk in chunks:
            assert chunk.metadata["locator_hash"] == _recomputed_epub_locator_hash(
                content_hash=source.metadata["content_hash"],
                document_id=document_id,
                chunk=chunk,
            )

        replayed_id = _ingest(repository, epub_path)
        replayed_source, replayed_document, replayed_chunks = _stored(repository, replayed_id)
        assert replayed_id == document_id
        assert replayed_source.id == source_id
        assert [chunk.id for chunk in replayed_chunks] == chunk_ids
        assert replayed_document.raw == raw_before
        assert _decode_epub_raw_envelope(replayed_document.raw) == original_bytes

        snapshot = tmp_path / "epub-raw-snapshot.json"
        save_snapshot(repository, snapshot)
        restored = load_snapshot(snapshot)
        restored_source, restored_document, restored_chunks = _stored(restored, document_id)
        assert restored_source.id == source_id
        assert [chunk.id for chunk in restored_chunks] == chunk_ids
        assert restored_document.raw == raw_before
        assert _decode_epub_raw_envelope(restored_document.raw) == original_bytes

    def test_epub_search_reuses_exact_persisted_ingestion_limits(
        self, epub_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from nexus_knowledge._citation_epub import EpubParseLimits
        from nexus_knowledge.ingestion import CitationIngestionLimits, CitationIngestionService
        from nexus_knowledge.retrieval import citation as citation_retrieval

        payload_size = len(epub_path.read_bytes())
        limits = CitationIngestionLimits(
            max_file_bytes=payload_size,
            max_epub_members=8,
            max_epub_expanded_bytes=64 * 1024,
            max_epub_compression_ratio=37.5,
        )
        expected_policy = _expected_epub_parse_limits(limits)
        repository = InMemoryKnowledgeRepository()
        ingestion = CitationIngestionService(repository=repository, limits=limits)
        document_id = ingestion.ingest_document(epub_path)
        source, document, chunks = _stored(repository, document_id)
        source_id = source.id
        chunk_ids = [chunk.id for chunk in chunks]

        assert document.metadata["epub_parse_limits"] == expected_policy
        monkeypatch.setattr(
            citation_retrieval,
            "_EPUB_PROVENANCE_LIMITS",
            EpubParseLimits(max_file_bytes=payload_size - 1),
        )
        assert _candidate_ids(_new_search(repository).search("alpha witness")) == [chunks[0].id]

        replayed_id = ingestion.ingest_document(epub_path)
        replayed_source, replayed_document, replayed_chunks = _stored(repository, replayed_id)
        assert replayed_id == document_id
        assert replayed_source.id == source_id
        assert [chunk.id for chunk in replayed_chunks] == chunk_ids
        assert replayed_document.metadata["epub_parse_limits"] == expected_policy

        snapshot = tmp_path / "epub-limit-policy-snapshot.json"
        save_snapshot(repository, snapshot)
        restored = load_snapshot(snapshot)
        restored_source, restored_document, restored_chunks = _stored(restored, document_id)
        assert restored_source.id == source_id
        assert [chunk.id for chunk in restored_chunks] == chunk_ids
        assert restored_document.metadata["epub_parse_limits"] == expected_policy
        assert _candidate_ids(_new_search(restored).search("alpha witness")) == [chunks[0].id]

    @pytest.mark.parametrize(
        "corruption",
        [
            "missing-policy",
            "non-mapping-policy",
            "missing-field",
            "extra-field",
            "wrong-version",
            "bool-file-limit",
            "too-small-file-limit",
            "too-small-member-limit",
            "zero-expanded-limit",
            "nan-ratio",
            "bool-segment-limit",
            "smaller-segment-limit",
            "larger-segment-limit",
            "string-text-limit",
            "smaller-text-limit",
            "larger-text-limit",
        ],
    )
    def test_epub_persisted_limit_corruption_fails_closed(
        self, corruption: str, epub_path: Path
    ) -> None:
        from nexus_knowledge.ingestion import CitationIngestionLimits

        repository = InMemoryKnowledgeRepository()
        document_id = _ingest(repository, epub_path)
        _, document, chunks = _stored(repository, document_id)
        policy: Any = _expected_epub_parse_limits(CitationIngestionLimits())
        document.metadata["epub_parse_limits"] = policy

        if corruption == "missing-policy":
            document.metadata.pop("epub_parse_limits")
        elif corruption == "non-mapping-policy":
            document.metadata["epub_parse_limits"] = "defaults"
        elif corruption == "missing-field":
            policy.pop("max_members")
        elif corruption == "extra-field":
            policy["unexpected"] = 1
        elif corruption == "wrong-version":
            policy["version"] = 2
        elif corruption == "bool-file-limit":
            policy["max_file_bytes"] = True
        elif corruption == "too-small-file-limit":
            policy["max_file_bytes"] = len(epub_path.read_bytes()) - 1
        elif corruption == "too-small-member-limit":
            policy["max_members"] = 1
        elif corruption == "zero-expanded-limit":
            policy["max_expanded_bytes"] = 0
        elif corruption == "nan-ratio":
            policy["max_compression_ratio"] = math.nan
        elif corruption == "bool-segment-limit":
            policy["max_segments"] = False
        elif corruption == "smaller-segment-limit":
            policy["max_segments"] = 9_999
        elif corruption == "larger-segment-limit":
            policy["max_segments"] = 10**12
        elif corruption == "string-text-limit":
            policy["max_text_chars"] = "20000000"
        elif corruption == "smaller-text-limit":
            policy["max_text_chars"] = 19_999_999
        else:
            policy["max_text_chars"] = 10**12

        search = _new_search(repository)
        assert search.search("opening chapter") == []
        assert search.search("findings chapter") == []

    @pytest.mark.parametrize("fixture_name", ["txt_path", "markdown_path", "pdf_path"])
    def test_non_epub_raw_remains_canonical_text(
        self, fixture_name: str, request: pytest.FixtureRequest
    ) -> None:
        path = request.getfixturevalue(fixture_name)
        repository = InMemoryKnowledgeRepository()
        document_id = _ingest(repository, path)
        source, document, _ = _stored(repository, document_id)

        assert source.kind != SourceKind.EPUB
        assert document.raw == document.text
        assert "epub_parse_limits" not in document.metadata

    def test_epub_body_line_with_recomputed_locator_hash_is_rejected(self, epub_path: Path) -> None:
        repository = InMemoryKnowledgeRepository()
        document_id = _ingest(repository, epub_path)
        source, _, chunks = _stored(repository, document_id)
        opening = chunks[0]
        original_hash = _assert_locator_hash(opening.metadata)
        body_line = opening.text.splitlines()[1].strip()
        assert body_line == EPUB_CHAPTERS[0][1]

        opening.metadata["section"] = body_line
        opening.metadata["section_line"] = 2
        opening.metadata["locator_hash"] = _recomputed_epub_locator_hash(
            content_hash=source.metadata["content_hash"],
            document_id=document_id,
            chunk=opening,
        )
        assert opening.metadata["locator_hash"] != original_hash

        search = _new_search(repository)
        assert search.search("opening chapter") == []
        assert search.search("findings chapter") == []

    def test_epub_coordinated_path_section_and_hash_mutation_is_rejected(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "fallback.epub"
        opening_xhtml = """<?xml version="1.0" encoding="UTF-8"?>
<html xmlns="http://www.w3.org/1999/xhtml"><body>
<p>Fallback opening contains fallbackunique evidence.</p>
</body></html>
"""
        _write_epub(path, opening_xhtml=opening_xhtml)
        repository = InMemoryKnowledgeRepository()
        document_id = _ingest(repository, path)
        source, _, chunks = _stored(repository, document_id)
        opening = chunks[0]
        original_hash = _assert_locator_hash(opening.metadata)
        assert opening.metadata["section_derivation"] == "chapter_path_stem"

        opening.metadata["chapter_path"] = "OEBPS/forged.xhtml"
        opening.metadata["section"] = "forged"
        opening.metadata["locator_hash"] = _recomputed_epub_locator_hash(
            content_hash=source.metadata["content_hash"],
            document_id=document_id,
            chunk=opening,
        )
        assert opening.metadata["locator_hash"] != original_hash
        assert Path(opening.metadata["chapter_path"]).stem == opening.metadata["section"]

        search = _new_search(repository)
        assert search.search("fallbackunique") == []
        assert search.search("findings chapter") == []

    def test_epub_altered_raw_payload_is_rejected_fail_closed(self, epub_path: Path) -> None:
        repository = InMemoryKnowledgeRepository()
        document_id = _ingest(repository, epub_path)
        source, document, _ = _stored(repository, document_id)
        original_bytes = _decode_epub_raw_envelope(document.raw)
        assert hashlib.sha256(original_bytes).hexdigest() == source.metadata["content_hash"]
        altered_bytes = original_bytes[:-1] + bytes([original_bytes[-1] ^ 1])
        document.raw = "citation-epub-raw-v1:" + base64.b64encode(altered_bytes).decode("ascii")
        assert hashlib.sha256(altered_bytes).hexdigest() != source.metadata["content_hash"]

        search = _new_search(repository)
        assert search.search("opening chapter") == []
        assert search.search("findings chapter") == []


@pytest.mark.parametrize("suffix", [".txt", ".md"])
@pytest.mark.parametrize("bom", ["", "\ufeff"])
@pytest.mark.parametrize("line_ending", ["\n", "\r\n", "\r"])
def test_exact_utf8_source_hash_rejects_coordinated_text_snapshot_corruption(
    tmp_path: Path, suffix: str, bom: str, line_ending: str
) -> None:
    text = f"{bom}Accepted evidence.{line_ending}Café 東京.{line_ending}"
    path = tmp_path / f"source{suffix}"
    path.write_bytes(text.encode("utf-8"))
    repository = InMemoryKnowledgeRepository()
    document_id = _ingest(repository, path)
    source, document, chunks = _stored(repository, document_id)
    search = _new_search(repository)
    assert search.search("Accepted")
    assert search.candidates()
    assert document.raw.encode("utf-8") == path.read_bytes()
    original_hash = source.metadata["content_hash"]

    document.raw = document.text = text.replace("Accepted", "Rejected")
    for chunk in chunks:
        chunk.text = chunk.text.replace("Accepted", "Rejected")
    snapshot = tmp_path / "forged.json"
    save_snapshot(repository, snapshot)
    restored = load_snapshot(snapshot)
    restored_search = _new_search(restored)

    assert source.metadata["content_hash"] == original_hash
    assert restored_search.candidates() == []
    assert restored_search.search("Rejected") == []


@pytest.mark.parametrize("empty_heading", ["#   ", "## \t", "###### \t \t"])
def test_empty_markdown_heading_uses_searchable_path_section_without_changing_text(
    tmp_path: Path, empty_heading: str
) -> None:
    text = f"{empty_heading}\r\nSearchable passage.\r\n# Named\r\nOther passage.\r\n"
    path = tmp_path / "blank-heading.md"
    path.write_bytes(text.encode("utf-8"))
    repository = InMemoryKnowledgeRepository()
    document_id = _ingest(repository, path)
    _, document, chunks = _stored(repository, document_id)
    expected_split = text.index("# Named")

    assert document.raw == document.text == text
    assert [chunk.text for chunk in chunks] == [text[:expected_split], text[expected_split:]]
    assert [(chunk.span.start, chunk.span.end) for chunk in chunks] == [
        (0, expected_split),
        (expected_split, len(text)),
    ]
    assert [chunk.metadata["section"] for chunk in chunks] == ["blank-heading", "Named"]
    assert len(_new_search(repository).candidates()) == 2
    assert _new_search(repository).search("Searchable")[0].section == "blank-heading"


@pytest.mark.parametrize("heading", ["Chapter\nOne", "Chapter <br/> One"])
def test_multiline_epub_heading_uses_explicit_fallback_without_changing_canonical_text(
    tmp_path: Path, heading: str
) -> None:
    path = tmp_path / "multiline-heading.epub"
    opening_xhtml = (
        '<html xmlns="http://www.w3.org/1999/xhtml"><body>'
        f"<h1>{heading}</h1><p>Searchable passage.</p></body></html>"
    )
    _write_epub(path, opening_xhtml=opening_xhtml)
    repository = InMemoryKnowledgeRepository()
    document_id = _ingest(repository, path)
    source, document, chunks = _stored(repository, document_id)
    opening = chunks[0]
    expected_opening = "Chapter\nOne\nSearchable passage.\n"
    expected_text = expected_opening + "\n".join(EPUB_CHAPTERS[1])
    metadata_before = dict(opening.metadata)

    assert document.text == expected_text
    assert opening.text == expected_opening
    assert (opening.span.start, opening.span.end) == (0, len(expected_opening))
    assert opening.metadata["section"] == "z-opening"
    assert opening.metadata["section_derivation"] == "chapter_path_stem"
    assert "section_line" not in opening.metadata
    _assert_locator_hash(opening.metadata)
    assert len(_new_search(repository).candidates()) == 2
    assert _new_search(repository).search("Searchable")[0].section == "z-opening"
    assert _ingest(repository, path) == document_id
    assert repository.chunks.by_document(document_id)[0].metadata == metadata_before

    opening.metadata.update(section="Chapter One", section_derivation="heading", section_line=1)
    opening.metadata["locator_hash"] = _recomputed_epub_locator_hash(
        content_hash=source.metadata["content_hash"], document_id=document_id, chunk=opening
    )
    assert _new_search(repository).candidates() == []


class TestFilteringAndRanking:
    def test_source_document_and_kind_filters_apply_before_limit(self, tmp_path: Path) -> None:
        repository = InMemoryKnowledgeRepository()
        allowed_path = tmp_path / "allowed.md"
        allowed_path.write_text(
            "# Allowed\nneedle " + "padding " * 200,
            encoding="utf-8",
        )
        allowed_document_id = _ingest(repository, allowed_path)
        allowed_source, _, allowed_chunks = _stored(repository, allowed_document_id)
        for index in range(6):
            path = tmp_path / f"strong-{index}.txt"
            path.write_text("needle needle needle", encoding="utf-8")
            _ingest(repository, path)
        search = _new_search(repository)
        expected_id = allowed_chunks[0].id

        filter_values = {
            "source_id": allowed_source.id,
            "document_id": allowed_document_id,
            "source_kind": SourceKind.MARKDOWN,
        }
        for filters in (
            {"source_id": allowed_source.id},
            {"document_id": allowed_document_id},
            {"source_kind": SourceKind.MARKDOWN},
            filter_values,
            _new_filters(**filter_values),
        ):
            candidates = search.search("needle", limit=1, filters=filters)
            assert _candidate_ids(candidates) == [expected_id]

    def test_valid_nonexistent_filters_return_empty(self, txt_path: Path) -> None:
        repository = InMemoryKnowledgeRepository()
        _ingest(repository, txt_path)
        search = _new_search(repository)

        for filters in (
            {"source_id": "src_missing"},
            {"document_id": "doc_missing"},
            {"source_kind": "nonexistent-kind"},
        ):
            assert search.search("Beta", filters=filters) == []

    def test_ties_use_explicit_stable_order_independent_of_insertion(self, tmp_path: Path) -> None:
        paths = [tmp_path / "zeta.txt", tmp_path / "alpha.txt", tmp_path / "middle.txt"]
        for path in paths:
            path.write_text("identical tie witness", encoding="utf-8")
        repositories = [InMemoryKnowledgeRepository(), InMemoryKnowledgeRepository()]
        for path in paths:
            _ingest(repositories[0], path)
        for path in reversed(paths):
            _ingest(repositories[1], path)

        first = _new_search(repositories[0]).search("identical tie")
        second = _new_search(repositories[1]).search("identical tie")

        assert first == second
        assert first == sorted(
            first,
            key=lambda candidate: (
                -candidate.score,
                candidate.source_id,
                candidate.document_id,
                candidate.segment_index,
                candidate.chunk_id,
            ),
        )

    def test_snapshot_restore_preserves_candidates_scores_and_order(self, tmp_path: Path) -> None:
        repository = InMemoryKnowledgeRepository()
        for name in ("first.txt", "second.txt"):
            path = tmp_path / name
            path.write_text("snapshot lexical witness", encoding="utf-8")
            _ingest(repository, path)
        expected = _new_search(repository).search("snapshot witness")
        snapshot = tmp_path / "snapshot.json"

        save_snapshot(repository, snapshot)
        restored = load_snapshot(snapshot)

        assert _new_search(restored).search("snapshot witness") == expected

    def test_duplicate_query_tokens_do_not_change_candidates_or_scores(
        self, txt_path: Path
    ) -> None:
        repository = InMemoryKnowledgeRepository()
        _ingest(repository, txt_path)
        search = _new_search(repository)

        assert search.search("Alpha") == search.search("Alpha alpha ALPHA")


class TestFreshnessAndIsolation:
    def test_index_observes_additions_after_first_search(self, tmp_path: Path) -> None:
        repository = InMemoryKnowledgeRepository()
        search = _new_search(repository)
        assert search.search("newlyadded") == []
        path = tmp_path / "added.txt"
        path.write_text("newlyadded citation witness", encoding="utf-8")
        document_id = _ingest(repository, path)
        expected = repository.chunks.by_document(document_id)[0]

        assert _candidate_ids(search.search("newlyadded")) == [expected.id]

    def test_index_observes_deletions_after_first_search(self, txt_path: Path) -> None:
        repository = InMemoryKnowledgeRepository()
        document_id = _ingest(repository, txt_path)
        chunk = repository.chunks.by_document(document_id)[0]
        search = _new_search(repository)
        assert search.search("Beta")

        assert repository.chunks.delete(chunk.id)

        assert search.search("Beta") == []

    def test_index_observes_same_count_delete_and_add(self, tmp_path: Path) -> None:
        repository = InMemoryKnowledgeRepository()
        first_path = tmp_path / "first.txt"
        first_path.write_text("firsttoken citation", encoding="utf-8")
        first_id = _ingest(repository, first_path)
        first_chunk = repository.chunks.by_document(first_id)[0]
        search = _new_search(repository)
        assert search.search("firsttoken")

        assert repository.chunks.delete(first_chunk.id)
        second_path = tmp_path / "second.txt"
        second_path.write_text("secondtoken citation", encoding="utf-8")
        second_id = _ingest(repository, second_path)
        second_chunk = repository.chunks.by_document(second_id)[0]
        assert repository.chunks.count() == 1

        assert search.search("firsttoken") == []
        assert _candidate_ids(search.search("secondtoken")) == [second_chunk.id]

    def test_search_is_read_only_and_accepts_narrow_repository(self, txt_path: Path) -> None:
        backing = InMemoryKnowledgeRepository()
        _ingest(backing, txt_path)
        before = _projection(backing)

        class NarrowRepository:
            sources = backing.sources
            documents = backing.documents
            chunks = backing.chunks

            def __getattr__(self, name: str) -> Any:
                if name in UNRELATED_REPOSITORIES:
                    raise AssertionError(f"citation search accessed unrelated store: {name}")
                raise AttributeError(name)

        candidates = _new_search(NarrowRepository()).search("Beta")

        assert candidates
        assert _projection(backing) == before

    def test_search_reads_one_chunk_snapshot_without_per_document_scans(
        self, tmp_path: Path
    ) -> None:
        backing = InMemoryKnowledgeRepository()
        expected_ids: set[str] = set()
        for index in range(8):
            path = tmp_path / f"snapshot-{index}.txt"
            path.write_text(f"shared traversal witness {index}", encoding="utf-8")
            document_id = _ingest(backing, path)
            expected_ids.add(backing.chunks.by_document(document_id)[0].id)

        class SnapshotChunkStore:
            def __init__(self) -> None:
                self.all_calls = 0
                self.by_document_calls = 0

            def all(self) -> list[Any]:
                self.all_calls += 1
                if self.all_calls > 1:
                    raise AssertionError("citation search read the chunk snapshot more than once")
                return backing.chunks.all()

            def by_document(self, document_id: str) -> list[Any]:
                self.by_document_calls += 1
                raise AssertionError(
                    f"citation search performed a per-document chunk scan: {document_id}"
                )

        chunks = SnapshotChunkStore()

        class SnapshotRepository:
            sources = backing.sources
            documents = backing.documents

            def __init__(self) -> None:
                self.chunks = chunks

        candidates = _new_search(SnapshotRepository()).search("shared traversal", limit=20)

        assert set(_candidate_ids(candidates)) == expected_ids
        assert chunks.all_calls == 1
        assert chunks.by_document_calls == 0


class TestPublicSerialization:
    def test_candidate_round_trips_through_public_json_codec(self, markdown_path: Path) -> None:
        repository = InMemoryKnowledgeRepository()
        _ingest(repository, markdown_path)
        candidate = _new_search(repository).search("source aligned")[0]

        restored = loads(dumps(candidate))

        assert restored == candidate
        assert restored.segment_id == candidate.segment_id == candidate.chunk_id
        assert isinstance(restored.source_metadata, dict)
        assert isinstance(restored.segment_metadata, dict)
        assert restored.source_metadata == candidate.source_metadata
        assert restored.segment_metadata == candidate.segment_metadata

    def test_filters_round_trip_through_public_json_codec(self) -> None:
        filters = _new_filters(
            source_id="src_public",
            document_id="doc_public",
            source_kind=SourceKind.MARKDOWN,
        )

        restored = loads(dumps(filters))

        assert restored == filters
        assert restored.source_id == "src_public"
        assert restored.document_id == "doc_public"
        assert restored.source_kind == SourceKind.MARKDOWN

    def test_candidate_and_filters_encode_as_json_primitive_payloads(self, txt_path: Path) -> None:
        repository = InMemoryKnowledgeRepository()
        _ingest(repository, txt_path)
        candidate = _new_search(repository).search("Beta")[0]
        filters = _new_filters(source_id=candidate.source_id)

        candidate_payload = json.loads(dumps(candidate))
        filter_payload = json.loads(dumps(filters))

        assert candidate_payload["$type"] == "citationcandidate"
        assert candidate_payload["chunk_id"] == candidate.chunk_id
        assert candidate_payload["source_metadata"] == candidate.source_metadata
        assert candidate_payload["segment_metadata"] == candidate.segment_metadata
        assert "segment_id" not in candidate_payload
        assert filter_payload == {
            "$type": "citationsearchfilters",
            "document_id": None,
            "source_id": candidate.source_id,
            "source_kind": None,
        }
        json.dumps({"candidate": candidate_payload, "filters": filter_payload})


class TestValidation:
    @pytest.mark.parametrize("query", [None, 1, 1.5, object()])
    def test_non_string_query_is_rejected(self, query: Any) -> None:
        with pytest.raises(TypeError):
            _new_search(InMemoryKnowledgeRepository()).search(query)

    @pytest.mark.parametrize("query", ["", "   ", "---", "___"])
    def test_empty_or_tokenless_query_returns_empty(self, query: str) -> None:
        assert _new_search(InMemoryKnowledgeRepository()).search(query) == []

    @pytest.mark.parametrize("limit", [True, False, 1.5, "2", None])
    def test_non_integer_limit_is_rejected(self, limit: Any) -> None:
        with pytest.raises(TypeError):
            _new_search(InMemoryKnowledgeRepository()).search("query", limit=limit)

    @pytest.mark.parametrize("limit", [0, -1, -100])
    def test_non_positive_limit_is_rejected(self, limit: int) -> None:
        with pytest.raises(ValueError):
            _new_search(InMemoryKnowledgeRepository()).search("query", limit=limit)

    @pytest.mark.parametrize("filters", [[], "source", 3, object()])
    def test_non_mapping_or_filter_object_is_rejected(self, filters: Any) -> None:
        with pytest.raises(TypeError):
            _new_search(InMemoryKnowledgeRepository()).search("query", filters=filters)

    def test_unknown_filter_key_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="unknown|unsupported|filter"):
            _new_search(InMemoryKnowledgeRepository()).search(
                "query", filters={"source_reference": "reference"}
            )

    @pytest.mark.parametrize(
        "filters",
        [
            {"source_id": None},
            {"source_id": 1},
            {"source_id": ""},
            {"document_id": "  "},
            {"source_kind": []},
        ],
    )
    def test_invalid_filter_values_are_rejected(self, filters: Any) -> None:
        with pytest.raises((TypeError, ValueError)):
            _new_search(InMemoryKnowledgeRepository()).search("query", filters=filters)

    def test_rejected_calls_do_not_mutate_repository(self, txt_path: Path) -> None:
        repository = InMemoryKnowledgeRepository()
        _ingest(repository, txt_path)
        before = _projection(repository)
        search = _new_search(repository)

        for call in (
            lambda: search.search(None),
            lambda: search.search("query", limit=0),
            lambda: search.search("query", filters={"unknown": "value"}),
        ):
            with pytest.raises((TypeError, ValueError)):
                call()
            assert _projection(repository) == before


class TestDependencyAndCompatibilityFences:
    @pytest.mark.parametrize(
        "import_statement",
        [
            "from nexus_knowledge.retrieval import CitationLexicalSearch",
            "from nexus_knowledge.retrieval.citation import CitationLexicalSearch",
        ],
    )
    def test_citation_search_imports_when_unrelated_capabilities_are_blocked(
        self, import_statement: str
    ) -> None:
        probe = _run_fresh_python(
            """
            import importlib.abc
            import os
            import sys
            from typing import get_args, get_origin, get_type_hints

            blocked_prefixes = (
                "nexus_knowledge.domain.entity",
                "nexus_knowledge.embedding",
                "nexus_knowledge.extraction",
                "nexus_knowledge.graph",
                "nexus_knowledge.port.embeddings",
                "nexus_knowledge.port.extractors",
                "nexus_knowledge.port.reranker",
                "nexus_knowledge.port.vector_store",
                "nexus_knowledge.retrieval.entity",
                "nexus_knowledge.retrieval.entity_retrieval",
                "nexus_knowledge.retrieval.features",
                "nexus_knowledge.retrieval.graphrag",
                "nexus_knowledge.retrieval.hybrid",
                "nexus_knowledge.retrieval.rerank",
                "nexus_knowledge.retrieval.vector_graph",
                "nexus_runtime",
            )

            def is_blocked(fullname):
                return any(
                    fullname == prefix or fullname.startswith(prefix + ".")
                    for prefix in blocked_prefixes
                )

            class BlockedCapabilityFinder(importlib.abc.MetaPathFinder):
                def find_spec(self, fullname, path=None, target=None):
                    del path, target
                    if is_blocked(fullname):
                        raise ModuleNotFoundError(
                            f"blocked unrelated citation-search capability: {fullname}"
                        )
                    return None

            sys.meta_path.insert(0, BlockedCapabilityFinder())
            exec(os.environ["CITATION_IMPORT_STATEMENT"])
            assert CitationLexicalSearch.__name__ == "CitationLexicalSearch"
            from nexus_knowledge.domain.document import Chunk
            from nexus_knowledge.port.citation_search import (
                CitationCandidate,
                CitationSearchFilters,
                CitationSearchPort,
                CitationSearchRepository,
            )
            from nexus_knowledge.retrieval.lexical import LexicalRetriever

            constructor_hints = get_type_hints(CitationLexicalSearch.__init__)
            concrete_hints = get_type_hints(CitationLexicalSearch.search)
            port_hints = get_type_hints(CitationSearchPort.search)
            lexical_hints = get_type_hints(LexicalRetriever.add_chunks)
            assert constructor_hints["repository"] is CitationSearchRepository
            assert constructor_hints["return"] is type(None)
            for hints in (concrete_hints, port_hints):
                assert hints["query"] is str
                assert hints["limit"] is int
                assert CitationSearchFilters in get_args(hints["filters"])
                assert type(None) in get_args(hints["filters"])
                assert get_origin(hints["return"]) is list
                assert get_args(hints["return"]) == (CitationCandidate,)
            assert get_origin(lexical_hints["chunks"]) is list
            assert get_args(lexical_hints["chunks"]) == (Chunk,)
            assert lexical_hints["return"] is type(None)
            unexpectedly_loaded = sorted(name for name in sys.modules if is_blocked(name))
            assert not unexpectedly_loaded, unexpectedly_loaded
            """,
            CITATION_IMPORT_STATEMENT=import_statement,
        )

        assert probe.returncode == 0, probe.stdout + probe.stderr

    def test_legacy_retrieval_exports_remain_lazy_and_resolve_to_existing_types(self) -> None:
        probe = _run_fresh_python(
            """
            import importlib
            import sys

            legacy = {
                "DeterministicReranker": (
                    "nexus_knowledge.retrieval.rerank",
                    "DeterministicReranker",
                ),
                "EntityIndex": ("nexus_knowledge.retrieval.entity", "EntityIndex"),
                "EntityRetriever": (
                    "nexus_knowledge.retrieval.entity_retrieval",
                    "EntityRetriever",
                ),
                "EvidenceGraph": ("nexus_knowledge.retrieval.graphrag", "EvidenceGraph"),
                "FeatureExtractor": ("nexus_knowledge.retrieval.features", "FeatureExtractor"),
                "GraphRAGEngine": ("nexus_knowledge.retrieval.graphrag", "GraphRAGEngine"),
                "GraphRetriever": ("nexus_knowledge.retrieval.vector_graph", "GraphRetriever"),
                "HybridRetriever": ("nexus_knowledge.retrieval.hybrid", "HybridRetriever"),
                "LexicalRetriever": ("nexus_knowledge.retrieval.lexical", "LexicalRetriever"),
                "MethodLatency": ("nexus_knowledge.retrieval.observability", "MethodLatency"),
                "QueryAnalysis": ("nexus_knowledge.retrieval.query", "QueryAnalysis"),
                "RankedCandidate": ("nexus_knowledge.retrieval.hybrid", "RankedCandidate"),
                "ReciprocalRankFusion": (
                    "nexus_knowledge.retrieval.fusion",
                    "ReciprocalRankFusion",
                ),
                "RetrievalHit": ("nexus_knowledge.retrieval.lexical", "RetrievalHit"),
                "RetrievalResult": ("nexus_knowledge.retrieval.hybrid", "RetrievalResult"),
                "RetrievalTrace": ("nexus_knowledge.retrieval.observability", "RetrievalTrace"),
                "VectorRetriever": ("nexus_knowledge.retrieval.vector_graph", "VectorRetriever"),
                "analyze_query": ("nexus_knowledge.retrieval.query", "analyze_query"),
            }
            legacy_modules = {module_name for module_name, _ in legacy.values()}

            retrieval = importlib.import_module("nexus_knowledge.retrieval")
            eager = sorted(legacy_modules.intersection(sys.modules))
            assert not eager, f"legacy retrieval modules loaded eagerly: {eager}"
            assert "CitationLexicalSearch" in retrieval.__all__
            assert set(legacy).issubset(retrieval.__all__)

            citation_type = retrieval.CitationLexicalSearch
            assert citation_type.__name__ == "CitationLexicalSearch"
            eager_after_citation = sorted(legacy_modules.intersection(sys.modules))
            assert not eager_after_citation, (
                f"citation export loaded legacy retrieval modules: {eager_after_citation}"
            )

            for public_name, (module_name, symbol_name) in legacy.items():
                expected = getattr(importlib.import_module(module_name), symbol_name)
                assert getattr(retrieval, public_name) is expected
            """
        )

        assert probe.returncode == 0, probe.stdout + probe.stderr

    def test_citation_search_has_no_semantic_or_pipeline_dependencies(self) -> None:
        concrete = _symbols()[3]
        module = inspect.getmodule(concrete)
        assert module is not None and module.__file__ is not None
        source_path = Path(module.__file__)
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
            "embedding",
            "embeddings",
            "entity",
            "extraction",
            "graph",
            "graphrag",
            "llm",
            "rerank",
            "reranker",
            "runtime",
            "vector",
            "vector_store",
        }

        assert not {
            imported
            for imported in imported_modules
            if forbidden_parts.intersection(imported.lower().split("."))
        }

    def test_existing_lexical_retriever_behavior_is_unchanged(self) -> None:
        from nexus_knowledge.embedding.hashing import tokenize
        from nexus_knowledge.retrieval.lexical import LexicalRetriever

        retriever = LexicalRetriever()
        chunk = Chunk(document_id="doc_existing", index=0, text="alpha café 東京")
        retriever.add_chunks([chunk])
        single = retriever.search(["alpha"])
        duplicate = retriever.search(["alpha", "alpha"])

        assert [hit.object_id for hit in single] == [chunk.id]
        assert duplicate[0].score == pytest.approx(single[0].score * 2.0)
        assert tokenize("café 東京") == ["caf"]
        assert retriever.search(["東京"]) == []
