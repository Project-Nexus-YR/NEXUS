"""Ingestion adapter, normalization and pipeline tests."""

import pytest

from nexus_knowledge.domain.document import Document
from nexus_knowledge.domain.source import Source, SourceKind
from nexus_knowledge.ingestion.adapters import (
    JsonAdapter,
    MarkdownAdapter,
    RepositoryAdapter,
    TextAdapter,
)
from nexus_knowledge.ingestion.normalization import RecursiveChunker, normalize_text


class TestNormalization:
    def test_whitespace_collapsed(self):
        assert normalize_text("a  b\n\n\n  c") == "a b\n\n c"

    def test_strips_surrounding_whitespace(self):
        assert normalize_text("  hello  ") == "hello"


class TestTextAdapter:
    def test_bytes_and_str(self):
        source = Source(title="doc", kind=SourceKind.TEXT, reference="r1")
        assert TextAdapter().read(source, b"hi")[0].text == "hi"
        assert TextAdapter().read(source, "hi")[0].text == "hi"

    def test_title_falls_back(self):
        source = Source(title="", kind=SourceKind.TEXT, reference="r1")
        assert TextAdapter().read(source, "hi")[0].title == "untitled"


class TestMarkdownAdapter:
    def test_splits_on_top_level_headings(self):
        source = Source(title="doc", kind=SourceKind.MARKDOWN, reference="r1")
        docs = MarkdownAdapter().read(source, b"# One\nbody1\n\n# Two\nbody2")
        assert len(docs) == 2
        assert docs[0].title == "One"
        assert docs[1].title == "Two"
        assert docs[0].text == "body1"
        assert docs[1].text == "body2"

    def test_no_headings_single_document(self):
        source = Source(title="doc", kind=SourceKind.MARKDOWN, reference="r1")
        docs = MarkdownAdapter().read(source, b"just text")
        assert len(docs) == 1


class TestJsonAdapter:
    def test_dict_flattens_nested(self):
        source = Source(title="cfg", kind=SourceKind.JSON, reference="r1")
        docs = JsonAdapter().read(source, b'{"company": {"name": "Acme"}, "hq": "London"}')
        assert len(docs) == 1
        assert "company.name: Acme" in docs[0].text
        assert "hq: London" in docs[0].text

    def test_list_becomes_one_document_per_item(self):
        source = Source(title="items", kind=SourceKind.JSON, reference="r1")
        docs = JsonAdapter().read(source, b'[{"name": "A"}, {"name": "B"}]')
        assert len(docs) == 2
        assert docs[1].title == "items #2"

    def test_invalid_json_falls_back_to_text(self):
        source = Source(title="bad", kind=SourceKind.JSON, reference="r1")
        docs = JsonAdapter().read(source, b"not json")
        assert docs[0].content_type == "text"


class TestRepositoryAdapter:
    def test_directory_glob(self, tmp_path):
        (tmp_path / "a.md").write_text("# A\nbody", encoding="utf-8")
        (tmp_path / "b.txt").write_text("plain", encoding="utf-8")
        (tmp_path / "skip.bin").write_bytes(b"\x00\x01")
        source = Source(title="repo", kind=SourceKind.REPOSITORY, reference=str(tmp_path))
        docs = RepositoryAdapter().read(source, tmp_path)
        assert len(docs) == 2
        assert {"a.md", "b.txt"} == {d.metadata.get("path") for d in docs}

    def test_file_payload(self, tmp_path):
        file_path = tmp_path / "a.txt"
        file_path.write_text("single", encoding="utf-8")
        source = Source(title="repo", kind=SourceKind.REPOSITORY, reference=str(file_path))
        docs = RepositoryAdapter().read(source, file_path)
        assert len(docs) == 1


class TestRecursiveChunker:
    def test_single_short_chunk(self):
        chunker = RecursiveChunker(max_chars=100, overlap=0)
        document = Document(source_id="s", title="t", content_type="text", text="hello world")
        chunks = chunker.chunk(document)
        assert len(chunks) == 1
        assert chunks[0].text == "hello world"

    def test_splits_into_multiple(self):
        chunker = RecursiveChunker(max_chars=10, overlap=2)
        document = Document(source_id="s", title="t", content_type="text", text="a " * 20)
        chunks = chunker.chunk(document)
        assert len(chunks) > 1
        assert all(len(c.text) <= 10 + chunker.overlap for c in chunks)

    def test_overlap_between_consecutive(self):
        chunker = RecursiveChunker(max_chars=10, overlap=3)
        document = Document(source_id="s", title="t", content_type="text", text="abcd efgh ijkl mnop")
        chunks = chunker.chunk(document)
        if len(chunks) > 1:
            assert chunks[0].text.endswith(chunks[1].text[:3])
