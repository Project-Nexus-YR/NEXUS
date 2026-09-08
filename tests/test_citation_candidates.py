"""Independent, validated citation candidate enumeration."""

from dataclasses import replace
from pathlib import Path

import pytest

from nexus_knowledge import CitationCandidatePort
from nexus_knowledge.ingestion import CitationIngestionService
from nexus_knowledge.persistence.memory import InMemoryKnowledgeRepository
from nexus_knowledge.retrieval import CitationLexicalSearch


def _corpus(tmp_path: Path):
    repository = InMemoryKnowledgeRepository()
    ingestion = CitationIngestionService(repository)
    for name, text in (
        ("first.txt", "He wept continuously."),
        ("second.md", "# Opening\nGrieving alone.\n# End\nHe departed."),
    ):
        path = tmp_path / name
        path.write_text(text)
        ingestion.ingest_document(path)
    return repository, CitationLexicalSearch(repository)


def test_candidates_have_no_query_or_lexical_gate(tmp_path: Path) -> None:
    repository, search = _corpus(tmp_path)
    assert isinstance(search, CitationCandidatePort)
    assert search.search("sad") == []
    candidates = search.candidates()
    assert len(candidates) == 3
    assert {candidate.chunk_id for candidate in candidates} == {
        chunk.id for chunk in repository.chunks.all()
    }
    assert all(candidate.score == 0.0 for candidate in candidates)
    assert candidates == sorted(
        candidates, key=lambda c: (c.source_id, c.document_id, c.segment_index, c.chunk_id)
    )
    candidates[0].source_metadata["tampered"] = True
    assert "tampered" not in search.candidates()[0].source_metadata


def test_candidates_apply_all_filters_and_validate_them(tmp_path: Path) -> None:
    _, search = _corpus(tmp_path)
    target = next(c for c in search.candidates() if c.source_kind == "text")
    for filters in (
        {"source_id": target.source_id},
        {"document_id": target.document_id},
        {"source_kind": "text"},
    ):
        assert search.candidates(filters) == [target]
    assert search.candidates({"document_id": "missing"}) == []
    for filters in ({"query": "sad"}, {"source_id": ""}, {"source_id": None}):
        with pytest.raises((TypeError, ValueError)):
            search.candidates(filters)


def test_corrupt_aggregate_is_excluded_from_independent_candidates(tmp_path: Path) -> None:
    repository, search = _corpus(tmp_path)
    chunk = next(c for c in repository.chunks.all() if c.text == "He wept continuously.")
    repository.chunks.save(replace(chunk, text="Hallucinated replacement"))
    assert all(c.document_id != chunk.document_id for c in search.candidates())
