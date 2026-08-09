"""Persistence: JSON codec and repository round trips."""

import pytest

from nexus_knowledge.domain.claim import Claim, Evidence
from nexus_knowledge.domain.common import Confidence, VerificationState
from nexus_knowledge.domain.document import Chunk, Document, Span
from nexus_knowledge.domain.entity import Entity, Relation
from nexus_knowledge.domain.source import Source, SourceKind
from nexus_knowledge.persistence.json_codec import dumps, load_snapshot, loads, save_snapshot
from nexus_knowledge.persistence.memory import InMemoryKnowledgeRepository


def _populated_repository() -> InMemoryKnowledgeRepository:
    repo = InMemoryKnowledgeRepository()
    source = Source(title="t", kind=SourceKind.TEXT, reference="r", id="src_1")
    repo.sources.save(source)
    document = Document(
        source_id=source.id,
        title="Doc",
        content_type="text",
        text="Ada works at Acme.",
        id="doc_1",
    )
    repo.documents.save(document)
    chunk = Chunk(document_id=document.id, index=0, text="Ada works at Acme.", span=Span(0, 20))
    chunk_id = chunk.id
    repo.chunks.save(chunk)
    ada = Entity(name="Ada", id="ada")
    acme = Entity(name="Acme", id="acme")
    repo.entities.save(ada)
    repo.entities.save(acme)
    relation = Relation(
        subject_id="ada",
        predicate="works_at",
        object_id="acme",
        confidence=Confidence(0.95),
        provenance=[chunk_id],
        id="rel_1",
    )
    repo.relations.save(relation)
    claim = Claim(
        text="Ada works at Acme",
        subject="Ada",
        predicate="works_at",
        object="Acme",
        supporting_evidence=["ev_1"],
        verification_state=VerificationState.SUPPORTED,
        id="claim_1",
    )
    repo.claims.save(claim)
    evidence = Evidence(
        claim_id="claim_1",
        chunk_id=chunk_id,
        document_id="doc_1",
        text="Ada works at Acme.",
        span=Span(0, 20),
        role="support",
        id="ev_1",
    )
    repo.evidence.save(evidence)
    return repo


class TestCodec:
    def test_plain_types_roundtrip(self):
        assert loads(dumps({"a": 1, "b": [1, 2]})) == {"a": 1, "b": [1, 2]}

    def test_claim_roundtrip(self):
        repo = _populated_repository()
        claim = repo.claims.get("claim_1")
        restored = loads(dumps(claim))
        assert restored.id == claim.id
        assert restored.verification_state == VerificationState.SUPPORTED
        assert restored.subject == "Ada"

    def test_relation_roundtrip(self):
        repo = _populated_repository()
        relation = repo.relations.get("rel_1")
        restored = loads(dumps(relation))
        assert restored.subject_id == "ada"
        assert float(restored.confidence) == 0.95
        assert restored.provenance == [repo.chunks.all()[0].id]

    def test_chunk_roundtrip_preserves_span(self):
        repo = _populated_repository()
        chunk = repo.chunks.all()[0]
        restored = loads(dumps(chunk))
        assert restored.id == chunk.id
        assert restored.span == Span(0, 20)

    def test_unknown_type_raises(self):
        with pytest.raises(TypeError):
            dumps(object())

    def test_unknown_payload_raises_on_load(self):
        with pytest.raises(TypeError):
            loads({"__nexus_type__": "MissingClass"})


class TestRepository:
    def test_crud(self):
        repo = InMemoryKnowledgeRepository()
        source = repo.sources.save(Source(title="t", kind=SourceKind.TEXT, reference="r"))
        assert repo.sources.get(source.id) == source
        assert repo.sources.delete(source.id) is True
        assert repo.sources.get(source.id) is None

    def test_by_subject_index(self):
        repo = _populated_repository()
        assert [c.id for c in repo.claims.by_subject("Ada")] == ["claim_1"]

    def test_by_claim_evidence_index(self):
        repo = _populated_repository()
        assert [e.id for e in repo.evidence.by_claim("claim_1")] == ["ev_1"]

    def test_entities_by_name(self):
        repo = _populated_repository()
        assert repo.entities.by_name("acme").id == "acme"

    def test_snapshot_roundtrip(self, tmp_path):
        repo = _populated_repository()
        path = tmp_path / "kb.json"
        save_snapshot(repo, path)
        restored = load_snapshot(path)
        assert restored.entities.by_name("acme").id == "acme"
        assert restored.claims.get("claim_1").id == "claim_1"
        assert restored.relations.get("rel_1").id == "rel_1"

    def test_snapshot_replaces_contents(self, tmp_path):
        repo = _populated_repository()
        path = tmp_path / "kb.json"
        save_snapshot(repo, path)
        restored = load_snapshot(path)
        restored.entities.delete("ada")
        save_snapshot(restored, path)
        again = load_snapshot(path)
        assert again.entities.get("ada") is None
