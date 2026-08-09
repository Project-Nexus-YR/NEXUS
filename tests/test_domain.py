"""Domain model tests."""

import pytest

from nexus_knowledge.domain.claim import Claim, Provenance
from nexus_knowledge.domain.common import Confidence, VerificationState
from nexus_knowledge.domain.document import Chunk, Document, Span
from nexus_knowledge.domain.entity import Entity, Relation
from nexus_knowledge.domain.ids import new_id, stable_id


class TestIds:
    def test_new_id_has_prefix(self):
        value = new_id("doc")
        assert value.startswith("doc_")
        assert len(value) > len("doc_")

    def test_stable_id_is_deterministic(self):
        assert stable_id("ent", "Alice") == stable_id("ent", "Alice")
        assert stable_id("ent", "Alice") != stable_id("ent", "Bob")

    def test_new_id_collision_resistance(self):
        assert new_id("x") != new_id("x")


class TestConfidence:
    def test_valid_range(self):
        assert float(Confidence(0.5)) == 0.5

    def test_clamps_rejects_out_of_range(self):
        with pytest.raises(ValueError):
            Confidence(1.5)
        with pytest.raises(ValueError):
            Confidence(-0.1)


class TestSpan:
    def test_valid(self):
        span = Span(2, 5)
        assert span.slice("hello world") == "llo"

    def test_invalid(self):
        with pytest.raises(ValueError):
            Span(5, 2)


class TestChunk:
    def test_id_is_deterministic(self):
        doc = Document(source_id="s", title="t", content_type="text", text="x")
        assert (
            Chunk(document_id=doc.id, index=1, text="x").id
            == Chunk(document_id=doc.id, index=1, text="x").id
        )

    def test_id_derives_from_document_and_index(self):
        chunk = Chunk(document_id="doc_a", index=0, text="x")
        assert chunk.id == stable_id("chunk", "doc_a", 0)


class TestEntityRelation:
    def test_entity_canonical_defaults_to_name(self):
        entity = Entity(name="Acme")
        assert entity.canonical == "Acme"

    def test_relation_tuple(self):
        relation = Relation(subject_id="a", predicate="p", object_id="b")
        assert relation.tuple == ("a", "p", "b")

    def test_relation_defaults_to_unverified(self):
        relation = Relation(subject_id="a", predicate="p", object_id="b")
        assert relation.verification_state == VerificationState.UNVERIFIED
        assert relation.provenance == []


class TestClaim:
    def test_default_verification_state(self):
        claim = Claim(text="x")
        assert claim.verification_state == VerificationState.UNVERIFIED

    def test_equality_by_id(self):
        a = Claim(text="x")
        b = Claim(text="x")
        assert a != b
        assert a == a


class TestProvenance:
    def test_chain_fields(self):
        provenance = Provenance(
            entity_id="c1",
            evidence_ids=("e1",),
            chunk_ids=("ch1",),
            document_ids=("d1",),
            source_ids=("s1",),
        )
        assert provenance.source_ids == ("s1",)
        assert provenance.chunk_ids == ("ch1",)
