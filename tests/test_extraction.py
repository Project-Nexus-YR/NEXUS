"""Extraction adapter tests."""

from nexus_knowledge.domain.document import Chunk, Document, Span
from nexus_knowledge.extraction.deterministic import (
    GazetteerEntityExtractor,
    PatternRelationExtractor,
)
from nexus_knowledge.extraction.llm_adapters import (
    CallbackEntityExtractor,
    CallbackRelationExtractor,
)


def _chunk(text: str) -> tuple[Document, Chunk]:
    document = Document(source_id="s", title="t", content_type="text", text=text)
    chunk = Chunk(document_id=document.id, index=0, text=text, span=Span(0, len(text)))
    return document, chunk


class TestGazetteerExtractor:
    def test_single_entity(self):
        document, chunk = _chunk("Ada Lovelace founded Acme Corp.")
        extractor = GazetteerEntityExtractor({"Person": ["Ada Lovelace"]})
        entities = extractor.extract(chunk, document)
        assert [e.name for e in entities] == ["Ada Lovelace"]
        assert entities[0].entity_type == "Person"

    def test_longest_match_wins(self):
        document, chunk = _chunk("Acme Corp is a company")
        extractor = GazetteerEntityExtractor({"Company": ["Acme"], "Org": ["Acme Corp"]})
        entities = extractor.extract(chunk, document)
        assert [e.name for e in entities] == ["Acme Corp"]
        assert entities[0].entity_type == "Org"

    def test_case_insensitive(self):
        document, chunk = _chunk("acme corp")
        extractor = GazetteerEntityExtractor({"Company": ["ACME CORP"]})
        assert [e.name for e in extractor.extract(chunk, document)] == ["acme corp"]

    def test_spans_are_source_aligned(self):
        document, chunk = _chunk("X Ada Lovelace Y")
        extractor = GazetteerEntityExtractor({"Person": ["Ada Lovelace"]})
        entity = extractor.extract(chunk, document)[0]
        assert entity.span == Span(2, 14)
        assert chunk.text[entity.span.start : entity.span.end] == "Ada Lovelace"

    def test_multiple_entity_types_for_one_name(self):
        extractor = GazetteerEntityExtractor()
        extractor.add_term("London", "City")
        extractor.add_term("London", "Place")
        document, chunk = _chunk("London")
        assert extractor.extract(chunk, document)[0].entity_type == "City"


class TestPatternRelationExtractor:
    def test_extracts_relation(self):
        document, chunk = _chunk("Ada Lovelace founded Acme Corp.")
        extractor = GazetteerEntityExtractor({"Person": ["Ada Lovelace"], "Company": ["Acme Corp"]})
        entities = extractor.extract(chunk, document)
        relations = PatternRelationExtractor().extract(chunk, document, entities)
        assert len(relations) == 1
        relation = relations[0]
        assert relation.subject == "Ada Lovelace"
        assert relation.predicate == "founded"
        assert relation.object == "Acme Corp"

    def test_does_not_cross_sentence_boundary(self):
        document, chunk = _chunk("Ada Lovelace is a person. Alan Turing founded Initech.")
        extractor = GazetteerEntityExtractor(
            {
                "Person": ["Ada Lovelace", "Alan Turing"],
                "Company": ["Initech"],
            }
        )
        entities = extractor.extract(chunk, document)
        relations = PatternRelationExtractor().extract(chunk, document, entities)
        assert len(relations) == 1
        assert relations[0].subject == "Alan Turing"

    def test_skips_self_referential(self):
        document, chunk = _chunk("Acme Corp is a type of Acme Corp.")
        extractor = GazetteerEntityExtractor({"Company": ["Acme Corp"]})
        entities = extractor.extract(chunk, document)
        relations = PatternRelationExtractor().extract(chunk, document, entities)
        assert relations == []

    def test_skips_when_missing_object_entity(self):
        document, chunk = _chunk("Acme Corp develops software.")
        extractor = GazetteerEntityExtractor({"Company": ["Acme Corp"]})
        entities = extractor.extract(chunk, document)
        relations = PatternRelationExtractor().extract(chunk, document, entities)
        assert relations == []


class TestCallbackAdapters:
    def test_callback_entity_extractor(self):
        document, chunk = _chunk("hello")
        extractor = CallbackEntityExtractor(
            lambda text: [{"name": "Acme", "entity_type": "Company"}]
        )
        entities = extractor.extract(chunk, document)
        assert entities[0].name == "Acme"

    def test_callback_relation_extractor(self):
        document, chunk = _chunk("hello")
        extractor = CallbackRelationExtractor(
            lambda text, entities: [{"subject": "A", "predicate": "p", "object": "B"}]
        )
        relations = extractor.extract(chunk, document, [])
        assert (relations[0].subject, relations[0].predicate, relations[0].object) == (
            "A",
            "p",
            "B",
        )
