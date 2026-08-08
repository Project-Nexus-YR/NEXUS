"""GraphRAG engine tests."""


class TestEvidenceGraph:
    def test_query_grounds_entities(self, ingested_engine):
        graph = ingested_engine.graphrag("Acme Corp")
        assert graph.query == "Acme Corp"
        assert graph.entities
        assert any(e.canonical == "Acme Corp" for e in graph.entities)

    def test_collects_relations(self, ingested_engine):
        graph = ingested_engine.graphrag("Ada Lovelace Acme Corp")
        predicates = {r.predicate for r in graph.relations}
        assert "works_at" in predicates

    def test_claims_evidence_sources(self, ingested_engine):
        graph = ingested_engine.graphrag("Acme Corp")
        assert graph.claims
        assert graph.evidence
        assert graph.sources

    def test_confidence_in_range(self, ingested_engine):
        graph = ingested_engine.graphrag("Acme Corp")
        assert 0.0 <= graph.confidence <= 1.0

    def test_paths_require_two_entities(self, ingested_engine):
        graph = ingested_engine.graphrag("Ada Lovelace Acme Corp London")
        if len({e.canonical for e in graph.entities}) >= 2:
            assert graph.paths
            assert all(p.nodes for p in graph.paths)

    def test_empty_query_returns_empty(self, engine):
        graph = engine.graphrag("completely unrelated nonsense query")
        assert graph.entities == []

    def test_to_dict_shape(self, ingested_engine):
        payload = ingested_engine.graphrag("Acme Corp").to_dict()
        assert set(payload) == {
            "query",
            "entities",
            "relations",
            "claims",
            "evidence_count",
            "sources",
            "paths",
            "confidence",
        }
