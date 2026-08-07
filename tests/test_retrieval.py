"""Retrieval pipeline tests: fusion, features, reranking, trace."""

import pytest

from nexus_knowledge.domain.document import Chunk
from nexus_knowledge.port.reranker import RerankCandidate
from nexus_knowledge.retrieval.entity import EntityIndex
from nexus_knowledge.retrieval.fusion import ReciprocalRankFusion
from nexus_knowledge.retrieval.hybrid import HybridRetriever
from nexus_knowledge.retrieval.lexical import LexicalRetriever, RetrievalHit
from nexus_knowledge.retrieval.query import analyze_query


def _chunks():
    chunks = []
    texts = [
        "Ada Lovelace founded Acme Corp",
        "Acme Corp is located in London",
        "Alan Turing worked at Bletchley Park",
    ]
    for i, text in enumerate(texts):
        chunks.append(Chunk(document_id="doc", index=i, text=text))
    return chunks


class TestQueryAnalysis:
    def test_entity_detection(self):
        analysis = analyze_query("Acme Corp", entity_ids=["e1"], entity_names=["Acme Corp"])
        assert analysis.entity_names == ("Acme Corp",)
        assert analysis.has_entities()

    def test_tokens(self):
        analysis = analyze_query("Who works at Acme?", entity_ids=[], entity_names=[])
        assert "acme" in analysis.tokens

    def test_filters(self):
        analysis = analyze_query("q", filters={"source": "s1"})
        assert analysis.filters == {"source": "s1"}


class TestLexicalRetriever:
    def test_bm25_scores(self):
        retriever = LexicalRetriever()
        retriever.add_chunks(_chunks())
        hits = retriever.search(["acme"], top_k=3)
        assert hits
        assert isinstance(hits[0], RetrievalHit)
        assert hits[0].score > 0

    def test_unknown_token_no_hits(self):
        retriever = LexicalRetriever()
        retriever.add_chunks(_chunks())
        assert retriever.search(["zzzz"], top_k=3) == []


class TestFusion:
    def test_rrf_prefers_present_in_all(self):
        fusion = ReciprocalRankFusion()
        lists = [
            [RetrievalHit("a", 1.0, "m1"), RetrievalHit("b", 0.5, "m1")],
            [RetrievalHit("b", 1.0, "m2"), RetrievalHit("a", 0.2, "m2")],
        ]
        fused = fusion.fuse(lists)
        assert fused[0][0] in {"a", "b"}
        assert set(fused[0][2]) == {"m1", "m2"}

    def test_method_weights(self):
        fusion = ReciprocalRankFusion(method_weights={"m1": 2.0})
        lists = [
            [RetrievalHit("a", 1.0, "m1")],
            [RetrievalHit("b", 1.0, "m2")],
        ]
        fused = fusion.fuse(lists)
        assert fused[0][0] == "a"


class TestEntityIndex:
    def test_lookup_by_canonical(self, ingested_engine):
        index = EntityIndex(ingested_engine.repository.entities)
        match = index.match("Acme Corp")
        assert match
        assert match[0].canonical == "Acme Corp"

    def test_lookup_unknown(self, ingested_engine):
        index = EntityIndex(ingested_engine.repository.entities)
        assert index.match("NoSuchThing") == []


class TestHybridRetriever:
    def test_unknown_method_rejected(self, ingested_engine):
        with pytest.raises(ValueError):
            HybridRetriever(
                ingested_engine.repository,
                ingested_engine.graph,
                ingested_engine.vector_store,
                ingested_engine.embedder,
                active_methods=("bogus",),
            )

    def test_retrieve_includes_all_methods(self, ingested_engine):
        result = ingested_engine.retrieve("Acme Corp located in London")
        methods = {m.method for m in result.trace.method_results}
        assert methods == {"lexical", "vector", "entity", "graph"}
        assert result.candidates
        assert result.analysis.entity_names == ("Acme Corp", "London")

    def test_method_configuration(self, ingested_engine):
        retriever = HybridRetriever(
            ingested_engine.repository,
            ingested_engine.graph,
            ingested_engine.vector_store,
            ingested_engine.embedder,
            active_methods=("lexical",),
        )
        result = retriever.retrieve("Acme Corp")
        assert {m.method for m in result.trace.method_results} == {"lexical"}

    def test_rerank_sorts_descending(self, ingested_engine):
        result = ingested_engine.retrieve("Acme Corp")
        scores = [c.score for c in result.candidates]
        assert scores == sorted(scores, reverse=True)

    def test_trace_fields(self, ingested_engine):
        result = ingested_engine.retrieve("Acme Corp")
        trace = result.trace.to_dict()
        assert trace["request_id"]
        assert trace["total_ms"] >= 0
        assert "method_results" in trace
        assert trace["reranker"]

    def test_metadata_filter_passthrough(self, ingested_engine):
        result = ingested_engine.retrieve("Acme Corp", metadata_filter={"source": "nope"})
        assert result.analysis.filters == {"source": "nope"}

    def test_candidate_to_dict(self, ingested_engine):
        result = ingested_engine.retrieve("Acme Corp")
        payload = result.to_dict()
        assert payload["query"] == "Acme Corp"
        assert payload["entities"]
        assert payload["candidates"]
