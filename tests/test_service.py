"""Service layer API contract tests."""

import pytest

from nexus_knowledge.domain.claim import Claim
from nexus_knowledge.domain.common import Confidence, VerificationState
from nexus_knowledge.domain.entity import Entity, Relation
from nexus_knowledge.domain.source import Source, SourceKind
from nexus_knowledge.service.engine import KnowledgeUpdate


class TestIngest:
    def test_ingest_text_updates_graph_and_indexes(self, ingested_engine):
        source = Source(title="probe", kind=SourceKind.TEXT, reference="probe/1")
        result = ingested_engine.ingest(source, "Ada Lovelace founded Acme Corp in London.")
        assert len(result.documents) == 1
        entities = {e.name for e in ingested_engine.repository.entities.all()}
        assert {"Ada Lovelace", "Acme Corp", "London"} <= entities
        assert ingested_engine.repository.chunks.count() >= 1
        assert ingested_engine.vector_store.size() >= 1

    def test_ingest_markdown(self, ingested_engine):
        source = Source(title="md", kind=SourceKind.MARKDOWN, reference="md/1")
        result = ingested_engine.ingest(source, "# Section\nAcme Corp develops software.")
        assert len(result.documents) == 1


class TestRetrieve:
    def test_returns_ranked_candidates(self, ingested_engine):
        result = ingested_engine.retrieve("who founded Acme Corp", top_k=3)
        assert result.candidates
        assert result.candidates[0].chunk.text

    def test_top_k_respected(self, ingested_engine):
        result = ingested_engine.retrieve("Acme Corp", top_k=2)
        assert len(result.candidates) <= 2


class TestQueryGraph:
    def test_filter_by_predicate(self, ingested_engine):
        relations = ingested_engine.query_graph(predicate="works_at")
        assert relations
        assert all(r["predicate"] == "works_at" for r in relations)

    def test_limit(self, ingested_engine):
        assert len(ingested_engine.query_graph(limit=1)) <= 1


class TestSubgraph:
    def test_subgraph_contains_seed(self, ingested_engine):
        acme = next(
            e for e in ingested_engine.repository.entities.all() if e.canonical == "Acme Corp"
        )
        subgraph = ingested_engine.get_subgraph([acme.id], depth=2)
        assert acme.id in subgraph.entity_ids()
        assert subgraph.edge_count() >= 1


class TestGapsAndScoring:
    def test_find_knowledge_gaps(self, ingested_engine):
        gaps = ingested_engine.find_knowledge_gaps()
        kinds = {g.kind for g in gaps}
        assert kinds  # non-empty
        assert all(g.candidate_investigations for g in gaps)

    def test_score_investigation(self, ingested_engine):
        scored = ingested_engine.score_investigation(top_k=3)
        assert scored
        assert scored[0].score > 0
        assert all(s.investigation.gap_id for s in scored)


class TestClaims:
    def test_propose_and_verify(self, ingested_engine):
        claim = ingested_engine.propose_claim(
            "Ada Lovelace founded Acme Corp",
            "Ada Lovelace",
            "founded",
            "Acme Corp",
            confidence=0.5,
        )
        assert claim.verification_state == VerificationState.UNVERIFIED
        assessment = ingested_engine.verify_claim(claim.id)
        updated = ingested_engine.repository.claims.get(claim.id)
        assert updated.verification_state == assessment.verification_state

    def test_verify_unknown_raises(self, ingested_engine):
        with pytest.raises(KeyError):
            ingested_engine.verify_claim("claim_does_not_exist")

    def test_propose_with_source_ref(self, ingested_engine):
        claim = ingested_engine.propose_claim("x", "a", "p", "b", source_ref="probe/alpha")
        assert len(claim.source_ids) == 1
        again = ingested_engine.propose_claim("y", "a", "p", "c", source_ref="probe/alpha")
        assert again.source_ids == claim.source_ids  # source reused


class TestProvenance:
    def test_provenance_resolution(self, ingested_engine):
        claim = next(c for c in ingested_engine.repository.claims.all() if c.provenance)
        response = ingested_engine.provenance(claim.id)
        assert response.claim_id == claim.id
        assert response.claim_text
        assert response.source_references
        assert response.provenance.document_ids
        assert response.provenance.chunk_ids

    def test_provenance_unknown_raises(self, ingested_engine):
        with pytest.raises(KeyError):
            ingested_engine.provenance("claim_does_not_exist")


class TestCommitUpdate:
    def test_atomic_commit(self, ingested_engine):
        entity = Entity(name="Grace Hopper", id="grace")
        relation = Relation(
            subject_id="grace",
            predicate="works_at",
            object_id="acme_id",
            confidence=Confidence(0.8),
        )
        claim = Claim(
            text="Grace Hopper works at Acme Corp",
            subject="Grace Hopper",
            predicate="works_at",
            object="Acme Corp",
        )
        update = KnowledgeUpdate(entities=[entity], relations=[relation], claims=[claim])
        receipt = ingested_engine.commit_knowledge_update(update)
        assert receipt.accepted == 3
        assert receipt.rejected == 0
        assert ingested_engine.repository.entities.get("grace") is not None

    def test_rejects_invalid(self, ingested_engine):
        update = KnowledgeUpdate(
            relations=[Relation(subject_id="", predicate="p", object_id="o")],
            claims=[Claim(text="")],
        )
        receipt = ingested_engine.commit_knowledge_update(update)
        assert receipt.accepted == 0
        assert receipt.rejected == 2
        assert len(receipt.errors) == 2


class TestSystem:
    def test_healthcheck_counts(self, ingested_engine):
        health = ingested_engine.healthcheck()
        assert health["documents"] >= 1
        assert health["chunks"] >= 1
        assert health["entities"] >= 1
        assert health["relations"] >= 1

    def test_graph_statistics(self, ingested_engine):
        stats = ingested_engine.graph_statistics()
        assert stats["num_entities"] >= 1
        assert stats["num_relations"] >= 1
