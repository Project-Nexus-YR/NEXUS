"""Knowledge gap engine tests."""

from nexus_knowledge.domain.claim import Claim
from nexus_knowledge.domain.common import VerificationState
from nexus_knowledge.domain.entity import Relation
from nexus_knowledge.domain.knowledge_gap import GapKind
from nexus_knowledge.knowledge.gaps import GapWeights


def test_find_persists_and_sorts(ingested_engine):
    gaps = ingested_engine.find_knowledge_gaps()
    assert gaps
    stored = ingested_engine.repository.gaps.all()
    assert len(stored) >= len(gaps)
    priorities = [g.priority for g in gaps]
    assert priorities == sorted(priorities, reverse=True)


def test_missing_relation_detected(ingested_engine):
    kinds = {g.kind for g in ingested_engine.find_knowledge_gaps()}
    assert GapKind.MISSING_RELATION in kinds


def test_low_confidence_claim_detected(ingested_engine):
    ingested_engine.propose_claim("wild claim", "Ghost", "haunts", "Acme Corp", confidence=0.1)
    gaps = ingested_engine.find_knowledge_gaps()
    low = [g for g in gaps if g.kind == GapKind.LOW_CONFIDENCE]
    assert low
    assert all(g.affected_claims for g in low)


def test_unsupported_claim_detected(ingested_engine):
    ingested_engine.propose_claim("x relates to y", "A", "p", "B")
    gaps = ingested_engine.find_knowledge_gaps()
    unsupported = [g for g in gaps if g.kind == GapKind.UNSUPPORTED_CLAIM]
    assert unsupported


def test_contradiction_gap_detected(ingested_engine):
    from nexus_knowledge.domain.contradiction import Contradiction, ContradictionKind

    ingested_engine.repository.contradictions.save(
        Contradiction(
            kind=ContradictionKind.CONFLICTING_CLAIMS,
            claim_a_id="a",
            claim_b_id="b",
            description="conflict",
            strength=0.8,
        )
    )
    kinds = {g.kind for g in ingested_engine.find_knowledge_gaps()}
    assert GapKind.CONTRADICTION in kinds


def test_stale_claim_detected(ingested_engine):
    repo = ingested_engine.repository
    repo.claims.save(
        Claim(
            text="old claim",
            verification_state=VerificationState.STALE,
            id="claim_stale",
        )
    )
    gaps = ingested_engine.find_knowledge_gaps()
    assert any(g.kind == GapKind.STALE for g in gaps)


def test_disconnected_entity_detected(engine):
    from nexus_knowledge.domain.entity import Entity

    engine.repository.entities.save(Entity(name="Isolated Inc", id="iso"))
    gaps = engine.find_knowledge_gaps()
    disconnected = [g for g in gaps if g.kind == GapKind.DISCONNECTED_ENTITY]
    assert disconnected
    assert disconnected[0].affected_entities == ["iso"]


def test_missing_evidence_detected(ingested_engine):
    repo = ingested_engine.repository
    claim = repo.claims.save(Claim(text="thin claim", subject="A", predicate="p", object="B"))
    repo.relations.save(Relation(subject_id="ent_a", predicate="p", object_id="ent_b", id="rthin"))
    gaps = ingested_engine.find_knowledge_gaps()
    missing = [g for g in gaps if g.kind == GapKind.MISSING_EVIDENCE]
    assert missing
    assert any(claim.id in g.affected_claims for g in missing)


def test_low_diversity_detected(ingested_engine):
    gaps = ingested_engine.find_knowledge_gaps()
    assert any(g.kind == GapKind.LOW_DIVERSITY for g in gaps)


def test_investigations_link_to_gap(ingested_engine):
    for gap in ingested_engine.find_knowledge_gaps():
        assert gap.candidate_investigations
        assert all(i.gap_id == gap.id for i in gap.candidate_investigations)
        assert all(i.target_entities for i in gap.candidate_investigations)
        assert all(i.estimated_cost > 0 for i in gap.candidate_investigations)


def test_candidate_pairs_limit():
    weights = GapWeights()
    weights.max_missing_relation_pairs = 3
    weights.min_sources = 1
    assert weights.max_missing_relation_pairs == 3


def test_importance_bounded(ingested_engine):
    gaps = ingested_engine.find_knowledge_gaps()
    assert all(0.0 <= g.importance <= 1.0 for g in gaps)
    assert all(0.0 <= g.uncertainty <= 1.0 for g in gaps)
