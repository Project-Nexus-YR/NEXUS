"""Contradiction detector tests."""

import pytest

from nexus_knowledge.domain.claim import Claim
from nexus_knowledge.domain.common import Confidence, VerificationState
from nexus_knowledge.domain.contradiction import ContradictionKind
from nexus_knowledge.domain.entity import Relation
from nexus_knowledge.knowledge.contradiction import ContradictionDetector
from nexus_knowledge.persistence.memory import InMemoryKnowledgeRepository


def _repo():
    return InMemoryKnowledgeRepository()


class TestConflictingClaims:
    def test_same_subject_predicate_different_objects(self):
        repo = _repo()
        repo.claims.save(Claim(text="A located in London", subject="A", predicate="located_in", object="London"))
        repo.claims.save(Claim(text="A located in Paris", subject="A", predicate="located_in", object="Paris"))
        detector = ContradictionDetector(
            repo.claims, repo.relations, repo.evidence, repo.contradictions
        )
        contradictions = detector.detect()
        kinds = {c.kind for c in contradictions}
        assert ContradictionKind.CONFLICTING_CLAIMS in kinds
        assert repo.contradictions.count() >= 1

    def test_same_object_is_not_a_conflict(self):
        repo = _repo()
        repo.claims.save(Claim(text="A located in London", subject="A", predicate="located_in", object="London"))
        repo.claims.save(Claim(text="A located in London", subject="A", predicate="located_in", object="London"))
        detector = ContradictionDetector(
            repo.claims, repo.relations, repo.evidence, repo.contradictions
        )
        contradictions = detector.detect()
        assert all(c.kind != ContradictionKind.CONFLICTING_CLAIMS for c in contradictions)

    def test_strength_scales_with_evidence(self):
        repo = _repo()
        a = Claim(
            text="A located in London", subject="A", predicate="located_in",
            object="London", confidence=Confidence(0.9), supporting_evidence=["e1", "e2"],
        )
        b = Claim(
            text="A located in Paris", subject="A", predicate="located_in",
            object="Paris", confidence=Confidence(0.9), supporting_evidence=["e1"],
        )
        repo.claims.save(a)
        repo.claims.save(b)
        detector = ContradictionDetector(
            repo.claims, repo.relations, repo.evidence, repo.contradictions
        )
        contradiction = next(
            c for c in detector.detect() if c.kind == ContradictionKind.CONFLICTING_CLAIMS
        )
        assert contradiction.strength > 0.5


class TestMutuallyExclusiveRelations:
    def test_detects_conflicting_relations(self):
        repo = _repo()
        repo.relations.save(Relation(subject_id="x", predicate="located_in", object_id="london", id="r1"))
        repo.relations.save(Relation(subject_id="x", predicate="located_in", object_id="paris", id="r2"))
        detector = ContradictionDetector(
            repo.claims, repo.relations, repo.evidence, repo.contradictions
        )
        contradictions = detector.detect()
        assert any(
            c.kind == ContradictionKind.MUTUALLY_EXCLUSIVE_RELATIONS
            for c in contradictions
        )


class TestStaleClaims:
    def test_stale_claim_flags_contradiction(self):
        repo = _repo()
        repo.claims.save(
            Claim(text="old", verification_state=VerificationState.STALE, id="cstale")
        )
        detector = ContradictionDetector(
            repo.claims, repo.relations, repo.evidence, repo.contradictions
        )
        contradictions = detector.detect()
        assert any(c.kind == ContradictionKind.STALE_CLAIM for c in contradictions)


class TestPersistedContradictions:
    def test_detect_is_idempotent(self):
        repo = _repo()
        repo.claims.save(Claim(text="A located in London", subject="A", predicate="located_in", object="London"))
        repo.claims.save(Claim(text="A located in Paris", subject="A", predicate="located_in", object="Paris"))
        detector = ContradictionDetector(
            repo.claims, repo.relations, repo.evidence, repo.contradictions
        )
        first = detector.detect()
        second = detector.detect()
        assert len(second) >= len(first)
        assert repo.contradictions.count() == len(first) + len(second)
