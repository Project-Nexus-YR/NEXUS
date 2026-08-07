"""Knowledge gap engine.

Detects measurable deficiencies in the knowledge graph:

* missing relations (observed co-occurrence without a direct edge)
* low-confidence claims
* unsupported claims (no supporting evidence)
* contradictions
* stale information
* disconnected entities
* missing evidence (sparse provenance)
* low source diversity

Every gap is derived from measurable graph/evidence properties; nothing
is generated arbitrarily. The engine also emits candidate
investigations that a planner can later score.
"""

from __future__ import annotations

from collections import defaultdict

from ..domain.common import VerificationState
from ..domain.knowledge_gap import GapKind, Investigation, KnowledgeGap
from ..graph.graph import KnowledgeGraph
from ..port.repository import KnowledgeRepository
from .uncertainty import UncertaintyModel

__all__ = ["GapEngine", "GapWeights"]


class GapWeights:
    """Heuristic weights for gap importance and cost estimation."""

    low_confidence_threshold = 0.4
    min_supporting_evidence = 1
    min_sources = 2
    max_missing_relation_pairs = 60
    cost_verify_claim = 1.0
    cost_find_evidence = 1.0
    cost_investigate_relation = 1.5
    cost_study_entity = 2.0


class GapEngine:
    """Derives knowledge gaps from measurable properties."""

    def __init__(
        self,
        repository: KnowledgeRepository,
        graph: KnowledgeGraph,
        uncertainty: UncertaintyModel | None = None,
        weights: GapWeights | None = None,
    ) -> None:
        self._repository = repository
        self._graph = graph
        self._uncertainty = uncertainty or UncertaintyModel()
        self._weights = weights or GapWeights()
        self._pagerank: dict[str, float] = {}

    def find(self) -> list[KnowledgeGap]:
        """Scan the knowledge base and persist all detected gaps."""
        self._pagerank = self._graph.pagerank()
        gaps: list[KnowledgeGap] = []
        gaps += self._missing_relations()
        gaps += self._low_confidence_claims()
        gaps += self._unsupported_claims()
        gaps += self._contradictions()
        gaps += self._stale_records()
        gaps += self._disconnected_entities()
        gaps += self._missing_evidence()
        gaps += self._low_diversity_claims()
        for gap in gaps:
            self._repository.gaps.save(gap)
        gaps.sort(key=lambda g: g.priority, reverse=True)
        return gaps

    # -- detectors ----------------------------------------------------
    def _missing_relations(self) -> list[KnowledgeGap]:
        pairs = self._candidate_pairs()
        gaps: list[KnowledgeGap] = []
        entities = {e.id: e for e in self._repository.entities.all()}
        for (a_id, b_id), degree in pairs:
            name_a = entities.get(a_id, None)
            name_b = entities.get(b_id, None)
            description = (
                f"no observed relation between "
                f"{name_a.canonical if name_a else a_id[:8]} and "
                f"{name_b.canonical if name_b else b_id[:8]}"
            )
            importance = self._normalized_centrality([a_id, b_id])
            gap = KnowledgeGap(
                kind=GapKind.MISSING_RELATION,
                description=description,
                reason="entities co-occur around shared neighbors with no direct edge",
                affected_entities=[a_id, b_id],
                uncertainty=0.5,
                importance=importance,
                estimated_cost=self._weights.cost_investigate_relation * (1.0 + 0.5 * degree),
            )
            gap.candidate_investigations = [
                self._candidate(
                    gap_id=gap.id,
                    kind=GapKind.MISSING_RELATION,
                    description=f"investigate relation between {a_id[:8]} and {b_id[:8]}",
                    target_entities=[a_id, b_id],
                    cost=self._weights.cost_investigate_relation,
                    importance=importance,
                )
            ]
            gaps.append(gap)
        return gaps

    def _low_confidence_claims(self) -> list[KnowledgeGap]:
        gaps: list[KnowledgeGap] = []
        threshold = self._weights.low_confidence_threshold
        for claim in self._repository.claims.all():
            if float(claim.confidence) >= threshold:
                continue
            entities = self._entities_for_claim(claim)
            importance = self._importance_for_claim(claim)
            gap = KnowledgeGap(
                kind=GapKind.LOW_CONFIDENCE,
                description=f"claim has low confidence: {claim.text[:80]}",
                reason=f"confidence {float(claim.confidence):.2f} below threshold {threshold}",
                affected_claims=[claim.id],
                affected_entities=entities,
                uncertainty=1.0 - float(claim.confidence),
                importance=importance,
                estimated_cost=self._weights.cost_verify_claim,
            )
            gap.candidate_investigations = [
                self._candidate(
                    gap_id=gap.id,
                    kind=GapKind.LOW_CONFIDENCE,
                    description=f"verify claim {claim.id[:8]}",
                    target_entities=entities,
                    cost=self._weights.cost_verify_claim,
                    importance=importance,
                )
            ]
            gaps.append(gap)
        return gaps

    def _unsupported_claims(self) -> list[KnowledgeGap]:
        gaps: list[KnowledgeGap] = []
        for claim in self._repository.claims.all():
            if claim.supporting_evidence:
                continue
            entities = self._entities_for_claim(claim)
            importance = self._importance_for_claim(claim)
            gap = KnowledgeGap(
                kind=GapKind.UNSUPPORTED_CLAIM,
                description=f"claim has no supporting evidence: {claim.text[:80]}",
                reason="claim is asserted without any supporting evidence",
                affected_claims=[claim.id],
                affected_entities=entities,
                uncertainty=1.0 - float(claim.confidence),
                importance=importance,
                estimated_cost=self._weights.cost_find_evidence,
            )
            gap.candidate_investigations = [
                self._candidate(
                    gap_id=gap.id,
                    kind=GapKind.UNSUPPORTED_CLAIM,
                    description=f"find evidence for claim {claim.id[:8]}",
                    target_entities=entities,
                    cost=self._weights.cost_find_evidence,
                    importance=importance,
                )
            ]
            gaps.append(gap)
        return gaps

    def _contradictions(self) -> list[KnowledgeGap]:
        gaps: list[KnowledgeGap] = []
        for contradiction in self._repository.contradictions.all():
            affected_claims = [
                c for c in (contradiction.claim_a_id, contradiction.claim_b_id) if c
            ]
            entities = self._entities_for_contradiction(contradiction)
            gap = KnowledgeGap(
                kind=GapKind.CONTRADICTION,
                description=f"{contradiction.kind}: {contradiction.description}",
                reason="conflicting knowledge was detected and preserved",
                affected_claims=affected_claims,
                uncertainty=min(1.0, contradiction.strength * 1.5),
                importance=contradiction.strength,
                estimated_cost=self._weights.cost_verify_claim,
            )
            gap.candidate_investigations = [
                self._candidate(
                    gap_id=gap.id,
                    kind=GapKind.CONTRADICTION,
                    description="resolve contradiction by gathering decisive evidence",
                    target_entities=entities,
                    cost=self._weights.cost_verify_claim,
                    importance=contradiction.strength,
                )
            ]
            gaps.append(gap)
        return gaps

    def _stale_records(self) -> list[KnowledgeGap]:
        gaps: list[KnowledgeGap] = []
        for claim in self._repository.claims.all():
            if claim.verification_state != VerificationState.STALE:
                continue
            entities = self._entities_for_claim(claim)
            importance = self._importance_for_claim(claim)
            gap = KnowledgeGap(
                kind=GapKind.STALE,
                description=f"stale claim: {claim.text[:80]}",
                reason="claim verification state is stale",
                affected_claims=[claim.id],
                affected_entities=entities,
                uncertainty=0.8,
                importance=importance,
                estimated_cost=self._weights.cost_find_evidence,
            )
            gap.candidate_investigations = [
                self._candidate(
                    gap_id=gap.id,
                    kind=GapKind.STALE,
                    description=f"refresh stale claim {claim.id[:8]}",
                    target_entities=entities,
                    cost=self._weights.cost_find_evidence,
                    importance=importance,
                )
            ]
            gaps.append(gap)
        return gaps

    def _disconnected_entities(self) -> list[KnowledgeGap]:
        gaps: list[KnowledgeGap] = []
        connected: set[str] = set()
        for relation in self._repository.relations.all():
            connected.add(relation.subject_id)
            connected.add(relation.object_id)
        entities = {e.id: e for e in self._repository.entities.all()}
        for entity_id in sorted(set(entities) - connected):
            importance = self._normalized_centrality([entity_id])
            gap = KnowledgeGap(
                kind=GapKind.DISCONNECTED_ENTITY,
                description=f"entity has no relations: {entities[entity_id].canonical}",
                reason="entity appears in no relation, isolating it from the graph",
                affected_entities=[entity_id],
                uncertainty=1.0,
                importance=importance,
                estimated_cost=self._weights.cost_study_entity,
            )
            gap.candidate_investigations = [
                self._candidate(
                    gap_id=gap.id,
                    kind=GapKind.DISCONNECTED_ENTITY,
                    description=f"study entity {entities[entity_id].canonical}",
                    target_entities=[entity_id],
                    cost=self._weights.cost_study_entity,
                    importance=importance,
                )
            ]
            gaps.append(gap)
        return gaps

    def _missing_evidence(self) -> list[KnowledgeGap]:
        gaps: list[KnowledgeGap] = []
        minimum = self._weights.min_supporting_evidence
        for claim in self._repository.claims.all():
            count = len(claim.supporting_evidence)
            if count >= minimum:
                continue
            entities = self._entities_for_claim(claim)
            importance = self._importance_for_claim(claim)
            cost = self._weights.cost_find_evidence * max(1, minimum - count)
            gap = KnowledgeGap(
                kind=GapKind.MISSING_EVIDENCE,
                description=f"claim has {count} evidence item(s): {claim.text[:80]}",
                reason=f"fewer than {minimum} supporting evidence items",
                affected_claims=[claim.id],
                affected_entities=entities,
                uncertainty=0.6,
                importance=importance,
                estimated_cost=cost,
            )
            gap.candidate_investigations = [
                self._candidate(
                    gap_id=gap.id,
                    kind=GapKind.MISSING_EVIDENCE,
                    description=f"gather evidence for claim {claim.id[:8]}",
                    target_entities=entities,
                    cost=cost,
                    importance=importance,
                )
            ]
            gaps.append(gap)
        return gaps

    def _low_diversity_claims(self) -> list[KnowledgeGap]:
        gaps: list[KnowledgeGap] = []
        minimum = self._weights.min_sources
        for claim in self._repository.claims.all():
            if len(set(claim.source_ids)) >= minimum:
                continue
            entities = self._entities_for_claim(claim)
            importance = self._importance_for_claim(claim)
            gap = KnowledgeGap(
                kind=GapKind.LOW_DIVERSITY,
                description=f"claim rests on a single source: {claim.text[:80]}",
                reason=f"fewer than {minimum} independent sources support the claim",
                affected_claims=[claim.id],
                affected_entities=entities,
                uncertainty=0.5,
                importance=importance,
                estimated_cost=self._weights.cost_find_evidence,
            )
            gap.candidate_investigations = [
                self._candidate(
                    gap_id=gap.id,
                    kind=GapKind.LOW_DIVERSITY,
                    description=f"find independent confirmation for claim {claim.id[:8]}",
                    target_entities=entities,
                    cost=self._weights.cost_find_evidence,
                    importance=importance,
                )
            ]
            gaps.append(gap)
        return gaps

    # -- helpers ------------------------------------------------------
    def _candidate_pairs(self) -> list[tuple[tuple[str, str], int]]:
        """Pairs of entities sharing a common neighbor without a direct edge."""
        adjacency: dict[str, set[str]] = defaultdict(set)
        for relation in self._repository.relations.all():
            adjacency[relation.subject_id].add(relation.object_id)
            adjacency[relation.object_id].add(relation.subject_id)
        direct: set[tuple[str, str]] = set()
        for relation in self._repository.relations.all():
            direct.add((relation.subject_id, relation.object_id))
            direct.add((relation.object_id, relation.subject_id))

        candidate_scores: dict[tuple[str, str], int] = defaultdict(int)
        for hub, neighbors in adjacency.items():
            ordered = sorted(neighbors)
            for i in range(len(ordered)):
                for j in range(i + 1, len(ordered)):
                    pair = (ordered[i], ordered[j])
                    if pair in direct or (pair[1], pair[0]) in direct:
                        continue
                    candidate_scores[pair] += 1
        limited = sorted(candidate_scores.items(), key=lambda item: item[1], reverse=True)
        return limited[: self._weights.max_missing_relation_pairs]

    def _normalized_centrality(self, entity_ids: list[str]) -> float:
        if not self._pagerank:
            return 0.5
        maximum = max(self._pagerank.values()) or 1.0
        return min(1.0, sum(self._pagerank.get(eid, 0.0) for eid in entity_ids) / maximum)

    def _entities_for_claim(self, claim) -> list[str]:
        entities = []
        for entity in self._repository.entities.all():
            lower = entity.canonical.lower()
            if lower == claim.subject.lower() or lower == claim.object.lower():
                entities.append(entity.id)
        return entities

    def _entities_for_contradiction(self, contradiction) -> list[str]:
        found: list[str] = []
        for claim_id in (contradiction.claim_a_id, contradiction.claim_b_id):
            claim = self._repository.claims.get(claim_id)
            if claim is not None:
                found.extend(self._entities_for_claim(claim))
        return sorted(set(found))

    def _importance_for_claim(self, claim) -> float:
        entities = self._entities_for_claim(claim)
        centrality = self._normalized_centrality(entities)
        return min(1.0, centrality * float(claim.confidence) + 0.2)

    def _candidate(
        self,
        gap_id: str,
        kind: str,
        description: str,
        target_entities: list[str],
        cost: float,
        importance: float,
    ) -> Investigation:
        return Investigation(
            gap_id=gap_id,
            description=description,
            target_entities=target_entities,
            expected_information_gain=0.0,
            uncertainty_reduction=0.0,
            importance=importance,
            estimated_cost=cost,
            score=0.0,
            metadata={"kind": kind},
        )
