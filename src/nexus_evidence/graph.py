"""Queryable citation graph linking claims, evidence spans, sources, and locators."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from .model import (
    AtomicClaim,
    Citation,
    EvidenceAssessment,
    EvidenceSource,
    EvidenceSpan,
)


@dataclass(slots=True)
class CitationGraph:
    claims: dict[str, AtomicClaim] = field(default_factory=dict)
    sources: dict[str, EvidenceSource] = field(default_factory=dict)
    evidence: dict[str, EvidenceSpan] = field(default_factory=dict)
    assessments: dict[str, EvidenceAssessment] = field(default_factory=dict)
    citations: dict[str, Citation] = field(default_factory=dict)

    def add_claim(self, claim: AtomicClaim) -> None:
        self._consistent(self.claims, claim.claim_id, claim, "claim")

    def add_evidence(self, source: EvidenceSource, evidence: EvidenceSpan) -> None:
        if evidence.source_id != source.source_id:
            raise ValueError("evidence source does not match its source record")
        existing_source = self.sources.get(source.source_id)
        if existing_source is not None and self._source_identity(
            existing_source
        ) != self._source_identity(source):
            raise ValueError(f"source identifier aliases different records: {source.source_id}")
        self.sources.setdefault(source.source_id, source)
        existing_evidence = self.evidence.get(evidence.evidence_id)
        if existing_evidence is not None and self._evidence_identity(
            existing_evidence
        ) != self._evidence_identity(evidence):
            raise ValueError(
                f"evidence identifier aliases different records: {evidence.evidence_id}"
            )
        # Search query, strategy, rank, and retrieval timestamp describe how the
        # same canonical span was rediscovered. Keep the first deterministic path.
        self.evidence.setdefault(evidence.evidence_id, evidence)

    def add_assessment(self, assessment: EvidenceAssessment) -> None:
        if assessment.claim_id not in self.claims or assessment.evidence_id not in self.evidence:
            raise ValueError("assessment endpoints must exist in the citation graph")
        self._consistent(self.assessments, assessment.assessment_id, assessment, "assessment")

    def add_citation(self, citation: Citation) -> None:
        evidence = self.evidence.get(citation.evidence_id)
        if (
            citation.claim_id not in self.claims
            or evidence is None
            or citation.source_id not in self.sources
            or evidence.source_id != citation.source_id
        ):
            raise ValueError("citation endpoints must form a valid claim-evidence-source path")
        self._consistent(self.citations, citation.citation_id, citation, "citation")

    def evidence_for_claim(self, claim_id: str) -> tuple[EvidenceSpan, ...]:
        evidence_ids = {
            item.evidence_id for item in self.assessments.values() if item.claim_id == claim_id
        }
        return tuple(self.evidence[key] for key in sorted(evidence_ids))

    def assessments_for_claim(self, claim_id: str) -> tuple[EvidenceAssessment, ...]:
        return tuple(
            sorted(
                (item for item in self.assessments.values() if item.claim_id == claim_id),
                key=lambda item: item.assessment_id,
            )
        )

    def citations_for_claim(self, claim_id: str) -> tuple[Citation, ...]:
        return tuple(
            sorted(
                (item for item in self.citations.values() if item.claim_id == claim_id),
                key=lambda item: item.citation_id,
            )
        )

    def claims_for_source(self, source_id: str) -> tuple[AtomicClaim, ...]:
        claim_ids = {
            citation.claim_id
            for citation in self.citations.values()
            if citation.source_id == source_id
        }
        return tuple(self.claims[key] for key in sorted(claim_ids))

    def adjacency(self) -> dict[str, tuple[str, ...]]:
        edges: dict[str, set[str]] = defaultdict(set)
        for assessment in self.assessments.values():
            edges[assessment.claim_id].add(assessment.evidence_id)
        for evidence in self.evidence.values():
            edges[evidence.evidence_id].add(evidence.source_id)
        for citation in self.citations.values():
            edges[citation.citation_id].update(
                {citation.claim_id, citation.evidence_id, citation.source_id}
            )
        return {key: tuple(sorted(value)) for key, value in sorted(edges.items())}

    def to_dict(self) -> dict[str, Any]:
        return {
            "claims": [self.claims[key].to_dict() for key in sorted(self.claims)],
            "sources": [self.sources[key].to_dict() for key in sorted(self.sources)],
            "evidence": [self.evidence[key].to_dict() for key in sorted(self.evidence)],
            "assessments": [self.assessments[key].to_dict() for key in sorted(self.assessments)],
            "citations": [self.citations[key].to_dict() for key in sorted(self.citations)],
            "adjacency": self.adjacency(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> CitationGraph:
        graph = cls()
        for item in payload.get("claims", []):
            graph.add_claim(AtomicClaim.from_dict(item))
        source_items = [EvidenceSource.from_dict(item) for item in payload.get("sources", [])]
        for source in source_items:
            graph.sources[source.source_id] = source
        for item in payload.get("evidence", []):
            evidence = EvidenceSpan.from_dict(item)
            graph.add_evidence(graph.sources[evidence.source_id], evidence)
        for item in payload.get("assessments", []):
            graph.add_assessment(EvidenceAssessment.from_dict(item))
        for item in payload.get("citations", []):
            graph.add_citation(Citation.from_dict(item))
        return graph

    @staticmethod
    def _consistent(store: dict[str, Any], key: str, value: Any, kind: str) -> None:
        previous = store.get(key)
        if previous is not None and previous != value:
            raise ValueError(f"{kind} identifier aliases different records: {key}")
        store[key] = value

    @staticmethod
    def _source_identity(source: EvidenceSource) -> tuple[object, ...]:
        return (
            source.title,
            source.reference,
            source.source_type,
            source.publisher,
            source.published_at,
            source.content_hash,
            source.quality,
            source.metadata,
        )

    @staticmethod
    def _evidence_identity(evidence: EvidenceSpan) -> tuple[object, ...]:
        return (
            evidence.source_id,
            evidence.text,
            evidence.document_id,
            evidence.chunk_id,
            evidence.page,
            evidence.section,
            evidence.paragraph,
            evidence.char_start,
            evidence.char_end,
            evidence.extractor_version,
            evidence.verifier_version,
        )
