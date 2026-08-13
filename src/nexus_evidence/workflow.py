"""Application service for claim extraction, evidence verification, and citation repair."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

from .audit import CitationAuditor, CitationSynthesizer
from .extraction import ClaimExtractor
from .graph import CitationGraph
from .metrics import EvidenceMetrics, InMemoryEvidenceMetrics
from .model import AtomicClaim, CitationAudit, ClaimAssessment, stable_id
from .persistence import EvidenceWorkflowStore, InMemoryEvidenceWorkflowStore
from .planning import SearchPlanner, SearchQuery
from .retrieval import EvidenceRetriever, RetrievedEvidence
from .verification import EvidenceVerifier


@dataclass(frozen=True, slots=True)
class EvidenceWorkflowPolicy:
    results_per_query: int = 5
    max_repair_rounds: int = 2
    citations_per_claim: int = 2
    max_claims: int = 100
    max_draft_chars: int = 100_000

    def __post_init__(self) -> None:
        if self.results_per_query < 1 or self.max_repair_rounds < 0:
            raise ValueError("evidence workflow bounds are invalid")
        if self.citations_per_claim < 1 or self.max_claims < 1 or self.max_draft_chars < 1:
            raise ValueError("evidence workflow limits must be positive")


@dataclass(frozen=True, slots=True)
class EvidenceWorkflowReport:
    workflow_id: str
    session_id: str
    draft_hash: str
    graph: CitationGraph
    audit: CitationAudit
    audit_history: tuple[CitationAudit, ...]
    queries_executed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "workflow_id": self.workflow_id,
            "session_id": self.session_id,
            "draft_hash": self.draft_hash,
            "graph": self.graph.to_dict(),
            "audit": self.audit.to_dict(),
            "audit_history": [item.to_dict() for item in self.audit_history],
            "queries_executed": self.queries_executed,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvidenceWorkflowReport:
        return cls(
            workflow_id=str(payload["workflow_id"]),
            session_id=str(payload["session_id"]),
            draft_hash=str(payload["draft_hash"]),
            graph=CitationGraph.from_dict(payload["graph"]),
            audit=CitationAudit.from_dict(payload["audit"]),
            audit_history=tuple(
                CitationAudit.from_dict(item) for item in payload.get("audit_history", [])
            ),
            queries_executed=int(payload["queries_executed"]),
        )


class EvidenceEngine:
    """Provider-neutral, bounded citation and verification application service."""

    def __init__(
        self,
        retriever: EvidenceRetriever,
        *,
        extractor: ClaimExtractor | None = None,
        planner: SearchPlanner | None = None,
        verifier: EvidenceVerifier | None = None,
        auditor: CitationAuditor | None = None,
        store: EvidenceWorkflowStore | None = None,
        policy: EvidenceWorkflowPolicy | None = None,
        metrics: EvidenceMetrics | None = None,
    ) -> None:
        self._retriever = retriever
        self._extractor = extractor or ClaimExtractor()
        self._planner = planner or SearchPlanner()
        self._verifier = verifier or EvidenceVerifier()
        self._auditor = auditor or CitationAuditor()
        self._synthesizer = CitationSynthesizer()
        self._store = store or InMemoryEvidenceWorkflowStore()
        self._policy = policy or EvidenceWorkflowPolicy()
        self._metrics = metrics or InMemoryEvidenceMetrics()

    def verify_answer(
        self, draft_answer: str, *, session_id: str = "standalone"
    ) -> EvidenceWorkflowReport:
        if not draft_answer.strip():
            raise ValueError("draft_answer must be a non-empty string")
        if len(draft_answer) > self._policy.max_draft_chars:
            raise ValueError("draft_answer exceeds the configured evidence workflow limit")
        draft_hash = hashlib.sha256(draft_answer.encode()).hexdigest()
        workflow_id = stable_id("evidence_workflow", session_id, draft_hash)
        completed = self._store.latest(workflow_id)
        if completed is not None:
            self._metrics.increment("workflow_cache_hits")
            return EvidenceWorkflowReport.from_dict(completed)

        claims = self._extractor.extract(draft_answer)
        if len(claims) > self._policy.max_claims:
            raise ValueError("atomic claim count exceeds the configured evidence workflow limit")
        self._metrics.increment("workflows_started")
        self._metrics.increment("claims_extracted", len(claims))
        graph = CitationGraph()
        for claim in claims:
            graph.add_claim(claim)
        self._store.save(
            workflow_id,
            session_id,
            "claims_extracted",
            {"draft_hash": draft_hash, "claims": [item.to_dict() for item in claims]},
        )

        query_count = 0
        for claim in self._planner.prioritize(claims):
            queries = self._planner.plan(claim)
            query_count += len(queries)
            self._metrics.increment("queries_executed", len(queries))
            self._retrieve_and_assess(graph, claim, queries)
        self._store.save(workflow_id, session_id, "evidence_collected", graph.to_dict())

        claim_assessments = self._aggregate(graph)
        self._replace_citations(graph, claim_assessments)
        audit = self._auditor.audit(graph, claim_assessments)
        history = [audit]
        previous_score = self._audit_score(audit)

        for repair_round in range(1, self._policy.max_repair_rounds + 1):
            claim_ids = self._auditor.repair_claims(audit)
            if not claim_ids:
                break
            evidence_before = len(graph.evidence)
            for claim_id in claim_ids:
                claim = graph.claims[claim_id]
                repair = SearchQuery(
                    claim_id,
                    f"{claim.text} authoritative primary source corroboration",
                    f"repair_{repair_round}",
                    1.0,
                )
                query_count += 1
                self._metrics.increment("queries_executed")
                self._retrieve_and_assess(graph, claim, (repair,))
            if len(graph.evidence) == evidence_before:
                break
            claim_assessments = self._aggregate(graph)
            self._replace_citations(graph, claim_assessments)
            candidate = self._auditor.audit(graph, claim_assessments, repair_round=repair_round)
            history.append(candidate)
            self._metrics.increment("repair_rounds")
            score = self._audit_score(candidate)
            audit = candidate
            self._store.save(
                workflow_id,
                session_id,
                f"repair_{repair_round}",
                {"graph": graph.to_dict(), "audit": audit.to_dict()},
            )
            if score <= previous_score:
                break
            previous_score = score

        report = EvidenceWorkflowReport(
            workflow_id,
            session_id,
            draft_hash,
            graph,
            audit,
            tuple(history),
            query_count,
        )
        self._store.save(workflow_id, session_id, "completed", report.to_dict())
        self._metrics.increment("workflows_completed")
        self._metrics.increment("evidence_spans", len(graph.evidence))
        self._metrics.increment("citations_emitted", len(graph.citations))
        for issue in audit.issues:
            self._metrics.increment(f"citation_issue_{issue.kind.value}")
        for name, value in audit.metrics.to_dict().items():
            self._metrics.observe(name, value)
        return report

    def _retrieve_and_assess(
        self,
        graph: CitationGraph,
        claim: AtomicClaim,
        queries: tuple[SearchQuery, ...],
    ) -> None:
        candidates: dict[str, RetrievedEvidence] = {}
        for query in queries:
            for item in self._retriever.search(query, limit=self._policy.results_per_query):
                previous = candidates.get(item.span.evidence_id)
                if previous is None or item.score > previous.score:
                    candidates[item.span.evidence_id] = item
        for candidate in sorted(
            candidates.values(), key=lambda item: (-item.score, item.span.evidence_id)
        ):
            graph.add_evidence(candidate.source, candidate.span)
            graph.add_assessment(self._verifier.assess(claim, candidate.span))

    def _aggregate(self, graph: CitationGraph) -> tuple[ClaimAssessment, ...]:
        source_by_evidence = {item.evidence_id: item.source_id for item in graph.evidence.values()}
        return tuple(
            self._verifier.aggregate(
                claim,
                graph.assessments_for_claim(claim.claim_id),
                source_by_evidence,
            )
            for claim in self._planner.prioritize(tuple(graph.claims.values()))
        )

    def _replace_citations(
        self, graph: CitationGraph, assessments: tuple[ClaimAssessment, ...]
    ) -> None:
        graph.citations.clear()
        for citation in self._synthesizer.synthesize(
            graph, assessments, citations_per_claim=self._policy.citations_per_claim
        ):
            graph.add_citation(citation)

    @staticmethod
    def _audit_score(audit: CitationAudit) -> float:
        metrics = audit.metrics
        return (
            metrics.weighted_coverage
            + metrics.citation_precision
            + metrics.evidence_support_rate
            - metrics.contradiction_rate
            - metrics.unsupported_rate
        )
