"""Knowledge engine service.

Transport-independent facade exposing the stable contract for the
autonomous research runtime and any other subsystem:

* ``retrieve``
* ``query_graph``
* ``get_subgraph``
* ``find_knowledge_gaps``
* ``score_investigation``
* ``propose_claim``
* ``verify_claim``
* ``commit_knowledge_update``
* ``provenance``

No external subsystem ever touches the internal repositories or graph
directly — only this service, its ports and its data objects.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..domain.claim import Claim, Evidence, Provenance
from ..domain.common import Confidence, VerificationState
from ..domain.document import Document
from ..domain.entity import Entity, Relation
from ..domain.knowledge_gap import Investigation, KnowledgeGap
from ..domain.source import Source
from ..graph.graph import KnowledgeGraph, KnowledgeSubgraph
from ..ingestion.pipeline import IngestionPipeline, IngestionResult
from ..knowledge.contradiction import ContradictionDetector
from ..knowledge.gaps import GapEngine
from ..knowledge.scorer import InvestigationScorer, ScoredInvestigation
from ..knowledge.uncertainty import UncertaintyAssessment, UncertaintyModel
from ..port.embeddings import EmbeddingProvider
from ..port.repository import KnowledgeRepository
from ..port.vector_store import VectorStore
from ..retrieval.graphrag import EvidenceGraph, GraphRAGEngine
from ..retrieval.hybrid import HybridRetriever, RetrievalResult
from ..retrieval.observability import RetrievalTrace

__all__ = [
    "KnowledgeEngine",
    "KnowledgeUpdate",
    "KnowledgeUpdateReceipt",
    "ProvenanceResponse",
]


@dataclass(slots=True)
class KnowledgeUpdate:
    """A batched set of knowledge changes to commit atomically."""

    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    documents: list[Document] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class KnowledgeUpdateReceipt:
    accepted: int
    rejected: int
    errors: list[str]

    def to_dict(self) -> dict[str, object]:
        return {"accepted": self.accepted, "rejected": self.rejected, "errors": self.errors}


@dataclass(frozen=True, slots=True)
class ProvenanceResponse:
    claim_id: str
    provenance: Provenance
    claim_text: str
    evidence: list[Evidence]
    source_references: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "claim_id": self.claim_id,
            "claim_text": self.claim_text,
            "evidence": [{"id": e.id, "text": e.text, "role": e.role} for e in self.evidence],
            "source_references": self.source_references,
            "provenance": {
                "evidence_ids": list(self.provenance.evidence_ids),
                "chunk_ids": list(self.provenance.chunk_ids),
                "document_ids": list(self.provenance.document_ids),
                "source_ids": list(self.provenance.source_ids),
            },
        }


class KnowledgeEngine:
    """High-level facade over the knowledge intelligence subsystem."""

    def __init__(
        self,
        repository: KnowledgeRepository,
        graph: KnowledgeGraph,
        vector_store: VectorStore,
        embedder: EmbeddingProvider,
        ingestion: IngestionPipeline,
        retriever: HybridRetriever,
        graphrag: GraphRAGEngine,
        uncertainty: UncertaintyModel,
        gap_engine: GapEngine,
        contradiction_detector: ContradictionDetector,
        scorer: InvestigationScorer | None = None,
    ) -> None:
        self.repository = repository
        self.graph = graph
        self.vector_store = vector_store
        self.embedder = embedder
        self.ingestion = ingestion
        self.retriever = retriever
        self._graphrag_engine = graphrag
        self.uncertainty = uncertainty
        self.gap_engine = gap_engine
        self.contradiction_detector = contradiction_detector
        self.scorer = scorer or InvestigationScorer()

    # -- ingestion ----------------------------------------------------
    def ingest(self, source: Source, payload: object) -> IngestionResult:
        """Ingest raw content, updating graph, embeddings and indexes."""
        result = self.ingestion.ingest(source, payload)
        self.retriever.invalidate_lexical_index()
        return result

    # -- retrieval ----------------------------------------------------
    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        metadata_filter: dict[str, object] | None = None,
    ) -> RetrievalResult:
        return self.retriever.retrieve(query, top_k=top_k, metadata_filter=metadata_filter)

    def graphrag(self, query: str, top_k: int = 8, depth: int = 2) -> EvidenceGraph:
        return self._graphrag_engine.query(query, top_k=top_k, depth=depth)

    # -- graph queries -------------------------------------------------
    def query_graph(
        self,
        predicate: str | None = None,
        subject: str | None = None,
        object: str | None = None,
        entity_type: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, object]]:
        """Filter-based graph query returning relation records."""
        results: list[dict[str, object]] = []
        for relation in self.graph.all_relations():
            if predicate and relation.predicate != predicate:
                continue
            if subject and relation.subject_id != subject:
                continue
            if object and relation.object_id != object:
                continue
            results.append(
                {
                    "relation_id": relation.id,
                    "subject_id": relation.subject_id,
                    "predicate": relation.predicate,
                    "object_id": relation.object_id,
                    "confidence": float(relation.confidence),
                    "verification_state": relation.verification_state.value,
                }
            )
            if len(results) >= limit:
                break
        if entity_type is not None:
            entities = {e.id: e for e in self.graph.all_entities() if e.entity_type == entity_type}
            results = [r for r in results if r["subject_id"] in entities or r["object_id"] in entities]
        return results

    def get_subgraph(
        self,
        entity_ids: list[str],
        depth: int = 1,
    ) -> KnowledgeSubgraph:
        return self.graph.subgraph(entity_ids, depth=depth)

    def graph_statistics(self) -> dict[str, int | float]:
        return self.graph.statistics().to_dict()

    # -- knowledge analysis --------------------------------------------
    def find_knowledge_gaps(self) -> list[KnowledgeGap]:
        return self.gap_engine.find()

    def score_investigation(self, top_k: int = 20) -> list[ScoredInvestigation]:
        gaps = self.repository.gaps.all()
        if not gaps:
            gaps = self.find_knowledge_gaps()
        return self.scorer.score_gaps(gaps)[:top_k]

    def detect_contradictions(self) -> list:
        return self.contradiction_detector.detect()

    # -- claims --------------------------------------------------------
    def propose_claim(
        self,
        text: str,
        subject: str,
        predicate: str,
        object: str,
        confidence: float = 0.5,
        source_ref: str = "",
        observed_at: str = "",
    ) -> Claim:
        """Create a claim; without provenance it stays unverified."""
        claim = Claim(
            text=text,
            subject=subject,
            predicate=predicate,
            object=object,
            confidence=Confidence(confidence),
            verification_state=VerificationState.UNVERIFIED,
            observed_at=observed_at,
        )
        if source_ref:
            source = self._find_or_create_source(source_ref)
            claim.source_ids = [source.id]
        self.repository.claims.save(claim)
        return claim

    def verify_claim(self, claim_id: str) -> UncertaintyAssessment:
        """Assess a claim's confidence and update its verification state."""
        claim = self.repository.claims.get(claim_id)
        if claim is None:
            raise KeyError(f"unknown claim id {claim_id}")
        assessment = self.uncertainty.evaluate(claim, self.repository.evidence, self.repository.sources)
        claim.confidence = Confidence(assessment.confidence)
        claim.verification_state = assessment.verification_state
        self.repository.claims.save(claim)
        return assessment

    def provenance(self, claim_id: str) -> ProvenanceResponse:
        """Return structured provenance answering 'why does the system believe this?'."""
        claim = self.repository.claims.get(claim_id)
        if claim is None:
            raise KeyError(f"unknown claim id {claim_id}")
        evidence = [self.repository.evidence.get(eid) for eid in claim.supporting_evidence]
        evidence = [e for e in evidence if e is not None]
        chunk_ids = list(claim.provenance)
        documents = {d.id: d for d in self.repository.documents.all()}
        document_ids = []
        source_ids = []
        for chunk_id in chunk_ids:
            chunk = self.repository.chunks.get(chunk_id)
            if chunk is None:
                continue
            if chunk.document_id not in document_ids:
                document_ids.append(chunk.document_id)
            document = documents.get(chunk.document_id)
            if document is not None and document.source_id not in source_ids:
                source_ids.append(document.source_id)
        sources = {s.id: s for s in self.repository.sources.all()}
        return ProvenanceResponse(
            claim_id=claim_id,
            provenance=Provenance(
                entity_id=claim_id,
                evidence_ids=tuple(e.id for e in evidence),
                chunk_ids=tuple(chunk_ids),
                document_ids=tuple(document_ids),
                source_ids=tuple(source_ids),
            ),
            claim_text=claim.text,
            evidence=evidence,
            source_references={sid: sources[sid].reference for sid in source_ids if sid in sources},
        )

    def validate_evidence_provenance(
        self,
        source_id: str,
        document_id: str,
        chunk_id: str,
        source_reference: str,
    ) -> bool:
        """Validate a source → document → chunk chain through the public service boundary."""
        source = self.repository.sources.get(source_id)
        document = self.repository.documents.get(document_id)
        chunk = self.repository.chunks.get(chunk_id)
        return bool(
            source is not None
            and document is not None
            and chunk is not None
            and source.reference == source_reference
            and document.source_id == source.id
            and chunk.document_id == document.id
        )

    # -- transactions ---------------------------------------------------
    def commit_knowledge_update(self, update: KnowledgeUpdate) -> KnowledgeUpdateReceipt:
        """Atomically apply a batch of knowledge changes."""
        accepted = 0
        errors: list[str] = []
        for entity in update.entities:
            if not entity.id:
                errors.append("entity without id rejected")
                continue
            self.repository.entities.save(entity)
            self.graph.add_entity(entity)
            accepted += 1
        for relation in update.relations:
            if not relation.subject_id or not relation.object_id:
                errors.append("relation without subject/object rejected")
                continue
            canonical = self.graph.add_relation(relation)
            self.repository.relations.save(canonical)
            accepted += 1
        for claim in update.claims:
            if not claim.text:
                errors.append("empty claim rejected")
                continue
            self.repository.claims.save(claim)
            accepted += 1
        for evidence in update.evidence:
            if not evidence.claim_id or not evidence.chunk_id:
                errors.append("evidence without claim/chunk rejected")
                continue
            self.repository.evidence.save(evidence)
            accepted += 1
        for document in update.documents:
            self.repository.documents.save(document)
            accepted += 1
        rejected = len(errors)
        self.retriever.invalidate_lexical_index()
        return KnowledgeUpdateReceipt(accepted=accepted, rejected=rejected, errors=errors)

    # -- system ---------------------------------------------------------
    def healthcheck(self) -> dict[str, int]:
        return self.repository.healthcheck()

    def _find_or_create_source(self, reference: str) -> Source:
        for source in self.repository.sources.all():
            if source.reference == reference:
                return source
        source = Source(title=reference, kind="other", reference=reference)
        return self.repository.sources.save(source)
