"""Ingestion pipeline.

Orchestrates the full chain for one source:

    source -> documents -> chunks -> entities -> relations -> claims
           -> evidence -> embeddings -> graph update

Every artifact retains source metadata and provenance references.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from ..domain.claim import Claim, Evidence
from ..domain.common import VerificationState
from ..domain.document import Chunk, Document, Span
from ..domain.entity import Entity, Relation
from ..domain.ids import stable_id
from ..domain.source import Source
from ..graph.graph import KnowledgeGraph
from ..port.chunker import Chunker
from ..port.embeddings import EmbeddingProvider
from ..port.extractors import EntityExtractor, RelationExtractor
from ..port.repository import KnowledgeRepository
from ..port.vector_store import VectorStore
from .adapters import RawDocument, SourceAdapter, TextAdapter
from .normalization import RecursiveChunker, normalize_text

__all__ = ["IngestionPipeline", "IngestionResult"]


@dataclass(slots=True)
class IngestionResult:
    source: Source | None = None
    documents: list[Document] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    claims: list[Claim] = field(default_factory=list)
    evidence: list[Evidence] = field(default_factory=list)
    embeddings_added: int = 0

    def summary(self) -> dict[str, int]:
        return {
            "documents": len(self.documents),
            "chunks": len(self.chunks),
            "entities": len(self.entities),
            "relations": len(self.relations),
            "claims": len(self.claims),
            "evidence": len(self.evidence),
            "embeddings": self.embeddings_added,
        }


class IngestionPipeline:
    """Provider-independent ingestion orchestrator."""

    def __init__(
        self,
        repository: KnowledgeRepository,
        graph: KnowledgeGraph,
        embedder: EmbeddingProvider,
        vector_store: VectorStore,
        entity_extractor: EntityExtractor,
        relation_extractor: RelationExtractor,
        chunker: Chunker | None = None,
        default_adapter: SourceAdapter = TextAdapter(),
    ) -> None:
        self._repository = repository
        self._graph = graph
        self._embedder = embedder
        self._vector_store = vector_store
        self._entity_extractor = entity_extractor
        self._relation_extractor = relation_extractor
        self._chunker = chunker or RecursiveChunker()
        self._adapter = default_adapter

    # -- public API ---------------------------------------------------
    def ingest(
        self,
        source: Source,
        payload: object,
        adapter: SourceAdapter | None = None,
    ) -> IngestionResult:
        """Ingest a raw payload (bytes, str or directory path)."""
        source = self._repository.sources.save(source)
        adapter = adapter or self._adapter
        raw_documents = adapter.read(source, payload)
        return self.ingest_raw(source, raw_documents)

    def ingest_raw(
        self,
        source: Source,
        raw_documents: list[RawDocument],
    ) -> IngestionResult:
        """Ingest pre-parsed :class:`RawDocument` records."""
        result = IngestionResult(source=source)
        for raw in raw_documents:
            self._ingest_document(source, raw, result)
        return result

    # -- internals ----------------------------------------------------
    def _ingest_document(
        self,
        source: Source,
        raw: RawDocument,
        result: IngestionResult,
    ) -> None:
        document = Document(
            source_id=source.id,
            title=raw.title,
            content_type=raw.content_type,
            text=normalize_text(raw.text),
            raw=raw.text,
            metadata=dict(raw.metadata),
        )
        self._repository.documents.save(document)
        result.documents.append(document)

        chunks = self._chunker.chunk(document)
        for chunk in chunks:
            self._repository.chunks.save(chunk)
            result.chunks.append(chunk)
            self._process_chunk(source, document, chunk, result)

    def _process_chunk(
        self,
        source: Source,
        document: Document,
        chunk: Chunk,
        result: IngestionResult,
    ) -> None:
        extracted_entities = self._entity_extractor.extract(chunk, document)
        entity_ids: dict[str, str] = {}
        for extracted in extracted_entities:
            entity = self._upsert_entity(extracted.name, extracted.entity_type)
            entity_ids[extracted.name.lower()] = entity.id
            if entity.id not in {e.id for e in result.entities}:
                result.entities.append(entity)

        extracted_relations = self._relation_extractor.extract(
            chunk, document, extracted_entities
        )
        for extracted in extracted_relations:
            subject_id = entity_ids.get(extracted.subject.lower())
            object_id = entity_ids.get(extracted.object.lower())
            if subject_id is None or object_id is None:
                continue
            claim = Claim(
                text=f"{extracted.subject} {extracted.predicate} {extracted.object}",
                subject=extracted.subject,
                predicate=extracted.predicate,
                object=extracted.object,
                confidence=extracted.confidence,
                provenance=[chunk.id],
                source_ids=[source.id],
                verification_state=VerificationState.UNVERIFIED,
            )
            evidence = self._build_evidence(
                claim.id, chunk, document, extracted.span, float(extracted.confidence),
            )
            claim.supporting_evidence = [evidence.id]
            self._repository.claims.save(claim)
            result.claims.append(claim)
            self._repository.evidence.save(evidence)
            result.evidence.append(evidence)

            relation = Relation(
                subject_id=subject_id,
                predicate=extracted.predicate,
                object_id=object_id,
                confidence=extracted.confidence,
                provenance=[chunk.id],
                source_ids=[source.id],
                supporting_evidence=[evidence.id],
                verification_state=VerificationState.UNVERIFIED,
            )
            canonical = self._graph.add_relation(relation)
            self._repository.relations.save(canonical)
            if canonical.id not in {r.id for r in result.relations}:
                result.relations.append(canonical)

        # embed the chunk regardless of extraction success
        embedding = self._embedder.embed(
            chunk.text,
            object_id=chunk.id,
        )
        embedding = replace(
            embedding,
            metadata={
                "chunk_id": chunk.id,
                "document_id": document.id,
                "source_id": source.id,
            },
        )
        self._vector_store.upsert(embedding)
        result.embeddings_added += 1

    def _upsert_entity(self, name: str, entity_type: str) -> Entity:
        existing = self._repository.entities.by_name(name)
        if existing is not None:
            return existing
        entity = Entity(
            name=name,
            entity_type=entity_type,
            id=stable_id("ent", name.lower()),
        )
        self._repository.entities.save(entity)
        self._graph.add_entity(entity)
        return entity

    def _build_evidence(
        self,
        claim_id: str,
        chunk: Chunk,
        document: Document,
        span: Span | None,
        confidence: float,
    ) -> Evidence:
        span_text = span.slice(chunk.text) if span else chunk.text
        return Evidence(
            claim_id=claim_id,
            chunk_id=chunk.id,
            document_id=document.id,
            text=span_text,
            role="support",
            span=span,
            quality=min(1.0, max(0.0, confidence)),
        )
