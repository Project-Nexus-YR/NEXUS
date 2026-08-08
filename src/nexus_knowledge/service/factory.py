"""Composition root.

Wires the knowledge engine's adapters together. Provider-independent:
the default composition uses only deterministic local adapters, and any
port can be swapped for a real provider (LLM embeddings, vector DB,
graph DB, search system) by replacing the corresponding argument.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..embedding.local_store import LocalVectorStore
from ..embedding.provider import LocalEmbeddingProvider
from ..extraction.deterministic import GazetteerEntityExtractor, PatternRelationExtractor
from ..graph.memory import InMemoryGraph
from ..ingestion.pipeline import IngestionPipeline
from ..knowledge.contradiction import ContradictionDetector
from ..knowledge.gaps import GapEngine
from ..knowledge.scorer import InvestigationScorer
from ..knowledge.uncertainty import UncertaintyModel
from ..persistence.memory import InMemoryKnowledgeRepository
from ..port.chunker import Chunker
from ..port.embeddings import EmbeddingProvider
from ..port.extractors import EntityExtractor, RelationExtractor
from ..port.repository import KnowledgeRepository
from ..port.vector_store import VectorStore
from ..retrieval.graphrag import GraphRAGEngine
from ..retrieval.hybrid import HybridRetriever
from .engine import KnowledgeEngine


@dataclass(slots=True)
class Adapters:
    """Optional provider overrides; defaults are deterministic and local."""

    repository: KnowledgeRepository = field(default_factory=InMemoryKnowledgeRepository)
    graph: InMemoryGraph = field(default_factory=InMemoryGraph)
    vector_store: VectorStore = field(default_factory=LocalVectorStore)
    embedder: EmbeddingProvider = field(default_factory=LocalEmbeddingProvider)
    entity_extractor: EntityExtractor = field(default_factory=GazetteerEntityExtractor)
    relation_extractor: RelationExtractor = field(default_factory=PatternRelationExtractor)
    chunker: Chunker | None = None
    gazetteer: dict[str, list[str]] | None = None
    active_methods: tuple[str, ...] = ("lexical", "vector", "entity", "graph")


def create_engine(adapters: Adapters | None = None) -> KnowledgeEngine:
    """Build a fully wired :class:`KnowledgeEngine` from adapters."""
    adapters = adapters or Adapters()
    if adapters.gazetteer and isinstance(adapters.entity_extractor, GazetteerEntityExtractor):
        for entity_type, names in adapters.gazetteer.items():
            for name in names:
                adapters.entity_extractor.add_term(name, entity_type)

    ingestion = IngestionPipeline(
        repository=adapters.repository,
        graph=adapters.graph,
        embedder=adapters.embedder,
        vector_store=adapters.vector_store,
        entity_extractor=adapters.entity_extractor,
        relation_extractor=adapters.relation_extractor,
        chunker=adapters.chunker,
    )
    retriever = HybridRetriever(
        repository=adapters.repository,
        graph=adapters.graph,
        vector_store=adapters.vector_store,
        embedder=adapters.embedder,
        active_methods=adapters.active_methods,
    )
    graphrag = GraphRAGEngine(adapters.repository, adapters.graph, retriever)
    uncertainty = UncertaintyModel()
    contradiction_detector = ContradictionDetector(
        claims=adapters.repository.claims,
        relations=adapters.repository.relations,
        evidence=adapters.repository.evidence,
        contradictions=adapters.repository.contradictions,
    )
    gap_engine = GapEngine(adapters.repository, adapters.graph, uncertainty)
    scorer = InvestigationScorer()
    return KnowledgeEngine(
        repository=adapters.repository,
        graph=adapters.graph,
        vector_store=adapters.vector_store,
        embedder=adapters.embedder,
        ingestion=ingestion,
        retriever=retriever,
        graphrag=graphrag,
        uncertainty=uncertainty,
        gap_engine=gap_engine,
        contradiction_detector=contradiction_detector,
        scorer=scorer,
    )
