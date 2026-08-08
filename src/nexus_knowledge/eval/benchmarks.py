"""Reproducible evaluation benchmarks.

Runs the retrieval, graph and knowledge benchmarks over the synthetic
corpus and reports deterministic metrics for a set of retrieval
configurations (lexical / vector / graph / hybrid / hybrid+reranker).

Benchmarks build their own engine from local adapters, so results are
reproducible without external services.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ..domain.source import Source, SourceKind
from ..service.factory import Adapters, create_engine
from . import metrics
from .fixtures import Corpus, build_corpus

__all__ = ["BenchmarkRunner", "run_benchmarks"]

RETRIEVAL_CONFIGS: dict[str, tuple[str, ...]] = {
    "lexical": ("lexical",),
    "vector": ("vector",),
    "graph": ("graph",),
    "hybrid": ("lexical", "vector", "entity", "graph"),
}


@dataclass(slots=True)
class BenchmarkReport:
    retrieval: dict[str, dict[str, float]] = field(default_factory=dict)
    graph: dict[str, dict[str, float]] = field(default_factory=dict)
    knowledge: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "retrieval": self.retrieval,
            "graph": self.graph,
            "knowledge": self.knowledge,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)


class BenchmarkRunner:
    """Executes retrieval, graph and knowledge benchmarks."""

    def __init__(self, corpus: Corpus | None = None) -> None:
        self.corpus = corpus or build_corpus()

    def run(self) -> BenchmarkReport:
        report = BenchmarkReport()
        report.retrieval = self._retrieval_benchmark()
        report.graph = self._graph_benchmark()
        report.knowledge = self._knowledge_benchmark()
        return report

    # -- engine builders ----------------------------------------------
    def _build_engine(self, active_methods: tuple[str, ...]):
        return create_engine(
            Adapters(
                gazetteer=self.corpus.gazetteer,
                active_methods=active_methods,
            )
        )

    def _ingest(self, engine) -> None:
        for reference, text in self.corpus.documents:
            engine.ingest(Source(title=reference, kind=SourceKind.TEXT, reference=reference), text)

    def _chunk_by_document(self, engine) -> dict[str, list[str]]:
        mapping: dict[str, list[str]] = {}
        for chunk in engine.repository.chunks.all():
            mapping.setdefault(chunk.document_id, []).append(chunk.id)
        return mapping

    # -- retrieval -----------------------------------------------------
    def _retrieval_benchmark(self) -> dict[str, dict[str, float]]:
        results: dict[str, dict[str, float]] = {}
        for name, methods in RETRIEVAL_CONFIGS.items():
            engine = self._build_engine(methods)
            self._ingest(engine)
            chunk_by_doc = self._chunk_by_document(engine)
            recalls: list[float] = []
            precisions: list[float] = []
            mrrs: list[float] = []
            ndcgs: list[float] = []
            for judgement in self.corpus.queries:
                retrieval = engine.retrieve(judgement.query, top_k=5)
                ranked = [c.chunk_id for c in retrieval.candidates]
                relevant: set[str] = set()
                for doc_title in judgement.relevant_documents:
                    doc_id = next(
                        (d.id for d in engine.repository.documents.all() if d.title == doc_title),
                        None,
                    )
                    if doc_id is not None:
                        relevant.update(chunk_by_doc.get(doc_id, []))
                recalls.append(metrics.recall_at_k(ranked, relevant))
                precisions.append(metrics.precision_at_k(ranked, relevant))
                mrrs.append(metrics.mean_reciprocal_rank(ranked, relevant))
                ndcgs.append(metrics.ndcg_at_k(ranked, relevant, k=5))
            results[name] = {
                "recall_at_5": round(metrics.mean(recalls), 4),
                "precision_at_5": round(metrics.mean(precisions), 4),
                "mrr": round(metrics.mean(mrrs), 4),
                "ndcg_at_5": round(metrics.mean(ndcgs), 4),
            }
        return results

    # -- graph ----------------------------------------------------------
    def _graph_benchmark(self) -> dict[str, dict[str, float]]:
        engine = self._build_engine(("lexical", "vector", "entity", "graph"))
        self._ingest(engine)
        entity_by_id = {e.id: e for e in engine.repository.entities.all()}
        entity_by_name = {e.canonical.lower(): e.id for e in engine.repository.entities.all()}

        entity_recalls: list[float] = []
        relation_recalls: list[float] = []
        evidence_precisions: list[float] = []

        for judgement in self.corpus.queries:
            evidence = engine.graphrag(judgement.query)
            relevant_entities = {
                entity_by_name.get(name, "missing") for name in judgement.relevant_entities
            } - {"missing"}
            found_entities = {e.id for e in evidence.entities}
            entity_recalls.append(metrics.entity_recall(found_entities, relevant_entities))

            found_relations = set()
            for relation in evidence.relations:
                subject = entity_by_id.get(relation.subject_id)
                object_ = entity_by_id.get(relation.object_id)
                if subject is None or object_ is None:
                    continue
                found_relations.add(
                    (subject.canonical.lower(), relation.predicate, object_.canonical.lower())
                )
            relation_recalls.append(
                metrics.relation_recall(found_relations, set(judgement.relevant_relations))
            )

            relevant_docs = set(judgement.relevant_documents)
            relevant_evidence = set()
            for item in evidence.evidence:
                doc = next(
                    (
                        d
                        for d in engine.repository.documents.all()
                        if any(
                            c.id == item.chunk_id
                            for c in engine.repository.chunks.by_document(d.id)
                        )
                    ),
                    None,
                )
                if doc is not None and doc.title in relevant_docs:
                    relevant_evidence.add(item.id)
            evidence_precisions.append(
                metrics.evidence_precision([e.id for e in evidence.evidence], relevant_evidence)
            )

        return {
            "entity_recall": round(metrics.mean(entity_recalls), 4),
            "relation_recall": round(metrics.mean(relation_recalls), 4),
            "evidence_precision": round(metrics.mean(evidence_precisions), 4),
        }

    # -- knowledge ------------------------------------------------------
    def _knowledge_benchmark(self) -> dict[str, float]:
        engine = self._build_engine(("lexical", "vector", "entity", "graph"))
        self._ingest(engine)
        entity_by_name: dict[str, str] = {}
        for entity in engine.repository.entities.all():
            entity_by_name.setdefault(entity.canonical.lower(), entity.id)

        predictions: list[bool] = []
        labels: list[bool] = []
        confidences: list[float] = []
        provenance_ok: list[bool] = []
        for text, subject, predicate, object_, is_true in self.corpus.claims:
            subject_id = entity_by_name.get(subject.lower())
            object_id = entity_by_name.get(object_.lower())
            claim = engine.propose_claim(
                text,
                subject,
                predicate,
                object_,
                confidence=0.5,
                source_ref="label-probe",
            )
            if subject_id is None or object_id is None:
                supported = False
                confidence = 0.0
            else:
                matched = [
                    r
                    for r in engine.repository.relations.all()
                    if r.subject_id == subject_id
                    and r.predicate == predicate
                    and r.object_id == object_id
                ]
                if matched:
                    claim.supporting_evidence = matched[0].supporting_evidence
                    claim.provenance = matched[0].provenance
                    claim.source_ids = matched[0].source_ids
                    engine.repository.claims.save(claim)
                    assessment = engine.verify_claim(claim.id)
                    supported = assessment.verification_state.value in ("verified", "supported")
                    confidence = assessment.confidence
                    provenance = engine.provenance(claim.id).to_dict()
                    provenance_ok.append(
                        bool(provenance["provenance"]["chunk_ids"])
                        and bool(provenance["provenance"]["document_ids"])
                        and bool(provenance["provenance"]["source_ids"])
                    )
                else:
                    supported = False
                    confidence = 0.0
            predictions.append(supported)
            labels.append(is_true)
            confidences.append(confidence)
        return {
            "claim_accuracy": round(metrics.claim_accuracy(predictions, labels), 4),
            "calibration_error": round(metrics.calibration_error(confidences, labels), 4),
            "provenance_correctness": round(metrics.provenance_correctness(provenance_ok), 4),
        }


def run_benchmarks(corpus: Corpus | None = None) -> BenchmarkReport:
    """Run all benchmarks and return the report."""
    return BenchmarkRunner(corpus).run()
