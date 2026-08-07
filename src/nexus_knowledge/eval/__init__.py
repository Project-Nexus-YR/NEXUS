"""Evaluation metrics and reproducible benchmarks."""

from .benchmarks import BenchmarkReport, BenchmarkRunner, run_benchmarks
from .fixtures import Corpus, QueryJudgement, build_corpus
from .metrics import (
    average_precision,
    calibration_error,
    claim_accuracy,
    entity_recall,
    evidence_precision,
    mean,
    mean_reciprocal_rank,
    ndcg_at_k,
    path_recall,
    precision_at_k,
    provenance_correctness,
    recall_at_k,
    relation_recall,
)

__all__ = [
    "BenchmarkReport",
    "BenchmarkRunner",
    "Corpus",
    "QueryJudgement",
    "average_precision",
    "build_corpus",
    "calibration_error",
    "claim_accuracy",
    "entity_recall",
    "evidence_precision",
    "mean",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "path_recall",
    "precision_at_k",
    "provenance_correctness",
    "recall_at_k",
    "relation_recall",
    "run_benchmarks",
]
