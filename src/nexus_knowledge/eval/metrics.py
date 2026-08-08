"""Ranking and knowledge evaluation metrics.

All metrics are deterministic and operate on sorted result lists of
object ids versus a ground-truth set. Implemented: Recall@K, Precision@K,
MRR, NDCG, graph recall/precision variants, claim accuracy, provenance
correctness and calibration error.
"""

from __future__ import annotations

import math

__all__ = [
    "recall_at_k",
    "precision_at_k",
    "mean_reciprocal_rank",
    "ndcg_at_k",
    "average_precision",
    "entity_recall",
    "relation_recall",
    "path_recall",
    "evidence_precision",
    "claim_accuracy",
    "provenance_correctness",
    "calibration_error",
    "mean",
]


def recall_at_k(results: list[str], relevant: set[str], k: int | None = None) -> float:
    k = k if k is not None else len(results)
    if not relevant:
        return 0.0
    retrieved = set(results[:k])
    return len(retrieved & relevant) / len(relevant)


def precision_at_k(results: list[str], relevant: set[str], k: int | None = None) -> float:
    k = k if k is not None else len(results)
    if k == 0:
        return 0.0
    retrieved = set(results[:k])
    return len(retrieved & relevant) / k


def mean_reciprocal_rank(results: list[str], relevant: set[str]) -> float:
    for rank, object_id in enumerate(results, start=1):
        if object_id in relevant:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(results: list[str], relevant: set[str], k: int | None = None) -> float:
    k = k if k is not None else len(results)
    if k == 0 or not relevant:
        return 0.0
    dcg = 0.0
    ideal = 0.0
    for i in range(min(k, len(results))):
        if results[i] in relevant:
            dcg += 1.0 / math.log2(i + 2)
    for i in range(min(k, len(relevant))):
        ideal += 1.0 / math.log2(i + 2)
    return dcg / ideal if ideal > 0 else 0.0


def average_precision(results: list[str], relevant: set[str]) -> float:
    hits = 0
    total = 0.0
    for rank, object_id in enumerate(results, start=1):
        if object_id in relevant:
            hits += 1
            total += hits / rank
    return total / len(relevant) if relevant else 0.0


def entity_recall(found_entities: set[str], relevant_entities: set[str]) -> float:
    if not relevant_entities:
        return 0.0
    return len(found_entities & relevant_entities) / len(relevant_entities)


def relation_recall(
    found_relations: set[tuple[str, str, str]], relevant_relations: set[tuple[str, str, str]]
) -> float:
    if not relevant_relations:
        return 0.0
    return len(found_relations & relevant_relations) / len(relevant_relations)


def path_recall(found_paths: list[list[str]], relevant_paths: set[tuple[str, ...]]) -> float:
    if not relevant_paths:
        return 0.0
    found = {tuple(path) for path in found_paths}
    return len(found & relevant_paths) / len(relevant_paths)


def evidence_precision(selected_evidence: list[str], relevant_evidence: set[str]) -> float:
    if not selected_evidence:
        return 0.0
    return len(set(selected_evidence) & relevant_evidence) / len(set(selected_evidence))


def claim_accuracy(predictions: list[bool], labels: list[bool]) -> float:
    if not labels:
        return 0.0
    correct = sum(1 for pred, label in zip(predictions, labels, strict=True) if pred == label)
    return correct / len(labels)


def provenance_correctness(traces: list[bool]) -> float:
    if not traces:
        return 0.0
    return sum(traces) / len(traces)


def calibration_error(predictions: list[float], labels: list[bool]) -> float:
    """Expected calibration error over 10 confidence buckets."""
    if not labels:
        return 0.0
    bins: list[tuple[list[float], list[bool]]] = [([], []) for _ in range(10)]
    for pred, label in zip(predictions, labels, strict=True):
        index = min(9, max(0, int(pred * 10)))
        bins[index][0].append(pred)
        bins[index][1].append(label)
    error = 0.0
    weight_total = 0.0
    for preds, labels_ in bins:
        if not preds:
            continue
        confidence = sum(preds) / len(preds)
        accuracy = sum(1 for label in labels_ if label) / len(labels_)
        weight_total += len(preds)
        error += len(preds) * abs(confidence - accuracy)
    return error / weight_total if weight_total else 0.0


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)
