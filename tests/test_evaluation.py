"""Evaluation metrics and benchmark determinism tests."""

import pytest

from nexus_knowledge.eval import metrics
from nexus_knowledge.eval.benchmarks import run_benchmarks


class TestRankingMetrics:
    def test_recall_at_k(self):
        assert metrics.recall_at_k(["a", "b", "c"], {"a", "b"}, k=1) == 0.5
        assert metrics.recall_at_k(["a", "b", "c"], {"a", "b"}, k=2) == 1.0
        assert metrics.recall_at_k(["a"], set()) == 0.0

    def test_precision_at_k(self):
        assert metrics.precision_at_k(["a", "b", "c"], {"a", "b"}, k=2) == 1.0
        assert metrics.precision_at_k(["a", "b"], {"c"}, k=2) == 0.0
        assert metrics.precision_at_k(["a"], {"a"}, k=0) == 0.0

    def test_mean_reciprocal_rank(self):
        assert metrics.mean_reciprocal_rank(["x", "a"], {"a"}) == 0.5
        assert metrics.mean_reciprocal_rank(["x"], {"a"}) == 0.0
        assert metrics.mean_reciprocal_rank(["a"], {"a"}) == 1.0

    def test_ndcg_at_k(self):
        assert metrics.ndcg_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0
        assert metrics.ndcg_at_k(["b", "x"], {"a", "b"}, k=2) < 1.0
        assert metrics.ndcg_at_k([], {"a"}) == 0.0

    def test_average_precision(self):
        assert metrics.average_precision(["a", "b", "c"], {"a"}) == 1.0
        assert metrics.average_precision(["b", "a"], {"a"}) == 0.5


class TestGraphMetrics:
    def test_entity_recall(self):
        assert metrics.entity_recall({"a", "b"}, {"a", "c"}) == 0.5
        assert metrics.entity_recall(set(), {"a"}) == 0.0

    def test_relation_recall(self):
        found = {("a", "p", "b")}
        relevant = {("a", "p", "b"), ("a", "p", "c")}
        assert metrics.relation_recall(found, relevant) == 0.5

    def test_path_recall(self):
        assert metrics.path_recall([["a", "b"]], {("a", "b")}) == 1.0
        assert metrics.path_recall([["a", "b"]], {("x", "y")}) == 0.0

    def test_evidence_precision(self):
        assert metrics.evidence_precision(["e1", "e2", "e3"], {"e1"}) == pytest.approx(1 / 3)
        assert metrics.evidence_precision([], {"e1"}) == 0.0


class TestKnowledgeMetrics:
    def test_claim_accuracy(self):
        assert metrics.claim_accuracy([True, False], [True, False]) == 1.0
        assert metrics.claim_accuracy([], []) == 0.0

    def test_provenance_correctness(self):
        assert metrics.provenance_correctness([True, False]) == 0.5
        assert metrics.provenance_correctness([]) == 0.0

    def test_calibration_error_perfect(self):
        assert metrics.calibration_error([1.0, 0.0], [True, False]) == 0.0

    def test_calibration_error_miscalibrated(self):
        error = metrics.calibration_error([0.9, 0.9, 0.1, 0.1], [True, False, True, False])
        assert error > 0.0

    def test_mean(self):
        assert metrics.mean([1.0, 2.0, 3.0]) == 2.0
        assert metrics.mean([]) == 0.0


class TestBenchmarks:
    def test_deterministic(self):
        first = run_benchmarks().to_dict()
        second = run_benchmarks().to_dict()
        assert first == second

    def test_report_shape(self):
        report = run_benchmarks().to_dict()
        assert set(report) == {"graph", "knowledge", "retrieval"}
        assert set(report["retrieval"]) == {"lexical", "vector", "graph", "hybrid"}
        for config, metrics_dict in report["retrieval"].items():
            assert set(metrics_dict) == {"recall_at_5", "precision_at_5", "mrr", "ndcg_at_5"}

    def test_graph_metrics_present(self):
        report = run_benchmarks().to_dict()
        assert set(report["graph"]) == {"entity_recall", "relation_recall", "evidence_precision"}

    def test_knowledge_metrics_present(self):
        report = run_benchmarks().to_dict()
        assert set(report["knowledge"]) == {"claim_accuracy", "calibration_error", "provenance_correctness"}
