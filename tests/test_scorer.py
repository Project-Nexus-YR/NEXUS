"""Investigation scorer tests."""

import pytest

from nexus_knowledge.domain.knowledge_gap import GapKind, Investigation, KnowledgeGap
from nexus_knowledge.knowledge.scorer import (
    BaselineGainEstimator,
    CentralityInvestigationScorer,
    InvestigationScorer,
    RandomInvestigationScorer,
)


def _gap(kind=GapKind.LOW_CONFIDENCE, uncertainty=0.8, importance=0.7, cost=2.0):
    gap = KnowledgeGap(
        kind=kind,
        description="d",
        reason="r",
        uncertainty=uncertainty,
        importance=importance,
        estimated_cost=cost,
    )
    gap.candidate_investigations = [
        Investigation(gap_id=gap.id, description="investigate", estimated_cost=cost)
    ]
    return gap


class TestBaselineGainEstimator:
    @pytest.mark.parametrize(
        "kind",
        [
            GapKind.MISSING_RELATION,
            GapKind.LOW_CONFIDENCE,
            GapKind.UNSUPPORTED_CLAIM,
            GapKind.CONTRADICTION,
            GapKind.STALE,
            GapKind.DISCONNECTED_ENTITY,
            GapKind.MISSING_EVIDENCE,
            GapKind.LOW_DIVERSITY,
        ],
    )
    def test_gain_in_unit_interval(self, kind):
        gap = _gap(kind=kind)
        gain = BaselineGainEstimator().expected_information_gain(gap)
        assert 0.0 <= gain <= 1.0

    def test_unknown_kind_default(self):
        gap = _gap(kind="UNKNOWN_KIND")
        assert BaselineGainEstimator().expected_information_gain(gap) == 0.5

    def test_uncertainty_reduction(self):
        assert BaselineGainEstimator().uncertainty_reduction(_gap(uncertainty=0.9)) == 0.9


class TestInvestigationScorer:
    def test_formula(self):
        gap = _gap(uncertainty=1.0, importance=1.0, cost=2.0)
        gap.candidate_investigations = [
            Investigation(gap_id=gap.id, description="i", estimated_cost=2.0)
        ]
        scored = InvestigationScorer().score_gaps(gaps=[gap])
        gain = BaselineGainEstimator().expected_information_gain(gap)
        expected = (gain * 1.0 * 1.0) / 2.0
        assert scored[0].score == pytest.approx(expected)
        assert scored[0].components["estimated_cost"] == 2.0

    def test_commits_components(self):
        gap = _gap()
        scored = InvestigationScorer().score_gaps([gap])[0]
        investigation = scored.investigation
        assert investigation.score == scored.score
        assert (
            investigation.expected_information_gain
            == scored.components["expected_information_gain"]
        )

    def test_scores_sort_descending(self):
        gaps = [
            _gap(uncertainty=0.9, importance=0.9, cost=1.0),
            _gap(uncertainty=0.1, importance=0.1, cost=10.0),
        ]
        scored = InvestigationScorer().score_gaps(gaps)
        scores = [s.score for s in scored]
        assert scores == sorted(scores, reverse=True)

    def test_to_dict(self):
        gap = _gap()
        payload = InvestigationScorer().score(gap.candidate_investigations[0], gap).to_dict()
        assert set(payload) == {"investigation_id", "gap_id", "description", "score", "components"}


class TestRandomInvestigationScorer:
    def test_deterministic(self):
        gap = _gap()
        first = RandomInvestigationScorer(seed=42).score_gaps([gap])[0].score
        second = RandomInvestigationScorer(seed=42).score_gaps([gap])[0].score
        assert first == second

    def test_seed_changes_order(self):
        gap = _gap()
        a = RandomInvestigationScorer(seed=1).score_gaps([gap])[0].score
        b = RandomInvestigationScorer(seed=2).score_gaps([gap])[0].score
        assert a != b


class TestCentralityInvestigationScorer:
    def test_importance_over_cost(self):
        gap = _gap(importance=0.8, cost=4.0)
        scored = CentralityInvestigationScorer().score(gap.candidate_investigations[0], gap)
        assert scored.score == pytest.approx(0.8 / 4.0)

    def test_zero_cost_protected(self):
        gap = _gap(importance=0.5, cost=0.0)
        investigation = Investigation(gap_id=gap.id, description="i", estimated_cost=0.0)
        scored = CentralityInvestigationScorer().score(investigation, gap)
        assert scored.score == 0.5 / 1e-6
