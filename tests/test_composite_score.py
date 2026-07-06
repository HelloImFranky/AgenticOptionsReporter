from datetime import datetime

import pytest

from agentic_options_reporter.analysis.composite_score import (
    WEIGHTING_PROFILES,
    compute_composite_score,
)
from agentic_options_reporter.models.schemas import DomainScore


def _domain_score(domain: str, score: float, confidence: float = 100.0) -> DomainScore:
    return DomainScore(
        domain=domain, score=score, confidence=confidence, evidence=[], factors=[],
        source="quant", generated_at=datetime(2026, 1, 1),
    )


@pytest.mark.parametrize("profile", list(WEIGHTING_PROFILES))
def test_weighting_profiles_sum_to_one(profile):
    assert sum(WEIGHTING_PROFILES[profile].values()) == pytest.approx(1.0, abs=1e-9)


def test_compute_composite_score_full_coverage_is_weighted_average():
    domain_scores = {domain: _domain_score(domain, 80.0) for domain in WEIGHTING_PROFILES["swing"]}
    result = compute_composite_score(domain_scores, source="quant", weighting_profile="swing")
    assert result.composite_score == pytest.approx(80.0)
    assert result.confidence == pytest.approx(100.0)
    assert result.source == "quant"
    assert set(result.domain_scores) == set(domain_scores)


def test_compute_composite_score_renormalizes_over_present_domains():
    """Only technical+liquidity present (swing weights 0.20+0.10=0.30): the
    composite should be the weighted average of just those two, not
    diluted by the missing domains' weight."""
    domain_scores = {
        "technical": _domain_score("technical", 100.0),
        "liquidity": _domain_score("liquidity", 0.0),
    }
    result = compute_composite_score(domain_scores, source="quant", weighting_profile="swing")
    # weighted: (0.20*100 + 0.10*0) / 0.30 = 66.67
    assert result.composite_score == pytest.approx(66.67, abs=0.01)


def test_compute_composite_score_confidence_discounted_by_coverage():
    """Two domains at full (100) confidence, but only 30% of swing's total
    weight is covered -> confidence should be discounted to ~30, not 100."""
    domain_scores = {
        "technical": _domain_score("technical", 90.0, confidence=100.0),
        "liquidity": _domain_score("liquidity", 90.0, confidence=100.0),
    }
    result = compute_composite_score(domain_scores, source="quant", weighting_profile="swing")
    assert result.confidence == pytest.approx(30.0, abs=0.5)


def test_compute_composite_score_empty_domains_is_zero_confidence_avoid():
    result = compute_composite_score({}, source="quant", weighting_profile="swing")
    assert result.composite_score == 0.0
    assert result.confidence == 0.0
    assert result.recommendation_action == "AVOID"
    assert result.domain_scores == {}


@pytest.mark.parametrize(
    "score,expected_action",
    [(90.0, "STRONG_BUY"), (65.0, "BUY"), (45.0, "HOLD"), (10.0, "AVOID")],
)
def test_compute_composite_score_action_thresholds(score, expected_action):
    domain_scores = {"technical": _domain_score("technical", score)}
    result = compute_composite_score(domain_scores, source="quant", weighting_profile="swing")
    assert result.recommendation_action == expected_action


def test_compute_composite_score_explainability_ranks_domains():
    domain_scores = {
        "technical": _domain_score("technical", 95.0),
        "risk": _domain_score("risk", 20.0),
    }
    result = compute_composite_score(domain_scores, source="quant", weighting_profile="swing")
    assert "strongest contributor" in result.explainability[0]
    assert "Technical" in result.explainability[0]
    assert "weakest contributor" in result.explainability[-1]
    assert "Risk" in result.explainability[-1]


def test_compute_composite_score_ignores_unknown_domain_ids():
    domain_scores = {
        "technical": _domain_score("technical", 80.0),
        "not_a_real_domain": _domain_score("technical", 0.0),  # arbitrary id, not in any profile
    }
    result = compute_composite_score(domain_scores, source="quant", weighting_profile="swing")
    assert "not_a_real_domain" not in result.domain_scores
