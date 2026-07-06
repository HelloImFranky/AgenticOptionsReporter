from agentic_options_reporter.analysis.indicators import compute_indicators
from agentic_options_reporter.analysis.options import evaluate_chain
from agentic_options_reporter.analysis.risk import compute_risk
from agentic_options_reporter.analysis.scoring import build_recommendation, score_candidates
from agentic_options_reporter.analysis.support_resistance import detect_levels
from agentic_options_reporter.analysis.trend import detect_trend
from agentic_options_reporter.analysis.volume import analyze_volume

# With no fundamentals/macro/news/benchmark/past-run data supplied, only
# these domains have anything to compute from: technical/risk/liquidity
# are always candidate-level; statistical_edge still gets a Monte Carlo
# bootstrap from the price history alone (historical_win_rate/expectancy/
# pattern_success are omitted below the 5-run minimum, but that's not the
# whole domain). fundamental/macro/sentiment/relative_strength all need
# data this test doesn't supply, so they're omitted entirely.
_MINIMAL_DOMAINS = {"technical", "risk", "liquidity", "statistical_edge"}


def _score(sample_option_chain, history):
    indicators = compute_indicators(history)
    trend = detect_trend(history, indicators)
    volume = analyze_volume(history, indicators)
    levels = detect_levels(history)

    evaluated = evaluate_chain(sample_option_chain, history)
    risk_profiles = compute_risk(evaluated)
    return score_candidates(evaluated, risk_profiles, trend, volume, levels, history, indicators)


def test_score_candidates_bounds_and_sorted(sample_option_chain, uptrend_history):
    scored = _score(sample_option_chain, uptrend_history)

    assert scored, "expected at least one scored candidate"
    for candidate in scored:
        assert 0 <= candidate.score <= 100
        assert set(candidate.domain_scores) == _MINIMAL_DOMAINS
        for domain_id, domain_score in candidate.domain_scores.items():
            assert domain_score.domain == domain_id
            assert domain_score.source == "quant"
            assert 0 <= domain_score.score <= 100
            assert 0 <= domain_score.confidence <= 100
    scores = [c.score for c in scored]
    assert scores == sorted(scores, reverse=True)


def test_score_candidates_excludes_illiquid(sample_option_chain, uptrend_history):
    sample_option_chain.contracts[0].open_interest = 0
    sample_option_chain.contracts[0].bid = 0.0
    target_symbol = sample_option_chain.contracts[0].contract_symbol

    scored = _score(sample_option_chain, uptrend_history)

    assert target_symbol not in {c.contract_symbol for c in scored}


def test_build_recommendation_empty_candidates_is_avoid():
    rec, trade_quality = build_recommendation([])
    assert rec.action == "AVOID"
    assert rec.contract_symbol is None
    assert rec.confidence == 0.0
    assert trade_quality is None


def test_build_recommendation_picks_top_scorer(sample_option_chain, uptrend_history):
    scored = _score(sample_option_chain, uptrend_history)

    rec, trade_quality = build_recommendation(scored)
    assert rec.contract_symbol == scored[0].contract_symbol
    assert rec.action in {"STRONG_BUY", "BUY", "HOLD", "AVOID"}
    assert trade_quality is not None
    assert trade_quality.contract_symbol == scored[0].contract_symbol
    assert trade_quality.composite_score == scored[0].score
    assert trade_quality.source == "quant"
    assert trade_quality.weighting_profile == "swing"
