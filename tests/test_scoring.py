from agentic_options_reporter.analysis.indicators import compute_indicators
from agentic_options_reporter.analysis.options import evaluate_chain
from agentic_options_reporter.analysis.risk import compute_risk
from agentic_options_reporter.analysis.scoring import build_recommendation, score_candidates
from agentic_options_reporter.analysis.support_resistance import detect_levels
from agentic_options_reporter.analysis.trend import detect_trend
from agentic_options_reporter.analysis.volume import analyze_volume


def test_score_candidates_bounds_and_sorted(sample_option_chain, uptrend_history):
    indicators = compute_indicators(uptrend_history)
    trend = detect_trend(uptrend_history, indicators)
    volume = analyze_volume(uptrend_history, indicators)
    levels = detect_levels(uptrend_history)

    evaluated = evaluate_chain(sample_option_chain, uptrend_history)
    risk_profiles = compute_risk(evaluated)

    scored = score_candidates(evaluated, risk_profiles, trend, volume, levels)

    assert scored, "expected at least one scored candidate"
    for candidate in scored:
        assert 0 <= candidate.score <= 100
        assert set(candidate.score_breakdown) == {
            "trend_alignment",
            "volume_confirmation",
            "support_resistance_proximity",
            "liquidity",
            "risk_reward",
        }
    scores = [c.score for c in scored]
    assert scores == sorted(scores, reverse=True)


def test_score_candidates_excludes_illiquid(sample_option_chain, uptrend_history):
    sample_option_chain.contracts[0].open_interest = 0
    sample_option_chain.contracts[0].bid = 0.0
    target_symbol = sample_option_chain.contracts[0].contract_symbol

    indicators = compute_indicators(uptrend_history)
    trend = detect_trend(uptrend_history, indicators)
    volume = analyze_volume(uptrend_history, indicators)
    levels = detect_levels(uptrend_history)

    evaluated = evaluate_chain(sample_option_chain, uptrend_history)
    risk_profiles = compute_risk(evaluated)
    scored = score_candidates(evaluated, risk_profiles, trend, volume, levels)

    assert target_symbol not in {c.contract_symbol for c in scored}


def test_build_recommendation_empty_candidates_is_avoid():
    rec = build_recommendation([])
    assert rec.action == "AVOID"
    assert rec.contract_symbol is None
    assert rec.confidence == 0.0


def test_build_recommendation_picks_top_scorer(sample_option_chain, uptrend_history):
    indicators = compute_indicators(uptrend_history)
    trend = detect_trend(uptrend_history, indicators)
    volume = analyze_volume(uptrend_history, indicators)
    levels = detect_levels(uptrend_history)

    evaluated = evaluate_chain(sample_option_chain, uptrend_history)
    risk_profiles = compute_risk(evaluated)
    scored = score_candidates(evaluated, risk_profiles, trend, volume, levels)

    rec = build_recommendation(scored)
    assert rec.contract_symbol == scored[0].contract_symbol
    assert rec.action in {"STRONG_BUY", "BUY", "HOLD", "AVOID"}
