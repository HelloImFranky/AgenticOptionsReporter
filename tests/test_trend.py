from agentic_options_reporter.analysis.indicators import compute_indicators
from agentic_options_reporter.analysis.trend import detect_trend


def test_detects_bullish_trend(uptrend_history):
    indicators = compute_indicators(uptrend_history)
    trend = detect_trend(uptrend_history, indicators)
    assert trend.direction == "bullish"
    assert trend.strength in {"weak", "moderate", "strong"}


def test_detects_bearish_trend(downtrend_history):
    indicators = compute_indicators(downtrend_history)
    trend = detect_trend(downtrend_history, indicators)
    assert trend.direction == "bearish"


def test_strength_buckets():
    from agentic_options_reporter.analysis.trend import _strength

    assert _strength(10) == "weak"
    assert _strength(25) == "moderate"
    assert _strength(45) == "strong"
