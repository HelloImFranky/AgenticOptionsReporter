from datetime import date, timedelta

from agentic_options_reporter.models.schemas import Bar, PriceHistory
from agentic_options_reporter.persistence import make_session_factory
from agentic_options_reporter.workflow import _price_range_metrics, run_analysis


def _history(highs_lows: list[tuple[float, float]]) -> PriceHistory:
    start = date(2026, 1, 1)
    bars = [
        Bar(dt=start + timedelta(days=i), open=lo, high=hi, low=lo, close=hi, volume=1000)
        for i, (hi, lo) in enumerate(highs_lows)
    ]
    return PriceHistory(symbol="TEST", bars=bars)


def test_price_range_metrics_uses_last_windows():
    # 25 ascending bars; the last 5 (1w) top out higher than the last 21 (1m) low.
    hl = [(100 + i, 90 + i) for i in range(25)]
    ranges = _price_range_metrics(_history(hl))
    assert ranges["week1_high"] == 124        # last bar high (100+24)
    assert ranges["week1_low"] == 110         # bar 20 low (90+20)
    assert ranges["month1_high"] == 124
    assert ranges["month1_low"] == 94         # bar 4 low (90+4), 21-bar window


def test_price_range_metrics_partial_history():
    # Only 6 bars: 1-week range available, 1-month is not.
    ranges = _price_range_metrics(_history([(10, 5)] * 6))
    assert "week1_high" in ranges and "week1_low" in ranges
    assert "month1_high" not in ranges
    # Fewer than a week of bars yields nothing.
    assert _price_range_metrics(_history([(10, 5)] * 3)) == {}


def test_run_analysis_end_to_end(fake_provider, fake_financial_provider):
    session_factory = make_session_factory("sqlite:///:memory:")

    result = run_analysis(
        symbol="TEST",
        lookback_days=260,
        provider=fake_provider,
        session_factory=session_factory,
        financial_provider=fake_financial_provider,
    )

    assert result.symbol == "TEST"
    assert result.run_id > 0
    assert result.recommendation.action in {"STRONG_BUY", "BUY", "HOLD", "AVOID"}
    assert result.candidates
    assert result.indicators.sma_20 > 0


def test_run_analysis_surfaces_merged_fundamentals(fake_provider, fake_financial_provider):
    session_factory = make_session_factory("sqlite:///:memory:")

    result = run_analysis(
        symbol="TEST",
        provider=fake_provider,
        session_factory=session_factory,
        financial_provider=fake_financial_provider,
    )

    assert result.fundamentals is not None
    assert result.fundamentals.profile.name == "Test Corp"
    assert result.fundamentals.metrics.pe_ratio == 25.0
    # Statements weren't advertised by the fake, so they're absent, not fatal.
    assert result.fundamentals.statements is None
    assert result.data_warnings == []


def test_run_analysis_derives_1w_and_1m_price_ranges(fake_provider, fake_financial_provider):
    """No provider serves 1w/1m high/low, so /analyze derives them from the
    price history and folds them into the fundamentals metrics."""
    session_factory = make_session_factory("sqlite:///:memory:")

    result = run_analysis(
        symbol="TEST",
        provider=fake_provider,
        session_factory=session_factory,
        financial_provider=fake_financial_provider,
    )

    metrics = result.fundamentals.metrics
    for field in ("week1_high", "week1_low", "month1_high", "month1_low"):
        assert getattr(metrics, field) is not None
    # Highs are >= lows, and the 1-month window spans at least the 1-week one.
    assert metrics.week1_high >= metrics.week1_low
    assert metrics.month1_high >= metrics.week1_high or metrics.month1_high >= metrics.month1_low
    assert metrics.month1_low <= metrics.week1_low
    # The provider-supplied metric is preserved alongside the derived ranges.
    assert metrics.pe_ratio == 25.0


def test_run_analysis_persists_fundamentals(fake_provider, fake_financial_provider):
    session_factory = make_session_factory("sqlite:///:memory:")

    result = run_analysis(
        symbol="TEST",
        provider=fake_provider,
        session_factory=session_factory,
        financial_provider=fake_financial_provider,
    )

    from agentic_options_reporter.models.db import AnalysisRun

    with session_factory() as session:
        run = session.get(AnalysisRun, result.run_id)
        # Fundamentals are persisted as JSON alongside the run.
        assert run.fundamentals is not None
        assert run.fundamentals["profile"]["name"] == "Test Corp"
        assert run.fundamentals["metrics"]["pe_ratio"] == 25.0


def test_run_analysis_persists_run(fake_provider, fake_financial_provider):
    session_factory = make_session_factory("sqlite:///:memory:")

    result = run_analysis(
        symbol="TEST",
        provider=fake_provider,
        session_factory=session_factory,
        financial_provider=fake_financial_provider,
    )

    from agentic_options_reporter.models.db import AnalysisRun

    with session_factory() as session:
        run = session.get(AnalysisRun, result.run_id)
        assert run is not None
        assert run.symbol == "TEST"
        assert run.recommendation is not None
        assert len(run.scored_candidates) == len(result.candidates)


def test_run_analysis_persists_trend_volume_and_levels(fake_provider, fake_financial_provider):
    """Regression test: these were previously placeholder-only on replay."""
    session_factory = make_session_factory("sqlite:///:memory:")

    result = run_analysis(
        symbol="TEST",
        provider=fake_provider,
        session_factory=session_factory,
        financial_provider=fake_financial_provider,
    )

    from agentic_options_reporter.models.db import AnalysisRun

    with session_factory() as session:
        run = session.get(AnalysisRun, result.run_id)
        assert run.trend_assessment is not None
        assert run.trend_assessment.direction == result.trend.direction
        assert run.trend_assessment.strength == result.trend.strength
        assert run.trend_assessment.adx == result.trend.adx

        assert run.volume_assessment is not None
        assert run.volume_assessment.relative_volume == result.volume.relative_volume
        assert run.volume_assessment.flags == result.volume.flags

        assert len(run.support_resistance_levels) == len(result.support_resistance)
