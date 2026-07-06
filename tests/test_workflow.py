from agentic_options_reporter.persistence import make_session_factory
from agentic_options_reporter.workflow import run_analysis


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
