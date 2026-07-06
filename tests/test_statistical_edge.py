from datetime import date, datetime

from agentic_options_reporter.analysis.statistical_edge import statistical_edge_domain_score
from agentic_options_reporter.models.schemas import PastRunOutcome


def _past_run_on(run_id: int, day_offset: int, action: str = "BUY", option_type: str = "call") -> PastRunOutcome:
    from datetime import timedelta

    generated_at = datetime(2024, 1, 1) + timedelta(days=day_offset)
    return PastRunOutcome(
        run_id=run_id, generated_at=generated_at, action=action,
        option_type=option_type, contract_symbol=f"TESTC{run_id:08d}",
    )


def test_statistical_edge_omits_historical_subfactors_below_minimum_sample(uptrend_history):
    """Fewer than 5 qualifying past runs -> historical_win_rate/expectancy/
    pattern_success are omitted, but the domain is still returned because
    the Monte Carlo bootstrap doesn't need past runs at all."""
    past_runs = [_past_run_on(i, day_offset=5 + i * 10) for i in range(3)]

    result = statistical_edge_domain_score(
        "call", uptrend_history, days_to_expiration=30, breakeven=105.0,
        underlying_price=120.0, past_runs=past_runs,
    )

    assert result is not None
    assert result.domain == "statistical_edge"
    factor_names = {f.name for f in result.factors}
    assert "monte_carlo_confidence" in factor_names
    assert "historical_win_rate" not in factor_names
    assert "expectancy" not in factor_names
    assert "pattern_success" not in factor_names
    assert result.confidence <= 70  # capped


def test_statistical_edge_includes_historical_subfactors_at_minimum_sample(uptrend_history):
    """5+ qualifying past runs (bullish calls in a steady uptrend) ->
    historical_win_rate/expectancy/pattern_success all present, and the
    domain reads favorably since the history is a clean uptrend."""
    past_runs = [_past_run_on(i, day_offset=5 + i * 20) for i in range(6)]

    result = statistical_edge_domain_score(
        "call", uptrend_history, days_to_expiration=30, breakeven=105.0,
        underlying_price=120.0, past_runs=past_runs,
    )

    assert result is not None
    factor_names = {f.name for f in result.factors}
    assert "historical_win_rate" in factor_names
    assert "expectancy" in factor_names
    assert "pattern_success" in factor_names
    win_rate_factor = next(f for f in result.factors if f.name == "historical_win_rate")
    # An uptrend should produce a high win rate for bullish (call) recommendations.
    assert win_rate_factor.value > 0.5


def test_statistical_edge_none_when_nothing_computable():
    """No past runs and too little price history for the Monte Carlo
    bootstrap -> the whole domain is omitted, not fabricated."""
    from agentic_options_reporter.models.schemas import Bar, PriceHistory

    tiny_history = PriceHistory(
        symbol="TEST",
        bars=[Bar(dt=date(2024, 1, i + 1), open=100, high=101, low=99, close=100, volume=1000) for i in range(5)],
    )
    result = statistical_edge_domain_score(
        "call", tiny_history, days_to_expiration=30, breakeven=105.0, underlying_price=100.0, past_runs=[],
    )
    assert result is None


def test_monte_carlo_bootstrap_is_deterministic(uptrend_history):
    """Same inputs -> same Monte Carlo confidence (fixed seed), so the
    domain score doesn't jitter between runs against identical data."""
    kwargs = dict(
        option_type="call", history=uptrend_history, days_to_expiration=30,
        breakeven=105.0, underlying_price=120.0, past_runs=[],
    )
    first = statistical_edge_domain_score(**kwargs)
    second = statistical_edge_domain_score(**kwargs)
    assert first.score == second.score
    assert first.confidence == second.confidence


def test_statistical_edge_weight_and_confidence_are_capped(uptrend_history):
    past_runs = [_past_run_on(i, day_offset=5 + i * 20) for i in range(6)]
    result = statistical_edge_domain_score(
        "call", uptrend_history, days_to_expiration=30, breakeven=105.0,
        underlying_price=120.0, past_runs=past_runs,
    )
    assert result.confidence <= 70
