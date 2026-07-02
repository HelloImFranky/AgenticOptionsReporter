from agentic_options_reporter.analysis.indicators import compute_indicators


def test_compute_indicators_returns_populated_snapshot(uptrend_history):
    snapshot = compute_indicators(uptrend_history)

    assert snapshot.sma_20 > 0
    assert snapshot.sma_50 > 0
    assert snapshot.sma_200 is not None
    assert 0 <= snapshot.rsi_14 <= 100
    assert snapshot.atr_14 > 0
    assert snapshot.bb_lower <= snapshot.bb_middle <= snapshot.bb_upper


def test_compute_indicators_requires_minimum_bars():
    from agentic_options_reporter.models.schemas import Bar, PriceHistory
    from datetime import date

    history = PriceHistory(
        symbol="X",
        bars=[Bar(dt=date(2024, 1, 1), open=1, high=1, low=1, close=1, volume=1)],
    )
    try:
        compute_indicators(history)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_uptrend_price_above_moving_averages(uptrend_history):
    snapshot = compute_indicators(uptrend_history)
    df = uptrend_history.to_dataframe()
    last_close = df["close"].iloc[-1]
    assert last_close > snapshot.sma_50
