"""Guards for Flet-layout gotchas in the frontend that unit tests can catch
without a running Flet session (constructing the controls is enough)."""

from agentic_options_reporter.frontend.app import (
    _fundamentals_controls,
    _insider_timeseries_chart,
)

_SERIES = [
    {"date": "2026-05-01", "net": 500.0, "is_buy": True},
    {"date": "2026-06-01", "net": -600.0, "is_buy": False},
]


def test_insider_chart_uses_fixed_height_not_expand():
    """Regression: the chart lives in the scrolling results column, where an
    expanding child throws Flutter's unbounded-height error and blanks the
    whole analysis result (recommendation + candidates included). It must use
    a fixed height and never expand."""
    chart = _insider_timeseries_chart(_SERIES)
    assert not chart.expand
    assert chart.height
    assert len(chart.bar_groups) == len(_SERIES)


def test_fundamentals_controls_build_without_error():
    fundamentals = {
        "ticker": "AAPL",
        "metrics": {"pe_ratio": 30.5, "week1_high": 214.3, "week1_low": 205.1},
        "earnings_calendar": {"next_date": "2026-08-01", "eps_estimate": 1.6},
        "insider_activity": {"net_shares": -100.0, "transactions": [
            {"name": "A", "transaction_type": "sell", "shares": 1000, "filed_at": "2026-06-01"},
            {"name": "B", "transaction_type": "buy", "shares": 900, "filed_at": "2026-05-01"},
        ]},
    }
    controls = _fundamentals_controls(fundamentals, [])
    assert controls  # renders something, no exception
