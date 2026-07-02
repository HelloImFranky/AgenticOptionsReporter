from datetime import date, timedelta

from agentic_options_reporter.analysis.indicators import compute_indicators
from agentic_options_reporter.analysis.volume import analyze_volume
from agentic_options_reporter.models.schemas import Bar, PriceHistory


def test_analyze_volume_flags_high_volume(uptrend_history):
    bars = list(uptrend_history.bars)
    last = bars[-1]
    bars[-1] = Bar(
        dt=last.dt,
        open=last.open,
        high=last.high,
        low=last.low,
        close=last.close,
        volume=last.volume * 5,
    )
    history = PriceHistory(symbol="TEST", bars=bars)

    indicators = compute_indicators(history)
    assessment = analyze_volume(history, indicators)

    assert assessment.relative_volume > 1.5
    assert "high_volume" in assessment.flags


def test_analyze_volume_flags_bearish_divergence():
    # Price grinds slightly higher on tiny up-day volume, but sells off hard
    # on massive down-day volume: net price is up, net OBV is down.
    start = date(2024, 1, 1)
    bars = []
    price = 100.0
    for i in range(40):
        if i % 2 == 0:
            price += 2.0
            volume = 1_000
        else:
            price -= 1.0
            volume = 5_000_000
        bars.append(
            Bar(dt=start + timedelta(days=i), open=price, high=price + 1, low=price - 1, close=price, volume=volume)
        )
    history = PriceHistory(symbol="TEST", bars=bars)
    indicators = compute_indicators(history)
    assessment = analyze_volume(history, indicators)

    assert "bearish_divergence" in assessment.flags
