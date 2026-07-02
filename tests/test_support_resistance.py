from datetime import date, timedelta

from agentic_options_reporter.analysis.support_resistance import detect_levels
from agentic_options_reporter.models.schemas import Bar, PriceHistory


def _price_at(i: int) -> float:
    if i <= 10:
        return 120 - 2 * i          # 120 -> 100
    if i <= 20:
        return 100 + 1 * (i - 10)   # 100 -> 110
    if i <= 30:
        return 110 - 1 * (i - 20)   # 110 -> 100
    if i <= 40:
        return 100 + 1 * (i - 30)   # 100 -> 110
    return 110 - 1 * (i - 40)       # 110 -> 105


def test_detect_levels_finds_repeated_support_and_resistance():
    start = date(2024, 1, 1)
    bars = [
        Bar(
            dt=start + timedelta(days=i),
            open=_price_at(i),
            high=_price_at(i) + 0.5,
            low=_price_at(i) - 0.5,
            close=_price_at(i),
            volume=1_000_000,
        )
        for i in range(46)
    ]
    history = PriceHistory(symbol="TEST", bars=bars)

    levels = detect_levels(history)

    supports = [lvl for lvl in levels if lvl.level_type == "support"]
    resistances = [lvl for lvl in levels if lvl.level_type == "resistance"]

    assert any(abs(lvl.price - 100) < 1 and lvl.touches >= 2 for lvl in supports)
    assert any(abs(lvl.price - 110) < 1 and lvl.touches >= 2 for lvl in resistances)


def test_detect_levels_returns_empty_for_short_history():
    bars = [
        Bar(dt=date(2024, 1, 1) + timedelta(days=i), open=1, high=1, low=1, close=1, volume=1)
        for i in range(3)
    ]
    history = PriceHistory(symbol="TEST", bars=bars)
    assert detect_levels(history) == []
