"""Support/resistance detection. See docs/indicators.md#support-resistance."""

from __future__ import annotations

from agentic_options_reporter.models.schemas import PriceHistory, SupportResistanceLevel

DEFAULT_PIVOT_WINDOW = 5
DEFAULT_TOLERANCE_PCT = 0.005


def _find_pivots(values: list[float], window: int, is_high: bool) -> list[tuple[int, float]]:
    pivots: list[tuple[int, float]] = []
    n = len(values)
    for i in range(window, n - window):
        segment = values[i - window : i + window + 1]
        target = max(segment) if is_high else min(segment)
        if values[i] == target:
            pivots.append((i, values[i]))
    return pivots


def _cluster(
    pivots: list[tuple[int, float]], tolerance_pct: float, level_type: str
) -> list[SupportResistanceLevel]:
    clusters: list[dict] = []
    for idx, price in sorted(pivots, key=lambda p: p[1]):
        for cluster in clusters:
            center = sum(cluster["prices"]) / len(cluster["prices"])
            if center > 0 and abs(price - center) / center <= tolerance_pct:
                cluster["prices"].append(price)
                cluster["indices"].append(idx)
                break
        else:
            clusters.append({"prices": [price], "indices": [idx]})

    return [
        SupportResistanceLevel(
            price=sum(cluster["prices"]) / len(cluster["prices"]),
            level_type=level_type,
            touches=len(cluster["prices"]),
            last_touch_index=max(cluster["indices"]),
        )
        for cluster in clusters
    ]


def detect_levels(
    history: PriceHistory,
    window: int = DEFAULT_PIVOT_WINDOW,
    tolerance_pct: float = DEFAULT_TOLERANCE_PCT,
) -> list[SupportResistanceLevel]:
    df = history.to_dataframe()
    if len(df) < 2 * window + 1:
        return []

    highs = df["high"].tolist()
    lows = df["low"].tolist()

    resistance = _cluster(_find_pivots(highs, window, is_high=True), tolerance_pct, "resistance")
    support = _cluster(_find_pivots(lows, window, is_high=False), tolerance_pct, "support")

    levels = support + resistance
    levels.sort(key=lambda lvl: (lvl.touches, lvl.last_touch_index), reverse=True)
    return levels
