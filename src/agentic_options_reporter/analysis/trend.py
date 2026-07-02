"""Trend classification. See docs/indicators.md#trend-classification."""

from __future__ import annotations

from agentic_options_reporter.models.schemas import (
    IndicatorSnapshot,
    PriceHistory,
    TrendAssessment,
    TrendStrength,
)


def _strength(adx: float) -> TrendStrength:
    if adx > 40:
        return "strong"
    if adx >= 20:
        return "moderate"
    return "weak"


def detect_trend(history: PriceHistory, indicators: IndicatorSnapshot) -> TrendAssessment:
    df = history.to_dataframe()
    if df.empty:
        raise ValueError("Cannot detect trend from empty price history")
    close = float(df["close"].iloc[-1])

    bullish = close > indicators.sma_20 > indicators.sma_50 and indicators.adx_14 > 20
    bearish = close < indicators.sma_20 < indicators.sma_50 and indicators.adx_14 > 20

    if bullish:
        direction = "bullish"
    elif bearish:
        direction = "bearish"
    else:
        direction = "neutral"

    return TrendAssessment(
        direction=direction,
        strength=_strength(indicators.adx_14),
        adx=indicators.adx_14,
    )
