"""Volume analysis. See docs/indicators.md#volume-analysis."""

from __future__ import annotations

from ta.volume import OnBalanceVolumeIndicator

from agentic_options_reporter.models.schemas import (
    IndicatorSnapshot,
    PriceHistory,
    VolumeAssessment,
)

DIVERGENCE_LOOKBACK = 10


def analyze_volume(
    history: PriceHistory, indicators: IndicatorSnapshot
) -> VolumeAssessment:
    df = history.to_dataframe()
    if df.empty:
        raise ValueError("Cannot analyze volume from empty price history")

    latest_volume = float(df["volume"].iloc[-1])
    relative_volume = (
        latest_volume / indicators.volume_sma_20 if indicators.volume_sma_20 else 1.0
    )

    flags: list[str] = []
    if relative_volume >= 1.5:
        flags.append("high_volume")
    elif relative_volume <= 0.5:
        flags.append("low_volume")
    else:
        flags.append("normal_volume")

    window = df.tail(DIVERGENCE_LOOKBACK)
    if len(window) >= 2:
        obv = OnBalanceVolumeIndicator(
            window["close"], window["volume"]
        ).on_balance_volume()
        price_change = window["close"].iloc[-1] - window["close"].iloc[0]
        obv_change = obv.iloc[-1] - obv.iloc[0]

        if price_change > 0 and obv_change < 0:
            flags.append("bearish_divergence")
        elif price_change < 0 and obv_change > 0:
            flags.append("bullish_divergence")

    return VolumeAssessment(relative_volume=relative_volume, flags=flags)
