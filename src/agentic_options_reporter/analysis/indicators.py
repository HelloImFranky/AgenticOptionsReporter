"""Technical indicators. See docs/indicators.md for definitions."""

from __future__ import annotations

import pandas as pd
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import ADXIndicator, EMAIndicator, MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator

from agentic_options_reporter.models.schemas import IndicatorSnapshot, PriceHistory


def compute_indicators(history: PriceHistory) -> IndicatorSnapshot:
    df = history.to_dataframe()
    if len(df) < 2:
        raise ValueError("Need at least 2 bars to compute indicators")

    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    sma_20 = SMAIndicator(close, window=min(20, len(df))).sma_indicator()
    sma_50 = SMAIndicator(close, window=min(50, len(df))).sma_indicator()
    sma_200 = (
        SMAIndicator(close, window=200).sma_indicator() if len(df) >= 200 else None
    )
    ema_12 = EMAIndicator(close, window=min(12, len(df))).ema_indicator()
    ema_26 = EMAIndicator(close, window=min(26, len(df))).ema_indicator()

    adx_window = min(14, max(2, len(df) - 1))
    adx = ADXIndicator(high, low, close, window=adx_window).adx()

    rsi = RSIIndicator(close, window=min(14, len(df) - 1)).rsi()

    macd_ind = MACD(close, window_slow=min(26, len(df)), window_fast=min(12, len(df)), window_sign=min(9, len(df)))

    stoch = StochasticOscillator(
        high, low, close, window=min(14, len(df)), smooth_window=min(3, len(df))
    )

    bb = BollingerBands(close, window=min(20, len(df)))

    atr = AverageTrueRange(high, low, close, window=min(14, len(df) - 1))

    obv = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    volume_sma_20 = SMAIndicator(volume, window=min(20, len(df))).sma_indicator()

    def last(series: pd.Series) -> float:
        value = series.dropna()
        return float(value.iloc[-1]) if not value.empty else 0.0

    return IndicatorSnapshot(
        sma_20=last(sma_20),
        sma_50=last(sma_50),
        sma_200=last(sma_200) if sma_200 is not None else None,
        ema_12=last(ema_12),
        ema_26=last(ema_26),
        adx_14=last(adx),
        rsi_14=last(rsi),
        macd=last(macd_ind.macd()),
        macd_signal=last(macd_ind.macd_signal()),
        macd_histogram=last(macd_ind.macd_diff()),
        stoch_k=last(stoch.stoch()),
        stoch_d=last(stoch.stoch_signal()),
        bb_upper=last(bb.bollinger_hband()),
        bb_middle=last(bb.bollinger_mavg()),
        bb_lower=last(bb.bollinger_lband()),
        atr_14=last(atr.average_true_range()),
        obv=last(obv),
        volume_sma_20=last(volume_sma_20),
    )
