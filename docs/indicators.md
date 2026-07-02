# Indicators

Computed by `src/agentic_options_reporter/analysis/indicators.py` from daily
OHLCV history using the `ta` library. All indicators are returned as a typed
`IndicatorSnapshot` (see `specs/database.yaml` for the persisted schema).

## Trend

- **SMA 20 / SMA 50 / SMA 200** — simple moving averages.
- **EMA 12 / EMA 26** — exponential moving averages (MACD inputs).
- **ADX 14** — trend strength (>25 considered trending).

## Momentum

- **RSI 14** — overbought (>70) / oversold (<30).
- **MACD (12, 26, 9)** — MACD line, signal line, histogram.
- **Stochastic %K/%D (14, 3)**.

## Volatility

- **Bollinger Bands (20, 2σ)** — upper/middle/lower bands, %B, bandwidth.
- **ATR 14** — average true range, used for stop/target sizing and risk
  calculations.

## Volume

- **OBV** — on-balance volume.
- **Volume SMA 20** — trailing average volume for relative-volume checks.

## Trend classification (`analysis/trend.py`)

Direction is `bullish` when price > SMA20 > SMA50 and ADX > 20; `bearish`
under the mirrored condition; otherwise `neutral`. Strength is `ADX`
bucketed into `weak` (<20), `moderate` (20-40), `strong` (>40).

## Volume analysis (`analysis/volume.py`)

Relative volume = latest volume / Volume SMA 20. Flags:
- `high_volume` when relative volume >= 1.5
- `low_volume` when relative volume <= 0.5
- `bullish_divergence` / `bearish_divergence` when price and OBV trend in
  opposite directions over the lookback window.

## Support/Resistance (`analysis/support_resistance.py`)

Levels are derived from local pivot highs/lows: a bar is a pivot high/low
if it is the max/min within a symmetric window (default 5 bars either
side). Levels within a tolerance band (default 0.5%) are merged and ranked
by touch count and recency.
