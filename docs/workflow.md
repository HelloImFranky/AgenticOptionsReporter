# Workflow

Narrative companion to `specs/workflow.yaml`, which is the authoritative
contract for step names, inputs, and outputs.

## Steps

1. **Download data** — fetch daily OHLCV history and the live option chain
   for the requested symbol via `MarketDataProvider`.
2. **Compute indicators** — trend (SMA/EMA, ADX), momentum (RSI, MACD,
   stochastic), volatility (Bollinger Bands, ATR), and volume (OBV, volume
   SMA) indicators from OHLCV history.
3. **Detect trend** — classify direction (`bullish`/`bearish`/`neutral`) and
   strength from moving-average alignment and ADX.
4. **Analyze volume** — flag volume relative to its trailing average and
   detect volume/price divergence.
5. **Detect support/resistance** — derive candidate levels from local
   swing highs/lows (pivot detection) over a lookback window.
6. **Evaluate option chain and Greeks** — for each contract, sanity-check
   pricing (bid/ask spread, open interest, volume) and compute Greeks
   (delta, gamma, theta, vega, rho) via Black-Scholes when the provider
   does not supply them.
7. **Compute risk** — max loss, max gain, breakeven(s), probability of
   profit approximation, and reward:risk ratio per candidate contract or
   spread.
8. **Score opportunity** — combine trend alignment, volume confirmation,
   proximity to support/resistance, liquidity, and risk:reward into a
   single 0-100 opportunity score (see `specs/scoring.yaml`).
9. **Generate recommendation** — rank scored candidates and produce a
   structured `Recommendation` (action, contract, rationale, confidence).
10. **Persist results** — write the run, indicator snapshot, scored
    candidates, and recommendation to the database via `persistence.py`.

## Orchestration

`workflow.py::run_analysis(symbol)` executes steps 1-10 in order and
returns an `AnalysisResult`. Each step is a pure function over typed
Pydantic models so it can be unit tested independently of network access.
