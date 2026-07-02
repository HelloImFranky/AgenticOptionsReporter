# Option Analysis

Implemented by `src/agentic_options_reporter/analysis/options.py`,
`risk.py`, and `scoring.py`.

## Chain evaluation

For each contract in the chain fetched via `MarketDataProvider`:

- **Liquidity filter**: reject contracts with open interest below a
  configurable floor, zero bid, or a bid/ask spread wider than a
  configurable percentage of the mid price.
- **Greeks**: if the data provider does not supply Greeks, compute delta,
  gamma, theta, vega, and rho with the Black-Scholes-Merton model using the
  underlying price, strike, time to expiration, risk-free rate, and implied
  volatility (from the provider, or back-solved via Newton-Raphson if
  absent).

## Risk (`analysis/risk.py`)

Per candidate (single-leg or defined-risk spread):

- **Max loss** / **max gain**
- **Breakeven price(s)**
- **Reward:risk ratio** = max gain / max loss (capped strategies) or a
  target-based ratio for undefined-risk strategies
- **Probability of profit (approx.)** — derived from delta as a proxy for
  probability of finishing in-the-money, or from a lognormal price
  distribution using the contract's implied volatility and days to
  expiration.

## Scoring (`analysis/scoring.py`, `specs/scoring.yaml`)

Each candidate receives a 0-100 opportunity score, a weighted sum of:

- Trend alignment with the proposed direction
- Volume confirmation
- Proximity to a supporting support/resistance level
- Liquidity (open interest, spread tightness)
- Risk:reward ratio

Weights are defined in `specs/scoring.yaml` so they can be tuned without
code changes. The top-ranked candidate(s) become the `Recommendation`
returned by the workflow, including a human-readable rationale built from
the contributing factors.
