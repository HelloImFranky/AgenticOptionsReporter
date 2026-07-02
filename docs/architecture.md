# Architecture

## Overview

AgenticOptionsReporter is a production-grade options analysis platform. It
downloads market data for an underlying equity, computes technical
indicators, evaluates the option chain (pricing, Greeks, liquidity), scores
candidate opportunities, and produces a persisted, structured recommendation.

Documentation in `docs/` is context. Specifications in `specs/` are the
single source of truth for interfaces and contracts. Code in `src/` must
conform to the specs; when architecture changes, the specs are updated
first and code follows.

## Layers

```
Documentation (docs/)
        |
        v
Planner Agent            -- decomposes a request into module-level tasks
        |
        v
Specialist Coding Agents -- backend, indicators, options engine, testing,
        |                    documentation, infrastructure
        v
Source Code (src/)
```

The **Planner Agent** reads `docs/` and `specs/` and breaks a feature or
analysis request into discrete, typed tasks scoped to a single module.
**Specialist Coding Agents** implement one concern each and must not invent
interfaces that are not already declared in `specs/`.

## Runtime pipeline

The runtime pipeline is defined authoritatively in `specs/workflow.yaml` and
narrated in `docs/workflow.md`. At a high level:

1. Download OHLCV price history and the option chain for a symbol.
2. Compute technical indicators.
3. Detect trend direction/strength.
4. Analyze volume behavior.
5. Detect support/resistance levels.
6. Evaluate the option chain: pricing sanity, Greeks, liquidity.
7. Compute position/portfolio risk.
8. Score the opportunity.
9. Generate a structured recommendation.
10. Persist all inputs/outputs for auditability.

## Module boundaries

| Module | Path | Responsibility |
|---|---|---|
| Data access | `src/agentic_options_reporter/data/` | Fetch and cache OHLCV + option chain data (yfinance, pluggable providers) |
| Indicators | `src/agentic_options_reporter/analysis/indicators.py` | Compute technical indicators (trend, momentum, volatility, volume) |
| Trend | `src/agentic_options_reporter/analysis/trend.py` | Classify trend direction and strength |
| Volume | `src/agentic_options_reporter/analysis/volume.py` | Volume trend and anomaly analysis |
| Support/Resistance | `src/agentic_options_reporter/analysis/support_resistance.py` | Detect price levels from pivots |
| Options engine | `src/agentic_options_reporter/analysis/options.py` | Option chain evaluation and Greeks |
| Risk | `src/agentic_options_reporter/analysis/risk.py` | Position and portfolio risk metrics |
| Scoring | `src/agentic_options_reporter/analysis/scoring.py` | Opportunity scoring and recommendation |
| Workflow | `src/agentic_options_reporter/workflow.py` | Orchestrates the pipeline end-to-end |
| Persistence | `src/agentic_options_reporter/persistence.py` | Writes results via SQLAlchemy models |
| API | `src/agentic_options_reporter/main.py` | FastAPI surface over the workflow |
| CLI client | `src/agentic_options_reporter/cli.py` | `requests`-based HTTP client with an `argparse` command interface for the API |

## Tooling

- Language: Python 3.13
- Frameworks: FastAPI, Pydantic, SQLAlchemy, asyncio
- Data: pandas, numpy, scipy, ta, yfinance
- HTTP client: requests (CLI only; server-side data access goes through `MarketDataProvider`)
- CLI: argparse, exposed as the `agentic-options-reporter` console script
- Visualization: Plotly, Matplotlib
- Testing: pytest
- Packaging: Poetry
- Containerization: Docker
- CI/CD: GitHub Actions
- Observability: Prometheus, Grafana, OpenTelemetry (future extension)

## Market data

Primary source is Yahoo Finance via `yfinance` for OHLCV and option chains.
The `MarketDataProvider` interface in `data/market_data.py` is written so
that Polygon.io, Alpaca, Tradier, Interactive Brokers, Finnhub, or Alpha
Vantage can be added as alternate providers without changing downstream
analysis code. Responses are cached to reduce rate-limit pressure.

## Claude Code instructions

- Always read `docs/` and `specs/` before writing or modifying code.
- Never invent interfaces; extend `specs/` first if a new interface is
  needed, then implement it.
- Write unit tests for every module.
- Keep modules small and typed (type hints throughout).
- Prefer dependency injection and Pydantic models over ad-hoc dicts.
