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

On top of that deterministic pipeline, a separate, optional
**investment-thesis agent pipeline** (`specs/agents.yaml`,
`src/agentic_options_reporter/thesis/`) interprets an already-persisted
run's output using an LLM. It is triggered explicitly
(`POST /runs/{run_id}/thesis`), never as part of `/analyze`, and it never
computes a number the quant engine hasn't already computed — see
`docs/investment_thesis.md`.

## Module boundaries

| Module | Path | Responsibility |
|---|---|---|
| Data access | `src/agentic_options_reporter/data/` | Fetch and cache OHLCV + option chain data (yfinance, pluggable providers), plus optional financial/news/macro/SEC research providers (see below) |
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
| API client | `src/agentic_options_reporter/api_client.py` | Shared `requests`-based HTTP client used by both the CLI and the Flet front end |
| CLI client | `src/agentic_options_reporter/cli.py` | `argparse` command interface over `api_client.ApiClient` |
| Front end | `src/agentic_options_reporter/frontend/` | Flet UI (`app.py`) over `api_client.ApiClient`, with display formatting isolated in `formatting.py` for testability |
| Investment-thesis pipeline | `src/agentic_options_reporter/thesis/` | LLM agents that interpret a persisted run's already-computed output (see docs/investment_thesis.md) |

## Tooling

- Language: Python 3.13
- Frameworks: FastAPI, Pydantic, SQLAlchemy, asyncio
- Data: pandas, numpy, scipy, ta, yfinance
- HTTP client: requests, shared by the CLI and front end (server-side data access goes through `MarketDataProvider`, not `api_client`)
- CLI: argparse, exposed as the `agentic-options-reporter` console script
- Front end: Flet, exposed as the `agentic-options-reporter-ui` console script
- LLM: provider-agnostic `thesis.llm_client.LlmClient` interface, selected per-request via `build_llm_client`; built-in providers are Anthropic, OpenAI, Groq, DeepSeek, and OpenRouter (all via the `anthropic`/`openai` packages — the latter four are OpenAI-API-compatible), plus Gemini (`google-genai` package). `provider="auto"` (default) builds an `LlmRouter` that fails over across every configured provider — see `specs/llm_providers.yaml`
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

## Research providers

Beyond price/option data, the investment-thesis pipeline optionally draws
on four more provider interfaces (`specs/providers.yaml`), each following
the same ABC-interface + concrete-implementation + env-var-API-key
pattern as `MarketDataProvider`:

| interface | implementations | env var(s) |
|---|---|---|
| `FinancialProvider` | Financial Modeling Prep, Alpha Vantage | `FMP_API_KEY`, `ALPHA_VANTAGE_API_KEY` |
| `NewsProvider` | Finnhub, Alpha Vantage, NewsAPI, GDELT (keyless) | `FINNHUB_API_KEY`, `ALPHA_VANTAGE_API_KEY`, `NEWSAPI_API_KEY` |
| `MacroProvider` | FRED, BLS, BEA | `FRED_API_KEY`, `BLS_API_KEY`, `BEA_API_KEY` |
| `SECProvider` | SEC EDGAR (free, keyless) | `SEC_EDGAR_USER_AGENT` (optional) |

Each interface's `build_<name>_provider()` factory (`data/*_provider.py`)
composes every currently-configured implementation into a
`<X>ProviderRouter` — the data-provider analog of
`thesis.llm_client.LlmRouter` (see Tooling above) — that fails over between them
per method call, configurable via `AOR_<NAME>_PROVIDER_FALLBACK_ORDER`.
`main.py`'s `_optional_*_provider()` helpers call these factories and
treat a `*ProviderError` (zero providers configured) as "not configured"
(`None`), not a startup or request failure. See `docs/investment_thesis.md`
for how the Financial/News/Macro Research agents consume these providers.
`SECProvider` has only one implementation and isn't wired into a router
or any agent yet (reserved for a future Catalyst agent).

## Claude Code instructions

- Always read `docs/` and `specs/` before writing or modifying code.
- Never invent interfaces; extend `specs/` first if a new interface is
  needed, then implement it.
- Write unit tests for every module.
- Keep modules small and typed (type hints throughout).
- Prefer dependency injection and Pydantic models over ad-hoc dicts.
- Keep all numerical calculations in `analysis/*.py`, deterministic and
  LLM-free. Agents in `thesis/*.py` interpret, challenge, and narrate
  already-computed results; they never compute or alter a score,
  indicator, Greek, or risk metric (see `specs/agents.yaml`).
