# AgenticOptionsReporter

A production-grade options analysis platform: downloads market data,
computes technical indicators, evaluates the option chain (pricing,
Greeks, liquidity), scores candidate opportunities, and produces a
persisted, structured recommendation.

This project is built using structured specifications instead of
conversational prompts. Documentation (`docs/`) is human-facing context;
specs (`specs/`) are the single source of truth for interfaces. See
`docs/architecture.md` for the full architecture guide.

## Repository layout

```
docs/     human-readable architecture, workflow, and analysis docs
specs/    machine-readable YAML contracts (workflow, api, scoring, database)
src/      application source (agentic_options_reporter package)
tests/    pytest unit tests, one file per module
docker/   container build
```

## Getting started

```bash
poetry install
poetry run uvicorn agentic_options_reporter.main:app --reload
```

Then, in another terminal, use the bundled CLI client (a `requests`-based
HTTP client with an `argparse` command interface — see
`src/agentic_options_reporter/cli.py`) instead of `curl`:

```bash
poetry run agentic-options-reporter health
poetry run agentic-options-reporter analyze AAPL
poetry run agentic-options-reporter analyze AAPL --lookback-days 90 --expiration 2026-01-16
poetry run agentic-options-reporter runs --symbol AAPL --limit 5
poetry run agentic-options-reporter run 1
poetry run agentic-options-reporter --base-url http://localhost:8000 health
```

Or launch the Flet front end (a desktop window by default; see
`src/agentic_options_reporter/frontend/app.py`), which drives the same API
through the same `ApiClient` the CLI uses. It's a Material 3 UI with an
Analyze tab (recommendation badge, trend/volume/indicator stat cards, a
scored-candidates table), an Agents tab (see below), and a History tab,
plus a light/dark mode toggle in the app bar:

```bash
poetry run agentic-options-reporter-ui
poetry run agentic-options-reporter-ui --web --port 8550   # serve in a browser instead
poetry run agentic-options-reporter-ui --base-url http://localhost:8000
```

### Investment thesis (LLM agent pipeline)

On top of the deterministic recommendation, an optional pipeline of LLM
agents (Quant Interpreter → Financial/News/Macro Research → Risk
Challenger → Options Strategy → Investment Thesis) can narrate,
challenge, and synthesize an already-persisted run into a written
thesis — see `docs/investment_thesis.md` and `specs/agents.yaml`. These
agents never compute a number the quant engine hasn't already computed.

Each research agent draws on multiple redundant data providers and fails
over between them the same way the LLM providers do below — a single
provider's outage, rate limit, or quota exhaustion no longer skips the
whole agent. The agent is only skipped (rendered as "not configured")
if *none* of its providers are configured. And if every configured
provider for one agent fails mid-run, the pipeline still completes:
that agent's finding is null and the failure is reported at the end of
the run in `pipeline_warnings` (an amber banner in the Agents tab)
rather than crashing the request.

```bash
# Market data (/analyze) — yfinance is keyless and serves both price
# history and option chains, so no key is required. These optional keys
# add price-history redundancy/failover (option chains stay yfinance-only):
export ALPHA_VANTAGE_API_KEY=... # alphavantage.co, daily price history
export TWELVE_DATA_API_KEY=...   # twelvedata.com, daily price history
export FINNHUB_API_KEY=...       # finnhub.io stock candles (may be premium-gated)

# Financial Research — all free tier; FINNHUB_API_KEY/ALPHA_VANTAGE_API_KEY
# do double duty for News Research below
export FMP_API_KEY=...           # financialmodelingprep.com, full coverage
export FINNHUB_API_KEY=...       # finnhub.io (profile/ratios/analyst consensus)
export ALPHA_VANTAGE_API_KEY=... # alphavantage.co (25 req/day, last resort)

# News Research — all free tier; Hacker News needs no key
export FINNHUB_API_KEY=...       # finnhub.io, financial news
export NEWSDATA_API_KEY=...      # newsdata.io, general news
export GUARDIAN_API_KEY=...      # open-platform.theguardian.com
export GNEWS_API_KEY=...         # gnews.io
export NEWSAPI_API_KEY=...       # newsapi.org (dev use only)

# Macro Research — all free; IMF and World Bank need no key
export FRED_API_KEY=...          # fred.stlouisfed.org, broadest + freshest
export BLS_API_KEY=...           # bls.gov, primary CPI source
export BEA_API_KEY=...           # bea.gov, primary GDP source
```

Six LLM providers are supported — `anthropic`, `openai`, `groq`, `gemini`,
`deepseek`, `openrouter` — each needing its own API key, set server-side:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GROQ_API_KEY=gsk_...
export GEMINI_API_KEY=...
export DEEPSEEK_API_KEY=sk-...
export OPENROUTER_API_KEY=sk-or-...
```

**By default (`--provider auto`), the CLI fails over across every one of
these that's configured**, in priority order (`anthropic` first, unless
`AOR_LLM_FALLBACK_ORDER` overrides it) — a quota exhaustion, rate limit,
or outage on one provider no longer blocks thesis generation, it just
tries the next one. See `docs/investment_thesis.md` for exactly which
failures trigger a retry vs. propagate immediately.

```bash
poetry run agentic-options-reporter thesis <run_id>                                     # generate, auto-failover across configured providers
poetry run agentic-options-reporter thesis <run_id> --regenerate                        # discard and regenerate
poetry run agentic-options-reporter thesis <run_id> --fetch-only                        # fetch without generating
poetry run agentic-options-reporter thesis <run_id> --stream                            # stream each agent's progress live (to stderr) as the pipeline runs
poetry run agentic-options-reporter thesis <run_id> --provider openai --api-key sk-...  # force one provider + your own key, this call only
```

Or use the Flet UI's **Agents** tab after running an analysis: pick a
**Provider** (Auto is the default and recommended choice) and, optionally,
paste your own **API key** (password-masked, sent only for that one
request and only enabled once a specific provider is chosen), then click
"Generate investment thesis"
to see a **Final output** verdict (the recommendation action + the
agents' consensus) and, below it, an **Agent conversation** — Quant
Interpreter, Financial Research, News Research, Macro Research, Catalyst
Research, Risk Challenger, Options Strategist, and Investment Thesis shown
in sequence as each agent's contribution, with a "skipped" message where
an agent had no candidate contract to work with (Risk/Strategy) or its
provider wasn't configured (the research agents). The conversation renders
**live** — each agent updates the moment it runs, with a status pill
(Queued → Running → Done/Skipped/Failed) and a collapsible "Under the
hood" panel exposing the exact prompt sent and raw response received, so
you can track and debug what each agent is doing.

## Testing

```bash
poetry run pytest
```

## Tooling

Python 3.13 · FastAPI · Pydantic · SQLAlchemy · pandas/numpy/scipy/ta ·
yfinance · Plotly/Matplotlib · requests · Flet · Anthropic Claude API ·
pytest · Poetry · Docker · GitHub Actions
