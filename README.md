# AgenticOptionsReporter

A production-grade options analysis platform: downloads market data,
computes technical indicators, evaluates the option chain (pricing,
Greeks, liquidity), scores candidate opportunities, and produces a
persisted, structured recommendation.

This project is built using structured specifications instead of
conversational prompts. Documentation (`docs/`) is human-facing context;
specs (`specs/`) are the single source of truth for interfaces. See
`docs/architecture.md` for the full architecture guide and
`agents/` for the roles that develop this codebase.

## Repository layout

```
docs/     human-readable architecture, workflow, and analysis docs
specs/    machine-readable YAML contracts (workflow, api, scoring, database)
agents/   specialist agent role descriptions (planner, backend, testing, docs)
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
scored-candidates table) and a History tab, plus a light/dark mode toggle
in the app bar:

```bash
poetry run agentic-options-reporter-ui
poetry run agentic-options-reporter-ui --web --port 8550   # serve in a browser instead
poetry run agentic-options-reporter-ui --base-url http://localhost:8000
```

### Investment thesis (LLM agent pipeline)

On top of the deterministic recommendation, an optional pipeline of LLM
agents (Quant Interpreter → Risk Challenger → Options Strategy →
Investment Thesis) can narrate, challenge, and synthesize an
already-persisted run into a written thesis — see
`docs/investment_thesis.md` and `specs/agents.yaml`. These agents never
compute a number the quant engine hasn't already computed. Requires
`ANTHROPIC_API_KEY` to be set:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
poetry run agentic-options-reporter thesis <run_id>              # generate
poetry run agentic-options-reporter thesis <run_id> --regenerate # discard and regenerate
poetry run agentic-options-reporter thesis <run_id> --fetch-only # fetch without generating
```

Or click "Generate investment thesis" in the Flet UI's Analyze tab after
running an analysis.

## Testing

```bash
poetry run pytest
```

## Tooling

Python 3.13 · FastAPI · Pydantic · SQLAlchemy · pandas/numpy/scipy/ta ·
yfinance · Plotly/Matplotlib · requests · Flet · Anthropic Claude API ·
pytest · Poetry · Docker · GitHub Actions
