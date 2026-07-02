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

## Testing

```bash
poetry run pytest
```

## Tooling

Python 3.13 · FastAPI · Pydantic · SQLAlchemy · pandas/numpy/scipy/ta ·
yfinance · Plotly/Matplotlib · pytest · Poetry · Docker · GitHub Actions
