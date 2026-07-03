# Investment Thesis Agent Pipeline

Narrative companion to `specs/agents.yaml`, which is the authoritative
contract for agent inputs/outputs. This is a separate, optional layer on
top of the deterministic quant engine described in `docs/workflow.md` — it
never replaces or recomputes anything the quant engine already produced.

## Why a separate pipeline

The deterministic engine (`analysis/*.py`) already produces a trend
classification, a volume read, an option chain evaluation, Greeks, risk
metrics, and a weighted opportunity score. An LLM has nothing useful to
add to *computing* those numbers, and asking it to would make the system
slower, non-reproducible, and harder to test.

What an LLM *is* good at: reading those already-computed numbers and
producing the plain-language interpretation, skepticism, and narrative
synthesis a human actually wants before acting on a recommendation. So
the pipeline is scoped to exactly that — see the "contract" note on each
agent in `specs/agents.yaml` for what it is and isn't allowed to author.

## Phase 1 scope

Phase 1 only reasons over data the quant engine already produced for a
persisted run — no external data sources (news, macro, social, filings)
are wired up yet. See `specs/agents.yaml: future_phases` for what a
Market Research agent and a News/Sentiment agent would need before they
could be added (primarily: picking and provisioning real data providers,
which is a product/cost decision, not an architecture one).

## The pipeline

```
AnalysisResult (persisted run)
        |
        v
Quant Interpreter  -- narrates the score breakdown; authors no numbers
        |
        v
Risk Challenger    -- argues against the trade; risk_level is its judgment call
        |
        v
Options Strategy   -- suggests a strategy shape, not a priced instrument
        |
        v
Investment Thesis  -- synthesizes everything into one paragraph + a consensus label
```

If the run's recommendation has no candidate contract (`AVOID`, or an
empty candidate list), Risk Challenger and Options Strategy are skipped
entirely — there's nothing for them to assess or size — and Investment
Thesis produces a short explanation of why no position is recommended.

## Execution model

Generation is a separate, explicit step from the deterministic
`/analyze` call:

```
POST /runs/{run_id}/thesis            # generate (404 if run missing, 409 if one exists and regenerate=false)
POST /runs/{run_id}/thesis?regenerate=true   # discard and regenerate
GET  /runs/{run_id}/thesis             # fetch a previously generated one
```

This keeps `/analyze` fast, cheap, and deterministic-only; a client (the
CLI's `thesis` subcommand, or the Flet UI's Agents tab) calls the thesis
endpoint afterward, once it has a `run_id`.

## Agents tab (Flet UI)

`frontend/app.py`'s Agents tab presents `AgentThesisResult` as two
sections rather than one undifferentiated blob:

- **Final output** — a compact verdict row: the deterministic
  recommendation's action badge, the agents' consensus badge, and the
  recommendation's confidence. This is the "read this and decide" part.
- **Agent conversation** — each agent shown as a labeled message in
  pipeline order (Quant Interpreter, Risk Challenger, Options Strategist,
  Investment Thesis), so the reasoning that produced the final output is
  inspectable rather than opaque. An agent skipped by the
  no-candidate short-circuit renders as a muted "Skipped — ..." message
  in its slot instead of being silently omitted.

## LLM access

`thesis.llm_client.LlmClient` is a small interface
(`complete(system_prompt, user_prompt) -> str`) — the same
dependency-injection pattern as `data.market_data.MarketDataProvider`.
`AnthropicLlmClient` is the default implementation; a different provider
can be added later without touching any agent module. Configure it via
the `ANTHROPIC_API_KEY` environment variable (or `AOR_LLM_MODEL` /
`AOR_LLM_MAX_TOKENS` to change the model/token budget).

Each agent instructs the model to respond with a single JSON object and
validates it against a Pydantic model (`thesis/parsing.py`). A malformed
or schema-mismatched response raises `ThesisGenerationError` rather than
silently falling back to a fabricated value; the API surfaces this as a
502.
