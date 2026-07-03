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

## Phase 1 scope (quant/risk/strategy only)

Phase 1 only reasoned over data the quant engine already produced for a
persisted run — no external data sources (news, macro, filings) were
wired up. Phase 2a (below) adds three optional research agents backed by
real external providers; see `specs/providers.yaml` for the provider
interfaces and `specs/agents.yaml: deferred` for what's still out of
scope (a Catalyst agent, and having Risk/Strategy incorporate the new
research findings).

## The pipeline

```
AnalysisResult (persisted run)
        |
        v
Quant Interpreter    -- narrates the score breakdown; authors no numbers
        |
        v
Financial Research   -- optional; skipped (null) if no FinancialProvider configured
News Research        -- optional; skipped (null) if no NewsProvider configured
Macro Research        -- optional; skipped (null) if no MacroProvider configured
        |
        v
Risk Challenger      -- argues against the trade; risk_level is its judgment call
        |
        v
Options Strategy     -- suggests a strategy shape, not a priced instrument
        |
        v
Investment Thesis    -- synthesizes everything into one paragraph + a consensus label
```

If the run's recommendation has no candidate contract (`AVOID`, or an
empty candidate list), Risk Challenger and Options Strategy are skipped
entirely — there's nothing for them to assess or size — and Investment
Thesis produces a short explanation of why no position is recommended.
The three research agents are ticker/market-wide rather than
contract-specific, so they still run in that case (as long as their
provider is configured).

## Research agents (Phase 2a)

Financial Research, News Research, and Macro Research each depend on a
provider interface (`FinancialProvider`, `NewsProvider`, `MacroProvider`
in `specs/providers.yaml`) via the same dependency-injection pattern as
`MarketDataProvider`. Each is **optional at runtime**: if the
corresponding provider's API key isn't configured, the FastAPI layer
constructs `None` instead of raising, `run_thesis_pipeline` skips that
agent, and the resulting `AgentThesisResult` field is `null` — mirrored
in the Agents tab as a muted "Skipped — no ... provider configured"
message rather than an error. A *configured* provider that fails at call
time (rate limited, bad ticker, network down) is a different case and
propagates as a real 502, since that's a genuine runtime problem rather
than expected absence.

| agent | provider interface | phase-2a implementation | env var |
|---|---|---|---|
| Financial Research | `FinancialProvider` | `FmpFinancialProvider` (Financial Modeling Prep) | `FMP_API_KEY` |
| News Research | `NewsProvider` | `FinnhubNewsProvider` (Finnhub) | `FINNHUB_API_KEY` |
| Macro Research | `MacroProvider` | `FredMacroProvider` (FRED) | `FRED_API_KEY` |

Financial Research's `analyst_consensus` field is passed through verbatim
from the provider's `AnalystEstimates.consensus_rating` — never
LLM-authored — the same "facts pass through, judgment is LLM-authored"
split used for `QuantInterpretation.overall_score`. `company_health`,
`growth`, `profitability`, and `cash_flow` are legitimate LLM judgment
calls over the given facts, analogous to `risk_level`.

A `SECProvider` interface (`SecEdgarProvider`, backed by the free,
keyless SEC EDGAR API) also exists in `data/sec_provider.py` for a future
Catalyst agent, but isn't wired into the pipeline yet (see
`specs/providers.yaml: deferred`).

## Execution model

Generation is a separate, explicit step from the deterministic
`/analyze` call:

```
POST /runs/{run_id}/thesis    body: ThesisGenerationRequest (all fields optional)
                               # 404 if run missing, 409 if one exists and regenerate=false
GET  /runs/{run_id}/thesis    # fetch a previously generated one
```

`ThesisGenerationRequest` is `{provider: str = "anthropic", api_key: str | null = null, regenerate: bool = false}`.
`api_key`, if supplied, overrides the server's configured key for that one
request only — it is never logged, echoed back, or persisted alongside
the generated thesis.

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
  pipeline order (Quant Interpreter, Financial Research, News Research,
  Macro Research, Risk Challenger, Options Strategist, Investment
  Thesis), so the reasoning that produced the final output is inspectable
  rather than opaque. An agent skipped by the no-candidate short-circuit,
  or a research agent whose provider isn't configured, renders as a muted
  "Skipped — ..." message in its slot instead of being silently omitted.

Above both sections, a **Provider** dropdown (Anthropic/OpenAI) and a
password-masked **API key** field let a user supply their own key for one
generation without touching server configuration — the key never leaves
that single request (see LLM access below).

## LLM access

`thesis.llm_client.LlmClient` is a small interface
(`complete(system_prompt, user_prompt) -> str`) — the same
dependency-injection pattern as `data.market_data.MarketDataProvider`.
`build_llm_client(provider, api_key=None, model=None, max_tokens=1024)`
selects a concrete implementation by name:

| provider | implementation | default model | env var |
|---|---|---|---|
| `anthropic` (default) | `AnthropicLlmClient` | `claude-sonnet-5` | `ANTHROPIC_API_KEY` |
| `openai` | `OpenAiLlmClient` | `gpt-4o-mini` | `OPENAI_API_KEY` |

An `api_key` passed to `build_llm_client` overrides the provider's
environment variable for that call only. A different provider can be
added later by implementing `LlmClient` and registering it in
`llm_client._PROVIDERS`, without touching any agent module. The server's
`AOR_LLM_MODEL` / `AOR_LLM_MAX_TOKENS` settings only apply to the default
`anthropic` provider; other providers use their own built-in default
model.

Each agent instructs the model to respond with a single JSON object and
validates it against a Pydantic model (`thesis/parsing.py`). A malformed
or schema-mismatched response raises `ThesisGenerationError` rather than
silently falling back to a fabricated value; the API surfaces this as a
502.
