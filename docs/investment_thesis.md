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

## Research agents

Financial Research, News Research, and Macro Research each depend on a
provider interface (`FinancialProvider`, `NewsProvider`, `MacroProvider`
in `specs/providers.yaml`) via the same dependency-injection pattern as
`MarketDataProvider`. Each is **optional at runtime**: if not a single
configured implementation exists for that interface, the FastAPI layer
constructs `None` instead of raising, `run_thesis_pipeline` skips that
agent, and the resulting `AgentThesisResult` field is `null` — mirrored
in the Agents tab as a muted "Skipped — no ... provider configured"
message rather than an error.

| agent | provider interface | implementations | env var(s) |
|---|---|---|---|
| Financial Research | `FinancialProvider` | Financial Modeling Prep, Finnhub, Alpha Vantage | `FMP_API_KEY`, `FINNHUB_API_KEY`, `ALPHA_VANTAGE_API_KEY` |
| News Research | `NewsProvider` | Finnhub, NewsData.io, The Guardian, GNews, Alpha Vantage, NewsAPI, Hacker News (keyless) | `FINNHUB_API_KEY`, `NEWSDATA_API_KEY`, `GUARDIAN_API_KEY`, `GNEWS_API_KEY`, `ALPHA_VANTAGE_API_KEY`, `NEWSAPI_API_KEY` |
| Macro Research | `MacroProvider` | FRED, BLS, BEA, IMF (keyless), World Bank (keyless) | `FRED_API_KEY`, `BLS_API_KEY`, `BEA_API_KEY` |

### Automatic failover across data providers

Each interface's `build_<name>_provider()` factory (e.g.
`build_news_provider()`) composes every implementation with a configured
API key into a `<X>ProviderRouter` — the data-provider analog of
`thesis.llm_client.LlmRouter` — configurable via a fallback-order env var
(`AOR_NEWS_PROVIDER_FALLBACK_ORDER`, `AOR_FINANCIAL_PROVIDER_FALLBACK_ORDER`,
`AOR_MACRO_PROVIDER_FALLBACK_ORDER`). Unlike the LLM router, routing
happens **per method call**, not per whole provider: BLS has no GDP data
and Alpha Vantage's OVERVIEW has no analyst consensus rating, so a
provider raises `<X>ProviderUnsupported` (retryable) for the methods
outside its specialty rather than being excluded from the router
entirely — it's still used for the methods it does support. A
transient failure (rate limit, quota, timeout, 5xx) advances to the next
configured provider the same way (see `specs/providers.yaml:
provider_router` for the full error-classification detail). Since Hacker
News (news) and IMF/World Bank (macro) need no API key, News Research
and Macro Research always have at least one provider available; only
Financial Research still requires a configured key.

All three research provider interfaces are **async** (one adapter
module per source under `data/news/`, `data/financial/`, and
`data/macro/`, sharing the infrastructure in `data/async_http.py` — see
`specs/providers.yaml`); each also exposes a `health()` probe, and the
routers check all their adapters' health concurrently. The sync
pipeline bridges with `asyncio.run()` at each research step — the
financial and macro steps fetch their four datasets concurrently via
`asyncio.gather`. All async adapters
share a process-wide 5-minute response cache, since free tiers meter by
the day and the provider objects are rebuilt per request — a
"Regenerate" click must not re-spend quota. GDELT was removed: its
per-IP throttle rate-limited routine pipeline use even with
caching/spacing/retry defenses, and the diverse news adapter set above
replaces it.

### When every data provider fails: warnings, not a crash

Even a fully-exhausted data-provider router doesn't fail the request.
The orchestrator wraps each research step: if its provider errors at
call time (e.g. every configured news source rate-limited at once),
that agent's finding stays `null`, the failure is
recorded in `AgentThesisResult.pipeline_warnings` (prefixed with the
agent name, e.g. `"news_research: provider failed during the run — …"`),
and the rest of the pipeline — quant, risk, strategy, the final thesis —
completes normally. Warnings are persisted with the thesis, returned by
both `POST` and `GET /runs/{run_id}/thesis`, and shown in the Agents tab
as an amber banner, with the affected agent's message reading "Skipped —
provider failed during the run". Only failures that make the synthesis
itself impossible (the LLM erroring, an unparseable LLM response) still
surface as a 502.

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

Above both sections, a **Provider** dropdown (Auto, plus every named
provider) and a password-masked **API key** field let a user override
the default behavior for one generation. Auto is the default and
recommended choice — it fails over across every configured provider; the
API key field is disabled while Auto is selected since there's no single
provider for a custom key to apply to. Picking a named provider enables
the field and forces that one provider, without touching server
configuration — the key never leaves that single request (see LLM access
below).

## LLM access

`thesis.llm_client.LlmClient` is a small interface
(`complete(system_prompt, user_prompt) -> str`) — the same
dependency-injection pattern as `data.market_data.MarketDataProvider`.
`build_llm_client(provider="auto", api_key=None, model=None, max_tokens=1024)`
either returns one named provider's client, or — when `provider="auto"`,
the default — an `LlmRouter` across every provider that has an API key
configured. Full detail (provider registry, error normalization, retry
strategy) is in `specs/llm_providers.yaml`; summary:

| provider | implementation | default model | env var |
|---|---|---|---|
| `anthropic` | `AnthropicLlmClient` | `claude-sonnet-5` | `ANTHROPIC_API_KEY` |
| `openai` | `OpenAiLlmClient` | `gpt-4o-mini` | `OPENAI_API_KEY` |
| `groq` | `GroqLlmClient` | `llama-3.3-70b-versatile` | `GROQ_API_KEY` |
| `gemini` | `GeminiLlmClient` | `gemini-2.5-pro` | `GEMINI_API_KEY` |
| `deepseek` | `DeepSeekLlmClient` | `deepseek-reasoner` | `DEEPSEEK_API_KEY` |
| `openrouter` | `OpenRouterLlmClient` | `deepseek/deepseek-r1` | `OPENROUTER_API_KEY` |

### Automatic failover (`provider="auto"`)

Relying on a single provider means one quota exhaustion, rate limit, or
outage blocks every thesis generation. `provider="auto"` builds an
`LlmRouter` from `AOR_LLM_FALLBACK_ORDER` (comma-separated provider
names; defaults to `anthropic,openai,groq,gemini,deepseek,openrouter`),
skipping any provider without a configured key. `LlmRouter.complete()`
tries each configured client in order; a *retryable* failure (rate
limit, quota exhaustion, timeout, or a 5xx/network failure) advances to
the next provider, while a bad-request or authentication failure is
raised immediately — another provider would reject the same malformed
request identically, and a bad key is a config problem specific to that
one provider, not the transient blip failover exists for. If every
configured provider fails, `LlmRouter` raises `LlmError` listing each
provider's failure. Agents never see any of this — they call the same
`LlmClient.complete()` either way.

A named provider (e.g. `openai`) still bypasses the router entirely —
used by the Agents tab's provider dropdown + custom API key fields for
one-off testing. `api_key` cannot be combined with `provider="auto"`
(422): there's no single provider for a custom key to apply to. The
server's `AOR_LLM_MODEL` / `AOR_LLM_MAX_TOKENS` settings only apply when
the explicit `anthropic` provider is chosen; every other provider (and
the router) uses its own built-in default model.

Each agent instructs the model to respond with a single JSON object and
validates it against a Pydantic model (`thesis/parsing.py`). A malformed
or schema-mismatched response raises `ThesisGenerationError` rather than
silently falling back to a fabricated value; the API surfaces this as a
502.
