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
wired up. Phase 2a added three optional research agents (Financial, News,
Macro) backed by real external providers, and phase 2b adds a **Catalyst
Research** agent that combines news + SEC filings + macro into a
structured list of dateable catalysts. See `specs/providers.yaml` for the
provider interfaces and `specs/agents.yaml: deferred` for what's still
out of scope (having Risk/Strategy incorporate the research findings).

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
Catalyst Research    -- optional; combines news + SEC filings + macro into dateable catalysts
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
The research agents (Financial, News, Macro, Catalyst) are
ticker/market-wide rather than contract-specific, so they still run in
that case (as long as their provider is configured).

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

**Catalyst Research** is the exception to the one-agent-one-provider
shape: it combines *three* streams — recent news (`NewsProvider`), recent
SEC filings (`SECProvider`, backed by the keyless SEC EDGAR API), and
macro indicators (`MacroProvider`) — into a structured list of discrete,
dateable catalysts (earnings, filings, corporate actions, macro releases)
each classified by category, timing horizon, and directional impact, plus
a net bias. It runs if *any* of the three streams is configured (each is
fetched under its own guard, so one failing only drops that stream, as a
`pipeline_warning`); because SEC EDGAR is keyless, in practice it always
has at least the filings stream. Distinct from News Research, which
authors an overall sentiment summary — Catalyst Research enumerates
individual events.

| agent | provider interface | implementations | env var(s) |
|---|---|---|---|
| Financial Research | `FinancialProvider` | Financial Modeling Prep, Finnhub, Alpha Vantage | `FMP_API_KEY`, `FINNHUB_API_KEY`, `ALPHA_VANTAGE_API_KEY` |
| News Research | `NewsProvider` | Finnhub, NewsData.io, The Guardian, GNews, Alpha Vantage, NewsAPI, Hacker News (keyless) | `FINNHUB_API_KEY`, `NEWSDATA_API_KEY`, `GUARDIAN_API_KEY`, `GNEWS_API_KEY`, `ALPHA_VANTAGE_API_KEY`, `NEWSAPI_API_KEY` |
| Macro Research | `MacroProvider` | FRED, BLS, BEA, IMF (keyless), World Bank (keyless) | `FRED_API_KEY`, `BLS_API_KEY`, `BEA_API_KEY` |
| Catalyst Research | `NewsProvider` + `SECProvider` + `MacroProvider` | (the above) + SEC EDGAR (keyless) | (the above) + `SEC_EDGAR_USER_AGENT` (optional) |

### Automatic failover across data providers

Each interface's `build_<name>_provider()` factory (e.g.
`build_news_provider()`) composes every implementation with a configured
API key into a `<X>ProviderRouter` — the data-provider analog of
`thesis.llm_client.LlmRouter` — configurable via a fallback-order env var
(`AOR_NEWS_PROVIDER_FALLBACK_ORDER`, `AOR_FINANCIAL_PROVIDER_FALLBACK_ORDER`,
`AOR_MACRO_PROVIDER_FALLBACK_ORDER`). Unlike the LLM router, routing
happens **per method call**, not per whole provider: routing for one
dataset/metric is independent of which provider answered the last one. A
transient failure (rate limit, quota, timeout, 5xx) advances to the next
configured provider (see `specs/providers.yaml: provider_router`). Since
Hacker News + Yahoo (news), Yahoo (fundamentals), and IMF/World Bank
(macro) need no API key, News Research, Financial Research, Macro Research,
and the `/analyze` fundamentals snapshot always have at least one provider
available — nothing requires a configured key anymore.

**Fan-out + merge (financial & news).** The financial and news routers go
a step further than failover: instead of stopping at the first provider
that answers, they query **every** supporting provider concurrently and
**combine** the results (`data.provider_router.acall_and_merge`). Record
datasets (company profile, ratios, estimates, metrics, earnings calendar)
are unioned **field-by-field** in priority order — a value present in
Finnhub but missing in Yahoo is filled in, and vice versa, so the merged
record is richer than any single source. List datasets (earnings history,
insider transactions, news articles) are unioned and de-duplicated. A
provider that fails simply doesn't contribute; the merge raises only if
*every* provider failed. Financial **statements** are the one exception —
period-bound, so merging across providers would mix reporting periods, and
they stay first-success failover. This is what "try all providers and get
all the data" means: the `/analyze` fundamentals snapshot and the Financial
Research agent both see the combined view, not whichever source happened to
answer first.

Yahoo Finance was added as a keyless source for **both** fundamentals
(`YFinanceFinancialProvider` — profile/statements/ratios/estimates/metrics/
earnings/calendar/insider) and company news (`YFinanceNewsProvider`), which
is what makes fundamentals and news available with zero configuration and
gives the merge a second source to combine on day one.

**Capability-based routing.** Rather than discovering "unsupported" by
catching an exception mid-call, every provider *declares what it serves*
and the router selects supporting providers *before* calling, so a
source is never asked for data it lacks. This applies to all three
research interfaces, in two flavors:

- **Macro & financial filter (hard).** Each macro provider declares its
  `supported_metrics` and serves any one through a single
  `fetch(metric_id) -> MacroObservation`; each financial provider
  declares its `supported_datasets` (`profile`, `statements`, `ratios`,
  `analyst_estimates`, plus the newer `metrics`, `earnings`,
  `earnings_calendar`, `insider` served by Finnhub and Yahoo). The router
  narrows to the providers that
  advertise a metric/dataset and drops the rest
  (`data.provider_router.filter_supporting`) — the fix for the "World
  Bank has no US fed funds rate" error: World Bank doesn't advertise
  `policy_rate`, so it's never queried for it, and a keyless-only
  deployment cleanly serves CPI/GDP while skipping the rate metrics no
  configured source provides. Likewise Finnhub's free tier doesn't
  advertise `statements`, so a statements request routes only to
  FMP/Alpha Vantage. A metric/dataset no configured provider serves is
  simply absent — the agent renders it "not available" and reasons over
  the rest, rather than erroring. Macro metrics are a structured registry
  (`data/macro/metrics.py`: `MacroMetric(id, category, country,
  frequency, unit)`), so adding one — unemployment, PPI — is a registry
  entry plus the adapters that happen to serve it. Per-metric/dataset
  priority is configurable via `AOR_MACRO_PRIORITY_<METRIC>` /
  `AOR_FINANCIAL_PRIORITY_<DATASET>` (e.g. prefer BEA's GDP over FRED's
  mirror).
- **News prioritizes (soft).** A ticker `search` prefers the
  company-news specialists whose endpoints are ticker-aware (Finnhub,
  Alpha Vantage advertise `company_news`) but *keeps* general-news
  providers as a fallback — they can still surface a keyword match
  (`data.provider_router.prioritize_supporting`). A general-news source
  genuinely can answer a ticker query, just less precisely, whereas World
  Bank genuinely has no policy rate — so news reorders where macro/
  financial exclude.

All three research provider interfaces are **async** (one adapter
module per source under `data/news/`, `data/financial/`, and
`data/macro/`, sharing the infrastructure in `data/async_http.py` — see
`specs/providers.yaml`); each also exposes a `health()` probe, and the
routers check all their adapters' health concurrently. The sync
pipeline bridges with `asyncio.run()` at each research step — the
financial and macro steps fetch their datasets concurrently via
`asyncio.gather` (macro fetches only the metrics some provider can
actually serve). All async adapters share a process-wide 5-minute
response cache, since free tiers meter by the day and the provider
objects are rebuilt per request — a "Regenerate" click must not
re-spend quota. GDELT was removed: its per-IP throttle rate-limited
routine pipeline use even with caching/spacing/retry defenses, and the
diverse news adapter set above replaces it.

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
split used for `QuantInterpretation.quant_trade_quality` (see
`specs/scoring.yaml` for the Trade Quality Score overhaul). `company_health`,
`growth`, `profitability`, and `cash_flow` are legitimate LLM judgment
calls over the given facts, analogous to `risk_level`. Since phase 3, most
agents ALSO author their own `domain_score` (specs/agents.yaml) — an
independent 0-100 judgment for one Trade Quality Score domain, never a
copy of the quant engine's number for that domain.

The async `SECProvider` interface (`SecEdgarProvider`, backed by the free,
keyless SEC EDGAR API, on the same `data/async_http.py` base as the other
three) in `data/sec_provider.py` feeds the Catalyst Research agent's
filings stream (see `specs/providers.yaml: SECProvider`).

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

### Live streaming (`POST /runs/{run_id}/thesis/stream`)

The blocking `POST /runs/{run_id}/thesis` only returns once the whole
pipeline finishes — for a full run that can be tens of seconds of model
latency with no feedback. The `/thesis/stream` variant runs the *same*
pipeline but reports progress live as **Server-Sent Events**
(`text/event-stream`), so a client can show each agent's status — and its
raw prompt/response — as it happens:

```
POST /runs/{run_id}/thesis/stream   body: ThesisGenerationRequest
  event: agent   data: AgentEvent          # one per agent phase transition
  event: agent   data: AgentEvent
  ...
  event: result  data: AgentThesisResult   # terminal; persisted server-side
```

An `AgentEvent` carries the agent id, a `phase`
(`started`/`completed`/`skipped`/`failed`), and — for an LLM-backed agent —
an `exchange` with the exact `system_prompt`, `user_prompt`, and
`raw_response` the model saw and returned (the "under the hood" view),
plus the parsed `output` on completion. Nothing about the stream is
persisted except the final `AgentThesisResult`, which is written exactly
as the blocking endpoint writes it (the design choice was "live only" — no
new schema or migration for the per-agent log).

How it's wired: `run_thesis_pipeline` takes an optional `on_event`
callback and emits one event per agent as it runs (a no-op, and the client
is left unwrapped, when `on_event is None`, so the blocking path is
byte-for-byte unchanged). To capture the raw exchange, the orchestrator
wraps the `LlmClient` in a `RecordingLlmClient` that remembers the last
`(system_prompt, user_prompt, response)` and clears it between agents so
each event carries only its own call. Because the pipeline is synchronous
and blocking, `thesis/streaming.py:run_thesis_streaming` runs it on a
worker thread that pushes events onto a queue, and the SSE endpoint drains
that queue into `text/event-stream` frames. The 404/409/422/502 guards run
*before* the stream opens (you can't raise an HTTP status once headers are
sent); a required-agent failure after that surfaces as a terminal `error`
frame instead.

Clients: `ApiClient.stream_thesis(run_id, ...)` yields
`{"event": ..., "data": ...}` dicts by parsing the SSE frames; the CLI's
`thesis --stream` prints each agent's phase to stderr as it runs and the
final result to stdout; the Flet Agents tab consumes it to update each
agent live (see below).

## Agents tab (Flet UI)

`frontend/app.py`'s Agents tab presents `AgentThesisResult` as two
sections rather than one undifferentiated blob:

- **Final output** — a compact verdict row: the deterministic
  recommendation's action badge, the agents' consensus badge, and the
  recommendation's confidence. This is the "read this and decide" part.
- **Agent conversation** — each agent shown as a labeled message in
  pipeline order (Quant Interpreter, Financial Research, News Research,
  Macro Research, Catalyst Research, Risk Challenger, Options Strategist,
  Investment Thesis), so the reasoning that produced the final output is
  inspectable rather than opaque. An agent skipped by the no-candidate
  short-circuit, or a research agent whose provider isn't configured,
  renders as a muted "Skipped — ..." message in its slot instead of being
  silently omitted.

  The tab renders **live**: it consumes `/thesis/stream` via
  `ApiClient.stream_thesis` on a background thread (`page.run_thread`), so
  each agent updates the moment its event arrives instead of the whole
  panel appearing at once after the run. Every agent carries a small status
  pill — Queued → Running… → Done / Skipped / Failed — and a collapsible
  **"Under the hood"** panel that reveals the exact system prompt, user
  prompt, and raw model response for that agent (from the event's
  `exchange`), for tracking and debugging what each agent actually said.
  The panel is hidden for agents with no LLM exchange (a skipped agent, or
  the deterministic no-candidate quant path).

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
