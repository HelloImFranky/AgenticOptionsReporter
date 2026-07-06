"""Orchestrates the investment-thesis agent pipeline (specs/agents.yaml).

Coordination only — no LLM calls of its own. Runs quant_interpreter, then
the optional financial/news/macro/catalyst research agents (each skipped
with a null finding if its provider wasn't supplied — see
specs/providers.yaml provider_availability), then (unless short-circuited)
risk_challenger, options_strategy, relative_strength_research, and
statistical_edge_research, then investment_thesis, and assembles the
result — including the agent-side composite Trade Quality Score, blended
from whichever agent DomainScores this run produced by the same
analysis/composite_score.py engine the quant path uses.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timezone

from agentic_options_reporter.analysis.composite_score import compute_composite_score
from agentic_options_reporter.analysis.domain_scoring import SECTOR_ETF_MAP
from agentic_options_reporter.data import financial as financial_data
from agentic_options_reporter.data.financial import FinancialProvider, FinancialProviderError
from agentic_options_reporter.data.macro import (
    DEFAULT_MACRO_METRICS,
    MacroProvider,
    MacroProviderError,
)
from agentic_options_reporter.data.market_data import MarketDataError, MarketDataProvider
from agentic_options_reporter.data.news import NewsProvider, NewsProviderError
from agentic_options_reporter.data.sec_provider import SECProvider, SecProviderError
from agentic_options_reporter.models.schemas import (
    AgentEvent,
    AgentExchange,
    AgentThesisResult,
    AnalysisResult,
    DomainScore,
    PriceHistory,
    QuantInterpretation,
)
from agentic_options_reporter.thesis import (
    catalyst_research,
    financial_research,
    investment_thesis,
    macro_research,
    news_research,
    options_strategy,
    quant_interpreter,
    relative_strength_research,
    risk_challenger,
    statistical_edge_research,
)
from agentic_options_reporter.thesis.llm_client import LlmClient, LlmError, RecordingLlmClient
from agentic_options_reporter.thesis.parsing import ThesisGenerationError

# Callback invoked with a per-agent AgentEvent as the pipeline runs, for a
# live view (thesis.streaming / the SSE endpoint). None = run silently, the
# original non-streaming behavior.
OnEvent = Callable[[AgentEvent], None]

# Enough trading-day history to comfortably cover the 21-bar relative
# strength lookback (analysis/domain_scoring.py _RS_LOOKBACK_BARS).
_RELATIVE_STRENGTH_HISTORY_DAYS = 60
_RELATIVE_STRENGTH_LOOKBACK_BARS = 21


async def _fetch_financial_inputs(provider: FinancialProvider, ticker: str) -> tuple:
    """Fetch the company profile (the required anchor) plus whichever of
    statements/ratios/estimates/metrics/earnings_calendar the configured
    providers actually serve, concurrently — each dataset merged across
    every provider that offers it. A dataset no provider covers comes back
    None and the agent omits it."""

    async def optional(dataset: str, method: str):
        if not provider.supports(dataset):
            return None
        return await getattr(provider, method)(ticker)

    return await asyncio.gather(
        provider.get_company_profile(ticker),
        optional(financial_data.STATEMENTS, "get_financial_statements"),
        optional(financial_data.RATIOS, "get_ratios"),
        optional(financial_data.ANALYST_ESTIMATES, "get_analyst_estimates"),
        optional(financial_data.METRICS, "get_company_metrics"),
        optional(financial_data.EARNINGS_CALENDAR, "get_earnings_calendar"),
    )


async def _fetch_macro_observations(provider: MacroProvider) -> list:
    """Fetch every default metric the router can actually serve,
    concurrently. Metrics no configured provider advertises are skipped
    (expected, not an error); a configured provider failing to serve a
    metric it does advertise propagates as MacroProviderError."""
    wanted = [m for m in DEFAULT_MACRO_METRICS if provider.supports(m)]
    observations = await asyncio.gather(*(provider.fetch(metric_id) for metric_id in wanted))
    return list(observations)


async def _fetch_catalyst_inputs(
    news_provider: NewsProvider | None,
    sec_provider: SECProvider | None,
    macro_provider: MacroProvider | None,
    ticker: str,
) -> tuple[list, list, list, list[str]]:
    """Gather the catalyst agent's three streams concurrently, each
    guarded independently so one source failing (or being unconfigured)
    doesn't lose the others. Returns (articles, filings, observations,
    errors); each error string names the failed stream and is surfaced as
    a pipeline_warning by the caller. News/macro fetches reuse whatever
    the news/macro steps already cached (shared TTL cache), so this adds
    no real network cost when those providers are also configured."""

    async def articles():
        if news_provider is None:
            return [], None
        try:
            return await news_provider.search(ticker), None
        except NewsProviderError as exc:
            return [], f"news — {exc}"

    async def filings():
        if sec_provider is None:
            return [], None
        try:
            return await sec_provider.get_recent_filings(ticker, limit=10), None
        except SecProviderError as exc:
            return [], f"SEC filings — {exc}"

    async def observations():
        if macro_provider is None:
            return [], None
        try:
            return await _fetch_macro_observations(macro_provider), None
        except MacroProviderError as exc:
            return [], f"macro — {exc}"

    (a, e_a), (f, e_f), (o, e_o) = await asyncio.gather(articles(), filings(), observations())
    errors = [e for e in (e_a, e_f, e_o) if e is not None]
    return a, f, o, errors


def _trailing_return(history: PriceHistory | None) -> float | None:
    """21-trading-day return from `history`, or None if unavailable —
    kept independent from analysis/domain_scoring.py's identical helper so
    the thesis layer doesn't reach into the analysis layer's internals."""
    if history is None:
        return None
    closes = [b.close for b in history.bars]
    if len(closes) <= _RELATIVE_STRENGTH_LOOKBACK_BARS:
        return None
    start, end = closes[-_RELATIVE_STRENGTH_LOOKBACK_BARS - 1], closes[-1]
    if start <= 0:
        return None
    return (end - start) / start


async def _fetch_relative_strength_inputs(
    market_data_provider: MarketDataProvider, symbol: str, sector: str | None
) -> tuple[float | None, float | None, float | None, str | None]:
    """Best-effort symbol/SPY/sector-ETF trailing returns for
    relative_strength_research — reuses MarketDataProvider, no new
    provider interface. A failed fetch for any one ticker just omits that
    return, never raises."""
    sector_etf = SECTOR_ETF_MAP.get(sector) if sector else None

    async def fetch(ticker: str | None) -> PriceHistory | None:
        if ticker is None:
            return None
        try:
            return await market_data_provider.get_price_history(ticker, _RELATIVE_STRENGTH_HISTORY_DAYS)
        except MarketDataError:
            return None

    symbol_history, benchmark_history, sector_history = await asyncio.gather(
        fetch(symbol), fetch("SPY"), fetch(sector_etf)
    )
    return (
        _trailing_return(symbol_history),
        _trailing_return(benchmark_history),
        _trailing_return(sector_history),
        sector_etf,
    )


def _run_fatal_agent(emit, reset_exchange, agent: str, run):
    """Run a required agent (quant / risk / strategy / investment_thesis)
    whose failure is fatal to the whole run: emit started → completed, or
    emit failed and re-raise so the caller surfaces the 502 (unchanged
    behavior — the emit is a no-op when no live view is attached)."""
    emit(agent, "started")
    reset_exchange()
    try:
        result = run()
    except (LlmError, ThesisGenerationError) as exc:
        emit(agent, "failed", detail=str(exc))
        raise
    emit(agent, "completed", output=result.model_dump(), with_exchange=True)
    return result


def run_thesis_pipeline(
    analysis_result: AnalysisResult,
    llm_client: LlmClient,
    market_data_provider: MarketDataProvider,
    financial_provider: FinancialProvider | None = None,
    news_provider: NewsProvider | None = None,
    macro_provider: MacroProvider | None = None,
    sec_provider: SECProvider | None = None,
    on_event: OnEvent | None = None,
) -> AgentThesisResult:
    # When a live view is requested, wrap the client so each agent's raw
    # prompt/response is captured for its event; otherwise run untouched.
    recorder = RecordingLlmClient(llm_client) if on_event is not None else None
    client: LlmClient = recorder if recorder is not None else llm_client

    def _emit(
        agent: str,
        phase: str,
        *,
        output: dict | None = None,
        detail: str | None = None,
        with_exchange: bool = False,
    ) -> None:
        if on_event is None:
            return
        exchange = None
        if with_exchange and recorder is not None and recorder.last_exchange is not None:
            system_prompt, user_prompt, raw_response = recorder.last_exchange
            exchange = AgentExchange(
                system_prompt=system_prompt, user_prompt=user_prompt, raw_response=raw_response
            )
        on_event(
            AgentEvent(
                agent=agent,
                phase=phase,
                at=datetime.now(timezone.utc).replace(tzinfo=None),
                output=output,
                detail=detail,
                exchange=exchange,
            )
        )

    def _reset_exchange() -> None:
        if recorder is not None:
            recorder.last_exchange = None

    recommendation = analysis_result.recommendation
    top_candidate = next(
        (c for c in analysis_result.candidates if c.contract_symbol == recommendation.contract_symbol),
        None,
    )

    if top_candidate is None:
        # No liquid candidate to assess or size — skip risk/strategy/
        # relative-strength/statistical-edge agents entirely rather than
        # asking an LLM to reason about data that doesn't exist (see
        # specs/agents.yaml: no_candidate_short_circuit).
        empty_trade_quality = compute_composite_score(
            {},
            source="quant",
            weighting_profile=analysis_result.weighting_profile,
            contract_symbol=None,
        )
        placeholder_technical = DomainScore(
            domain="technical",
            score=0.0,
            confidence=0.0,
            evidence=["No candidate contract to assess."],
            factors=[],
            source="quant",
            generated_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        quant = QuantInterpretation(
            narrative=recommendation.rationale,
            key_factors=[],
            quant_trade_quality=empty_trade_quality,
            technical_domain_score=placeholder_technical,
        )
        _emit("quant_interpreter", "completed", output=quant.model_dump())  # deterministic, no LLM call
        risk = None
        strategy = None
        relative_strength_finding = None
        statistical_edge_finding = None
        _emit("risk_challenger", "skipped", detail="No candidate contract to assess.")
        _emit("options_strategy", "skipped", detail="No candidate contract to build a strategy around.")
        _emit("relative_strength_research", "skipped", detail="No candidate contract to assess.")
        _emit("statistical_edge_research", "skipped", detail="No candidate contract to assess.")
    else:
        quant = _run_fatal_agent(
            _emit, _reset_exchange, "quant_interpreter",
            lambda: quant_interpreter.run(
                client, analysis_result.indicators, analysis_result.trend,
                analysis_result.volume, top_candidate, analysis_result.weighting_profile,
            ),
        )
        risk = _run_fatal_agent(
            _emit, _reset_exchange, "risk_challenger",
            lambda: risk_challenger.run(
                client, top_candidate, analysis_result.trend, analysis_result.support_resistance
            ),
        )
        strategy = _run_fatal_agent(
            _emit, _reset_exchange, "options_strategy",
            lambda: options_strategy.run(client, analysis_result.trend, top_candidate, risk),
        )

    # A configured provider failing mid-run (rate limited, network down,
    # bad ticker) must not throw away the work above: the affected finding
    # stays null and the failure is reported in pipeline_warnings at the
    # end of the run instead (see specs/agents.yaml provider_availability).
    pipeline_warnings: list[str] = []

    financial_finding = None
    company_sector: str | None = None
    if financial_provider is not None:
        _emit("financial_research", "started")
        ticker = analysis_result.symbol
        try:
            # FinancialProvider is async (specs/providers.yaml); this
            # pipeline is sync, so bridge with a private event loop and
            # fetch the four datasets concurrently.
            profile, statements, ratios, estimates, metrics, calendar = asyncio.run(
                _fetch_financial_inputs(financial_provider, ticker)
            )
            company_sector = profile.sector or None
            _reset_exchange()
            financial_finding = financial_research.run(
                client, profile, statements, ratios, estimates,
                metrics=metrics, earnings_calendar=calendar,
            )
        except FinancialProviderError as exc:
            pipeline_warnings.append(f"financial_research: provider failed during the run — {exc}")
            _emit("financial_research", "failed", detail=str(exc))
        except ThesisGenerationError as exc:
            pipeline_warnings.append(f"financial_research: unusable model response — {exc}")
            _emit("financial_research", "failed", detail=str(exc))
        else:
            _emit("financial_research", "completed", output=financial_finding.model_dump(), with_exchange=True)
    else:
        _emit("financial_research", "skipped", detail="No financial data provider configured.")

    news_finding = None
    if news_provider is not None:
        _emit("news_research", "started")
        ticker = analysis_result.symbol
        try:
            # NewsProvider is async (specs/providers.yaml); this pipeline is
            # sync, so bridge with a private event loop for the fetch.
            articles = asyncio.run(news_provider.search(ticker))
            _reset_exchange()
            news_finding = news_research.run(client, articles)
        except NewsProviderError as exc:
            pipeline_warnings.append(f"news_research: provider failed during the run — {exc}")
            _emit("news_research", "failed", detail=str(exc))
        except ThesisGenerationError as exc:
            pipeline_warnings.append(f"news_research: unusable model response — {exc}")
            _emit("news_research", "failed", detail=str(exc))
        else:
            _emit("news_research", "completed", output=news_finding.model_dump(), with_exchange=True)
    else:
        _emit("news_research", "skipped", detail="No news data provider configured.")

    macro_finding = None
    if macro_provider is not None:
        _emit("macro_research", "started")
        try:
            # MacroProvider is async and capability-based
            # (specs/providers.yaml); this pipeline is sync, so bridge
            # with a private event loop and fetch every serveable metric
            # concurrently.
            observations = asyncio.run(_fetch_macro_observations(macro_provider))
            if observations:
                _reset_exchange()
                macro_finding = macro_research.run(client, observations)
                _emit("macro_research", "completed", output=macro_finding.model_dump(), with_exchange=True)
            else:
                # No configured provider serves any requested metric — leave
                # the finding null (nothing to report), no warning.
                _emit("macro_research", "skipped", detail="No configured provider serves any requested metric.")
        except MacroProviderError as exc:
            pipeline_warnings.append(f"macro_research: provider failed during the run — {exc}")
            _emit("macro_research", "failed", detail=str(exc))
        except ThesisGenerationError as exc:
            pipeline_warnings.append(f"macro_research: unusable model response — {exc}")
            _emit("macro_research", "failed", detail=str(exc))
    else:
        _emit("macro_research", "skipped", detail="No macro data provider configured.")

    # Catalyst research combines all three research streams (news + SEC
    # filings + macro). It runs if ANY of them is configured, reasoning
    # over whatever subset is present; each stream is fetched under its own
    # guard so one failing only drops that stream (recorded as a warning),
    # not the whole finding. An unusable model response is likewise
    # non-fatal — the finding stays null and a warning is recorded.
    catalyst_finding = None
    if news_provider is not None or sec_provider is not None or macro_provider is not None:
        _emit("catalyst_research", "started")
        ticker = analysis_result.symbol
        articles, filings, observations, catalyst_errors = asyncio.run(
            _fetch_catalyst_inputs(news_provider, sec_provider, macro_provider, ticker)
        )
        for err in catalyst_errors:
            pipeline_warnings.append(f"catalyst_research: provider failed during the run — {err}")
        if articles or filings or observations:
            try:
                _reset_exchange()
                catalyst_finding = catalyst_research.run(client, articles, filings, observations)
            except ThesisGenerationError as exc:
                pipeline_warnings.append(f"catalyst_research: unusable model response — {exc}")
                _emit("catalyst_research", "failed", detail=str(exc))
            else:
                # A malformed individual catalyst is dropped rather than failing
                # the whole finding (models/schemas.py) — surface that it
                # happened so a silent drop is still visible at the end of the run.
                if catalyst_finding.dropped_count:
                    pipeline_warnings.append(
                        f"catalyst_research: dropped {catalyst_finding.dropped_count} malformed "
                        f"catalyst item(s) from the model response"
                    )
                _emit("catalyst_research", "completed", output=catalyst_finding.model_dump(), with_exchange=True)
        else:
            _emit("catalyst_research", "skipped", detail="No news, SEC, or macro data available for this ticker.")
    else:
        _emit("catalyst_research", "skipped", detail="No news, SEC, or macro provider configured.")

    if top_candidate is not None:
        _emit("relative_strength_research", "started")
        try:
            symbol_return, benchmark_return, sector_return, sector_etf = asyncio.run(
                _fetch_relative_strength_inputs(market_data_provider, analysis_result.symbol, company_sector)
            )
            _reset_exchange()
            relative_strength_finding = relative_strength_research.run(
                client,
                top_candidate.option_type,
                analysis_result.symbol,
                symbol_return,
                benchmark_return,
                sector_return,
                sector_etf,
            )
        except ThesisGenerationError as exc:
            pipeline_warnings.append(f"relative_strength_research: unusable model response — {exc}")
            _emit("relative_strength_research", "failed", detail=str(exc))
        else:
            _emit(
                "relative_strength_research", "completed",
                output=relative_strength_finding.model_dump(), with_exchange=True,
            )

        _emit("statistical_edge_research", "started")
        quant_statistical_edge = top_candidate.domain_scores.get("statistical_edge")
        try:
            _reset_exchange()
            statistical_edge_finding = statistical_edge_research.run(
                client, quant_statistical_edge, analysis_result.trend, top_candidate
            )
        except ThesisGenerationError as exc:
            pipeline_warnings.append(f"statistical_edge_research: unusable model response — {exc}")
            _emit("statistical_edge_research", "failed", detail=str(exc))
        else:
            _emit(
                "statistical_edge_research", "completed",
                output=statistical_edge_finding.model_dump(), with_exchange=True,
            )

    thesis = _run_fatal_agent(
        _emit, _reset_exchange, "investment_thesis",
        lambda: investment_thesis.run(
            client, quant, financial_finding, news_finding, macro_finding, catalyst_finding,
            risk, strategy, recommendation, analysis_result.trend, analysis_result.volume,
        ),
    )

    # Agent-side composite Trade Quality Score: the same engine the quant
    # path uses (analysis/composite_score.py), fed whichever agent
    # DomainScores this run actually produced.
    agent_domain_scores: dict[str, DomainScore] = {}
    if top_candidate is not None:
        agent_domain_scores["technical"] = quant.technical_domain_score
    if financial_finding is not None:
        agent_domain_scores["fundamental"] = financial_finding.domain_score
    if news_finding is not None:
        agent_domain_scores["sentiment"] = news_finding.domain_score
    if macro_finding is not None:
        agent_domain_scores["macro"] = macro_finding.domain_score
    if risk is not None:
        agent_domain_scores["risk"] = risk.domain_score
    if strategy is not None:
        agent_domain_scores["liquidity"] = strategy.domain_score
    if relative_strength_finding is not None:
        agent_domain_scores["relative_strength"] = relative_strength_finding.domain_score
    if statistical_edge_finding is not None:
        agent_domain_scores["statistical_edge"] = statistical_edge_finding.domain_score

    agent_trade_quality = (
        compute_composite_score(
            agent_domain_scores,
            source="agent",
            weighting_profile=analysis_result.weighting_profile,
            contract_symbol=recommendation.contract_symbol,
        )
        if agent_domain_scores
        else None
    )

    return AgentThesisResult(
        run_id=analysis_result.run_id,
        generated_at=datetime.now(timezone.utc).replace(tzinfo=None),
        quant_interpretation=quant,
        financial_research=financial_finding,
        news_research=news_finding,
        macro_research=macro_finding,
        catalyst_research=catalyst_finding,
        relative_strength_research=relative_strength_finding,
        statistical_edge_research=statistical_edge_finding,
        risk_assessment=risk,
        strategy_suggestion=strategy,
        investment_thesis=thesis,
        agent_trade_quality=agent_trade_quality,
        pipeline_warnings=pipeline_warnings,
    )
