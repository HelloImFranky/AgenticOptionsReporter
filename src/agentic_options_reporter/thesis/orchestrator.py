"""Orchestrates the investment-thesis agent pipeline (specs/agents.yaml).

Coordination only — no LLM calls of its own. Runs quant_interpreter, then
the optional financial/news/macro research agents (each skipped with a
null finding if its provider wasn't supplied — see specs/providers.yaml
provider_availability), then (unless short-circuited) risk_challenger and
options_strategy, then investment_thesis, and assembles the result.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from agentic_options_reporter.data import financial as financial_data
from agentic_options_reporter.data.financial import FinancialProvider, FinancialProviderError
from agentic_options_reporter.data.macro import (
    DEFAULT_MACRO_METRICS,
    MacroProvider,
    MacroProviderError,
)
from agentic_options_reporter.data.news import NewsProvider, NewsProviderError
from agentic_options_reporter.data.sec_provider import SECProvider, SecProviderError
from agentic_options_reporter.models.schemas import (
    AgentThesisResult,
    AnalysisResult,
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
    risk_challenger,
)
from agentic_options_reporter.thesis.llm_client import LlmClient
from agentic_options_reporter.thesis.parsing import ThesisGenerationError


async def _fetch_financial_inputs(provider: FinancialProvider, ticker: str) -> tuple:
    """Fetch the company profile (the required anchor) plus whichever of
    statements/ratios/estimates the configured providers actually serve,
    concurrently. A dataset no provider covers (e.g. statements when only
    Finnhub is configured) comes back None and the agent omits it."""

    async def optional(dataset: str, method: str):
        if not provider.supports(dataset):
            return None
        return await getattr(provider, method)(ticker)

    return await asyncio.gather(
        provider.get_company_profile(ticker),
        optional(financial_data.STATEMENTS, "get_financial_statements"),
        optional(financial_data.RATIOS, "get_ratios"),
        optional(financial_data.ANALYST_ESTIMATES, "get_analyst_estimates"),
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


def run_thesis_pipeline(
    analysis_result: AnalysisResult,
    llm_client: LlmClient,
    financial_provider: FinancialProvider | None = None,
    news_provider: NewsProvider | None = None,
    macro_provider: MacroProvider | None = None,
    sec_provider: SECProvider | None = None,
) -> AgentThesisResult:
    recommendation = analysis_result.recommendation
    top_candidate = next(
        (c for c in analysis_result.candidates if c.contract_symbol == recommendation.contract_symbol),
        None,
    )

    if top_candidate is None:
        # No liquid candidate to assess or size — skip risk/strategy agents
        # entirely rather than asking an LLM to reason about data that
        # doesn't exist (see specs/agents.yaml: no_candidate_short_circuit).
        quant = QuantInterpretation(
            narrative=recommendation.rationale,
            key_factors=[],
            score_breakdown={},
            overall_score=0.0,
        )
        risk = None
        strategy = None
    else:
        quant = quant_interpreter.run(
            llm_client,
            analysis_result.indicators,
            analysis_result.trend,
            analysis_result.volume,
            top_candidate,
        )
        risk = risk_challenger.run(
            llm_client, top_candidate, analysis_result.trend, analysis_result.support_resistance
        )
        strategy = options_strategy.run(llm_client, analysis_result.trend, top_candidate, risk)

    # A configured provider failing mid-run (rate limited, network down,
    # bad ticker) must not throw away the work above: the affected finding
    # stays null and the failure is reported in pipeline_warnings at the
    # end of the run instead (see specs/agents.yaml provider_availability).
    pipeline_warnings: list[str] = []

    financial_finding = None
    if financial_provider is not None:
        ticker = analysis_result.symbol
        try:
            # FinancialProvider is async (specs/providers.yaml); this
            # pipeline is sync, so bridge with a private event loop and
            # fetch the four datasets concurrently.
            profile, statements, ratios, estimates = asyncio.run(
                _fetch_financial_inputs(financial_provider, ticker)
            )
            financial_finding = financial_research.run(
                llm_client, profile, statements, ratios, estimates
            )
        except FinancialProviderError as exc:
            pipeline_warnings.append(f"financial_research: provider failed during the run — {exc}")
        except ThesisGenerationError as exc:
            pipeline_warnings.append(f"financial_research: unusable model response — {exc}")

    news_finding = None
    if news_provider is not None:
        ticker = analysis_result.symbol
        try:
            # NewsProvider is async (specs/providers.yaml); this pipeline is
            # sync, so bridge with a private event loop for the fetch.
            articles = asyncio.run(news_provider.search(ticker))
            news_finding = news_research.run(llm_client, articles)
        except NewsProviderError as exc:
            pipeline_warnings.append(f"news_research: provider failed during the run — {exc}")
        except ThesisGenerationError as exc:
            pipeline_warnings.append(f"news_research: unusable model response — {exc}")

    macro_finding = None
    if macro_provider is not None:
        try:
            # MacroProvider is async and capability-based
            # (specs/providers.yaml); this pipeline is sync, so bridge
            # with a private event loop and fetch every serveable metric
            # concurrently.
            observations = asyncio.run(_fetch_macro_observations(macro_provider))
            if observations:
                macro_finding = macro_research.run(llm_client, observations)
            # else: no configured provider serves any requested metric —
            # leave the finding null (nothing to report), no warning.
        except MacroProviderError as exc:
            pipeline_warnings.append(f"macro_research: provider failed during the run — {exc}")
        except ThesisGenerationError as exc:
            pipeline_warnings.append(f"macro_research: unusable model response — {exc}")

    # Catalyst research combines all three research streams (news + SEC
    # filings + macro). It runs if ANY of them is configured, reasoning
    # over whatever subset is present; each stream is fetched under its own
    # guard so one failing only drops that stream (recorded as a warning),
    # not the whole finding. An unusable model response is likewise
    # non-fatal — the finding stays null and a warning is recorded.
    catalyst_finding = None
    if news_provider is not None or sec_provider is not None or macro_provider is not None:
        ticker = analysis_result.symbol
        articles, filings, observations, catalyst_errors = asyncio.run(
            _fetch_catalyst_inputs(news_provider, sec_provider, macro_provider, ticker)
        )
        for err in catalyst_errors:
            pipeline_warnings.append(f"catalyst_research: provider failed during the run — {err}")
        if articles or filings or observations:
            try:
                catalyst_finding = catalyst_research.run(llm_client, articles, filings, observations)
            except ThesisGenerationError as exc:
                pipeline_warnings.append(f"catalyst_research: unusable model response — {exc}")
            else:
                # A malformed individual catalyst is dropped rather than failing
                # the whole finding (models/schemas.py) — surface that it
                # happened so a silent drop is still visible at the end of the run.
                if catalyst_finding.dropped_count:
                    pipeline_warnings.append(
                        f"catalyst_research: dropped {catalyst_finding.dropped_count} malformed "
                        f"catalyst item(s) from the model response"
                    )

    thesis = investment_thesis.run(
        llm_client,
        quant,
        financial_finding,
        news_finding,
        macro_finding,
        catalyst_finding,
        risk,
        strategy,
        recommendation,
        analysis_result.trend,
        analysis_result.volume,
    )

    return AgentThesisResult(
        run_id=analysis_result.run_id,
        generated_at=datetime.now(timezone.utc).replace(tzinfo=None),
        quant_interpretation=quant,
        financial_research=financial_finding,
        news_research=news_finding,
        macro_research=macro_finding,
        catalyst_research=catalyst_finding,
        risk_assessment=risk,
        strategy_suggestion=strategy,
        investment_thesis=thesis,
        pipeline_warnings=pipeline_warnings,
    )
