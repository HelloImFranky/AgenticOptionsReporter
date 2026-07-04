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

from agentic_options_reporter.data.financial import FinancialProvider, FinancialProviderError
from agentic_options_reporter.data.macro_provider import MacroProvider, MacroProviderError
from agentic_options_reporter.data.news import NewsProvider, NewsProviderError
from agentic_options_reporter.models.schemas import (
    AgentThesisResult,
    AnalysisResult,
    QuantInterpretation,
)
from agentic_options_reporter.thesis import (
    financial_research,
    investment_thesis,
    macro_research,
    news_research,
    options_strategy,
    quant_interpreter,
    risk_challenger,
)
from agentic_options_reporter.thesis.llm_client import LlmClient


async def _fetch_financial_inputs(provider: FinancialProvider, ticker: str) -> tuple:
    return await asyncio.gather(
        provider.get_company_profile(ticker),
        provider.get_financial_statements(ticker),
        provider.get_ratios(ticker),
        provider.get_analyst_estimates(ticker),
    )


def run_thesis_pipeline(
    analysis_result: AnalysisResult,
    llm_client: LlmClient,
    financial_provider: FinancialProvider | None = None,
    news_provider: NewsProvider | None = None,
    macro_provider: MacroProvider | None = None,
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

    macro_finding = None
    if macro_provider is not None:
        try:
            macro_finding = macro_research.run(
                llm_client,
                macro_provider.get_interest_rates(),
                macro_provider.get_cpi(),
                macro_provider.get_gdp(),
                macro_provider.get_macro_calendar(),
            )
        except MacroProviderError as exc:
            pipeline_warnings.append(f"macro_research: provider failed during the run — {exc}")

    thesis = investment_thesis.run(
        llm_client,
        quant,
        financial_finding,
        news_finding,
        macro_finding,
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
        risk_assessment=risk,
        strategy_suggestion=strategy,
        investment_thesis=thesis,
        pipeline_warnings=pipeline_warnings,
    )
