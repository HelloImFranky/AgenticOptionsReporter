"""Pipeline orchestration. Authoritative step order in specs/workflow.yaml."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from sqlalchemy.orm import sessionmaker

from agentic_options_reporter.analysis.indicators import compute_indicators
from agentic_options_reporter.analysis.options import evaluate_chain
from agentic_options_reporter.analysis.risk import compute_risk
from agentic_options_reporter.analysis.scoring import build_recommendation, score_candidates
from agentic_options_reporter.analysis.support_resistance import detect_levels
from agentic_options_reporter.analysis.trend import detect_trend
from agentic_options_reporter.analysis.volume import analyze_volume
from agentic_options_reporter.config import get_settings
from agentic_options_reporter.data.financial import (
    FinancialProvider,
    FinancialProviderError,
    build_financial_provider,
)
from agentic_options_reporter.data.financial.snapshot import gather_fundamentals
from agentic_options_reporter.data.market_data import MarketDataProvider, build_market_data_provider
from agentic_options_reporter.models.schemas import (
    AnalysisResult,
    FundamentalsSnapshot,
    OptionChain,
    PriceHistory,
)
from agentic_options_reporter.persistence import make_session_factory, persist_analysis_run


async def _fetch_market_data(
    provider: MarketDataProvider, symbol: str, lookback_days: int, expiration: str | None
) -> tuple[PriceHistory, OptionChain]:
    """Fetch price history and the option chain concurrently — they're
    independent, so overlap the two round-trips (the MarketDataProvider is
    async; see specs/providers.yaml)."""
    return await asyncio.gather(
        provider.get_price_history(symbol, lookback_days),
        provider.get_option_chain(symbol, expiration),
    )


async def _fetch_inputs(
    market_provider: MarketDataProvider,
    financial_provider: FinancialProvider | None,
    symbol: str,
    lookback_days: int,
    expiration: str | None,
) -> tuple[PriceHistory, OptionChain, FundamentalsSnapshot | None, list[str]]:
    """Fetch the technical inputs (required) and the cross-provider
    fundamentals (best-effort) together. Fundamentals never block the
    analysis: any failure gathering them is returned as a warning and the
    snapshot comes back with whatever succeeded (or None if no provider)."""

    async def fundamentals() -> tuple[FundamentalsSnapshot | None, list[str]]:
        if financial_provider is None:
            return None, []
        try:
            return await gather_fundamentals(financial_provider, symbol)
        except FinancialProviderError as exc:
            return None, [f"fundamentals: {exc}"]

    (history, chain), (snapshot, warnings) = await asyncio.gather(
        _fetch_market_data(market_provider, symbol, lookback_days, expiration),
        fundamentals(),
    )
    return history, chain, snapshot, warnings


def _optional_financial_provider() -> FinancialProvider | None:
    """The fundamentals router (keyless Yahoo is always available, so this
    normally returns a live provider); None only if construction fails."""
    try:
        return build_financial_provider()
    except FinancialProviderError:
        return None


def run_analysis(
    symbol: str,
    lookback_days: int = 365,
    expiration: str | None = None,
    provider: MarketDataProvider | None = None,
    session_factory: sessionmaker | None = None,
    financial_provider: FinancialProvider | None = None,
) -> AnalysisResult:
    settings = get_settings()
    provider = provider or build_market_data_provider()
    if financial_provider is None:
        financial_provider = _optional_financial_provider()
    session_factory = session_factory or make_session_factory(settings.database_url)

    # The providers are async; this pipeline is sync, so bridge with a
    # private event loop and fetch the technicals + fundamentals concurrently.
    history, chain, fundamentals, data_warnings = asyncio.run(
        _fetch_inputs(provider, financial_provider, symbol, lookback_days, expiration)
    )

    indicators = compute_indicators(history)
    trend = detect_trend(history, indicators)
    volume = analyze_volume(history, indicators)
    levels = detect_levels(history)

    evaluated_contracts = evaluate_chain(chain, history)
    risk_profiles = compute_risk(evaluated_contracts)
    candidates = score_candidates(evaluated_contracts, risk_profiles, trend, volume, levels)
    recommendation = build_recommendation(candidates)

    with session_factory() as session:
        run_id = persist_analysis_run(
            session,
            symbol,
            lookback_days,
            expiration,
            indicators,
            trend,
            volume,
            levels,
            candidates,
            recommendation,
            fundamentals=fundamentals,
            data_warnings=data_warnings,
        )

    return AnalysisResult(
        symbol=symbol,
        run_id=run_id,
        generated_at=datetime.now(timezone.utc).replace(tzinfo=None),
        indicators=indicators,
        trend=trend,
        volume=volume,
        support_resistance=levels,
        candidates=candidates,
        recommendation=recommendation,
        # Cross-provider fundamentals, gathered best-effort alongside the
        # technicals and persisted with the run (see persist_analysis_run),
        # so /runs/{id} replays the same snapshot.
        fundamentals=fundamentals,
        data_warnings=data_warnings,
    )
