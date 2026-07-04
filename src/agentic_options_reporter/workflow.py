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
from agentic_options_reporter.data.market_data import MarketDataProvider, build_market_data_provider
from agentic_options_reporter.models.schemas import AnalysisResult, OptionChain, PriceHistory
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


def run_analysis(
    symbol: str,
    lookback_days: int = 365,
    expiration: str | None = None,
    provider: MarketDataProvider | None = None,
    session_factory: sessionmaker | None = None,
) -> AnalysisResult:
    settings = get_settings()
    provider = provider or build_market_data_provider()
    session_factory = session_factory or make_session_factory(settings.database_url)

    # The MarketDataProvider is async; this pipeline is sync, so bridge
    # with a private event loop and fetch the two inputs concurrently.
    history, chain = asyncio.run(
        _fetch_market_data(provider, symbol, lookback_days, expiration)
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
    )
