"""Pipeline orchestration. Authoritative step order in specs/workflow.yaml."""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from sqlalchemy.orm import sessionmaker

from agentic_options_reporter.analysis.domain_scoring import SECTOR_ETF_MAP
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
from agentic_options_reporter.data.macro import (
    DEFAULT_MACRO_METRICS,
    MacroProvider,
    MacroProviderError,
    build_macro_provider,
)
from agentic_options_reporter.data.market_data import (
    MarketDataError,
    MarketDataProvider,
    build_market_data_provider,
)
from agentic_options_reporter.data.news import NewsProvider, NewsProviderError, build_news_provider
from agentic_options_reporter.models.schemas import (
    AnalysisResult,
    CompanyMetrics,
    FundamentalsSnapshot,
    MacroObservation,
    NewsArticle,
    OptionChain,
    PriceHistory,
    WeightingProfileId,
)
from agentic_options_reporter.persistence import (
    fetch_recent_runs_for_symbol,
    make_session_factory,
    persist_analysis_run,
)


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


async def _fetch_macro_observations(provider: MacroProvider) -> list[MacroObservation]:
    """Fetch every default metric the router can actually serve,
    concurrently (mirrors thesis/orchestrator.py's identically-named
    helper, kept independent so workflow.py doesn't depend on the thesis
    layer)."""
    wanted = [m for m in DEFAULT_MACRO_METRICS if provider.supports(m)]
    if not wanted:
        return []
    return list(await asyncio.gather(*(provider.fetch(metric_id) for metric_id in wanted)))


async def _fetch_benchmark_history(
    provider: MarketDataProvider, symbol: str, lookback_days: int
) -> PriceHistory | None:
    """Best-effort SPY/sector-ETF price history for the Relative Strength
    domain (analysis/domain_scoring.py) — reuses MarketDataProvider, no new
    provider interface. A failure here never fails /analyze."""
    try:
        return await provider.get_price_history(symbol, lookback_days)
    except MarketDataError:
        return None


async def _fetch_inputs(
    market_provider: MarketDataProvider,
    financial_provider: FinancialProvider | None,
    macro_provider: MacroProvider | None,
    news_provider: NewsProvider | None,
    symbol: str,
    lookback_days: int,
    expiration: str | None,
) -> tuple[
    PriceHistory,
    OptionChain,
    FundamentalsSnapshot | None,
    list[MacroObservation],
    list[NewsArticle],
    PriceHistory | None,
    PriceHistory | None,
    list[str],
]:
    """Fetch the technical inputs (required) alongside every best-effort,
    cross-provider signal the Trade Quality Score domains need: cross-
    provider fundamentals, macro observations, news articles, and a SPY
    benchmark price history. None of the best-effort fetches ever block or
    fail the analysis — a failure is recorded as a warning and the
    corresponding domain is simply omitted downstream (analysis/scoring.py)."""

    async def fundamentals() -> tuple[FundamentalsSnapshot | None, list[str]]:
        if financial_provider is None:
            return None, []
        try:
            return await gather_fundamentals(financial_provider, symbol)
        except FinancialProviderError as exc:
            return None, [f"fundamentals: {exc}"]

    async def macro() -> tuple[list[MacroObservation], str | None]:
        if macro_provider is None:
            return [], None
        try:
            return await _fetch_macro_observations(macro_provider), None
        except MacroProviderError as exc:
            return [], f"macro: {exc}"

    async def news() -> tuple[list[NewsArticle], str | None]:
        if news_provider is None:
            return [], None
        try:
            return await news_provider.search(symbol), None
        except NewsProviderError as exc:
            return [], f"news: {exc}"

    (
        (history, chain),
        (snapshot, fund_warnings),
        (macro_observations, macro_warning),
        (articles, news_warning),
        benchmark_history,
    ) = await asyncio.gather(
        _fetch_market_data(market_provider, symbol, lookback_days, expiration),
        fundamentals(),
        macro(),
        news(),
        _fetch_benchmark_history(market_provider, "SPY", lookback_days),
    )

    warnings = list(fund_warnings)
    if macro_warning:
        warnings.append(macro_warning)
    if news_warning:
        warnings.append(news_warning)

    # The sector-ETF benchmark depends on the sector fundamentals just
    # resolved, so it's a second, still best-effort round trip rather than
    # part of the gather above.
    sector = snapshot.profile.sector if snapshot and snapshot.profile else None
    sector_etf = SECTOR_ETF_MAP.get(sector) if sector else None
    sector_history = (
        await _fetch_benchmark_history(market_provider, sector_etf, lookback_days)
        if sector_etf
        else None
    )

    return history, chain, snapshot, macro_observations, articles, benchmark_history, sector_history, warnings


def _optional_financial_provider() -> FinancialProvider | None:
    """The fundamentals router (keyless Yahoo is always available, so this
    normally returns a live provider); None only if construction fails."""
    try:
        return build_financial_provider()
    except FinancialProviderError:
        return None


def _optional_macro_provider() -> MacroProvider | None:
    """Mirrors _optional_financial_provider: None only if construction
    fails (e.g. no macro adapter configured at all)."""
    try:
        return build_macro_provider()
    except MacroProviderError:
        return None


def _optional_news_provider() -> NewsProvider | None:
    """Mirrors _optional_financial_provider: None only if construction
    fails (e.g. no news adapter configured at all)."""
    try:
        return build_news_provider()
    except NewsProviderError:
        return None


# Trading-day windows for the derived price ranges (markets are closed
# weekends/holidays, so count bars rather than calendar days).
_WEEK_BARS = 5
_MONTH_BARS = 21


def _price_range_metrics(history: PriceHistory) -> dict[str, float]:
    """1-week and 1-month high/low derived from the most recent daily bars.
    Returns only the ranges the available history supports (a brand-new
    listing with <5 bars yields nothing)."""
    bars = history.bars
    ranges: dict[str, float] = {}
    if len(bars) >= _WEEK_BARS:
        window = bars[-_WEEK_BARS:]
        ranges["week1_high"] = max(b.high for b in window)
        ranges["week1_low"] = min(b.low for b in window)
    if len(bars) >= _MONTH_BARS:
        window = bars[-_MONTH_BARS:]
        ranges["month1_high"] = max(b.high for b in window)
        ranges["month1_low"] = min(b.low for b in window)
    return ranges


def _augment_price_ranges(
    fundamentals: FundamentalsSnapshot | None, history: PriceHistory, symbol: str
) -> FundamentalsSnapshot | None:
    """Fold the derived 1w/1m high/low into the snapshot's metrics. Creates
    a metrics record if the providers returned none, so the ranges show
    even when fundamentals coverage is otherwise thin."""
    ranges = _price_range_metrics(history)
    if not ranges:
        return fundamentals
    if fundamentals is None:
        fundamentals = FundamentalsSnapshot(ticker=symbol.upper())
    metrics = fundamentals.metrics or CompanyMetrics(ticker=symbol.upper())
    fundamentals = fundamentals.model_copy(
        update={"metrics": metrics.model_copy(update=ranges)}
    )
    return fundamentals


def run_analysis(
    symbol: str,
    lookback_days: int = 365,
    expiration: str | None = None,
    provider: MarketDataProvider | None = None,
    session_factory: sessionmaker | None = None,
    financial_provider: FinancialProvider | None = None,
    macro_provider: MacroProvider | None = None,
    news_provider: NewsProvider | None = None,
    weighting_profile: WeightingProfileId = "swing",
) -> AnalysisResult:
    settings = get_settings()
    provider = provider or build_market_data_provider()
    if financial_provider is None:
        financial_provider = _optional_financial_provider()
    if macro_provider is None:
        macro_provider = _optional_macro_provider()
    if news_provider is None:
        news_provider = _optional_news_provider()
    session_factory = session_factory or make_session_factory(settings.database_url)

    # The providers are async; this pipeline is sync, so bridge with a
    # private event loop and fetch the technicals + every best-effort
    # Trade Quality Score signal concurrently.
    (
        history,
        chain,
        fundamentals,
        macro_observations,
        news_articles,
        benchmark_history,
        sector_history,
        data_warnings,
    ) = asyncio.run(
        _fetch_inputs(
            provider, financial_provider, macro_provider, news_provider, symbol, lookback_days, expiration
        )
    )
    # No provider serves 1w/1m high/low, but we already have the daily bars,
    # so derive them and fold them into the fundamentals metrics snapshot.
    fundamentals = _augment_price_ranges(fundamentals, history, symbol)

    indicators = compute_indicators(history)
    trend = detect_trend(history, indicators)
    volume = analyze_volume(history, indicators)
    levels = detect_levels(history)

    evaluated_contracts = evaluate_chain(chain, history)
    risk_profiles = compute_risk(evaluated_contracts)

    # Opened before scoring (not just before persisting) so the Statistical
    # Edge domain scorer (analysis/statistical_edge.py) can look up this
    # symbol's past runs in the same session that will persist this one.
    with session_factory() as session:
        past_runs = fetch_recent_runs_for_symbol(session, symbol)

        candidates = score_candidates(
            evaluated_contracts,
            risk_profiles,
            trend,
            volume,
            levels,
            history,
            indicators,
            fundamentals=fundamentals,
            macro_observations=macro_observations,
            news_articles=news_articles,
            benchmark_history=benchmark_history,
            sector_history=sector_history,
            past_runs=past_runs,
            weighting_profile=weighting_profile,
        )
        recommendation, trade_quality = build_recommendation(candidates, weighting_profile)

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
            trade_quality,
            weighting_profile,
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
        trade_quality=trade_quality,
        weighting_profile=weighting_profile,
        # Cross-provider fundamentals, gathered best-effort alongside the
        # technicals and persisted with the run (see persist_analysis_run),
        # so /runs/{id} replays the same snapshot.
        fundamentals=fundamentals,
        data_warnings=data_warnings,
    )
