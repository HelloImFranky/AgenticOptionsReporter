"""Market-data capability-filtering router and configuration-driven factory.

Like the macro/financial routers, this SELECTS providers by declared
capability before calling: for `price_history` it narrows to the sources
that advertise it (all of them), and for `option_chain` to the sources
that advertise it (yfinance today), applies any per-capability priority
override, then fails over among just those on transient errors. A
price-only source is never asked for an option chain.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

from agentic_options_reporter.data.market_data.alphavantage import AlphaVantageMarketDataProvider
from agentic_options_reporter.data.market_data.base import (
    OPTION_CHAIN,
    PRICE_HISTORY,
    MarketDataError,
    MarketDataProvider,
    MarketDataUnsupported,
    ProviderHealth,
)
from agentic_options_reporter.data.market_data.finnhub import FinnhubMarketDataProvider
from agentic_options_reporter.data.market_data.twelvedata import TwelveDataMarketDataProvider
from agentic_options_reporter.data.market_data.yfinance_provider import YFinanceProvider
from agentic_options_reporter.data.provider_router import acall_with_fallback, filter_supporting
from agentic_options_reporter.models.schemas import OptionChain, PriceHistory


class MarketDataProviderRouter(MarketDataProvider):
    """Capability-filtering failover router across configured market-data
    adapters. Implements MarketDataProvider itself, so the analysis
    pipeline can't tell whether it's talking to one adapter or many."""

    def __init__(self, clients: list[tuple[str, MarketDataProvider]]) -> None:
        if not clients:
            raise MarketDataError(
                "No market-data providers are configured for automatic failover. Set at "
                f"least one provider's API key (supported: {', '.join(sorted(_PROVIDERS))})."
            )
        self._clients = clients

    @property
    def provider_names(self) -> list[str]:
        return [name for name, _ in self._clients]

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset().union(*(client.capabilities for _, client in self._clients))

    def _candidates_for(self, capability: str) -> list[tuple[str, MarketDataProvider]]:
        candidates = filter_supporting(self._clients, capability)
        override = _capability_priority_override(capability)
        if override:
            rank = {name: i for i, name in enumerate(override)}
            candidates.sort(key=lambda nc: rank.get(nc[0], len(override)))
        return candidates

    async def get_price_history(self, symbol: str, lookback_days: int = 365) -> PriceHistory:
        candidates = self._candidates_for(PRICE_HISTORY)
        if not candidates:
            raise MarketDataUnsupported("No configured market-data provider serves price history.")
        return await acall_with_fallback(
            candidates, "get_price_history", MarketDataError, symbol, lookback_days
        )

    async def get_option_chain(self, symbol: str, expiration: str | None = None) -> OptionChain:
        candidates = self._candidates_for(OPTION_CHAIN)
        if not candidates:
            raise MarketDataUnsupported("No configured market-data provider serves option chains.")
        return await acall_with_fallback(
            candidates, "get_option_chain", MarketDataError, symbol, expiration
        )

    async def health(self) -> ProviderHealth:
        """Probe every adapter concurrently; the router is healthy if any
        adapter is. `detail` carries the per-adapter breakdown."""
        results = await asyncio.gather(*(client.health() for _, client in self._clients))
        healthy = any(result.healthy for result in results)
        detail = "; ".join(
            f"{result.provider}: {'ok' if result.healthy else result.detail or 'unhealthy'}"
            for result in results
        )
        return ProviderHealth(
            provider="router",
            healthy=healthy,
            latency_ms=max((r.latency_ms or 0.0) for r in results) if results else None,
            detail=detail,
            checked_at=datetime.now(timezone.utc),
        )


_PROVIDERS: dict[str, type[MarketDataProvider]] = {
    "yfinance": YFinanceProvider,
    "alphavantage": AlphaVantageMarketDataProvider,
    "twelvedata": TwelveDataMarketDataProvider,
    "finnhub": FinnhubMarketDataProvider,
}

# yfinance first: keyless, serves both capabilities, and the only source
# of option chains. The keyed HTTP sources add price-history redundancy.
_DEFAULT_FALLBACK_ORDER = ["yfinance", "alphavantage", "twelvedata", "finnhub"]


def _fallback_order() -> list[str]:
    raw = os.environ.get(
        "AOR_MARKET_DATA_PROVIDER_FALLBACK_ORDER", ",".join(_DEFAULT_FALLBACK_ORDER)
    )
    return [name.strip().lower() for name in raw.split(",") if name.strip()]


def _capability_priority_override(capability: str) -> list[str]:
    """Optional per-capability provider priority, e.g.
    AOR_MARKET_DATA_PRIORITY_PRICE_HISTORY="alphavantage,yfinance" to
    prefer Alpha Vantage's price history. Falls back to the global order
    when unset (see specs/providers.yaml: configurable_priority)."""
    raw = os.environ.get(f"AOR_MARKET_DATA_PRIORITY_{capability.upper()}", "")
    return [name.strip().lower() for name in raw.split(",") if name.strip()]


def build_market_data_provider() -> MarketDataProviderRouter:
    """Build a MarketDataProviderRouter from
    AOR_MARKET_DATA_PROVIDER_FALLBACK_ORDER, skipping any provider without
    a configured API key. yfinance is keyless, so with the default order
    the router is never empty — market data is always available."""
    clients: list[tuple[str, MarketDataProvider]] = []
    for name in _fallback_order():
        provider_cls = _PROVIDERS.get(name)
        if provider_cls is None:
            continue
        try:
            clients.append((name, provider_cls()))
        except MarketDataError:
            continue  # not configured (missing API key) — skip, don't fail the request
    return MarketDataProviderRouter(clients)
