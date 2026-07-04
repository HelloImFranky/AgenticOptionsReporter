"""Market-data provider interface and adapter base.

`MarketDataProvider` is the async interface the analysis pipeline depends
on (dependency injection — same pattern as data.news/financial/macro).
One adapter per source lives in this package (see specs/providers.yaml);
`router.build_market_data_provider()` composes whichever are configured
into a capability-filtering failover router.

Capability-based, like the fundamentals layer: a provider declares which
capabilities it serves (`capabilities`), and the router filters to
supporters before calling. Most sources serve `price_history` but not
free `option_chain` data, so an option-chain request routes only to the
sources that advertise it (yfinance today) — the same
filter-before-calling model as "Finnhub has no financial statements."

`_HttpMarketDataProvider` binds the shared async-HTTP infrastructure
(data.async_http: key handling, error normalization, class-level TTL
response cache, health probe) to this interface's error hierarchy.
YFinanceProvider is the exception — it wraps the synchronous `yfinance`
library via asyncio.to_thread rather than the httpx base (see
yfinance_provider.py).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from agentic_options_reporter.data.async_http import AsyncHttpProviderBase, ProviderHealth
from agentic_options_reporter.data.provider_errors import (
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
    ProviderUnsupported,
)
from agentic_options_reporter.models.schemas import OptionChain, PriceHistory

# Capabilities a market-data provider may serve. Small and fixed (like
# the fundamentals datasets, unlike macro's open-ended metric registry).
# Every source serves PRICE_HISTORY; free OPTION_CHAIN data is rare, so in
# practice only yfinance advertises it and the router filters chain
# requests to it (see specs/providers.yaml).
PRICE_HISTORY = "price_history"
OPTION_CHAIN = "option_chain"
MARKET_DATA_CAPABILITIES = frozenset({PRICE_HISTORY, OPTION_CHAIN})

# Probe ticker for health checks: one cheap price-history request against
# a symbol every market-data source covers.
_HEALTH_PROBE_TICKER = "AAPL"

__all__ = [
    "MARKET_DATA_CAPABILITIES",
    "OPTION_CHAIN",
    "PRICE_HISTORY",
    "MarketDataError",
    "MarketDataProvider",
    "MarketDataRateLimited",
    "MarketDataTimeout",
    "MarketDataUnavailable",
    "MarketDataUnsupported",
    "ProviderHealth",
]


class MarketDataError(RuntimeError):
    """Raised when a provider cannot return the requested data."""


class MarketDataRateLimited(MarketDataError, ProviderRateLimited):
    """The provider rejected the request for exceeding its rate limit (HTTP 429)."""


class MarketDataTimeout(MarketDataError, ProviderTimeout):
    """The request to the provider timed out."""


class MarketDataUnavailable(MarketDataError, ProviderUnavailable):
    """The provider is unreachable or returned a server error (5xx / network failure)."""


class MarketDataUnsupported(MarketDataError, ProviderUnsupported):
    """This provider doesn't offer the requested data at all (e.g. a
    price-only source asked for an option chain)."""


class MarketDataProvider(ABC):
    """Interface implemented by all market-data providers.

    Capability-based: a provider declares which capabilities it serves
    (`capabilities`), and the router filters to supporters before calling
    — so a price-only source is never asked for an option chain, rather
    than raising Unsupported mid-call.
    """

    @property
    @abstractmethod
    def capabilities(self) -> frozenset[str]:
        """The MARKET_DATA_CAPABILITIES this provider serves."""
        raise NotImplementedError

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    @abstractmethod
    async def get_price_history(self, symbol: str, lookback_days: int = 365) -> PriceHistory:
        raise NotImplementedError

    @abstractmethod
    async def get_option_chain(self, symbol: str, expiration: str | None = None) -> OptionChain:
        raise NotImplementedError

    @abstractmethod
    async def health(self) -> ProviderHealth:
        raise NotImplementedError


class _HttpMarketDataProvider(AsyncHttpProviderBase, MarketDataProvider):
    """Base for HTTP-backed market-data adapters. A subclass sets
    `CAPABILITIES` (the ids it serves). These sources cover price history
    only; free option-chain endpoints are rare, so `get_option_chain`
    here raises MarketDataUnsupported (retryable — the router falls
    through to a source that does serve chains)."""

    ERROR_CLS = MarketDataError
    RATE_LIMITED_CLS = MarketDataRateLimited
    TIMEOUT_CLS = MarketDataTimeout
    UNAVAILABLE_CLS = MarketDataUnavailable

    CAPABILITIES: frozenset[str] = frozenset({PRICE_HISTORY})

    @property
    def capabilities(self) -> frozenset[str]:
        return self.CAPABILITIES

    async def get_option_chain(self, symbol: str, expiration: str | None = None) -> OptionChain:
        raise MarketDataUnsupported(
            f"{self.PROVIDER_LABEL} does not provide option-chain data."
        )

    async def _health_probe(self) -> None:
        await self.get_price_history(_HEALTH_PROBE_TICKER, lookback_days=5)
