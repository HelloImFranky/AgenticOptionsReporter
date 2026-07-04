"""Macroeconomic provider interface and adapter base.

`MacroProvider` is the async interface the macro_research agent depends
on (dependency injection — same pattern as data.news/data.financial).
One adapter per source lives in this package (see specs/providers.yaml);
`router.build_macro_provider()` composes whichever are configured into a
failover router.

Most macro sources are specialists, not full-interface providers: BLS
publishes CPI but not GDP or rates; BEA publishes GDP but not CPI; IMF
and the World Bank publish CPI/GDP but not US policy rates. Each raises
`MacroProviderUnsupported` — retryable — for the methods outside its
domain, so the router still uses every source for what it does cover.

`_HttpMacroProvider` binds the shared async-HTTP infrastructure
(data.async_http: key handling, error normalization, class-level TTL
response cache, health probe) to this interface's error hierarchy.
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
from agentic_options_reporter.models.schemas import (
    CpiSnapshot,
    GdpSnapshot,
    InterestRates,
    MacroEvent,
)

__all__ = [
    "MacroProvider",
    "MacroProviderError",
    "MacroProviderRateLimited",
    "MacroProviderTimeout",
    "MacroProviderUnavailable",
    "MacroProviderUnsupported",
    "ProviderHealth",
]


class MacroProviderError(RuntimeError):
    """Raised when a MacroProvider cannot return the requested data."""


class MacroProviderRateLimited(MacroProviderError, ProviderRateLimited):
    """The provider rejected the request for exceeding its rate limit (HTTP 429)."""


class MacroProviderTimeout(MacroProviderError, ProviderTimeout):
    """The request to the provider timed out."""


class MacroProviderUnavailable(MacroProviderError, ProviderUnavailable):
    """The provider is unreachable or returned a server error (5xx / network failure)."""


class MacroProviderUnsupported(MacroProviderError, ProviderUnsupported):
    """This provider doesn't publish the requested data at all (e.g. BLS has no GDP series)."""


class MacroProvider(ABC):
    """Interface implemented by all macroeconomic data providers."""

    @abstractmethod
    async def get_interest_rates(self) -> InterestRates:
        raise NotImplementedError

    @abstractmethod
    async def get_cpi(self) -> CpiSnapshot:
        raise NotImplementedError

    @abstractmethod
    async def get_gdp(self) -> GdpSnapshot:
        raise NotImplementedError

    @abstractmethod
    async def get_macro_calendar(self) -> list[MacroEvent]:
        raise NotImplementedError

    @abstractmethod
    async def health(self) -> ProviderHealth:
        raise NotImplementedError


def yoy_change_pct(latest: float, year_ago: float | None) -> float | None:
    """Year-over-year percentage change from two provider-supplied
    observations — an arithmetic derivation, not fabrication."""
    if year_ago in (None, 0):
        return None
    return (latest - year_ago) / abs(year_ago) * 100


class _HttpMacroProvider(AsyncHttpProviderBase, MacroProvider):
    """Base for HTTP-backed macro adapters. Each adapter implements
    `_health_probe` against a series it actually publishes (a full-
    interface default would mark specialists unhealthy for data they
    never claimed to have)."""

    ERROR_CLS = MacroProviderError
    RATE_LIMITED_CLS = MacroProviderRateLimited
    TIMEOUT_CLS = MacroProviderTimeout
    UNAVAILABLE_CLS = MacroProviderUnavailable
