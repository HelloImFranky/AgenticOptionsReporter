"""Macroeconomic provider interface and adapter base.

`MacroProvider` is the async, CAPABILITY-BASED interface the
macro_research agent depends on (via the router). Rather than a fixed
method per metric, a provider declares which metric ids it serves
(`supported_metrics`) and fetches any one of them through a single
`fetch(metric_id)` coroutine returning a normalized `MacroObservation`.

This is the redesign from the provider-architecture doc: the router
FILTERS to providers that advertise a metric before calling, so a
specialist is never asked for data it doesn't have (the "World Bank has
no US policy rate" case). Adding a metric is a registry entry plus the
adapters that serve it — no new interface method, no provider forced to
stub out data it lacks. Unsupported metrics are structural (a provider
simply isn't in the candidate list), not an exception caught mid-call.

`_HttpMacroProvider` binds the shared async-HTTP infrastructure
(data.async_http: key handling, error normalization, class-level TTL
response cache, health probe) to this interface's error hierarchy and
provides the metric-dispatch + health-probe scaffolding.
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
from agentic_options_reporter.models.schemas import MacroObservation

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
    """This provider doesn't publish the requested metric (e.g. BLS has no GDP series).

    In the capability model the router filters these out up front, so
    this is mostly a defensive guard for a direct `fetch` of an
    unadvertised metric.
    """


class MacroProvider(ABC):
    """Interface implemented by all macroeconomic data providers."""

    @property
    @abstractmethod
    def supported_metrics(self) -> frozenset[str]:
        """The metric ids (data.macro.metrics) this provider serves."""
        raise NotImplementedError

    def supports(self, metric_id: str) -> bool:
        return metric_id in self.supported_metrics

    @abstractmethod
    async def fetch(self, metric_id: str) -> MacroObservation:
        """Return the latest normalized observation for `metric_id`, or
        raise MacroProviderUnsupported if this provider doesn't serve it."""
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
    """Base for HTTP-backed macro adapters.

    A subclass sets `METRICS` (the ids it serves) and implements
    `_fetch(metric_id)`. This base validates the id against `METRICS`
    (defensive — the router already filters), and probes the first
    supported metric for `health()` so a specialist is never marked
    unhealthy for data it never claimed to have.
    """

    ERROR_CLS = MacroProviderError
    RATE_LIMITED_CLS = MacroProviderRateLimited
    TIMEOUT_CLS = MacroProviderTimeout
    UNAVAILABLE_CLS = MacroProviderUnavailable

    METRICS: frozenset[str] = frozenset()

    @property
    def supported_metrics(self) -> frozenset[str]:
        return self.METRICS

    async def fetch(self, metric_id: str) -> MacroObservation:
        if metric_id not in self.METRICS:
            raise MacroProviderUnsupported(
                f"{self.PROVIDER_LABEL} does not publish '{metric_id}'."
            )
        return await self._fetch(metric_id)

    @abstractmethod
    async def _fetch(self, metric_id: str) -> MacroObservation:
        raise NotImplementedError

    async def _health_probe(self) -> None:
        metric_id = next(iter(self.METRICS), None)
        if metric_id is not None:
            await self.fetch(metric_id)
