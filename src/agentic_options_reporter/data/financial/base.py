"""Company-fundamentals provider interface and adapter base.

`FinancialProvider` is the async interface the financial_research agent
depends on (dependency injection — same pattern as data.news). One
adapter per source lives in this package (see specs/providers.yaml);
`router.build_financial_provider()` composes whichever are configured
into a failover router.

`_HttpFinancialProvider` binds the shared async-HTTP infrastructure
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
    AnalystEstimates,
    CompanyMetrics,
    CompanyProfile,
    EarningsCalendar,
    EarningsHistory,
    FinancialRatios,
    FinancialStatementSummary,
    InsiderActivity,
)

# Dataset "capabilities" a fundamentals provider may serve. Small and
# fixed (unlike macro's open-ended metric registry), so plain constants
# rather than a registry — but the same capability-declaration idea: a
# provider advertises which of these it covers and the router filters to
# supporters before calling (see specs/providers.yaml).
PROFILE = "profile"
STATEMENTS = "statements"
RATIOS = "ratios"
ANALYST_ESTIMATES = "analyst_estimates"
# Newer datasets — not every provider serves them, so they're advertised
# via `supported_datasets` the same way and default to Unsupported below.
METRICS = "metrics"
EARNINGS = "earnings"
EARNINGS_CALENDAR = "earnings_calendar"
INSIDER = "insider"
# The original four datasets every full-coverage source (FMP, Alpha
# Vantage) serves. The newer four (metrics/earnings/calendar/insider) are
# served by Finnhub and Yahoo only, so they're advertised individually
# rather than folded into a "full coverage" set.
CORE_FINANCIAL_DATASETS = frozenset({PROFILE, STATEMENTS, RATIOS, ANALYST_ESTIMATES})
# Every dataset the interface knows about (the universe).
FINANCIAL_DATASETS = frozenset(
    {PROFILE, STATEMENTS, RATIOS, ANALYST_ESTIMATES, METRICS, EARNINGS, EARNINGS_CALENDAR, INSIDER}
)

__all__ = [
    "ANALYST_ESTIMATES",
    "CORE_FINANCIAL_DATASETS",
    "EARNINGS",
    "EARNINGS_CALENDAR",
    "FINANCIAL_DATASETS",
    "INSIDER",
    "METRICS",
    "PROFILE",
    "RATIOS",
    "STATEMENTS",
    "FinancialProvider",
    "FinancialProviderError",
    "FinancialProviderRateLimited",
    "FinancialProviderTimeout",
    "FinancialProviderUnavailable",
    "FinancialProviderUnsupported",
    "ProviderHealth",
]


class FinancialProviderError(RuntimeError):
    """Raised when a FinancialProvider cannot return the requested data."""


class FinancialProviderRateLimited(FinancialProviderError, ProviderRateLimited):
    """The provider rejected the request for exceeding its rate limit (HTTP 429)."""


class FinancialProviderTimeout(FinancialProviderError, ProviderTimeout):
    """The request to the provider timed out."""


class FinancialProviderUnavailable(FinancialProviderError, ProviderUnavailable):
    """The provider is unreachable or returned a server error (5xx / network failure)."""


class FinancialProviderUnsupported(FinancialProviderError, ProviderUnsupported):
    """This provider doesn't offer the requested data at all (e.g. Finnhub's
    free tier has no raw financial statements)."""


class FinancialProvider(ABC):
    """Interface implemented by all company-fundamentals providers.

    Capability-based: a provider declares which datasets it serves
    (`supported_datasets`), and the router filters to supporters before
    calling a dataset's method — so Finnhub (no statements on the free
    tier) is never asked for statements, rather than raising Unsupported
    mid-call.
    """

    @property
    @abstractmethod
    def supported_datasets(self) -> frozenset[str]:
        """The FINANCIAL_DATASETS ids this provider serves."""
        raise NotImplementedError

    def supports(self, dataset: str) -> bool:
        return dataset in self.supported_datasets

    @abstractmethod
    async def get_company_profile(self, ticker: str) -> CompanyProfile:
        raise NotImplementedError

    @abstractmethod
    async def get_financial_statements(self, ticker: str) -> FinancialStatementSummary:
        raise NotImplementedError

    @abstractmethod
    async def get_ratios(self, ticker: str) -> FinancialRatios:
        raise NotImplementedError

    @abstractmethod
    async def get_analyst_estimates(self, ticker: str) -> AnalystEstimates:
        raise NotImplementedError

    # The four datasets below arrived after the original interface. They
    # are NOT abstract — a provider serves one only if it advertises it in
    # `supported_datasets`, and the router never calls an un-advertised
    # dataset. The default raises Unsupported (retryable) as a defensive
    # guard, so existing adapters that don't serve them need no changes.
    async def get_company_metrics(self, ticker: str) -> CompanyMetrics:
        raise FinancialProviderUnsupported(
            f"{type(self).__name__} does not provide company metrics."
        )

    async def get_earnings_history(self, ticker: str) -> EarningsHistory:
        raise FinancialProviderUnsupported(
            f"{type(self).__name__} does not provide earnings history."
        )

    async def get_earnings_calendar(self, ticker: str) -> EarningsCalendar:
        raise FinancialProviderUnsupported(
            f"{type(self).__name__} does not provide an earnings calendar."
        )

    async def get_insider_activity(self, ticker: str) -> InsiderActivity:
        raise FinancialProviderUnsupported(
            f"{type(self).__name__} does not provide insider transactions."
        )

    @abstractmethod
    async def health(self) -> ProviderHealth:
        raise NotImplementedError


# Probe ticker for health checks: one cheap, cacheable profile request
# against a symbol every fundamentals source covers.
_HEALTH_PROBE_TICKER = "AAPL"


class _HttpFinancialProvider(AsyncHttpProviderBase, FinancialProvider):
    """Base for HTTP-backed fundamentals adapters. A subclass sets
    `DATASETS` (the ids it serves) — every one serves at least PROFILE,
    the health-probe dataset."""

    ERROR_CLS = FinancialProviderError
    RATE_LIMITED_CLS = FinancialProviderRateLimited
    TIMEOUT_CLS = FinancialProviderTimeout
    UNAVAILABLE_CLS = FinancialProviderUnavailable

    DATASETS: frozenset[str] = frozenset()

    @property
    def supported_datasets(self) -> frozenset[str]:
        return self.DATASETS

    async def _health_probe(self) -> None:
        await self.get_company_profile(_HEALTH_PROBE_TICKER)
