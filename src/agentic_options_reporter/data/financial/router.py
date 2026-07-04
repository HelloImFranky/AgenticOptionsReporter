"""Financial provider capability-filtering router and factory.

Each of the four datasets (profile/statements/ratios/analyst_estimates)
routes only among the providers that ADVERTISE it: the router filters to
supporters before calling, so Finnhub — which has no statements on the
free tier — is never asked for statements, rather than raising
Unsupported mid-call. An optional per-dataset priority override
(AOR_FINANCIAL_PRIORITY_<DATASET>) reorders candidates for one dataset.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

from agentic_options_reporter.data.financial.alphavantage import AlphaVantageFinancialProvider
from agentic_options_reporter.data.financial.base import (
    ANALYST_ESTIMATES,
    PROFILE,
    RATIOS,
    STATEMENTS,
    FinancialProvider,
    FinancialProviderError,
    FinancialProviderUnsupported,
    ProviderHealth,
)
from agentic_options_reporter.data.financial.finnhub import FinnhubFinancialProvider
from agentic_options_reporter.data.financial.fmp import FmpFinancialProvider
from agentic_options_reporter.data.provider_router import acall_with_fallback, filter_supporting
from agentic_options_reporter.models.schemas import (
    AnalystEstimates,
    CompanyProfile,
    FinancialRatios,
    FinancialStatementSummary,
)


class FinancialProviderRouter(FinancialProvider):
    """Capability-filtering failover router across configured fundamentals
    adapters. Implements FinancialProvider, so the financial_research
    consumer can't tell whether it's one adapter or many."""

    def __init__(self, clients: list[tuple[str, FinancialProvider]]) -> None:
        if not clients:
            raise FinancialProviderError(
                "No financial providers are configured for automatic failover. Set at "
                f"least one provider's API key (supported: {', '.join(sorted(_PROVIDERS))})."
            )
        self._clients = clients

    @property
    def provider_names(self) -> list[str]:
        return [name for name, _ in self._clients]

    @property
    def supported_datasets(self) -> frozenset[str]:
        return frozenset().union(*(client.supported_datasets for _, client in self._clients))

    def _candidates_for(self, dataset: str) -> list[tuple[str, FinancialProvider]]:
        candidates = filter_supporting(self._clients, dataset)
        override = _dataset_priority_override(dataset)
        if override:
            rank = {name: i for i, name in enumerate(override)}
            candidates.sort(key=lambda nc: rank.get(nc[0], len(override)))
        return candidates

    async def _route(self, dataset: str, method: str, ticker: str):
        candidates = self._candidates_for(dataset)
        if not candidates:
            raise FinancialProviderUnsupported(
                f"No configured financial provider serves '{dataset}'."
            )
        return await acall_with_fallback(candidates, method, FinancialProviderError, ticker)

    async def get_company_profile(self, ticker: str) -> CompanyProfile:
        return await self._route(PROFILE, "get_company_profile", ticker)

    async def get_financial_statements(self, ticker: str) -> FinancialStatementSummary:
        return await self._route(STATEMENTS, "get_financial_statements", ticker)

    async def get_ratios(self, ticker: str) -> FinancialRatios:
        return await self._route(RATIOS, "get_ratios", ticker)

    async def get_analyst_estimates(self, ticker: str) -> AnalystEstimates:
        return await self._route(ANALYST_ESTIMATES, "get_analyst_estimates", ticker)

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


_PROVIDERS: dict[str, type[FinancialProvider]] = {
    "fmp": FmpFinancialProvider,
    "finnhub": FinnhubFinancialProvider,
    "alphavantage": AlphaVantageFinancialProvider,
}

# FMP first (full coverage, ~250 requests/day), Finnhub next (~60/min but
# partial coverage), Alpha Vantage last (~25 requests/day).
_DEFAULT_FALLBACK_ORDER = ["fmp", "finnhub", "alphavantage"]


def _fallback_order() -> list[str]:
    raw = os.environ.get(
        "AOR_FINANCIAL_PROVIDER_FALLBACK_ORDER", ",".join(_DEFAULT_FALLBACK_ORDER)
    )
    return [name.strip().lower() for name in raw.split(",") if name.strip()]


def _dataset_priority_override(dataset: str) -> list[str]:
    """Optional per-dataset provider priority, e.g.
    AOR_FINANCIAL_PRIORITY_ANALYST_ESTIMATES="finnhub,fmp" to prefer
    Finnhub's derived consensus over FMP's. Falls back to the global
    order when unset."""
    raw = os.environ.get(f"AOR_FINANCIAL_PRIORITY_{dataset.upper()}", "")
    return [name.strip().lower() for name in raw.split(",") if name.strip()]


def build_financial_provider() -> FinancialProviderRouter:
    """Build a FinancialProviderRouter from
    AOR_FINANCIAL_PROVIDER_FALLBACK_ORDER, skipping any provider without a
    configured API key. Raises FinancialProviderError if the resulting
    router would have zero clients."""
    clients: list[tuple[str, FinancialProvider]] = []
    for name in _fallback_order():
        provider_cls = _PROVIDERS.get(name)
        if provider_cls is None:
            continue
        try:
            clients.append((name, provider_cls()))
        except FinancialProviderError:
            continue  # not configured (missing API key) — skip, don't fail the request
    return FinancialProviderRouter(clients)
