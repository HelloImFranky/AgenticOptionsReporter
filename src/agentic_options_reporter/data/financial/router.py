"""Financial provider failover router and configuration-driven factory."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

from agentic_options_reporter.data.financial.alphavantage import AlphaVantageFinancialProvider
from agentic_options_reporter.data.financial.base import (
    FinancialProvider,
    FinancialProviderError,
    ProviderHealth,
)
from agentic_options_reporter.data.financial.finnhub import FinnhubFinancialProvider
from agentic_options_reporter.data.financial.fmp import FmpFinancialProvider
from agentic_options_reporter.data.provider_router import acall_with_fallback
from agentic_options_reporter.models.schemas import (
    AnalystEstimates,
    CompanyProfile,
    FinancialRatios,
    FinancialStatementSummary,
)


class FinancialProviderRouter(FinancialProvider):
    """Tries a priority-ordered list of already-constructed
    FinancialProvider adapters per method call, advancing to the next on
    a retryable failure (see data.provider_router) — so a
    partial-coverage adapter like Finnhub (no statements on the free
    tier) is still used for the methods it does support."""

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

    async def get_company_profile(self, ticker: str) -> CompanyProfile:
        return await acall_with_fallback(
            self._clients, "get_company_profile", FinancialProviderError, ticker
        )

    async def get_financial_statements(self, ticker: str) -> FinancialStatementSummary:
        return await acall_with_fallback(
            self._clients, "get_financial_statements", FinancialProviderError, ticker
        )

    async def get_ratios(self, ticker: str) -> FinancialRatios:
        return await acall_with_fallback(
            self._clients, "get_ratios", FinancialProviderError, ticker
        )

    async def get_analyst_estimates(self, ticker: str) -> AnalystEstimates:
        return await acall_with_fallback(
            self._clients, "get_analyst_estimates", FinancialProviderError, ticker
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


_PROVIDERS: dict[str, type[FinancialProvider]] = {
    "fmp": FmpFinancialProvider,
    "finnhub": FinnhubFinancialProvider,
    "alphavantage": AlphaVantageFinancialProvider,
}

# FMP first (full interface coverage, ~250 requests/day), Finnhub next
# (~60/min but partial coverage), Alpha Vantage last (~25 requests/day).
_DEFAULT_FALLBACK_ORDER = ["fmp", "finnhub", "alphavantage"]


def _fallback_order() -> list[str]:
    raw = os.environ.get(
        "AOR_FINANCIAL_PROVIDER_FALLBACK_ORDER", ",".join(_DEFAULT_FALLBACK_ORDER)
    )
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
