"""Financial provider capability-filtering + fan-out-merge router and factory.

Each dataset routes only among the providers that ADVERTISE it: the router
filters to supporters before calling, so Finnhub — which has no statements
on the free tier — is never asked for statements, rather than raising
Unsupported mid-call. An optional per-dataset priority override
(AOR_FINANCIAL_PRIORITY_<DATASET>) reorders candidates for one dataset.

Most datasets are then MERGED across every supporting provider rather than
taken from the first that answers: profile/ratios/estimates/metrics/calendar
are unioned field-by-field (a value present in Finnhub but missing in Yahoo
is filled in, and vice versa), and earnings/insider lists are unioned and
de-duplicated — so "try all providers and get all the data" holds. Only
statements stay failover: they're period-bound, and merging a Q3 revenue
from one source with an FY net income from another would silently corrupt
the record.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

from agentic_options_reporter.data.financial.alphavantage import AlphaVantageFinancialProvider
from agentic_options_reporter.data.financial.base import (
    ANALYST_ESTIMATES,
    EARNINGS,
    EARNINGS_CALENDAR,
    INSIDER,
    METRICS,
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
from agentic_options_reporter.data.financial.yfinance_provider import YFinanceFinancialProvider
from agentic_options_reporter.data.provider_router import (
    acall_and_merge,
    acall_with_fallback,
    filter_supporting,
    merge_lists,
    merge_models,
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


def _combine_earnings(results: list[EarningsHistory]) -> EarningsHistory:
    surprises = merge_lists((r.surprises for r in results), key=lambda s: s.period)
    return EarningsHistory(ticker=results[0].ticker, surprises=surprises)


def _combine_insider(results: list[InsiderActivity]) -> InsiderActivity:
    transactions = merge_lists(
        (r.transactions for r in results),
        key=lambda t: (t.name, t.filed_at, t.shares, t.transaction_type),
    )
    total = 0.0
    seen = False
    for tx in transactions:
        if tx.shares is None:
            continue
        seen = True
        total += tx.shares if tx.transaction_type != "sell" else -tx.shares
    return InsiderActivity(
        ticker=results[0].ticker,
        transactions=transactions,
        net_shares=total if seen else None,
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

    def _require_candidates(self, dataset: str, method: str):
        candidates = self._candidates_for(dataset)
        if not candidates:
            raise FinancialProviderUnsupported(
                f"No configured financial provider serves '{dataset}'."
            )
        return candidates

    async def _route_failover(self, dataset: str, method: str, ticker: str):
        """First-success routing — for datasets that must NOT be merged."""
        candidates = self._require_candidates(dataset, method)
        return await acall_with_fallback(candidates, method, FinancialProviderError, ticker)

    async def _route_merged(self, dataset: str, method: str, combine, ticker: str):
        """Fan-out across every supporting provider and merge the results."""
        candidates = self._require_candidates(dataset, method)
        return await acall_and_merge(candidates, method, FinancialProviderError, combine, ticker)

    async def get_company_profile(self, ticker: str) -> CompanyProfile:
        return await self._route_merged(PROFILE, "get_company_profile", merge_models, ticker)

    async def get_financial_statements(self, ticker: str) -> FinancialStatementSummary:
        # Period-bound — merging across providers would mix reporting
        # periods, so this one stays first-success failover.
        return await self._route_failover(STATEMENTS, "get_financial_statements", ticker)

    async def get_ratios(self, ticker: str) -> FinancialRatios:
        return await self._route_merged(RATIOS, "get_ratios", merge_models, ticker)

    async def get_analyst_estimates(self, ticker: str) -> AnalystEstimates:
        return await self._route_merged(
            ANALYST_ESTIMATES, "get_analyst_estimates", merge_models, ticker
        )

    async def get_company_metrics(self, ticker: str) -> CompanyMetrics:
        return await self._route_merged(METRICS, "get_company_metrics", merge_models, ticker)

    async def get_earnings_history(self, ticker: str) -> EarningsHistory:
        return await self._route_merged(
            EARNINGS, "get_earnings_history", _combine_earnings, ticker
        )

    async def get_earnings_calendar(self, ticker: str) -> EarningsCalendar:
        return await self._route_merged(
            EARNINGS_CALENDAR, "get_earnings_calendar", merge_models, ticker
        )

    async def get_insider_activity(self, ticker: str) -> InsiderActivity:
        return await self._route_merged(INSIDER, "get_insider_activity", _combine_insider, ticker)

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
    "yfinance": YFinanceFinancialProvider,
    "alphavantage": AlphaVantageFinancialProvider,
}

# FMP first (full coverage, ~250 requests/day), Finnhub next (~60/min,
# metrics/earnings/insider), Yahoo (keyless, broad coverage incl. all the
# newer datasets — always available), Alpha Vantage last (~25 requests/day).
# With merge routing, order is a priority for field-level tie-breaks, not a
# stop-at-first list.
_DEFAULT_FALLBACK_ORDER = ["fmp", "finnhub", "yfinance", "alphavantage"]


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
