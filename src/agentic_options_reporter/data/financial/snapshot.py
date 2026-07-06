"""Best-effort, cross-provider fundamentals gathering for one ticker.

`gather_fundamentals` fetches every dataset a configured FinancialProvider
(usually the merge router) advertises — profile, statements, ratios,
analyst estimates, metrics, earnings history, earnings calendar, insider
activity — concurrently and independently: a dataset no provider serves
comes back None, and one that fails is recorded as a warning rather than
losing the rest. It returns the merged FundamentalsSnapshot plus the list
of warnings, so /analyze can surface as much as is available without any
single source failure aborting the run.
"""

from __future__ import annotations

import asyncio

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
)
from agentic_options_reporter.models.schemas import FundamentalsSnapshot

# (dataset capability, provider method, FundamentalsSnapshot field).
_DATASETS = [
    (PROFILE, "get_company_profile", "profile"),
    (STATEMENTS, "get_financial_statements", "statements"),
    (RATIOS, "get_ratios", "ratios"),
    (ANALYST_ESTIMATES, "get_analyst_estimates", "estimates"),
    (METRICS, "get_company_metrics", "metrics"),
    (EARNINGS, "get_earnings_history", "earnings_history"),
    (EARNINGS_CALENDAR, "get_earnings_calendar", "earnings_calendar"),
    (INSIDER, "get_insider_activity", "insider_activity"),
]


async def gather_fundamentals(
    provider: FinancialProvider, ticker: str
) -> tuple[FundamentalsSnapshot, list[str]]:
    """Fetch every advertised fundamentals dataset concurrently and return
    the merged snapshot plus any per-dataset warnings (never raises for a
    single dataset failing)."""

    async def fetch(dataset: str, method: str, field: str):
        if not provider.supports(dataset):
            return field, None, None
        try:
            return field, await getattr(provider, method)(ticker), None
        except FinancialProviderError as exc:
            return field, None, f"{field}: {exc}"

    results = await asyncio.gather(*(fetch(*d) for d in _DATASETS))

    fields: dict[str, object] = {"ticker": ticker.upper()}
    warnings: list[str] = []
    for field, value, warning in results:
        fields[field] = value
        if warning is not None:
            warnings.append(warning)
    return FundamentalsSnapshot(**fields), warnings
