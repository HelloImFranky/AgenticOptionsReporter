"""Company fundamentals data access.

`FinancialProvider` is the interface the financial_research agent
depends on (dependency injection — the same pattern as
`market_data.MarketDataProvider`). `FmpFinancialProvider` is the
phase-2a implementation (see specs/providers.yaml), backed by Financial
Modeling Prep's free tier.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from agentic_options_reporter.models.schemas import (
    AnalystEstimates,
    CompanyProfile,
    FinancialRatios,
    FinancialStatementSummary,
)


class FinancialProviderError(RuntimeError):
    """Raised when a FinancialProvider cannot return the requested data."""


class FinancialProvider(ABC):
    """Interface implemented by all company-fundamentals providers."""

    @abstractmethod
    def get_company_profile(self, ticker: str) -> CompanyProfile:
        raise NotImplementedError

    @abstractmethod
    def get_financial_statements(self, ticker: str) -> FinancialStatementSummary:
        raise NotImplementedError

    @abstractmethod
    def get_ratios(self, ticker: str) -> FinancialRatios:
        raise NotImplementedError

    @abstractmethod
    def get_analyst_estimates(self, ticker: str) -> AnalystEstimates:
        raise NotImplementedError


class FmpFinancialProvider(FinancialProvider):
    """FinancialProvider implementation backed by Financial Modeling Prep."""

    BASE_URL = "https://financialmodelingprep.com/api/v3"

    def __init__(self, api_key: str | None = None, timeout_seconds: int = 15) -> None:
        self._api_key = api_key or os.environ.get("FMP_API_KEY")
        if not self._api_key:
            raise FinancialProviderError(
                "No Financial Modeling Prep API key configured. Set FMP_API_KEY, "
                "or supply one explicitly."
            )
        self._timeout = timeout_seconds

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        import requests

        url = f"{self.BASE_URL}{path}"
        query = dict(params or {})
        query["apikey"] = self._api_key
        try:
            response = requests.get(url, params=query, timeout=self._timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise FinancialProviderError(f"FMP request to {path} failed: {exc}") from exc
        return response.json()

    def _first_or_empty(self, data: Any) -> dict[str, Any]:
        if isinstance(data, list) and data:
            return data[0]
        return {}

    def get_company_profile(self, ticker: str) -> CompanyProfile:
        item = self._first_or_empty(self._get(f"/profile/{ticker}"))
        return CompanyProfile(
            ticker=ticker.upper(),
            name=item.get("companyName", ""),
            sector=item.get("sector", ""),
            industry=item.get("industry", ""),
            market_cap=item.get("mktCap"),
            description=item.get("description", ""),
        )

    def get_financial_statements(self, ticker: str) -> FinancialStatementSummary:
        income = self._first_or_empty(
            self._get(f"/income-statement/{ticker}", {"period": "annual", "limit": 1})
        )
        cash_flow = self._first_or_empty(
            self._get(f"/cash-flow-statement/{ticker}", {"period": "annual", "limit": 1})
        )
        return FinancialStatementSummary(
            ticker=ticker.upper(),
            period=income.get("calendarYear", income.get("date", "")),
            revenue=income.get("revenue"),
            net_income=income.get("netIncome"),
            operating_cash_flow=cash_flow.get("operatingCashFlow"),
            free_cash_flow=cash_flow.get("freeCashFlow"),
        )

    def get_ratios(self, ticker: str) -> FinancialRatios:
        item = self._first_or_empty(self._get(f"/ratios/{ticker}", {"limit": 1}))
        return FinancialRatios(
            ticker=ticker.upper(),
            pe_ratio=item.get("priceEarningsRatio"),
            pb_ratio=item.get("priceToBookRatio"),
            debt_to_equity=item.get("debtEquityRatio"),
            current_ratio=item.get("currentRatio"),
            return_on_equity=item.get("returnOnEquity"),
            gross_margin=item.get("grossProfitMargin"),
            net_margin=item.get("netProfitMargin"),
        )

    def get_analyst_estimates(self, ticker: str) -> AnalystEstimates:
        item = self._first_or_empty(self._get(f"/analyst-estimates/{ticker}", {"limit": 1}))
        return AnalystEstimates(
            ticker=ticker.upper(),
            consensus_rating=item.get("consensusRating", "N/A"),
            price_target_mean=item.get("estimatedPriceTargetAvg"),
            price_target_high=item.get("estimatedPriceTargetHigh"),
            price_target_low=item.get("estimatedPriceTargetLow"),
            num_analysts=int(item.get("numberAnalystEstimatedRevenue") or 0),
        )
