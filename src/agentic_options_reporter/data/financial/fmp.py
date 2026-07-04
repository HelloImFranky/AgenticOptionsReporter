"""Financial Modeling Prep adapter (financialmodelingprep.com — free tier
~250 requests/day).

Full coverage of the interface: profile, statements, ratios, and
analyst estimates.
"""

from __future__ import annotations

from typing import Any

from agentic_options_reporter.data.financial.base import _HttpFinancialProvider
from agentic_options_reporter.models.schemas import (
    AnalystEstimates,
    CompanyProfile,
    FinancialRatios,
    FinancialStatementSummary,
)


class FmpFinancialProvider(_HttpFinancialProvider):
    BASE_URL = "https://financialmodelingprep.com/api/v3"
    PROVIDER_LABEL = "Financial Modeling Prep"
    API_KEY_ENV_VAR = "FMP_API_KEY"

    async def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        query = dict(params or {})
        query["apikey"] = self._api_key
        return await self._get_json(f"{self.BASE_URL}{path}", query)

    @staticmethod
    def _first_or_empty(data: Any) -> dict[str, Any]:
        if isinstance(data, list) and data:
            return data[0]
        return {}

    async def get_company_profile(self, ticker: str) -> CompanyProfile:
        item = self._first_or_empty(await self._get(f"/profile/{ticker.upper()}"))
        return CompanyProfile(
            ticker=ticker.upper(),
            name=item.get("companyName", ""),
            sector=item.get("sector", ""),
            industry=item.get("industry", ""),
            market_cap=item.get("mktCap"),
            description=item.get("description", ""),
        )

    async def get_financial_statements(self, ticker: str) -> FinancialStatementSummary:
        income = self._first_or_empty(
            await self._get(f"/income-statement/{ticker.upper()}", {"period": "annual", "limit": 1})
        )
        cash_flow = self._first_or_empty(
            await self._get(f"/cash-flow-statement/{ticker.upper()}", {"period": "annual", "limit": 1})
        )
        return FinancialStatementSummary(
            ticker=ticker.upper(),
            period=income.get("calendarYear", income.get("date", "")),
            revenue=income.get("revenue"),
            net_income=income.get("netIncome"),
            operating_cash_flow=cash_flow.get("operatingCashFlow"),
            free_cash_flow=cash_flow.get("freeCashFlow"),
        )

    async def get_ratios(self, ticker: str) -> FinancialRatios:
        item = self._first_or_empty(await self._get(f"/ratios/{ticker.upper()}", {"limit": 1}))
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

    async def get_analyst_estimates(self, ticker: str) -> AnalystEstimates:
        item = self._first_or_empty(
            await self._get(f"/analyst-estimates/{ticker.upper()}", {"limit": 1})
        )
        return AnalystEstimates(
            ticker=ticker.upper(),
            consensus_rating=item.get("consensusRating", "N/A"),
            price_target_mean=item.get("estimatedPriceTargetAvg"),
            price_target_high=item.get("estimatedPriceTargetHigh"),
            price_target_low=item.get("estimatedPriceTargetLow"),
            num_analysts=int(item.get("numberAnalystEstimatedRevenue") or 0),
        )
