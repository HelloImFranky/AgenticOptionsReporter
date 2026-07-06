"""Alpha Vantage fundamentals adapter (alphavantage.co — free tier ~25
requests/day, why it sits last in the default fallback order).

Uses the OVERVIEW / INCOME_STATEMENT / CASH_FLOW endpoints. OVERVIEW
doesn't include debt_to_equity or current_ratio and has no dedicated
analyst-estimates endpoint (only a single AnalystTargetPrice figure) —
those fields stay at their schema defaults rather than being guessed,
the same "missing data stays missing" convention as the other adapters.
The free tier returns HTTP 200 with an "Information"/"Note" field
instead of a proper 429 when rate limited; `_check_payload` treats that
the same as a real 429.
"""

from __future__ import annotations

from typing import Any

from agentic_options_reporter.data.financial.base import (
    CORE_FINANCIAL_DATASETS,
    FinancialProviderRateLimited,
    _HttpFinancialProvider,
)
from agentic_options_reporter.models.schemas import (
    AnalystEstimates,
    CompanyProfile,
    FinancialRatios,
    FinancialStatementSummary,
)


class AlphaVantageFinancialProvider(_HttpFinancialProvider):
    BASE_URL = "https://www.alphavantage.co/query"
    PROVIDER_LABEL = "Alpha Vantage"
    API_KEY_ENV_VAR = "ALPHA_VANTAGE_API_KEY"

    # Serves the core four, though analyst_estimates is price-target-only
    # (no consensus rating) and ratios omit debt/current — data-quality
    # gaps kept null, not capability gaps.
    DATASETS = CORE_FINANCIAL_DATASETS

    def _check_payload(self, payload: Any) -> None:
        if isinstance(payload, dict) and ("Information" in payload or "Note" in payload):
            raise FinancialProviderRateLimited(
                f"{self.PROVIDER_LABEL} rate limited or restricted: "
                f"{payload.get('Information') or payload.get('Note')}"
            )

    async def _get(self, function: str, ticker: str) -> Any:
        return await self._get_json(
            self.BASE_URL,
            {"function": function, "symbol": ticker.upper(), "apikey": self._api_key},
        )

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value in (None, "", "None", "-"):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    async def get_company_profile(self, ticker: str) -> CompanyProfile:
        item = await self._get("OVERVIEW", ticker)
        return CompanyProfile(
            ticker=ticker.upper(),
            name=item.get("Name", ""),
            sector=item.get("Sector", ""),
            industry=item.get("Industry", ""),
            market_cap=self._to_float(item.get("MarketCapitalization")),
            description=item.get("Description", ""),
        )

    async def get_financial_statements(self, ticker: str) -> FinancialStatementSummary:
        income_reports = (await self._get("INCOME_STATEMENT", ticker)).get("annualReports") or []
        cash_flow_reports = (await self._get("CASH_FLOW", ticker)).get("annualReports") or []
        income = income_reports[0] if income_reports else {}
        cash_flow = cash_flow_reports[0] if cash_flow_reports else {}

        operating_cash_flow = self._to_float(cash_flow.get("operatingCashflow"))
        capital_expenditures = self._to_float(cash_flow.get("capitalExpenditures"))
        free_cash_flow = (
            operating_cash_flow - capital_expenditures
            if operating_cash_flow is not None and capital_expenditures is not None
            else None
        )
        return FinancialStatementSummary(
            ticker=ticker.upper(),
            period=income.get("fiscalDateEnding", ""),
            revenue=self._to_float(income.get("totalRevenue")),
            net_income=self._to_float(income.get("netIncome")),
            operating_cash_flow=operating_cash_flow,
            free_cash_flow=free_cash_flow,
        )

    async def get_ratios(self, ticker: str) -> FinancialRatios:
        item = await self._get("OVERVIEW", ticker)
        revenue_ttm = self._to_float(item.get("RevenueTTM"))
        gross_profit_ttm = self._to_float(item.get("GrossProfitTTM"))
        gross_margin = (
            gross_profit_ttm / revenue_ttm
            if gross_profit_ttm is not None and revenue_ttm not in (None, 0)
            else None
        )
        return FinancialRatios(
            ticker=ticker.upper(),
            pe_ratio=self._to_float(item.get("PERatio")),
            pb_ratio=self._to_float(item.get("PriceToBookRatio")),
            debt_to_equity=None,  # not provided by OVERVIEW
            current_ratio=None,  # not provided by OVERVIEW
            return_on_equity=self._to_float(item.get("ReturnOnEquityTTM")),
            gross_margin=gross_margin,
            net_margin=self._to_float(item.get("ProfitMargin")),
        )

    async def get_analyst_estimates(self, ticker: str) -> AnalystEstimates:
        item = await self._get("OVERVIEW", ticker)
        return AnalystEstimates(
            ticker=ticker.upper(),
            consensus_rating="N/A",  # OVERVIEW has no consensus-rating field
            price_target_mean=self._to_float(item.get("AnalystTargetPrice")),
            price_target_high=None,
            price_target_low=None,
            num_analysts=0,
        )
