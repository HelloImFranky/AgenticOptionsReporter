"""Finnhub fundamentals adapter (finnhub.io — free tier ~60 requests/min).

Partial coverage, per Finnhub's free tier:

- get_company_profile → /stock/profile2
- get_ratios → /stock/metric (TTM ratio metrics)
- get_analyst_estimates → /stock/recommendation — the consensus rating
  is DERIVED as the largest bucket of the latest month's analyst counts
  (strongBuy/buy/hold/sell/strongSell), which is a summary of
  provider-supplied tallies, not a fabricated opinion; price targets
  are a premium endpoint and stay null.
- get_financial_statements → raises FinancialProviderUnsupported
  (retryable), so the router falls through to FMP/Alpha Vantage for
  statements while still using Finnhub for what it covers — the same
  partial-coverage pattern as BLS/BEA in the macro package.
"""

from __future__ import annotations

from typing import Any

from agentic_options_reporter.data.financial.base import (
    FinancialProviderUnsupported,
    _HttpFinancialProvider,
)
from agentic_options_reporter.models.schemas import (
    AnalystEstimates,
    CompanyProfile,
    FinancialRatios,
    FinancialStatementSummary,
)

# Ordered so ties break toward the more cautious label.
_RECOMMENDATION_BUCKETS = [
    ("strongSell", "Strong Sell"),
    ("sell", "Sell"),
    ("hold", "Hold"),
    ("buy", "Buy"),
    ("strongBuy", "Strong Buy"),
]


class FinnhubFinancialProvider(_HttpFinancialProvider):
    BASE_URL = "https://finnhub.io/api/v1"
    PROVIDER_LABEL = "Finnhub"
    API_KEY_ENV_VAR = "FINNHUB_API_KEY"

    async def _get(self, path: str, params: dict[str, Any]) -> Any:
        return await self._get_json(
            f"{self.BASE_URL}{path}", {**params, "token": self._api_key}
        )

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value in (None, ""):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    async def get_company_profile(self, ticker: str) -> CompanyProfile:
        item = await self._get("/stock/profile2", {"symbol": ticker.upper()})
        market_cap = self._to_float(item.get("marketCapitalization"))
        return CompanyProfile(
            ticker=ticker.upper(),
            name=item.get("name", ""),
            sector="",  # profile2 exposes only an industry classification
            industry=item.get("finnhubIndustry", ""),
            # profile2 reports market cap in millions.
            market_cap=market_cap * 1_000_000 if market_cap is not None else None,
            description="",
        )

    async def get_financial_statements(self, ticker: str) -> FinancialStatementSummary:
        raise FinancialProviderUnsupported(
            "Finnhub's free tier does not expose raw financial statements."
        )

    async def get_ratios(self, ticker: str) -> FinancialRatios:
        data = await self._get("/stock/metric", {"symbol": ticker.upper(), "metric": "all"})
        metric = data.get("metric") or {}
        # Finnhub reports margins as percentages (e.g. 45.2), not fractions.
        gross_margin = self._to_float(metric.get("grossMarginTTM"))
        net_margin = self._to_float(metric.get("netProfitMarginTTM"))
        roe = self._to_float(metric.get("roeTTM"))
        return FinancialRatios(
            ticker=ticker.upper(),
            pe_ratio=self._to_float(metric.get("peTTM")),
            pb_ratio=self._to_float(metric.get("pb")),
            debt_to_equity=self._to_float(metric.get("totalDebt/totalEquityQuarterly")),
            current_ratio=self._to_float(metric.get("currentRatioQuarterly")),
            return_on_equity=roe / 100 if roe is not None else None,
            gross_margin=gross_margin / 100 if gross_margin is not None else None,
            net_margin=net_margin / 100 if net_margin is not None else None,
        )

    async def get_analyst_estimates(self, ticker: str) -> AnalystEstimates:
        data = await self._get("/stock/recommendation", {"symbol": ticker.upper()})
        latest = data[0] if isinstance(data, list) and data else {}
        counts = {
            field: int(latest.get(field) or 0) for field, _ in _RECOMMENDATION_BUCKETS
        }
        num_analysts = sum(counts.values())
        if num_analysts:
            top_field = max(counts, key=counts.get)
            consensus_rating = dict(_RECOMMENDATION_BUCKETS)[top_field]
        else:
            consensus_rating = "N/A"
        return AnalystEstimates(
            ticker=ticker.upper(),
            consensus_rating=consensus_rating,
            price_target_mean=None,  # /stock/price-target is a premium endpoint
            price_target_high=None,
            price_target_low=None,
            num_analysts=num_analysts,
        )
