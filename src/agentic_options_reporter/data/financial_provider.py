"""Company fundamentals data access.

`FinancialProvider` is the interface the financial_research agent
depends on (dependency injection — the same pattern as
`market_data.MarketDataProvider`). Two concrete implementations exist
(Financial Modeling Prep, Alpha Vantage — see specs/providers.yaml);
`build_financial_provider()` composes whichever are currently configured
into a `FinancialProviderRouter` that fails over between them per method
call, the data-provider analog of `thesis.llm_client.LlmRouter`.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

from agentic_options_reporter.data.provider_errors import (
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
    ProviderUnsupported,
)
from agentic_options_reporter.data.provider_router import call_with_fallback, classify_requests_error
from agentic_options_reporter.models.schemas import (
    AnalystEstimates,
    CompanyProfile,
    FinancialRatios,
    FinancialStatementSummary,
)


class FinancialProviderError(RuntimeError):
    """Raised when a FinancialProvider cannot return the requested data."""


class FinancialProviderRateLimited(FinancialProviderError, ProviderRateLimited):
    """The provider rejected the request for exceeding its rate limit (HTTP 429)."""


class FinancialProviderTimeout(FinancialProviderError, ProviderTimeout):
    """The request to the provider timed out."""


class FinancialProviderUnavailable(FinancialProviderError, ProviderUnavailable):
    """The provider is unreachable or returned a server error (5xx / network failure)."""


class FinancialProviderUnsupported(FinancialProviderError, ProviderUnsupported):
    """This provider doesn't offer the requested data at all."""


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
    PROVIDER_LABEL = "Financial Modeling Prep"

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
            raise classify_requests_error(
                exc,
                self.PROVIDER_LABEL,
                base_error_cls=FinancialProviderError,
                rate_limited_cls=FinancialProviderRateLimited,
                timeout_cls=FinancialProviderTimeout,
                unavailable_cls=FinancialProviderUnavailable,
            ) from exc
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


class AlphaVantageFinancialProvider(FinancialProvider):
    """FinancialProvider implementation backed by Alpha Vantage's
    OVERVIEW / INCOME_STATEMENT / CASH_FLOW endpoints.

    Alpha Vantage's OVERVIEW payload doesn't include debt_to_equity or
    current_ratio, and has no dedicated analyst-estimates endpoint (only
    a single AnalystTargetPrice figure) — those fields are left at their
    schema defaults (None / "N/A" / 0) rather than fabricated, the same
    "missing data stays missing" convention FmpFinancialProvider already
    follows for absent fields. Alpha Vantage's free tier returns HTTP 200
    with an "Information"/"Note" field instead of a proper 429 when rate
    limited; `_get` treats that the same as a real 429.
    """

    BASE_URL = "https://www.alphavantage.co/query"
    PROVIDER_LABEL = "Alpha Vantage"

    def __init__(self, api_key: str | None = None, timeout_seconds: int = 15) -> None:
        self._api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY")
        if not self._api_key:
            raise FinancialProviderError(
                "No Alpha Vantage API key configured. Set ALPHA_VANTAGE_API_KEY, "
                "or supply one explicitly."
            )
        self._timeout = timeout_seconds

    def _get(self, function: str, ticker: str) -> Any:
        import requests

        params = {"function": function, "symbol": ticker.upper(), "apikey": self._api_key}
        try:
            response = requests.get(self.BASE_URL, params=params, timeout=self._timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise classify_requests_error(
                exc,
                self.PROVIDER_LABEL,
                base_error_cls=FinancialProviderError,
                rate_limited_cls=FinancialProviderRateLimited,
                timeout_cls=FinancialProviderTimeout,
                unavailable_cls=FinancialProviderUnavailable,
            ) from exc

        data = response.json()
        if "Information" in data or "Note" in data:
            raise FinancialProviderRateLimited(
                f"{self.PROVIDER_LABEL} rate limited or restricted: "
                f"{data.get('Information') or data.get('Note')}"
            )
        return data

    @staticmethod
    def _to_float(value: Any) -> float | None:
        if value in (None, "", "None", "-"):
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def get_company_profile(self, ticker: str) -> CompanyProfile:
        item = self._get("OVERVIEW", ticker)
        return CompanyProfile(
            ticker=ticker.upper(),
            name=item.get("Name", ""),
            sector=item.get("Sector", ""),
            industry=item.get("Industry", ""),
            market_cap=self._to_float(item.get("MarketCapitalization")),
            description=item.get("Description", ""),
        )

    def get_financial_statements(self, ticker: str) -> FinancialStatementSummary:
        income_reports = self._get("INCOME_STATEMENT", ticker).get("annualReports") or []
        cash_flow_reports = self._get("CASH_FLOW", ticker).get("annualReports") or []
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

    def get_ratios(self, ticker: str) -> FinancialRatios:
        item = self._get("OVERVIEW", ticker)
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
            debt_to_equity=None,  # not provided by Alpha Vantage's OVERVIEW endpoint
            current_ratio=None,  # not provided by Alpha Vantage's OVERVIEW endpoint
            return_on_equity=self._to_float(item.get("ReturnOnEquityTTM")),
            gross_margin=gross_margin,
            net_margin=self._to_float(item.get("ProfitMargin")),
        )

    def get_analyst_estimates(self, ticker: str) -> AnalystEstimates:
        item = self._get("OVERVIEW", ticker)
        return AnalystEstimates(
            ticker=ticker.upper(),
            consensus_rating="N/A",  # Alpha Vantage's OVERVIEW has no consensus-rating field
            price_target_mean=self._to_float(item.get("AnalystTargetPrice")),
            price_target_high=None,
            price_target_low=None,
            num_analysts=0,
        )


class FinancialProviderRouter(FinancialProvider):
    """Tries a priority-ordered list of already-constructed
    FinancialProvider clients per method call, advancing to the next on a
    retryable failure (see data.provider_router)."""

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

    def get_company_profile(self, ticker: str) -> CompanyProfile:
        return call_with_fallback(self._clients, "get_company_profile", FinancialProviderError, ticker)

    def get_financial_statements(self, ticker: str) -> FinancialStatementSummary:
        return call_with_fallback(
            self._clients, "get_financial_statements", FinancialProviderError, ticker
        )

    def get_ratios(self, ticker: str) -> FinancialRatios:
        return call_with_fallback(self._clients, "get_ratios", FinancialProviderError, ticker)

    def get_analyst_estimates(self, ticker: str) -> AnalystEstimates:
        return call_with_fallback(
            self._clients, "get_analyst_estimates", FinancialProviderError, ticker
        )


_PROVIDERS: dict[str, type[FinancialProvider]] = {
    "fmp": FmpFinancialProvider,
    "alphavantage": AlphaVantageFinancialProvider,
}

_DEFAULT_FALLBACK_ORDER = ["fmp", "alphavantage"]


def _fallback_order() -> list[str]:
    raw = os.environ.get("AOR_FINANCIAL_PROVIDER_FALLBACK_ORDER", ",".join(_DEFAULT_FALLBACK_ORDER))
    return [name.strip().lower() for name in raw.split(",") if name.strip()]


def build_financial_provider() -> FinancialProvider:
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
