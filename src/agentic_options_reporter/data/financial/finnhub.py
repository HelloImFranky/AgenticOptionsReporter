"""Finnhub fundamentals adapter (finnhub.io — free tier ~60 requests/min).

Partial coverage, per Finnhub's free tier:

- get_company_profile → /stock/profile2
- get_ratios → /stock/metric (TTM ratio metrics)
- get_analyst_estimates → /stock/recommendation — the consensus rating
  is DERIVED as the largest bucket of the latest month's analyst counts
  (strongBuy/buy/hold/sell/strongSell), which is a summary of
  provider-supplied tallies, not a fabricated opinion; price targets
  are a premium endpoint and stay null.
- get_financial_statements → NOT advertised (statements not in DATASETS),
  so the capability-filtering router never asks Finnhub for statements
  and routes them to FMP/Alpha Vantage instead. The method remains as a
  defensive guard that raises FinancialProviderUnsupported.
"""

from __future__ import annotations

from typing import Any

from datetime import date, datetime

from agentic_options_reporter.data.financial.base import (
    ANALYST_ESTIMATES,
    EARNINGS,
    EARNINGS_CALENDAR,
    INSIDER,
    METRICS,
    PROFILE,
    RATIOS,
    FinancialProviderUnsupported,
    _HttpFinancialProvider,
)
from agentic_options_reporter.models.schemas import (
    AnalystEstimates,
    CompanyMetrics,
    CompanyProfile,
    EarningsCalendar,
    EarningsHistory,
    EarningsSurprise,
    FinancialRatios,
    FinancialStatementSummary,
    InsiderActivity,
    InsiderTransaction,
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

    # Free tier: no raw statements, but rich metrics/earnings/insider data.
    DATASETS = frozenset(
        {PROFILE, RATIOS, ANALYST_ESTIMATES, METRICS, EARNINGS, EARNINGS_CALENDAR, INSIDER}
    )

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

    async def get_company_metrics(self, ticker: str) -> CompanyMetrics:
        data = await self._get("/stock/metric", {"symbol": ticker.upper(), "metric": "all"})
        metric = data.get("metric") or {}
        # Finnhub reports margins and dividend yield as percentages.
        gross = self._to_float(metric.get("grossMarginTTM"))
        operating = self._to_float(metric.get("operatingMarginTTM"))
        net = self._to_float(metric.get("netProfitMarginTTM"))
        div_yield = self._to_float(metric.get("dividendYieldIndicatedAnnual"))
        market_cap = self._to_float(metric.get("marketCapitalization"))
        return CompanyMetrics(
            ticker=ticker.upper(),
            market_cap=market_cap * 1_000_000 if market_cap is not None else None,
            pe_ratio=self._to_float(metric.get("peTTM")),
            forward_pe=self._to_float(metric.get("forwardPE")),
            peg_ratio=self._to_float(metric.get("pegTTM")),
            price_to_book=self._to_float(metric.get("pb")),
            price_to_sales=self._to_float(metric.get("psTTM")),
            beta=self._to_float(metric.get("beta")),
            dividend_yield=div_yield / 100 if div_yield is not None else None,
            week52_high=self._to_float(metric.get("52WeekHigh")),
            week52_low=self._to_float(metric.get("52WeekLow")),
            gross_margin=gross / 100 if gross is not None else None,
            operating_margin=operating / 100 if operating is not None else None,
            profit_margin=net / 100 if net is not None else None,
            revenue_growth=self._to_float(metric.get("revenueGrowthTTMYoy")),
            earnings_growth=self._to_float(metric.get("epsGrowthTTMYoy")),
        )

    async def get_earnings_history(self, ticker: str) -> EarningsHistory:
        data = await self._get("/stock/earnings", {"symbol": ticker.upper()})
        rows = data if isinstance(data, list) else []
        surprises: list[EarningsSurprise] = []
        for row in rows:
            actual = self._to_float(row.get("actual"))
            estimate = self._to_float(row.get("estimate"))
            surprise = self._to_float(row.get("surprise"))
            pct = self._to_float(row.get("surprisePercent"))
            period = row.get("period") or (
                f"Q{row.get('quarter')} {row.get('year')}" if row.get("year") else ""
            )
            surprises.append(
                EarningsSurprise(
                    period=str(period),
                    actual_eps=actual,
                    estimate_eps=estimate,
                    surprise=surprise if surprise is not None
                    else (actual - estimate if actual is not None and estimate is not None else None),
                    surprise_percent=pct / 100 if pct is not None else None,
                )
            )
        return EarningsHistory(ticker=ticker.upper(), surprises=surprises)

    async def get_earnings_calendar(self, ticker: str) -> EarningsCalendar:
        data = await self._get("/calendar/earnings", {"symbol": ticker.upper()})
        entries = (data or {}).get("earningsCalendar") or []
        # Prefer the nearest upcoming report; Finnhub returns most-recent
        # first, so scan for the earliest date on/after today, else the first.
        today = date.today()
        chosen = None
        for entry in entries:
            entry_date = self._parse_date(entry.get("date"))
            if entry_date is not None and entry_date >= today:
                if chosen is None or entry_date < chosen[0]:
                    chosen = (entry_date, entry)
        if chosen is None and entries:
            first = entries[0]
            chosen = (self._parse_date(first.get("date")), first)
        if chosen is None:
            return EarningsCalendar(ticker=ticker.upper())
        entry_date, entry = chosen
        return EarningsCalendar(
            ticker=ticker.upper(),
            next_date=entry_date,
            eps_estimate=self._to_float(entry.get("epsEstimate")),
            revenue_estimate=self._to_float(entry.get("revenueEstimate")),
        )

    async def get_insider_activity(self, ticker: str) -> InsiderActivity:
        data = await self._get("/stock/insider-transactions", {"symbol": ticker.upper()})
        rows = (data or {}).get("data") or []
        transactions: list[InsiderTransaction] = []
        for row in rows:
            shares = self._to_float(row.get("share") or row.get("change"))
            price = self._to_float(row.get("transactionPrice"))
            value = shares * price if shares is not None and price else None
            code = str(row.get("transactionCode") or "")
            transactions.append(
                InsiderTransaction(
                    name=row.get("name") or "",
                    relationship="",  # not provided by this endpoint
                    # Finnhub codes: S = sale, P = purchase; others passed through.
                    transaction_type=("sell" if code == "S" else "buy" if code == "P" else code),
                    shares=shares,
                    value=value,
                    filed_at=self._parse_date(row.get("filingDate") or row.get("transactionDate")),
                )
            )
        net_shares = _net_shares(transactions)
        return InsiderActivity(
            ticker=ticker.upper(), transactions=transactions, net_shares=net_shares
        )

    @staticmethod
    def _parse_date(value: object) -> date | None:
        if not value:
            return None
        try:
            return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
        except ValueError:
            return None


def _net_shares(transactions: list[InsiderTransaction]) -> float | None:
    """Net insider share flow (buys positive, sells negative); None when no
    transaction carries a share count."""
    total = 0.0
    seen = False
    for tx in transactions:
        if tx.shares is None:
            continue
        seen = True
        total += tx.shares if tx.transaction_type != "sell" else -tx.shares
    return total if seen else None
