"""Yahoo Finance fundamentals adapter, backed by the synchronous
`yfinance` package.

Like the market-data Yahoo adapter, `yfinance` is a synchronous,
pandas-returning library with no async API, so it implements the async
`FinancialProvider` interface directly and offloads each blocking call to
a worker thread via `asyncio.to_thread`. Keyless, and the only source here
that serves all eight datasets — profile, statements, ratios,
analyst_estimates, metrics, earnings history, earnings calendar, and
insider transactions — most of them off the single `.info` dict, which is
cached per ticker so profile/metrics/ratios/estimates share one fetch.
"""

from __future__ import annotations

import asyncio
import math
import time
from datetime import date, datetime, timezone
from typing import Any

from agentic_options_reporter.data.async_http import ProviderHealth
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
    FinancialProviderUnavailable,
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

_HEALTH_PROBE_TICKER = "AAPL"


def _to_float(value: Any) -> float | None:
    """Coerce a yfinance value to float, treating None and NaN as missing."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(result) else result


class _TTLCache:
    def __init__(self, ttl_seconds: int = 300) -> None:
        self._ttl = ttl_seconds
        self._store: dict[Any, tuple[float, Any]] = {}

    def get(self, key: Any) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: Any, value: Any) -> None:
        self._store[key] = (time.monotonic() + self._ttl, value)


class YFinanceFinancialProvider(FinancialProvider):
    """Yahoo Finance fundamentals via the `yfinance` package. Keyless."""

    PROVIDER_LABEL = "Yahoo Finance"

    DATASETS = frozenset(
        {PROFILE, STATEMENTS, RATIOS, ANALYST_ESTIMATES, METRICS, EARNINGS, EARNINGS_CALENDAR, INSIDER}
    )

    def __init__(self, cache_ttl_seconds: int = 300) -> None:
        self._cache = _TTLCache(cache_ttl_seconds)

    @property
    def supported_datasets(self) -> frozenset[str]:
        return self.DATASETS

    # -- shared .info fetch (profile/metrics/ratios/estimates read from it) --

    async def _info(self, ticker: str) -> dict[str, Any]:
        cache_key = ("info", ticker.upper())
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        info = await asyncio.to_thread(self._info_sync, ticker)
        self._cache.set(cache_key, info)
        return info

    def _info_sync(self, ticker: str) -> dict[str, Any]:
        import yfinance as yf

        try:
            info = yf.Ticker(ticker).info
        except FinancialProviderError:
            raise
        except Exception as exc:  # noqa: BLE001 — normalize yfinance/network errors for the router
            raise FinancialProviderUnavailable(
                f"Yahoo Finance request failed for {ticker!r}: {exc}"
            ) from exc
        if not info:
            raise FinancialProviderError(f"No profile data returned for {ticker!r}")
        return info

    # -- profile / metrics / ratios / estimates (from .info) --

    async def get_company_profile(self, ticker: str) -> CompanyProfile:
        info = await self._info(ticker)
        return CompanyProfile(
            ticker=ticker.upper(),
            name=info.get("longName") or info.get("shortName") or "",
            sector=info.get("sector") or "",
            industry=info.get("industry") or "",
            market_cap=_to_float(info.get("marketCap")),
            description=info.get("longBusinessSummary") or "",
        )

    async def get_company_metrics(self, ticker: str) -> CompanyMetrics:
        info = await self._info(ticker)
        return CompanyMetrics(
            ticker=ticker.upper(),
            market_cap=_to_float(info.get("marketCap")),
            pe_ratio=_to_float(info.get("trailingPE")),
            forward_pe=_to_float(info.get("forwardPE")),
            peg_ratio=_to_float(info.get("pegRatio") or info.get("trailingPegRatio")),
            price_to_book=_to_float(info.get("priceToBook")),
            price_to_sales=_to_float(info.get("priceToSalesTrailing12Months")),
            beta=_to_float(info.get("beta")),
            dividend_yield=_to_float(info.get("dividendYield")),
            week52_high=_to_float(info.get("fiftyTwoWeekHigh")),
            week52_low=_to_float(info.get("fiftyTwoWeekLow")),
            gross_margin=_to_float(info.get("grossMargins")),
            operating_margin=_to_float(info.get("operatingMargins")),
            profit_margin=_to_float(info.get("profitMargins")),
            revenue_growth=_to_float(info.get("revenueGrowth")),
            earnings_growth=_to_float(info.get("earningsGrowth")),
        )

    async def get_ratios(self, ticker: str) -> FinancialRatios:
        info = await self._info(ticker)
        return FinancialRatios(
            ticker=ticker.upper(),
            pe_ratio=_to_float(info.get("trailingPE")),
            pb_ratio=_to_float(info.get("priceToBook")),
            debt_to_equity=_to_float(info.get("debtToEquity")),
            current_ratio=_to_float(info.get("currentRatio")),
            return_on_equity=_to_float(info.get("returnOnEquity")),
            gross_margin=_to_float(info.get("grossMargins")),
            net_margin=_to_float(info.get("profitMargins")),
        )

    async def get_analyst_estimates(self, ticker: str) -> AnalystEstimates:
        info = await self._info(ticker)
        rating = info.get("recommendationKey") or "N/A"
        return AnalystEstimates(
            ticker=ticker.upper(),
            consensus_rating=str(rating).replace("_", " ").title() if rating != "N/A" else "N/A",
            price_target_mean=_to_float(info.get("targetMeanPrice")),
            price_target_high=_to_float(info.get("targetHighPrice")),
            price_target_low=_to_float(info.get("targetLowPrice")),
            num_analysts=int(info.get("numberOfAnalystOpinions") or 0),
        )

    # -- statements (from .income_stmt / .cashflow DataFrames) --

    async def get_financial_statements(self, ticker: str) -> FinancialStatementSummary:
        return await asyncio.to_thread(self._statements_sync, ticker)

    def _statements_sync(self, ticker: str) -> FinancialStatementSummary:
        import yfinance as yf

        try:
            t = yf.Ticker(ticker)
            income = t.income_stmt
            cashflow = t.cashflow
        except FinancialProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise FinancialProviderUnavailable(
                f"Yahoo Finance request failed for {ticker!r}: {exc}"
            ) from exc

        def _line(frame: Any, label: str) -> float | None:
            try:
                if frame is None or getattr(frame, "empty", True) or label not in frame.index:
                    return None
                return _to_float(frame.loc[label].iloc[0])
            except (KeyError, IndexError, AttributeError):
                return None

        period = "latest"
        try:
            if income is not None and not income.empty:
                period = str(income.columns[0].date().year)
        except (AttributeError, IndexError):
            pass

        return FinancialStatementSummary(
            ticker=ticker.upper(),
            period=period,
            revenue=_line(income, "Total Revenue"),
            net_income=_line(income, "Net Income"),
            operating_cash_flow=_line(cashflow, "Operating Cash Flow"),
            free_cash_flow=_line(cashflow, "Free Cash Flow"),
        )

    # -- earnings history + calendar --

    async def get_earnings_history(self, ticker: str) -> EarningsHistory:
        return await asyncio.to_thread(self._earnings_history_sync, ticker)

    def _earnings_history_sync(self, ticker: str) -> EarningsHistory:
        import yfinance as yf

        try:
            frame = yf.Ticker(ticker).earnings_dates
        except FinancialProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise FinancialProviderUnavailable(
                f"Yahoo Finance request failed for {ticker!r}: {exc}"
            ) from exc

        surprises: list[EarningsSurprise] = []
        if frame is not None and not getattr(frame, "empty", True):
            for idx, row in frame.iterrows():
                actual = _to_float(row.get("Reported EPS"))
                estimate = _to_float(row.get("EPS Estimate"))
                if actual is None and estimate is None:
                    continue  # a future/undisclosed date — skip
                surprise = actual - estimate if actual is not None and estimate is not None else None
                pct = _to_float(row.get("Surprise(%)"))
                surprises.append(
                    EarningsSurprise(
                        period=str(getattr(idx, "date", lambda: idx)()),
                        actual_eps=actual,
                        estimate_eps=estimate,
                        surprise=surprise,
                        surprise_percent=pct / 100 if pct is not None else None,
                    )
                )
        return EarningsHistory(ticker=ticker.upper(), surprises=surprises)

    async def get_earnings_calendar(self, ticker: str) -> EarningsCalendar:
        return await asyncio.to_thread(self._earnings_calendar_sync, ticker)

    def _earnings_calendar_sync(self, ticker: str) -> EarningsCalendar:
        import yfinance as yf

        try:
            calendar = yf.Ticker(ticker).calendar
        except FinancialProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise FinancialProviderUnavailable(
                f"Yahoo Finance request failed for {ticker!r}: {exc}"
            ) from exc

        next_date: date | None = None
        eps_estimate: float | None = None
        revenue_estimate: float | None = None
        if isinstance(calendar, dict):
            dates = calendar.get("Earnings Date")
            if isinstance(dates, list) and dates:
                first = dates[0]
                next_date = first if isinstance(first, date) else _coerce_date(first)
            elif isinstance(dates, date):
                next_date = dates
            eps_estimate = _to_float(calendar.get("Earnings Average"))
            revenue_estimate = _to_float(calendar.get("Revenue Average"))
        return EarningsCalendar(
            ticker=ticker.upper(),
            next_date=next_date,
            eps_estimate=eps_estimate,
            revenue_estimate=revenue_estimate,
        )

    # -- insider transactions --

    async def get_insider_activity(self, ticker: str) -> InsiderActivity:
        return await asyncio.to_thread(self._insider_sync, ticker)

    def _insider_sync(self, ticker: str) -> InsiderActivity:
        import yfinance as yf

        try:
            frame = yf.Ticker(ticker).insider_transactions
        except FinancialProviderError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise FinancialProviderUnavailable(
                f"Yahoo Finance request failed for {ticker!r}: {exc}"
            ) from exc

        transactions: list[InsiderTransaction] = []
        if frame is not None and not getattr(frame, "empty", True):
            for _, row in frame.iterrows():
                shares = _to_float(row.get("Shares"))
                text = str(row.get("Text") or "")
                transactions.append(
                    InsiderTransaction(
                        name=str(row.get("Insider") or ""),
                        relationship=str(row.get("Position") or ""),
                        transaction_type=("sell" if "Sale" in text else "buy" if "Purchase" in text else text),
                        shares=shares,
                        value=_to_float(row.get("Value")),
                        filed_at=_coerce_date(row.get("Start Date")),
                    )
                )
        net_shares = _signed_net_shares(transactions)
        return InsiderActivity(
            ticker=ticker.upper(), transactions=transactions, net_shares=net_shares
        )

    async def health(self) -> ProviderHealth:
        started = time.monotonic()
        try:
            await self.get_company_profile(_HEALTH_PROBE_TICKER)
        except Exception as exc:  # noqa: BLE001 — health() reports, never raises
            return ProviderHealth(
                provider=self.PROVIDER_LABEL,
                healthy=False,
                latency_ms=(time.monotonic() - started) * 1000,
                detail=str(exc),
                checked_at=datetime.now(timezone.utc),
            )
        return ProviderHealth(
            provider=self.PROVIDER_LABEL,
            healthy=True,
            latency_ms=(time.monotonic() - started) * 1000,
            checked_at=datetime.now(timezone.utc),
        )


def _coerce_date(value: Any) -> date | None:
    if value is None:
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    to_pydatetime = getattr(value, "to_pydatetime", None)
    if callable(to_pydatetime):
        try:
            return to_pydatetime().date()
        except (ValueError, AttributeError):
            return None
    try:
        return datetime.strptime(str(value)[:10], "%Y-%m-%d").date()
    except ValueError:
        return None


def _signed_net_shares(transactions: list[InsiderTransaction]) -> float | None:
    """Net insider share flow: buys positive, sells negative. None when no
    transaction carries a share count."""
    total = 0.0
    seen = False
    for tx in transactions:
        if tx.shares is None:
            continue
        seen = True
        total += tx.shares if tx.transaction_type != "sell" else -tx.shares
    return total if seen else None
