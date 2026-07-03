"""Macroeconomic data access.

`MacroProvider` is the interface the macro_research agent depends on
(dependency injection — the same pattern as
`market_data.MarketDataProvider`). Three concrete implementations exist
(FRED, BLS, BEA — see specs/providers.yaml); `build_macro_provider()`
composes whichever are currently configured into a `MacroProviderRouter`
that fails over between them per method call, the data-provider analog
of `thesis.llm_client.LlmRouter`.

BLS and BEA are specialist agencies, not full FRED replacements: BLS is
the primary source for CPI (and doesn't publish interest rates or GDP);
BEA is the primary source for GDP (and doesn't publish CPI or interest
rates). Each raises `MacroProviderUnsupported` — retryable — for the
methods outside its domain, so the router still uses FRED for those
while gaining real redundancy on the CPI/GDP methods FRED shares with
BLS/BEA respectively.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any

from agentic_options_reporter.data.provider_errors import (
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
    ProviderUnsupported,
)
from agentic_options_reporter.data.provider_router import call_with_fallback, classify_requests_error
from agentic_options_reporter.models.schemas import CpiSnapshot, GdpSnapshot, InterestRates, MacroEvent


class MacroProviderError(RuntimeError):
    """Raised when a MacroProvider cannot return the requested data."""


class MacroProviderRateLimited(MacroProviderError, ProviderRateLimited):
    """The provider rejected the request for exceeding its rate limit (HTTP 429)."""


class MacroProviderTimeout(MacroProviderError, ProviderTimeout):
    """The request to the provider timed out."""


class MacroProviderUnavailable(MacroProviderError, ProviderUnavailable):
    """The provider is unreachable or returned a server error (5xx / network failure)."""


class MacroProviderUnsupported(MacroProviderError, ProviderUnsupported):
    """This provider doesn't publish the requested data at all (e.g. BLS has no GDP series)."""


class MacroProvider(ABC):
    """Interface implemented by all macroeconomic data providers."""

    @abstractmethod
    def get_interest_rates(self) -> InterestRates:
        raise NotImplementedError

    @abstractmethod
    def get_cpi(self) -> CpiSnapshot:
        raise NotImplementedError

    @abstractmethod
    def get_gdp(self) -> GdpSnapshot:
        raise NotImplementedError

    @abstractmethod
    def get_macro_calendar(self) -> list[MacroEvent]:
        raise NotImplementedError


class FredMacroProvider(MacroProvider):
    """MacroProvider implementation backed by the FRED API."""

    BASE_URL = "https://api.stlouisfed.org/fred"
    PROVIDER_LABEL = "FRED"

    def __init__(self, api_key: str | None = None, timeout_seconds: int = 15) -> None:
        self._api_key = api_key or os.environ.get("FRED_API_KEY")
        if not self._api_key:
            raise MacroProviderError(
                "No FRED API key configured. Set FRED_API_KEY, or supply one explicitly."
            )
        self._timeout = timeout_seconds

    def _fetch_series(self, series_id: str, limit: int = 1) -> list[dict[str, Any]]:
        import requests

        url = f"{self.BASE_URL}/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "sort_order": "desc",
            "limit": limit,
        }
        try:
            response = requests.get(url, params=params, timeout=self._timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise classify_requests_error(
                exc,
                self.PROVIDER_LABEL,
                base_error_cls=MacroProviderError,
                rate_limited_cls=MacroProviderRateLimited,
                timeout_cls=MacroProviderTimeout,
                unavailable_cls=MacroProviderUnavailable,
            ) from exc

        observations = response.json().get("observations", [])
        # FRED uses "." to mark a missing value; drop those.
        return [obs for obs in observations if obs.get("value") not in (None, ".")]

    def _latest_value(self, series_id: str) -> tuple[float, date] | None:
        observations = self._fetch_series(series_id, limit=1)
        if not observations:
            return None
        obs = observations[0]
        return float(obs["value"]), datetime.strptime(obs["date"], "%Y-%m-%d").date()

    def _yoy_change_pct(self, series_id: str, periods_per_year: int) -> tuple[float, float | None, date] | None:
        observations = self._fetch_series(series_id, limit=periods_per_year + 1)
        if not observations:
            return None
        latest = observations[0]
        latest_value = float(latest["value"])
        latest_date = datetime.strptime(latest["date"], "%Y-%m-%d").date()

        yoy_change_pct = None
        if len(observations) > periods_per_year:
            year_ago_value = float(observations[periods_per_year]["value"])
            if year_ago_value:
                yoy_change_pct = (latest_value - year_ago_value) / abs(year_ago_value) * 100

        return latest_value, yoy_change_pct, latest_date

    def get_interest_rates(self) -> InterestRates:
        fed_funds = self._latest_value("FEDFUNDS")
        ten_year = self._latest_value("DGS10")
        two_year = self._latest_value("DGS2")

        as_of = next(
            (result[1] for result in (fed_funds, ten_year, two_year) if result is not None),
            date.today(),
        )
        return InterestRates(
            fed_funds_rate=fed_funds[0] if fed_funds else None,
            ten_year_yield=ten_year[0] if ten_year else None,
            two_year_yield=two_year[0] if two_year else None,
            as_of=as_of,
        )

    def get_cpi(self) -> CpiSnapshot:
        result = self._yoy_change_pct("CPIAUCSL", periods_per_year=12)
        if result is None:
            raise MacroProviderError("FRED returned no CPI observations")
        value, yoy_change_pct, as_of = result
        return CpiSnapshot(value=value, yoy_change_pct=yoy_change_pct, as_of=as_of)

    def get_gdp(self) -> GdpSnapshot:
        result = self._yoy_change_pct("GDP", periods_per_year=4)
        if result is None:
            raise MacroProviderError("FRED returned no GDP observations")
        value, yoy_growth_pct, as_of = result
        return GdpSnapshot(value=value, yoy_growth_pct=yoy_growth_pct, as_of=as_of)

    def get_macro_calendar(self) -> list[MacroEvent]:
        # FRED is a historical-series API, not a forward economic calendar.
        # See specs/providers.yaml for the documented gap.
        return []


class BlsMacroProvider(MacroProvider):
    """MacroProvider implementation backed by the Bureau of Labor
    Statistics API v2 — the primary source for the CPI series FRED
    mirrors. BLS doesn't publish interest rates or GDP: get_interest_rates
    and get_gdp raise MacroProviderUnsupported (retryable) so a
    MacroProviderRouter falls through to FRED/BEA for those.
    """

    BASE_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data"
    PROVIDER_LABEL = "BLS"
    CPI_SERIES_ID = "CUUR0000SA0"  # CPI-U, US city average, all items, not seasonally adjusted

    def __init__(self, api_key: str | None = None, timeout_seconds: int = 15) -> None:
        self._api_key = api_key or os.environ.get("BLS_API_KEY")
        if not self._api_key:
            raise MacroProviderError(
                "No BLS API key configured. Set BLS_API_KEY, or supply one explicitly."
            )
        self._timeout = timeout_seconds

    def _fetch_series(self, series_id: str) -> list[dict[str, Any]]:
        import requests

        current_year = date.today().year
        url = f"{self.BASE_URL}/{series_id}"
        params = {
            "registrationkey": self._api_key,
            "startyear": str(current_year - 1),
            "endyear": str(current_year),
        }
        try:
            response = requests.get(url, params=params, timeout=self._timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise classify_requests_error(
                exc,
                self.PROVIDER_LABEL,
                base_error_cls=MacroProviderError,
                rate_limited_cls=MacroProviderRateLimited,
                timeout_cls=MacroProviderTimeout,
                unavailable_cls=MacroProviderUnavailable,
            ) from exc

        payload = response.json()
        if payload.get("status") not in (None, "REQUEST_SUCCEEDED"):
            raise MacroProviderError(f"BLS request failed: {payload.get('message')}")

        series_list = (payload.get("Results") or {}).get("series") or []
        data = series_list[0].get("data") if series_list else []
        # BLS labels months "M01".."M12"; sort descending by year+period so
        # index 0 is always the most recent observation.
        return sorted(data or [], key=lambda obs: (obs.get("year", ""), obs.get("period", "")), reverse=True)

    def get_interest_rates(self) -> InterestRates:
        raise MacroProviderUnsupported("BLS does not publish interest rate data.")

    def get_cpi(self) -> CpiSnapshot:
        observations = self._fetch_series(self.CPI_SERIES_ID)
        if not observations:
            raise MacroProviderError("BLS returned no CPI observations")

        latest = observations[0]
        latest_value = float(latest["value"])
        as_of = date(int(latest["year"]), int(latest["period"].lstrip("M")), 1)

        yoy_change_pct = None
        if len(observations) > 12:
            year_ago_value = float(observations[12]["value"])
            if year_ago_value:
                yoy_change_pct = (latest_value - year_ago_value) / abs(year_ago_value) * 100

        return CpiSnapshot(value=latest_value, yoy_change_pct=yoy_change_pct, as_of=as_of)

    def get_gdp(self) -> GdpSnapshot:
        raise MacroProviderUnsupported("BLS does not publish GDP data.")

    def get_macro_calendar(self) -> list[MacroEvent]:
        return []


class BeaMacroProvider(MacroProvider):
    """MacroProvider implementation backed by the Bureau of Economic
    Analysis API — the primary source for the (nominal) GDP series FRED
    mirrors. BEA doesn't publish CPI or interest rates: get_cpi and
    get_interest_rates raise MacroProviderUnsupported (retryable) so a
    MacroProviderRouter falls through to FRED/BLS for those.
    """

    BASE_URL = "https://apps.bea.gov/api/data"
    PROVIDER_LABEL = "BEA"

    def __init__(self, api_key: str | None = None, timeout_seconds: int = 15) -> None:
        self._api_key = api_key or os.environ.get("BEA_API_KEY")
        if not self._api_key:
            raise MacroProviderError(
                "No BEA API key configured. Set BEA_API_KEY, or supply one explicitly."
            )
        self._timeout = timeout_seconds

    def get_interest_rates(self) -> InterestRates:
        raise MacroProviderUnsupported("BEA does not publish interest rate data.")

    def get_cpi(self) -> CpiSnapshot:
        raise MacroProviderUnsupported("BEA does not publish CPI data.")

    def get_gdp(self) -> GdpSnapshot:
        current_year = date.today().year
        params = {
            "UserID": self._api_key,
            "method": "GetData",
            "datasetname": "NIPA",
            "TableName": "T10101",  # line 1: Gross domestic product (nominal, quarterly)
            "Frequency": "Q",
            "Year": f"{current_year - 2},{current_year - 1},{current_year}",
            "ResultFormat": "JSON",
        }

        import requests

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=self._timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise classify_requests_error(
                exc,
                self.PROVIDER_LABEL,
                base_error_cls=MacroProviderError,
                rate_limited_cls=MacroProviderRateLimited,
                timeout_cls=MacroProviderTimeout,
                unavailable_cls=MacroProviderUnavailable,
            ) from exc

        payload = response.json()
        results = ((payload.get("BEAAPI") or {}).get("Results")) or {}
        rows = [row for row in (results.get("Data") or []) if row.get("LineNumber") == "1"]
        # BEA's TimePeriod ("2026Q2") sorts correctly as a plain string.
        rows.sort(key=lambda row: row.get("TimePeriod", ""), reverse=True)
        if not rows:
            raise MacroProviderError("BEA returned no GDP observations")

        latest = rows[0]
        latest_value = self._parse_value(latest["DataValue"])
        as_of = self._period_end_date(latest["TimePeriod"])

        yoy_growth_pct = None
        if len(rows) > 4:
            year_ago_value = self._parse_value(rows[4]["DataValue"])
            if year_ago_value:
                yoy_growth_pct = (latest_value - year_ago_value) / abs(year_ago_value) * 100

        return GdpSnapshot(value=latest_value, yoy_growth_pct=yoy_growth_pct, as_of=as_of)

    def get_macro_calendar(self) -> list[MacroEvent]:
        return []

    @staticmethod
    def _parse_value(value: str) -> float:
        return float(str(value).replace(",", ""))

    @staticmethod
    def _period_end_date(time_period: str) -> date:
        year_str, quarter_str = time_period.split("Q")
        year = int(year_str)
        quarter_end_month = int(quarter_str) * 3
        return date(year, quarter_end_month, 1)


class MacroProviderRouter(MacroProvider):
    """Tries a priority-ordered list of already-constructed MacroProvider
    clients per method call, advancing to the next on a retryable failure
    (see data.provider_router)."""

    def __init__(self, clients: list[tuple[str, MacroProvider]]) -> None:
        if not clients:
            raise MacroProviderError(
                "No macro providers are configured for automatic failover. Set at least "
                f"one provider's API key (supported: {', '.join(sorted(_PROVIDERS))})."
            )
        self._clients = clients

    @property
    def provider_names(self) -> list[str]:
        return [name for name, _ in self._clients]

    def get_interest_rates(self) -> InterestRates:
        return call_with_fallback(self._clients, "get_interest_rates", MacroProviderError)

    def get_cpi(self) -> CpiSnapshot:
        return call_with_fallback(self._clients, "get_cpi", MacroProviderError)

    def get_gdp(self) -> GdpSnapshot:
        return call_with_fallback(self._clients, "get_gdp", MacroProviderError)

    def get_macro_calendar(self) -> list[MacroEvent]:
        return call_with_fallback(self._clients, "get_macro_calendar", MacroProviderError)


_PROVIDERS: dict[str, type[MacroProvider]] = {
    "fred": FredMacroProvider,
    "bls": BlsMacroProvider,
    "bea": BeaMacroProvider,
}

_DEFAULT_FALLBACK_ORDER = ["fred", "bls", "bea"]


def _fallback_order() -> list[str]:
    raw = os.environ.get("AOR_MACRO_PROVIDER_FALLBACK_ORDER", ",".join(_DEFAULT_FALLBACK_ORDER))
    return [name.strip().lower() for name in raw.split(",") if name.strip()]


def build_macro_provider() -> MacroProvider:
    """Build a MacroProviderRouter from AOR_MACRO_PROVIDER_FALLBACK_ORDER,
    skipping any provider without a configured API key. Raises
    MacroProviderError if the resulting router would have zero clients."""
    clients: list[tuple[str, MacroProvider]] = []
    for name in _fallback_order():
        provider_cls = _PROVIDERS.get(name)
        if provider_cls is None:
            continue
        try:
            clients.append((name, provider_cls()))
        except MacroProviderError:
            continue  # not configured (missing API key) — skip, don't fail the request
    return MacroProviderRouter(clients)
