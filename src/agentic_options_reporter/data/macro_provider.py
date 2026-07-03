"""Macroeconomic data access.

`MacroProvider` is the interface the macro_research agent depends on
(dependency injection — the same pattern as
`market_data.MarketDataProvider`). `FredMacroProvider` is the phase-2a
implementation (see specs/providers.yaml), backed by the Federal
Reserve's FRED API — a historical-series API, not a forward economic
calendar, so `get_macro_calendar` returns an empty list rather than
fabricating one (see specs/providers.yaml for the documented gap).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import date, datetime
from typing import Any

from agentic_options_reporter.models.schemas import CpiSnapshot, GdpSnapshot, InterestRates, MacroEvent


class MacroProviderError(RuntimeError):
    """Raised when a MacroProvider cannot return the requested data."""


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
            raise MacroProviderError(f"FRED request for series {series_id!r} failed: {exc}") from exc

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
