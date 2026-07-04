"""FRED adapter (fred.stlouisfed.org — free key, generous limits).

The broadest single macro source: interest rates (FEDFUNDS/DGS10/DGS2),
CPI (CPIAUCSL, monthly), and GDP (GDP, quarterly). FRED is a
historical-series API, not a forward economic calendar, so
`get_macro_calendar` returns an empty list rather than fabricating one
(a documented gap, see specs/providers.yaml).
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime
from typing import Any

from agentic_options_reporter.data.macro.base import (
    MacroProviderError,
    _HttpMacroProvider,
    yoy_change_pct,
)
from agentic_options_reporter.models.schemas import (
    CpiSnapshot,
    GdpSnapshot,
    InterestRates,
    MacroEvent,
)


class FredMacroProvider(_HttpMacroProvider):
    BASE_URL = "https://api.stlouisfed.org/fred"
    PROVIDER_LABEL = "FRED"
    API_KEY_ENV_VAR = "FRED_API_KEY"

    async def _fetch_series(self, series_id: str, limit: int = 1) -> list[dict[str, Any]]:
        payload = await self._get_json(
            f"{self.BASE_URL}/series/observations",
            {
                "series_id": series_id,
                "api_key": self._api_key,
                "file_type": "json",
                "sort_order": "desc",
                "limit": limit,
            },
        )
        observations = payload.get("observations", [])
        # FRED uses "." to mark a missing value; drop those.
        return [obs for obs in observations if obs.get("value") not in (None, ".")]

    async def _latest_value(self, series_id: str) -> tuple[float, date] | None:
        observations = await self._fetch_series(series_id, limit=1)
        if not observations:
            return None
        obs = observations[0]
        return float(obs["value"]), datetime.strptime(obs["date"], "%Y-%m-%d").date()

    async def _latest_with_yoy(
        self, series_id: str, periods_per_year: int
    ) -> tuple[float, float | None, date] | None:
        observations = await self._fetch_series(series_id, limit=periods_per_year + 1)
        if not observations:
            return None
        latest = observations[0]
        latest_value = float(latest["value"])
        latest_date = datetime.strptime(latest["date"], "%Y-%m-%d").date()
        year_ago = (
            float(observations[periods_per_year]["value"])
            if len(observations) > periods_per_year
            else None
        )
        return latest_value, yoy_change_pct(latest_value, year_ago), latest_date

    async def get_interest_rates(self) -> InterestRates:
        fed_funds, ten_year, two_year = await asyncio.gather(
            self._latest_value("FEDFUNDS"),
            self._latest_value("DGS10"),
            self._latest_value("DGS2"),
        )
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

    async def get_cpi(self) -> CpiSnapshot:
        result = await self._latest_with_yoy("CPIAUCSL", periods_per_year=12)
        if result is None:
            raise MacroProviderError("FRED returned no CPI observations")
        value, change, as_of = result
        return CpiSnapshot(value=value, yoy_change_pct=change, as_of=as_of)

    async def get_gdp(self) -> GdpSnapshot:
        result = await self._latest_with_yoy("GDP", periods_per_year=4)
        if result is None:
            raise MacroProviderError("FRED returned no GDP observations")
        value, growth, as_of = result
        return GdpSnapshot(value=value, yoy_growth_pct=growth, as_of=as_of)

    async def get_macro_calendar(self) -> list[MacroEvent]:
        return []

    async def _health_probe(self) -> None:
        await self.get_interest_rates()
