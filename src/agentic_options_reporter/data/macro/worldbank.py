"""World Bank adapter (api.worldbank.org — free, keyless).

Covers US CPI (indicator ``FP.CPI.TOTL``, an annual index, 2010=100) and
nominal GDP (``NY.GDP.MKTP.CD``, annual, current US$). World Bank data
is ANNUAL and lags the primary US agencies by a year or more — a
last-resort fallback for when every fresher source is down, which is why
it sits at the end of the default fallback order. Interest rates aren't
covered (the Bank's real-interest-rate indicator is a different metric
from US policy rates), so that method raises MacroProviderUnsupported.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from agentic_options_reporter.data.macro.base import (
    MacroProviderError,
    MacroProviderUnsupported,
    _HttpMacroProvider,
    yoy_change_pct,
)
from agentic_options_reporter.models.schemas import (
    CpiSnapshot,
    GdpSnapshot,
    InterestRates,
    MacroEvent,
)


class WorldBankMacroProvider(_HttpMacroProvider):
    BASE_URL = "https://api.worldbank.org/v2/country/USA/indicator"
    PROVIDER_LABEL = "World Bank"
    API_KEY_ENV_VAR = None  # keyless

    CPI_INDICATOR = "FP.CPI.TOTL"
    GDP_INDICATOR = "NY.GDP.MKTP.CD"

    async def _fetch_indicator(self, indicator: str) -> list[dict[str, Any]]:
        payload = await self._get_json(
            f"{self.BASE_URL}/{indicator}", {"format": "json", "per_page": 5}
        )
        # Response shape: [pagination_metadata, rows]; rows are
        # most-recent-first and may carry null values for years not yet
        # published.
        rows = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
        return [row for row in (rows or []) if row.get("value") is not None]

    async def _latest_with_yoy(self, indicator: str) -> tuple[float, float | None, date]:
        rows = await self._fetch_indicator(indicator)
        if not rows:
            raise MacroProviderError(f"World Bank returned no observations for {indicator}")
        latest = rows[0]
        latest_value = float(latest["value"])
        year_ago = float(rows[1]["value"]) if len(rows) > 1 else None
        as_of = date(int(latest["date"]), 12, 31)
        return latest_value, yoy_change_pct(latest_value, year_ago), as_of

    async def get_interest_rates(self) -> InterestRates:
        raise MacroProviderUnsupported(
            "World Bank does not publish US policy rates; use FRED."
        )

    async def get_cpi(self) -> CpiSnapshot:
        value, change, as_of = await self._latest_with_yoy(self.CPI_INDICATOR)
        return CpiSnapshot(value=value, yoy_change_pct=change, as_of=as_of)

    async def get_gdp(self) -> GdpSnapshot:
        value, growth, as_of = await self._latest_with_yoy(self.GDP_INDICATOR)
        return GdpSnapshot(value=value, yoy_growth_pct=growth, as_of=as_of)

    async def get_macro_calendar(self) -> list[MacroEvent]:
        return []

    async def _health_probe(self) -> None:
        await self.get_gdp()
