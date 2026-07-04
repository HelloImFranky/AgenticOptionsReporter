"""Bureau of Labor Statistics adapter (bls.gov — free key).

The primary source for the CPI series FRED mirrors — real redundancy on
get_cpi, not just an alternate. BLS is a labor-statistics agency: it
publishes neither interest rates nor GDP, so those methods raise
MacroProviderUnsupported (retryable) and the router falls through.
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


class BlsMacroProvider(_HttpMacroProvider):
    BASE_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data"
    PROVIDER_LABEL = "BLS"
    API_KEY_ENV_VAR = "BLS_API_KEY"
    CPI_SERIES_ID = "CUUR0000SA0"  # CPI-U, US city average, all items, not seasonally adjusted

    def _check_payload(self, payload: Any) -> None:
        if isinstance(payload, dict) and payload.get("status") not in (None, "REQUEST_SUCCEEDED"):
            raise MacroProviderError(f"BLS request failed: {payload.get('message')}")

    async def _fetch_series(self, series_id: str) -> list[dict[str, Any]]:
        current_year = date.today().year
        payload = await self._get_json(
            f"{self.BASE_URL}/{series_id}",
            {
                "registrationkey": self._api_key,
                "startyear": str(current_year - 1),
                "endyear": str(current_year),
            },
        )
        series_list = (payload.get("Results") or {}).get("series") or []
        data = series_list[0].get("data") if series_list else []
        # BLS labels months "M01".."M12"; sort descending by year+period so
        # index 0 is always the most recent observation.
        return sorted(
            data or [], key=lambda obs: (obs.get("year", ""), obs.get("period", "")), reverse=True
        )

    async def get_interest_rates(self) -> InterestRates:
        raise MacroProviderUnsupported("BLS does not publish interest rate data.")

    async def get_cpi(self) -> CpiSnapshot:
        observations = await self._fetch_series(self.CPI_SERIES_ID)
        if not observations:
            raise MacroProviderError("BLS returned no CPI observations")

        latest = observations[0]
        latest_value = float(latest["value"])
        as_of = date(int(latest["year"]), int(latest["period"].lstrip("M")), 1)
        year_ago = float(observations[12]["value"]) if len(observations) > 12 else None
        return CpiSnapshot(
            value=latest_value, yoy_change_pct=yoy_change_pct(latest_value, year_ago), as_of=as_of
        )

    async def get_gdp(self) -> GdpSnapshot:
        raise MacroProviderUnsupported("BLS does not publish GDP data.")

    async def get_macro_calendar(self) -> list[MacroEvent]:
        return []

    async def _health_probe(self) -> None:
        await self.get_cpi()
