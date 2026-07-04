"""International Monetary Fund adapter (dataservices.imf.org — free,
keyless SDMX-JSON service).

Covers US CPI (IFS series ``M.US.PCPI_IX``, a monthly index) and nominal
GDP (``Q.US.NGDP_SA_XDC``, quarterly, seasonally adjusted). US policy
rates aren't cleanly exposed through IFS, so get_interest_rates raises
MacroProviderUnsupported (retryable) and the router falls through to
FRED. Keyless, so it's always available — but IMF data lags the primary
US agencies, which is why it sits behind FRED/BLS/BEA in the default
fallback order.
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


class ImfMacroProvider(_HttpMacroProvider):
    BASE_URL = "https://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/IFS"
    PROVIDER_LABEL = "IMF"
    API_KEY_ENV_VAR = None  # keyless

    CPI_SERIES_KEY = "M.US.PCPI_IX"
    GDP_SERIES_KEY = "Q.US.NGDP_SA_XDC"

    async def _fetch_observations(self, series_key: str) -> list[dict[str, Any]]:
        start_year = date.today().year - 2
        payload = await self._get_json(
            f"{self.BASE_URL}/{series_key}", {"startPeriod": str(start_year)}
        )
        series = ((payload.get("CompactData") or {}).get("DataSet") or {}).get("Series") or {}
        observations = series.get("Obs") or []
        # SDMX-JSON returns a bare dict (not a one-item list) for a
        # single observation.
        if isinstance(observations, dict):
            observations = [observations]
        observations = [obs for obs in observations if obs.get("@OBS_VALUE") not in (None, "")]
        # Most-recent-first; "YYYY-MM" and "YYYY-Qn" both sort correctly
        # as plain strings within one series.
        return sorted(observations, key=lambda obs: obs.get("@TIME_PERIOD", ""), reverse=True)

    @staticmethod
    def _period_end_date(time_period: str) -> date:
        if "Q" in time_period:
            year_str, quarter_str = time_period.split("-Q")
            return date(int(year_str), int(quarter_str) * 3, 1)
        year_str, month_str = time_period.split("-")
        return date(int(year_str), int(month_str), 1)

    async def get_interest_rates(self) -> InterestRates:
        raise MacroProviderUnsupported(
            "IMF IFS does not cleanly expose US policy rates; use FRED."
        )

    async def get_cpi(self) -> CpiSnapshot:
        observations = await self._fetch_observations(self.CPI_SERIES_KEY)
        if not observations:
            raise MacroProviderError("IMF returned no CPI observations")
        latest = observations[0]
        latest_value = float(latest["@OBS_VALUE"])
        year_ago = float(observations[12]["@OBS_VALUE"]) if len(observations) > 12 else None
        return CpiSnapshot(
            value=latest_value,
            yoy_change_pct=yoy_change_pct(latest_value, year_ago),
            as_of=self._period_end_date(latest["@TIME_PERIOD"]),
        )

    async def get_gdp(self) -> GdpSnapshot:
        observations = await self._fetch_observations(self.GDP_SERIES_KEY)
        if not observations:
            raise MacroProviderError("IMF returned no GDP observations")
        latest = observations[0]
        latest_value = float(latest["@OBS_VALUE"])
        year_ago = float(observations[4]["@OBS_VALUE"]) if len(observations) > 4 else None
        return GdpSnapshot(
            value=latest_value,
            yoy_growth_pct=yoy_change_pct(latest_value, year_ago),
            as_of=self._period_end_date(latest["@TIME_PERIOD"]),
        )

    async def get_macro_calendar(self) -> list[MacroEvent]:
        return []

    async def _health_probe(self) -> None:
        await self.get_cpi()
