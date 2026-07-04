"""Bureau of Labor Statistics adapter (bls.gov — free key).

A specialist: it serves only `cpi` (the primary series FRED mirrors —
real redundancy, not just an alternate). BLS publishes neither rates nor
GDP, so it simply doesn't advertise them and the router never asks.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from agentic_options_reporter.data.macro.base import (
    MacroProviderError,
    _HttpMacroProvider,
    yoy_change_pct,
)
from agentic_options_reporter.data.macro.metrics import get_metric
from agentic_options_reporter.models.schemas import MacroObservation


class BlsMacroProvider(_HttpMacroProvider):
    BASE_URL = "https://api.bls.gov/publicAPI/v2/timeseries/data"
    PROVIDER_LABEL = "BLS"
    API_KEY_ENV_VAR = "BLS_API_KEY"
    CPI_SERIES_ID = "CUUR0000SA0"  # CPI-U, US city average, all items, not seasonally adjusted

    METRICS = frozenset({"cpi"})

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

    async def _fetch(self, metric_id: str) -> MacroObservation:
        observations = await self._fetch_series(self.CPI_SERIES_ID)
        if not observations:
            raise MacroProviderError("BLS returned no CPI observations")

        latest = observations[0]
        latest_value = float(latest["value"])
        as_of = date(int(latest["year"]), int(latest["period"].lstrip("M")), 1)
        year_ago = float(observations[12]["value"]) if len(observations) > 12 else None

        metric = get_metric("cpi")
        return MacroObservation(
            metric_id="cpi",
            label=metric.label,
            value=latest_value,
            unit=metric.unit,
            as_of=as_of,
            source=self.PROVIDER_LABEL,
            yoy_change_pct=yoy_change_pct(latest_value, year_ago),
        )
