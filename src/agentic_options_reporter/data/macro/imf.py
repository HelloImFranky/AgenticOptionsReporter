"""International Monetary Fund adapter (dataservices.imf.org — free,
keyless SDMX-JSON service).

Serves `cpi` (IFS series ``M.US.PCPI_IX``, monthly index) and `gdp`
(``Q.US.NGDP_SA_XDC``, quarterly nominal, seasonally adjusted). US policy
rates aren't cleanly exposed through IFS, so it doesn't advertise rate
metrics. Keyless, so it's always available — but IMF data lags the
primary US agencies, which is why it sits behind FRED/BLS/BEA in the
default order.
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


class ImfMacroProvider(_HttpMacroProvider):
    BASE_URL = "https://dataservices.imf.org/REST/SDMX_JSON.svc/CompactData/IFS"
    PROVIDER_LABEL = "IMF"
    API_KEY_ENV_VAR = None  # keyless

    # metric id -> (IFS series key, YoY lookback in periods)
    _SERIES: dict[str, tuple[str, int]] = {
        "cpi": ("M.US.PCPI_IX", 12),
        "gdp": ("Q.US.NGDP_SA_XDC", 4),
    }
    METRICS = frozenset(_SERIES)

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

    async def _fetch(self, metric_id: str) -> MacroObservation:
        series_key, yoy_periods = self._SERIES[metric_id]
        observations = await self._fetch_observations(series_key)
        if not observations:
            raise MacroProviderError(f"IMF returned no observations for {metric_id}")

        latest = observations[0]
        latest_value = float(latest["@OBS_VALUE"])
        year_ago = (
            float(observations[yoy_periods]["@OBS_VALUE"])
            if len(observations) > yoy_periods
            else None
        )

        metric = get_metric(metric_id)
        return MacroObservation(
            metric_id=metric_id,
            label=metric.label,
            value=latest_value,
            unit=metric.unit,
            as_of=self._period_end_date(latest["@TIME_PERIOD"]),
            source=self.PROVIDER_LABEL,
            yoy_change_pct=yoy_change_pct(latest_value, year_ago),
        )
