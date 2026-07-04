"""FRED adapter (fred.stlouisfed.org — free key, generous limits).

The broadest single macro source: it serves every metric in the
registry — the fed funds rate (FEDFUNDS) and Treasury yields
(DGS10/DGS2), CPI (CPIAUCSL), and GDP (GDP).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from agentic_options_reporter.data.macro.base import (
    MacroProviderError,
    _HttpMacroProvider,
    yoy_change_pct,
)
from agentic_options_reporter.data.macro.metrics import get_metric
from agentic_options_reporter.models.schemas import MacroObservation


class FredMacroProvider(_HttpMacroProvider):
    BASE_URL = "https://api.stlouisfed.org/fred"
    PROVIDER_LABEL = "FRED"
    API_KEY_ENV_VAR = "FRED_API_KEY"

    # metric id -> (FRED series id, periods-per-year for YoY or None).
    # Rates are point-in-time levels, so no YoY; CPI/GDP get YoY.
    _SERIES: dict[str, tuple[str, int | None]] = {
        "policy_rate": ("FEDFUNDS", None),
        "treasury_10y": ("DGS10", None),
        "treasury_2y": ("DGS2", None),
        "cpi": ("CPIAUCSL", 12),
        "gdp": ("GDP", 4),
    }
    METRICS = frozenset(_SERIES)

    async def _fetch_series(self, series_id: str, limit: int) -> list[dict[str, Any]]:
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

    async def _fetch(self, metric_id: str) -> MacroObservation:
        series_id, yoy_periods = self._SERIES[metric_id]
        limit = 1 if yoy_periods is None else yoy_periods + 1
        observations = await self._fetch_series(series_id, limit=limit)
        if not observations:
            raise MacroProviderError(f"FRED returned no observations for {metric_id}")

        latest = observations[0]
        latest_value = float(latest["value"])
        as_of = datetime.strptime(latest["date"], "%Y-%m-%d").date()
        change = None
        if yoy_periods is not None and len(observations) > yoy_periods:
            change = yoy_change_pct(latest_value, float(observations[yoy_periods]["value"]))

        metric = get_metric(metric_id)
        return MacroObservation(
            metric_id=metric_id,
            label=metric.label,
            value=latest_value,
            unit=metric.unit,
            as_of=as_of,
            source=self.PROVIDER_LABEL,
            yoy_change_pct=change,
        )
