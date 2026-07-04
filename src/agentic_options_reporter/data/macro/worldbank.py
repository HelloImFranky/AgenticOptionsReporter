"""World Bank adapter (api.worldbank.org — free, keyless).

Serves `cpi` (indicator ``FP.CPI.TOTL``, annual index, 2010=100) and
`gdp` (``NY.GDP.MKTP.CD``, annual nominal, current US$). Its data is
ANNUAL and lags the US agencies by a year or more — the last-resort
fallback, at the end of the default order. It does NOT advertise any
rate metric (the Bank's real-interest-rate indicator is a different
metric from US policy rates), so the router never asks it for one — the
capability model's answer to the original "World Bank has no fed funds"
error.
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


class WorldBankMacroProvider(_HttpMacroProvider):
    BASE_URL = "https://api.worldbank.org/v2/country/USA/indicator"
    PROVIDER_LABEL = "World Bank"
    API_KEY_ENV_VAR = None  # keyless

    _INDICATORS: dict[str, str] = {
        "cpi": "FP.CPI.TOTL",
        "gdp": "NY.GDP.MKTP.CD",
    }
    METRICS = frozenset(_INDICATORS)

    async def _fetch_indicator(self, indicator: str) -> list[dict[str, Any]]:
        payload = await self._get_json(
            f"{self.BASE_URL}/{indicator}", {"format": "json", "per_page": 5}
        )
        # Response shape: [pagination_metadata, rows]; rows are
        # most-recent-first and may carry null values for years not yet
        # published.
        rows = payload[1] if isinstance(payload, list) and len(payload) > 1 else []
        return [row for row in (rows or []) if row.get("value") is not None]

    async def _fetch(self, metric_id: str) -> MacroObservation:
        rows = await self._fetch_indicator(self._INDICATORS[metric_id])
        if not rows:
            raise MacroProviderError(f"World Bank returned no observations for {metric_id}")

        latest_value = float(rows[0]["value"])
        year_ago = float(rows[1]["value"]) if len(rows) > 1 else None

        metric = get_metric(metric_id)
        return MacroObservation(
            metric_id=metric_id,
            label=metric.label,
            value=latest_value,
            unit=metric.unit,
            as_of=date(int(rows[0]["date"]), 12, 31),
            source=self.PROVIDER_LABEL,
            yoy_change_pct=yoy_change_pct(latest_value, year_ago),
        )
