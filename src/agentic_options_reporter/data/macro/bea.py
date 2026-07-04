"""Bureau of Economic Analysis adapter (bea.gov — free key).

A specialist: it serves only `gdp` (the nominal series FRED mirrors —
real redundancy). BEA publishes neither CPI (its PCE price index is a
related but distinct metric) nor rates, so it doesn't advertise them.
"""

from __future__ import annotations

from datetime import date

from agentic_options_reporter.data.macro.base import (
    MacroProviderError,
    _HttpMacroProvider,
    yoy_change_pct,
)
from agentic_options_reporter.data.macro.metrics import get_metric
from agentic_options_reporter.models.schemas import MacroObservation


class BeaMacroProvider(_HttpMacroProvider):
    BASE_URL = "https://apps.bea.gov/api/data"
    PROVIDER_LABEL = "BEA"
    API_KEY_ENV_VAR = "BEA_API_KEY"

    METRICS = frozenset({"gdp"})

    async def _fetch(self, metric_id: str) -> MacroObservation:
        current_year = date.today().year
        payload = await self._get_json(
            self.BASE_URL,
            {
                "UserID": self._api_key,
                "method": "GetData",
                "datasetname": "NIPA",
                "TableName": "T10101",  # line 1: Gross domestic product (nominal, quarterly)
                "Frequency": "Q",
                "Year": f"{current_year - 2},{current_year - 1},{current_year}",
                "ResultFormat": "JSON",
            },
        )
        results = ((payload.get("BEAAPI") or {}).get("Results")) or {}
        rows = [row for row in (results.get("Data") or []) if row.get("LineNumber") == "1"]
        # BEA's TimePeriod ("2026Q2") sorts correctly as a plain string.
        rows.sort(key=lambda row: row.get("TimePeriod", ""), reverse=True)
        if not rows:
            raise MacroProviderError("BEA returned no GDP observations")

        latest_value = self._parse_value(rows[0]["DataValue"])
        year_ago = self._parse_value(rows[4]["DataValue"]) if len(rows) > 4 else None

        metric = get_metric("gdp")
        return MacroObservation(
            metric_id="gdp",
            label=metric.label,
            value=latest_value,
            unit=metric.unit,
            as_of=self._period_end_date(rows[0]["TimePeriod"]),
            source=self.PROVIDER_LABEL,
            yoy_change_pct=yoy_change_pct(latest_value, year_ago),
        )

    @staticmethod
    def _parse_value(value: str) -> float:
        return float(str(value).replace(",", ""))

    @staticmethod
    def _period_end_date(time_period: str) -> date:
        year_str, quarter_str = time_period.split("Q")
        return date(int(year_str), int(quarter_str) * 3, 1)
