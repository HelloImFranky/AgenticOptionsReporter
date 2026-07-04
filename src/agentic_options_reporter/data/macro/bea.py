"""Bureau of Economic Analysis adapter (bea.gov — free key).

The primary source for the (nominal) GDP series FRED mirrors — real
redundancy on get_gdp. BEA publishes neither CPI (its PCE price index is
a related but distinct metric) nor interest rates, so those methods
raise MacroProviderUnsupported (retryable) and the router falls through.
"""

from __future__ import annotations

from datetime import date

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


class BeaMacroProvider(_HttpMacroProvider):
    BASE_URL = "https://apps.bea.gov/api/data"
    PROVIDER_LABEL = "BEA"
    API_KEY_ENV_VAR = "BEA_API_KEY"

    async def get_interest_rates(self) -> InterestRates:
        raise MacroProviderUnsupported("BEA does not publish interest rate data.")

    async def get_cpi(self) -> CpiSnapshot:
        raise MacroProviderUnsupported("BEA does not publish CPI data.")

    async def get_gdp(self) -> GdpSnapshot:
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
        return GdpSnapshot(
            value=latest_value,
            yoy_growth_pct=yoy_change_pct(latest_value, year_ago),
            as_of=self._period_end_date(rows[0]["TimePeriod"]),
        )

    async def get_macro_calendar(self) -> list[MacroEvent]:
        return []

    async def _health_probe(self) -> None:
        await self.get_gdp()

    @staticmethod
    def _parse_value(value: str) -> float:
        return float(str(value).replace(",", ""))

    @staticmethod
    def _period_end_date(time_period: str) -> date:
        year_str, quarter_str = time_period.split("Q")
        return date(int(year_str), int(quarter_str) * 3, 1)
