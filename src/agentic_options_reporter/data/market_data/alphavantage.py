"""Alpha Vantage market-data adapter (alphavantage.co — free tier ~25
requests/day). Serves daily price history via TIME_SERIES_DAILY; no
option-chain data (inherits the base's Unsupported guard). The free tier
returns HTTP 200 with an "Information"/"Note" field instead of a real 429
when rate limited; `_check_payload` treats that the same as a 429.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any

from agentic_options_reporter.data.market_data.base import (
    PRICE_HISTORY,
    MarketDataError,
    MarketDataRateLimited,
    _HttpMarketDataProvider,
)
from agentic_options_reporter.models.schemas import Bar, PriceHistory


class AlphaVantageMarketDataProvider(_HttpMarketDataProvider):
    BASE_URL = "https://www.alphavantage.co/query"
    PROVIDER_LABEL = "Alpha Vantage"
    API_KEY_ENV_VAR = "ALPHA_VANTAGE_API_KEY"

    CAPABILITIES = frozenset({PRICE_HISTORY})

    def _check_payload(self, payload: Any) -> None:
        if isinstance(payload, dict) and ("Information" in payload or "Note" in payload):
            raise MarketDataRateLimited(
                f"{self.PROVIDER_LABEL} rate limited or restricted: "
                f"{payload.get('Information') or payload.get('Note')}"
            )

    async def get_price_history(self, symbol: str, lookback_days: int = 365) -> PriceHistory:
        # compact returns the latest 100 points; full returns 20+ years.
        output_size = "compact" if lookback_days <= 100 else "full"
        data = await self._get_json(
            self.BASE_URL,
            {
                "function": "TIME_SERIES_DAILY",
                "symbol": symbol.upper(),
                "outputsize": output_size,
                "apikey": self._api_key,
            },
        )
        series = data.get("Time Series (Daily)") if isinstance(data, dict) else None
        if not series:
            raise MarketDataError(f"{self.PROVIDER_LABEL} returned no price history for {symbol!r}")

        cutoff = date.today() - timedelta(days=lookback_days)
        bars: list[Bar] = []
        for day, row in series.items():
            bar_date = datetime.strptime(day, "%Y-%m-%d").date()
            if bar_date < cutoff:
                continue
            bars.append(
                Bar(
                    dt=bar_date,
                    open=float(row["1. open"]),
                    high=float(row["2. high"]),
                    low=float(row["3. low"]),
                    close=float(row["4. close"]),
                    volume=float(row["5. volume"]),
                )
            )
        bars.sort(key=lambda b: b.dt)
        if not bars:
            raise MarketDataError(f"{self.PROVIDER_LABEL} returned no price history for {symbol!r}")
        return PriceHistory(symbol=symbol, bars=bars)
