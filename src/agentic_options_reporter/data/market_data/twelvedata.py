"""Twelve Data market-data adapter (twelvedata.com — free tier ~800
requests/day, 8/min). Serves daily price history via /time_series; no
option-chain data (inherits the base's Unsupported guard). Errors arrive
as HTTP 200 with {"status": "error", "message": ...}; a rate-limit
message is normalized to MarketDataRateLimited so the router fails over.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from agentic_options_reporter.data.market_data.base import (
    PRICE_HISTORY,
    MarketDataError,
    MarketDataRateLimited,
    _HttpMarketDataProvider,
)
from agentic_options_reporter.models.schemas import Bar, PriceHistory

# Twelve Data caps outputsize at 5000 points per request.
_MAX_OUTPUT_SIZE = 5000


class TwelveDataMarketDataProvider(_HttpMarketDataProvider):
    BASE_URL = "https://api.twelvedata.com/time_series"
    PROVIDER_LABEL = "Twelve Data"
    API_KEY_ENV_VAR = "TWELVE_DATA_API_KEY"

    CAPABILITIES = frozenset({PRICE_HISTORY})

    def _check_payload(self, payload: Any) -> None:
        if isinstance(payload, dict) and payload.get("status") == "error":
            message = payload.get("message", "")
            if "limit" in message.lower() or "credit" in message.lower():
                raise MarketDataRateLimited(f"{self.PROVIDER_LABEL} rate limited: {message}")
            raise MarketDataError(f"{self.PROVIDER_LABEL} request failed: {message}")

    async def get_price_history(self, symbol: str, lookback_days: int = 365) -> PriceHistory:
        data = await self._get_json(
            self.BASE_URL,
            {
                "symbol": symbol.upper(),
                "interval": "1day",
                "outputsize": min(lookback_days, _MAX_OUTPUT_SIZE),
                "apikey": self._api_key,
            },
        )
        values = data.get("values") if isinstance(data, dict) else None
        if not values:
            raise MarketDataError(f"{self.PROVIDER_LABEL} returned no price history for {symbol!r}")

        bars = [
            Bar(
                dt=datetime.strptime(row["datetime"], "%Y-%m-%d").date(),
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                # Some assets (e.g. indices) omit volume; default to 0.
                volume=float(row.get("volume") or 0.0),
            )
            for row in values
        ]
        bars.sort(key=lambda b: b.dt)  # Twelve Data returns newest-first
        return PriceHistory(symbol=symbol, bars=bars)
