"""Finnhub market-data adapter (finnhub.io — same key as the Finnhub news
and fundamentals adapters). Serves daily price history via /stock/candle;
no option-chain data (inherits the base's Unsupported guard).

Note: Finnhub moved /stock/candle behind a paid tier for many accounts;
where it's unavailable the endpoint returns {"s": "no_data"} or a 403,
which normalize to MarketDataError / MarketDataUnavailable so the router
fails over to another price-history source. Implemented against the
documented free contract regardless.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from agentic_options_reporter.data.market_data.base import (
    PRICE_HISTORY,
    MarketDataError,
    _HttpMarketDataProvider,
)
from agentic_options_reporter.models.schemas import Bar, PriceHistory


class FinnhubMarketDataProvider(_HttpMarketDataProvider):
    BASE_URL = "https://finnhub.io/api/v1/stock/candle"
    PROVIDER_LABEL = "Finnhub"
    API_KEY_ENV_VAR = "FINNHUB_API_KEY"

    CAPABILITIES = frozenset({PRICE_HISTORY})

    async def get_price_history(self, symbol: str, lookback_days: int = 365) -> PriceHistory:
        to_ts = int(datetime.now(timezone.utc).timestamp())
        from_ts = int((datetime.now(timezone.utc) - timedelta(days=lookback_days)).timestamp())
        data = await self._get_json(
            self.BASE_URL,
            {
                "symbol": symbol.upper(),
                "resolution": "D",
                "from": from_ts,
                "to": to_ts,
                "token": self._api_key,
            },
        )
        if not isinstance(data, dict) or data.get("s") != "ok":
            raise MarketDataError(
                f"{self.PROVIDER_LABEL} returned no price history for {symbol!r} "
                f"(status: {data.get('s') if isinstance(data, dict) else 'unknown'})"
            )

        opens = data.get("o", [])
        highs = data.get("h", [])
        lows = data.get("l", [])
        closes = data.get("c", [])
        volumes = data.get("v", [])
        timestamps = data.get("t", [])
        bars = [
            Bar(
                dt=datetime.fromtimestamp(timestamps[i], tz=timezone.utc).date(),
                open=float(opens[i]),
                high=float(highs[i]),
                low=float(lows[i]),
                close=float(closes[i]),
                volume=float(volumes[i]),
            )
            for i in range(len(timestamps))
        ]
        if not bars:
            raise MarketDataError(f"{self.PROVIDER_LABEL} returned no price history for {symbol!r}")
        return PriceHistory(symbol=symbol, bars=bars)
