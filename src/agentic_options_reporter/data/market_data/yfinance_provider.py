"""Yahoo Finance adapter, backed by the synchronous `yfinance` package.

The odd one out in this package: `yfinance` is a synchronous library
(pandas-returning, `requests`-backed) with no async API, so it can't sit
on the httpx `_HttpMarketDataProvider` base. Instead it implements the
async `MarketDataProvider` interface directly and offloads each blocking
call to a worker thread via `asyncio.to_thread`, so it still joins the
failover router and exposes a `health()` probe like every other adapter.

It's the only source here that serves OPTION_CHAIN — free option-chain
data is rare — so the router filters chain requests to it while
load-balancing/failing-over price history across the HTTP sources too.
Keyless.
"""

from __future__ import annotations

import asyncio
import math
import time
from datetime import datetime, timezone
from typing import Any

from agentic_options_reporter.data.async_http import ProviderHealth
from agentic_options_reporter.data.market_data.base import (
    MARKET_DATA_CAPABILITIES,
    MarketDataError,
    MarketDataProvider,
    MarketDataUnavailable,
)
from agentic_options_reporter.models.schemas import (
    Bar,
    OptionChain,
    OptionContract,
    PriceHistory,
)


class _TTLCache:
    """Minimal in-process TTL cache to reduce provider rate-limit pressure."""

    def __init__(self, ttl_seconds: int = 300) -> None:
        self._ttl = ttl_seconds
        self._store: dict[Any, tuple[float, Any]] = {}

    def get(self, key: Any) -> Any | None:
        entry = self._store.get(key)
        if entry is None:
            return None
        expires_at, value = entry
        if time.monotonic() > expires_at:
            del self._store[key]
            return None
        return value

    def set(self, key: Any, value: Any) -> None:
        self._store[key] = (time.monotonic() + self._ttl, value)


class YFinanceProvider(MarketDataProvider):
    """Yahoo Finance backed provider using the `yfinance` package."""

    PROVIDER_LABEL = "Yahoo Finance"

    def __init__(self, cache_ttl_seconds: int = 300) -> None:
        self._cache = _TTLCache(cache_ttl_seconds)

    @property
    def capabilities(self) -> frozenset[str]:
        return MARKET_DATA_CAPABILITIES  # both price history and option chains

    async def get_price_history(self, symbol: str, lookback_days: int = 365) -> PriceHistory:
        cache_key = ("history", symbol, lookback_days)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        history = await asyncio.to_thread(self._get_price_history_sync, symbol, lookback_days)
        self._cache.set(cache_key, history)
        return history

    def _get_price_history_sync(self, symbol: str, lookback_days: int) -> PriceHistory:
        import yfinance as yf

        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=f"{lookback_days}d", auto_adjust=False)
        except MarketDataError:
            raise
        except Exception as exc:  # noqa: BLE001 — normalize yfinance/network errors for failover
            raise MarketDataUnavailable(f"Yahoo Finance request failed for {symbol!r}: {exc}") from exc
        if df.empty:
            raise MarketDataError(f"No price history returned for symbol {symbol!r}")

        bars = [
            Bar(
                dt=idx.date(),
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]),
            )
            for idx, row in df.iterrows()
        ]
        return PriceHistory(symbol=symbol, bars=bars)

    async def get_option_chain(self, symbol: str, expiration: str | None = None) -> OptionChain:
        cache_key = ("chain", symbol, expiration)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        chain = await asyncio.to_thread(self._get_option_chain_sync, symbol, expiration)
        self._cache.set(cache_key, chain)
        return chain

    def _get_option_chain_sync(self, symbol: str, expiration: str | None) -> OptionChain:
        import yfinance as yf

        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
        except MarketDataError:
            raise
        except Exception as exc:  # noqa: BLE001 — normalize yfinance/network errors for failover
            raise MarketDataUnavailable(f"Yahoo Finance request failed for {symbol!r}: {exc}") from exc
        if not expirations:
            raise MarketDataError(f"No option expirations available for {symbol!r}")

        target_expiration = expiration or expirations[0]
        if target_expiration not in expirations:
            raise MarketDataError(
                f"Expiration {target_expiration!r} not available for {symbol!r}; "
                f"available: {list(expirations)}"
            )

        chain = ticker.option_chain(target_expiration)
        underlying_price = _last_close(ticker)

        contracts: list[OptionContract] = []
        for option_type, frame in (("call", chain.calls), ("put", chain.puts)):
            for _, row in frame.iterrows():
                contracts.append(
                    OptionContract(
                        contract_symbol=row["contractSymbol"],
                        option_type=option_type,
                        strike=float(row["strike"]),
                        expiration=datetime.strptime(target_expiration, "%Y-%m-%d").date(),
                        bid=float(row.get("bid") or 0.0),
                        ask=float(row.get("ask") or 0.0),
                        last_price=float(row.get("lastPrice") or 0.0),
                        volume=_safe_int(row.get("volume")),
                        open_interest=_safe_int(row.get("openInterest")),
                        implied_volatility=(
                            float(row["impliedVolatility"])
                            if row.get("impliedVolatility") is not None
                            else None
                        ),
                        in_the_money=bool(row.get("inTheMoney", False)),
                    )
                )

        return OptionChain(
            symbol=symbol,
            underlying_price=underlying_price,
            as_of=datetime.now(timezone.utc),
            contracts=contracts,
        )

    async def health(self) -> ProviderHealth:
        """Probe a cheap price-history fetch. Never raises — an unhealthy
        provider is a result, not an error (mirrors AsyncHttpProviderBase)."""
        started = time.monotonic()
        try:
            await self.get_price_history(_HEALTH_PROBE_TICKER, lookback_days=5)
        except Exception as exc:  # noqa: BLE001 — health() reports, never raises
            return ProviderHealth(
                provider=self.PROVIDER_LABEL,
                healthy=False,
                latency_ms=(time.monotonic() - started) * 1000,
                detail=str(exc),
                checked_at=datetime.now(timezone.utc),
            )
        return ProviderHealth(
            provider=self.PROVIDER_LABEL,
            healthy=True,
            latency_ms=(time.monotonic() - started) * 1000,
            checked_at=datetime.now(timezone.utc),
        )


_HEALTH_PROBE_TICKER = "AAPL"


def _safe_int(value: Any, default: int = 0) -> int:
    """Coerce yfinance's option-chain fields to int, treating NaN as missing.

    yfinance reports NaN (not None) for volume/open interest on contracts
    with no trades, and NaN is truthy in Python so `value or default` does
    not catch it — it falls through to `int(nan)`, which raises ValueError.
    """
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return default
    return int(value)


def _last_close(ticker: Any) -> float:
    fast_info = getattr(ticker, "fast_info", None)
    if fast_info is not None:
        price = fast_info.get("lastPrice") if hasattr(fast_info, "get") else None
        if price:
            return float(price)
    history = ticker.history(period="1d")
    if history.empty:
        raise MarketDataError("Unable to determine underlying price")
    return float(history["Close"].iloc[-1])
