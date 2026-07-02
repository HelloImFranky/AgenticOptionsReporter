"""Market data access.

`MarketDataProvider` is the interface the rest of the codebase depends on
(dependency injection — see agents/backend.md). `YFinanceProvider` is the
default implementation backed by Yahoo Finance via `yfinance`. Additional
providers (Polygon.io, Alpaca, Tradier, Interactive Brokers, Finnhub,
Alpha Vantage) can be added by implementing the same interface without
touching analysis code.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from agentic_options_reporter.models.schemas import (
    Bar,
    OptionChain,
    OptionContract,
    PriceHistory,
)


class MarketDataError(RuntimeError):
    """Raised when a provider cannot return the requested data."""


class MarketDataProvider(ABC):
    """Interface implemented by all market data providers."""

    @abstractmethod
    def get_price_history(self, symbol: str, lookback_days: int = 365) -> PriceHistory:
        raise NotImplementedError

    @abstractmethod
    def get_option_chain(
        self, symbol: str, expiration: str | None = None
    ) -> OptionChain:
        raise NotImplementedError


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

    def __init__(self, cache_ttl_seconds: int = 300) -> None:
        self._cache = _TTLCache(cache_ttl_seconds)

    def get_price_history(self, symbol: str, lookback_days: int = 365) -> PriceHistory:
        cache_key = ("history", symbol, lookback_days)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        import yfinance as yf

        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f"{lookback_days}d", auto_adjust=False)
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
        history = PriceHistory(symbol=symbol, bars=bars)
        self._cache.set(cache_key, history)
        return history

    def get_option_chain(
        self, symbol: str, expiration: str | None = None
    ) -> OptionChain:
        cache_key = ("chain", symbol, expiration)
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached

        import yfinance as yf

        ticker = yf.Ticker(symbol)
        expirations = ticker.options
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
                        expiration=datetime.strptime(
                            target_expiration, "%Y-%m-%d"
                        ).date(),
                        bid=float(row.get("bid") or 0.0),
                        ask=float(row.get("ask") or 0.0),
                        last_price=float(row.get("lastPrice") or 0.0),
                        volume=int(row.get("volume") or 0),
                        open_interest=int(row.get("openInterest") or 0),
                        implied_volatility=(
                            float(row["impliedVolatility"])
                            if row.get("impliedVolatility") is not None
                            else None
                        ),
                        in_the_money=bool(row.get("inTheMoney", False)),
                    )
                )

        result = OptionChain(
            symbol=symbol,
            underlying_price=underlying_price,
            as_of=datetime.now(timezone.utc),
            contracts=contracts,
        )
        self._cache.set(cache_key, result)
        return result


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
