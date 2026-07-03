from __future__ import annotations

import math
import sys
import types
from datetime import date, datetime, timedelta, timezone

import pytest

from agentic_options_reporter.data.market_data import MarketDataProvider
from agentic_options_reporter.models.schemas import (
    Bar,
    OptionChain,
    OptionContract,
    PriceHistory,
)
from agentic_options_reporter.thesis.llm_client import LlmClient


def _make_bars(n: int, start_price: float = 100.0, trend_per_day: float = 0.05) -> list[Bar]:
    bars = []
    start_date = date(2024, 1, 1)
    price = start_price
    for i in range(n):
        price = start_price + trend_per_day * i + 2 * math.sin(i / 5)
        price = max(price, 1.0)
        open_ = price - 0.3
        high = price + 0.8
        low = price - 0.8
        close = price
        volume = 1_000_000 + (500_000 if i % 7 == 0 else 0)
        bars.append(
            Bar(
                dt=start_date + timedelta(days=i),
                open=open_,
                high=high,
                low=low,
                close=close,
                volume=volume,
            )
        )
    return bars


@pytest.fixture
def uptrend_history() -> PriceHistory:
    return PriceHistory(symbol="TEST", bars=_make_bars(260, start_price=100.0, trend_per_day=0.4))


@pytest.fixture
def downtrend_history() -> PriceHistory:
    return PriceHistory(symbol="TEST", bars=_make_bars(260, start_price=200.0, trend_per_day=-0.4))


@pytest.fixture
def flat_history() -> PriceHistory:
    return PriceHistory(symbol="TEST", bars=_make_bars(260, start_price=100.0, trend_per_day=0.0))


@pytest.fixture
def sample_option_chain() -> OptionChain:
    underlying_price = 100.0
    expiration = date.today() + timedelta(days=30)
    strikes = [90, 95, 100, 105, 110]
    contracts = []
    for strike in strikes:
        for option_type in ("call", "put"):
            mid = max(1.0, abs(underlying_price - strike) * 0.1 + 2.0)
            contracts.append(
                OptionContract(
                    contract_symbol=f"TEST{expiration:%y%m%d}{option_type[0].upper()}{int(strike*1000):08d}",
                    option_type=option_type,
                    strike=float(strike),
                    expiration=expiration,
                    bid=round(mid - 0.05, 2),
                    ask=round(mid + 0.05, 2),
                    last_price=round(mid, 2),
                    volume=100,
                    open_interest=800,
                    implied_volatility=0.35,
                    in_the_money=(strike < underlying_price if option_type == "call" else strike > underlying_price),
                )
            )
    return OptionChain(
        symbol="TEST",
        underlying_price=underlying_price,
        as_of=datetime.now(timezone.utc),
        contracts=contracts,
    )


class FakeMarketDataProvider(MarketDataProvider):
    def __init__(self, history: PriceHistory, chain: OptionChain) -> None:
        self._history = history
        self._chain = chain

    def get_price_history(self, symbol: str, lookback_days: int = 365) -> PriceHistory:
        return self._history

    def get_option_chain(self, symbol: str, expiration: str | None = None) -> OptionChain:
        return self._chain


@pytest.fixture
def fake_provider(uptrend_history: PriceHistory, sample_option_chain: OptionChain) -> FakeMarketDataProvider:
    return FakeMarketDataProvider(uptrend_history, sample_option_chain)


class FakeLlmClient(LlmClient):
    """Dispatches canned JSON responses by matching a substring in the
    system prompt, so a single fake can serve a whole agent pipeline run."""

    def __init__(self, responses: dict[str, str]) -> None:
        self._responses = responses
        self.calls: list[tuple[str, str]] = []

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        for key, response in self._responses.items():
            if key in system_prompt:
                return response
        raise AssertionError(f"No fake response configured for prompt: {system_prompt[:60]!r}")


class FakeHttpResponse:
    """Stand-in for a `requests.Response`, used to test the provider
    modules (news/financial/macro/sec) without live network calls."""

    def __init__(self, json_data, status_code: int = 200, raise_exc: Exception | None = None):
        self._json_data = json_data
        self.status_code = status_code
        self._raise_exc = raise_exc

    def raise_for_status(self) -> None:
        if self._raise_exc is not None:
            raise self._raise_exc

    def json(self):
        return self._json_data


class FakeRequestsGet:
    """Callable stand-in for `requests.get` that returns queued responses
    in order and records every call for assertions."""

    def __init__(self, *responses: FakeHttpResponse):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, url, params=None, headers=None, timeout=None):
        self.calls.append({"url": url, "params": params, "headers": headers, "timeout": timeout})
        if not self._responses:
            raise AssertionError("No more fake HTTP responses configured")
        return self._responses.pop(0)


@pytest.fixture
def fake_requests_module(monkeypatch):
    """Injects a fake `requests` module so provider modules (which do a
    lazy `import requests` inside their HTTP methods) never touch the
    network. Configure `.get` per test with a `FakeRequestsGet`."""
    fake_module = types.SimpleNamespace(
        get=None,
        exceptions=types.SimpleNamespace(RequestException=Exception),
    )
    monkeypatch.setitem(sys.modules, "requests", fake_module)
    return fake_module
