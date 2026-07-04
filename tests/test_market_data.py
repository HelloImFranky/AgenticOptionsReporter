import asyncio
import sys
import types
from datetime import date, datetime, timezone

import httpx
import pandas as pd
import pytest

from agentic_options_reporter.data.market_data import (
    OPTION_CHAIN,
    PRICE_HISTORY,
    AlphaVantageMarketDataProvider,
    FinnhubMarketDataProvider,
    MarketDataError,
    MarketDataProvider,
    MarketDataProviderRouter,
    MarketDataRateLimited,
    MarketDataUnavailable,
    MarketDataUnsupported,
    TwelveDataMarketDataProvider,
    YFinanceProvider,
    build_market_data_provider,
)
from agentic_options_reporter.data.market_data.base import _HttpMarketDataProvider
from agentic_options_reporter.models.schemas import OptionChain, PriceHistory


@pytest.fixture(autouse=True)
def _reset_market_data_cache():
    _HttpMarketDataProvider.clear_shared_cache()
    yield
    _HttpMarketDataProvider.clear_shared_cache()


_ALL_KEY_ENV_VARS = ("ALPHA_VANTAGE_API_KEY", "TWELVE_DATA_API_KEY", "FINNHUB_API_KEY")


@pytest.fixture(autouse=True)
def _clear_market_data_env(monkeypatch):
    for var in _ALL_KEY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("AOR_MARKET_DATA_PROVIDER_FALLBACK_ORDER", raising=False)
    for cap in (PRICE_HISTORY, OPTION_CHAIN):
        monkeypatch.delenv(f"AOR_MARKET_DATA_PRIORITY_{cap.upper()}", raising=False)


# -- yfinance adapter (sync lib wrapped via asyncio.to_thread) --


class _FakeTicker:
    def __init__(self, symbol):
        self.symbol = symbol
        self.options = ("2099-01-01",)

    def history(self, period="1y", auto_adjust=False):
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        return pd.DataFrame(
            {
                "Open": [100, 101, 102, 103, 104],
                "High": [101, 102, 103, 104, 105],
                "Low": [99, 100, 101, 102, 103],
                "Close": [100.5, 101.5, 102.5, 103.5, 104.5],
                "Volume": [1000, 1100, 1200, 1300, 1400],
            },
            index=idx,
        )

    def option_chain(self, expiration):
        calls = pd.DataFrame(
            [
                {
                    "contractSymbol": "FAKE990101C00100000",
                    "strike": 100.0,
                    "bid": 2.0,
                    "ask": 2.2,
                    "lastPrice": 2.1,
                    "volume": 10,
                    "openInterest": 500,
                    "impliedVolatility": 0.3,
                    "inTheMoney": False,
                }
            ]
        )
        puts = pd.DataFrame(
            [
                {
                    "contractSymbol": "FAKE990101P00100000",
                    "strike": 100.0,
                    "bid": 1.8,
                    "ask": 2.0,
                    "lastPrice": 1.9,
                    "volume": 8,
                    "openInterest": 400,
                    "impliedVolatility": 0.32,
                    "inTheMoney": False,
                }
            ]
        )
        return types.SimpleNamespace(calls=calls, puts=puts)


@pytest.fixture
def fake_yfinance(monkeypatch):
    fake_module = types.SimpleNamespace(Ticker=_FakeTicker)
    monkeypatch.setitem(sys.modules, "yfinance", fake_module)
    return fake_module


def test_yfinance_is_keyless_and_serves_both_capabilities():
    provider = YFinanceProvider()
    assert isinstance(provider, MarketDataProvider)
    assert provider.supports(PRICE_HISTORY)
    assert provider.supports(OPTION_CHAIN)


def test_yfinance_get_price_history(fake_yfinance):
    provider = YFinanceProvider()
    history = asyncio.run(provider.get_price_history("FAKE", lookback_days=5))
    assert history.symbol == "FAKE"
    assert len(history.bars) == 5
    assert history.bars[-1].close == 104.5


def test_yfinance_price_history_is_cached(fake_yfinance):
    provider = YFinanceProvider()
    first = asyncio.run(provider.get_price_history("FAKE"))
    second = asyncio.run(provider.get_price_history("FAKE"))
    assert first is second


def test_yfinance_get_option_chain(fake_yfinance):
    provider = YFinanceProvider()
    chain = asyncio.run(provider.get_option_chain("FAKE"))
    assert chain.symbol == "FAKE"
    assert len(chain.contracts) == 2
    assert {c.option_type for c in chain.contracts} == {"call", "put"}


def test_yfinance_unknown_expiration_raises(fake_yfinance):
    provider = YFinanceProvider()
    with pytest.raises(MarketDataError):
        asyncio.run(provider.get_option_chain("FAKE", expiration="1999-01-01"))


def test_yfinance_empty_history_raises(monkeypatch):
    class _EmptyTicker(_FakeTicker):
        def history(self, period="1y", auto_adjust=False):
            return pd.DataFrame()

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=_EmptyTicker))
    with pytest.raises(MarketDataError):
        asyncio.run(YFinanceProvider().get_price_history("FAKE"))


# -- HTTP price-history adapters --


class RecordingTransport:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("No more fake HTTP responses queued")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        status_code, payload = item
        return httpx.Response(status_code, json=payload)


def _client(transport: RecordingTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(transport))


_KEYED_PROVIDERS = [
    AlphaVantageMarketDataProvider,
    TwelveDataMarketDataProvider,
    FinnhubMarketDataProvider,
]


@pytest.mark.parametrize("provider_cls", _KEYED_PROVIDERS)
def test_http_provider_requires_api_key(provider_cls):
    with pytest.raises(MarketDataError):
        provider_cls()


@pytest.mark.parametrize("provider_cls", _KEYED_PROVIDERS)
def test_http_providers_serve_price_only_not_chains(provider_cls):
    provider = provider_cls(api_key="k")
    assert provider.supports(PRICE_HISTORY)
    assert not provider.supports(OPTION_CHAIN)
    with pytest.raises(MarketDataUnsupported):
        asyncio.run(provider.get_option_chain("AAPL"))


def test_alpha_vantage_maps_daily_series():
    transport = RecordingTransport(
        (200, {
            "Time Series (Daily)": {
                date.today().isoformat(): {
                    "1. open": "100.0", "2. high": "102.0", "3. low": "99.0",
                    "4. close": "101.0", "5. volume": "1000000",
                }
            }
        })
    )
    provider = AlphaVantageMarketDataProvider(api_key="k", client=_client(transport))
    history = asyncio.run(provider.get_price_history("AAPL", lookback_days=30))
    assert history.bars[-1].close == 101.0
    assert transport.requests[0].url.params["function"] == "TIME_SERIES_DAILY"


def test_alpha_vantage_rate_limit_note_raises():
    transport = RecordingTransport((200, {"Note": "rate limit reached"}))
    provider = AlphaVantageMarketDataProvider(api_key="k", client=_client(transport))
    with pytest.raises(MarketDataRateLimited):
        asyncio.run(provider.get_price_history("AAPL"))


def test_twelve_data_maps_values_oldest_first():
    transport = RecordingTransport(
        (200, {
            "status": "ok",
            "values": [
                {"datetime": "2026-01-02", "open": "101", "high": "103", "low": "100", "close": "102", "volume": "1200"},
                {"datetime": "2026-01-01", "open": "100", "high": "102", "low": "99", "close": "101", "volume": "1000"},
            ],
        })
    )
    provider = TwelveDataMarketDataProvider(api_key="k", client=_client(transport))
    history = asyncio.run(provider.get_price_history("AAPL"))
    assert [b.dt for b in history.bars] == [date(2026, 1, 1), date(2026, 1, 2)]


def test_twelve_data_error_status_rate_limit_raises():
    transport = RecordingTransport((200, {"status": "error", "message": "You have run out of API credits"}))
    provider = TwelveDataMarketDataProvider(api_key="k", client=_client(transport))
    with pytest.raises(MarketDataRateLimited):
        asyncio.run(provider.get_price_history("AAPL"))


def test_finnhub_maps_candles():
    ts = int(datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp())
    transport = RecordingTransport(
        (200, {"s": "ok", "o": [100], "h": [102], "l": [99], "c": [101], "v": [1000], "t": [ts]})
    )
    provider = FinnhubMarketDataProvider(api_key="k", client=_client(transport))
    history = asyncio.run(provider.get_price_history("AAPL"))
    assert history.bars[0].close == 101.0


def test_finnhub_no_data_status_raises():
    transport = RecordingTransport((200, {"s": "no_data"}))
    provider = FinnhubMarketDataProvider(api_key="k", client=_client(transport))
    with pytest.raises(MarketDataError):
        asyncio.run(provider.get_price_history("AAPL"))


def test_http_429_raises_rate_limited():
    transport = RecordingTransport((429, {}))
    provider = TwelveDataMarketDataProvider(api_key="k", client=_client(transport))
    with pytest.raises(MarketDataRateLimited):
        asyncio.run(provider.get_price_history("AAPL"))


def test_http_5xx_raises_unavailable():
    transport = RecordingTransport((503, {}))
    provider = TwelveDataMarketDataProvider(api_key="k", client=_client(transport))
    with pytest.raises(MarketDataUnavailable):
        asyncio.run(provider.get_price_history("AAPL"))


# -- Router --


class _StubProvider(MarketDataProvider):
    def __init__(self, caps, history=None, chain=None, error=None, name="stub"):
        self._caps = frozenset(caps)
        self._history = history
        self._chain = chain
        self._error = error
        self._name = name
        self.price_calls = 0

    @property
    def capabilities(self):
        return self._caps

    async def get_price_history(self, symbol, lookback_days=365):
        self.price_calls += 1
        if self._error is not None:
            raise self._error
        return self._history

    async def get_option_chain(self, symbol, expiration=None):
        if OPTION_CHAIN not in self._caps:
            raise MarketDataUnsupported("no chains")
        return self._chain

    async def health(self):
        from agentic_options_reporter.data.market_data import ProviderHealth

        return ProviderHealth(
            provider=self._name, healthy=self._error is None, checked_at=datetime.now(timezone.utc)
        )


def _history() -> PriceHistory:
    from agentic_options_reporter.models.schemas import Bar

    return PriceHistory(symbol="AAPL", bars=[Bar(dt=date(2026, 1, 1), open=1, high=1, low=1, close=1, volume=1)])


def _chain() -> OptionChain:
    return OptionChain(symbol="AAPL", underlying_price=1.0, as_of=datetime.now(timezone.utc), contracts=[])


def test_router_rejects_empty_client_list():
    with pytest.raises(MarketDataError):
        MarketDataProviderRouter([])


def test_router_option_chain_filtered_to_chain_provider():
    """Price-only providers must never be asked for an option chain."""
    price_only = _StubProvider({PRICE_HISTORY}, history=_history(), name="price")
    full = _StubProvider({PRICE_HISTORY, OPTION_CHAIN}, history=_history(), chain=_chain(), name="full")
    router = MarketDataProviderRouter([("price", price_only), ("full", full)])

    chain = asyncio.run(router.get_option_chain("AAPL"))
    assert chain.symbol == "AAPL"  # served by the only chain-capable provider


def test_router_raises_unsupported_when_no_provider_serves_chains():
    price_only = _StubProvider({PRICE_HISTORY}, history=_history(), name="price")
    router = MarketDataProviderRouter([("price", price_only)])
    with pytest.raises(MarketDataUnsupported):
        asyncio.run(router.get_option_chain("AAPL"))


def test_router_price_history_fails_over():
    down = _StubProvider({PRICE_HISTORY}, error=MarketDataUnavailable("down"), name="down")
    up = _StubProvider({PRICE_HISTORY}, history=_history(), name="up")
    router = MarketDataProviderRouter([("down", down), ("up", up)])

    history = asyncio.run(router.get_price_history("AAPL"))
    assert history.symbol == "AAPL"
    assert down.price_calls == 1 and up.price_calls == 1


def test_router_capabilities_is_union():
    price_only = _StubProvider({PRICE_HISTORY}, name="price")
    full = _StubProvider({PRICE_HISTORY, OPTION_CHAIN}, name="full")
    router = MarketDataProviderRouter([("price", price_only), ("full", full)])
    assert router.capabilities == frozenset({PRICE_HISTORY, OPTION_CHAIN})


def test_router_applies_per_capability_priority_override(monkeypatch):
    monkeypatch.setenv("AOR_MARKET_DATA_PRIORITY_PRICE_HISTORY", "b,a")
    a = _StubProvider({PRICE_HISTORY}, history=_history(), name="a")
    b = _StubProvider({PRICE_HISTORY}, history=_history(), name="b")
    router = MarketDataProviderRouter([("a", a), ("b", b)])

    asyncio.run(router.get_price_history("AAPL"))
    # Override puts b first, so b answers and a is never called.
    assert b.price_calls == 1 and a.price_calls == 0


# -- build_market_data_provider --


def test_build_includes_only_yfinance_when_no_keys():
    provider = build_market_data_provider()
    assert provider.provider_names == ["yfinance"]


def test_build_includes_configured_keyed_providers(monkeypatch):
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "k")
    monkeypatch.setenv("TWELVE_DATA_API_KEY", "k")
    provider = build_market_data_provider()
    assert provider.provider_names == ["yfinance", "alphavantage", "twelvedata"]


def test_build_respects_fallback_order_env(monkeypatch):
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "k")
    monkeypatch.setenv("AOR_MARKET_DATA_PROVIDER_FALLBACK_ORDER", "alphavantage,yfinance")
    provider = build_market_data_provider()
    assert provider.provider_names == ["alphavantage", "yfinance"]
