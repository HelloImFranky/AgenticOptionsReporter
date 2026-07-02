import sys
import types
from datetime import date, timedelta

import pandas as pd
import pytest

from agentic_options_reporter.data.market_data import MarketDataError, YFinanceProvider


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


def test_get_price_history(fake_yfinance):
    provider = YFinanceProvider()
    history = provider.get_price_history("FAKE", lookback_days=5)
    assert history.symbol == "FAKE"
    assert len(history.bars) == 5
    assert history.bars[-1].close == 104.5


def test_get_price_history_is_cached(fake_yfinance):
    provider = YFinanceProvider()
    first = provider.get_price_history("FAKE")
    second = provider.get_price_history("FAKE")
    assert first is second


def test_get_option_chain(fake_yfinance):
    provider = YFinanceProvider()
    chain = provider.get_option_chain("FAKE")
    assert chain.symbol == "FAKE"
    assert len(chain.contracts) == 2
    option_types = {c.option_type for c in chain.contracts}
    assert option_types == {"call", "put"}


def test_get_option_chain_unknown_expiration_raises(fake_yfinance):
    provider = YFinanceProvider()
    with pytest.raises(MarketDataError):
        provider.get_option_chain("FAKE", expiration="1999-01-01")
