"""Tests for cross-provider fundamentals: the fan-out-merge helpers, the
Yahoo/Finnhub new datasets, the merge router, and the /analyze snapshot."""

import asyncio
import sys
import types
from datetime import date

import httpx
import pytest

from agentic_options_reporter.data.async_http import AsyncHttpProviderBase
from agentic_options_reporter.data.financial import (
    FinancialProvider,
    FinancialProviderError,
    FinancialProviderRateLimited,
    FinancialProviderRouter,
    FinnhubFinancialProvider,
    YFinanceFinancialProvider,
)
from agentic_options_reporter.data.financial.base import (
    EARNINGS,
    INSIDER,
    METRICS,
    PROFILE,
)
from agentic_options_reporter.data.financial.snapshot import gather_fundamentals
from agentic_options_reporter.data.provider_router import (
    acall_and_merge,
    merge_lists,
    merge_models,
)
from agentic_options_reporter.models.schemas import (
    CompanyMetrics,
    CompanyProfile,
    EarningsHistory,
    EarningsSurprise,
    FundamentalsSnapshot,
)


@pytest.fixture(autouse=True)
def _reset_cache_and_env(monkeypatch):
    AsyncHttpProviderBase.clear_shared_cache()
    for var in ("FMP_API_KEY", "FINNHUB_API_KEY", "ALPHA_VANTAGE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    yield
    AsyncHttpProviderBase.clear_shared_cache()


# --------------------------------------------------------------------------
# merge helpers
# --------------------------------------------------------------------------


def test_merge_models_fills_missing_fields_in_priority_order():
    a = CompanyProfile(ticker="AAPL", name="Apple", sector="", industry="Electronics")
    b = CompanyProfile(ticker="AAPL", name="Apple Inc", sector="Technology", market_cap=3e12)
    merged = merge_models([a, b])
    # First present value wins per field.
    assert merged.name == "Apple"              # a wins (present)
    assert merged.sector == "Technology"        # a empty -> b fills
    assert merged.industry == "Electronics"     # only a has it
    assert merged.market_cap == 3e12            # only b has it


def test_merge_models_treats_na_and_zero_analysts_appropriately():
    a = CompanyMetrics(ticker="X", dividend_yield=0.0, pe_ratio=None)
    b = CompanyMetrics(ticker="X", dividend_yield=0.02, pe_ratio=30.0)
    merged = merge_models([a, b])
    # A real 0.0 is present, so a wins for dividend_yield; pe filled from b.
    assert merged.dividend_yield == 0.0
    assert merged.pe_ratio == 30.0


def test_merge_lists_dedupes_by_key_preserving_order():
    merged = merge_lists([[1, 2], [2, 3], [3, 4]], key=lambda x: x)
    assert merged == [1, 2, 3, 4]


def test_acall_and_merge_combines_successes_and_ignores_failures():
    class Ok:
        def __init__(self, value):
            self._value = value

        async def m(self, arg):
            return self._value

    class Boom:
        async def m(self, arg):
            raise FinancialProviderRateLimited("429")

    clients = [("ok1", Ok("a")), ("bad", Boom()), ("ok2", Ok("b"))]
    result = asyncio.run(
        acall_and_merge(clients, "m", FinancialProviderError, lambda xs: xs, "AAPL")
    )
    assert result == ["a", "b"]  # both successes, failure skipped


def test_acall_and_merge_raises_only_when_all_fail():
    class Boom:
        def __init__(self, name):
            self._name = name

        async def m(self, arg):
            raise FinancialProviderRateLimited(f"{self._name} 429")

    clients = [("first", Boom("first")), ("second", Boom("second"))]
    with pytest.raises(FinancialProviderError, match="first.*second"):
        asyncio.run(acall_and_merge(clients, "m", FinancialProviderError, lambda xs: xs, "AAPL"))


# --------------------------------------------------------------------------
# Yahoo (yfinance) fundamentals adapter
# --------------------------------------------------------------------------


class _FakeInfoTicker:
    _INFO = {
        "longName": "Apple Inc.",
        "sector": "Technology",
        "industry": "Consumer Electronics",
        "marketCap": 3_000_000_000_000,
        "trailingPE": 30.5,
        "forwardPE": 28.0,
        "priceToBook": 45.0,
        "beta": 1.25,
        "dividendYield": 0.005,
        "fiftyTwoWeekHigh": 200.0,
        "fiftyTwoWeekLow": 150.0,
        "grossMargins": 0.44,
        "profitMargins": 0.25,
        "recommendationKey": "buy",
        "targetMeanPrice": 220.0,
        "numberOfAnalystOpinions": 30,
    }

    def __init__(self, symbol):
        self.symbol = symbol

    @property
    def info(self):
        return dict(self._INFO)


@pytest.fixture
def fake_yfinance_info(monkeypatch):
    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=_FakeInfoTicker))


def test_yfinance_serves_all_datasets():
    provider = YFinanceFinancialProvider()
    assert provider.supports(METRICS)
    assert provider.supports(EARNINGS)
    assert provider.supports(INSIDER)
    assert provider.supports(PROFILE)


def test_yfinance_profile_and_metrics_from_info(fake_yfinance_info):
    provider = YFinanceFinancialProvider()
    profile = asyncio.run(provider.get_company_profile("aapl"))
    metrics = asyncio.run(provider.get_company_metrics("aapl"))

    assert profile.name == "Apple Inc."
    assert profile.sector == "Technology"
    assert profile.market_cap == 3_000_000_000_000
    assert metrics.pe_ratio == 30.5
    assert metrics.beta == 1.25
    assert metrics.week52_high == 200.0


def test_yfinance_analyst_estimates_titlecases_rating(fake_yfinance_info):
    provider = YFinanceFinancialProvider()
    estimates = asyncio.run(provider.get_analyst_estimates("aapl"))
    assert estimates.consensus_rating == "Buy"
    assert estimates.price_target_mean == 220.0
    assert estimates.num_analysts == 30


def test_yfinance_insider_transactions_and_net_shares(monkeypatch):
    import pandas as pd

    class _InsiderTicker:
        def __init__(self, symbol):
            pass

        @property
        def insider_transactions(self):
            return pd.DataFrame(
                [
                    {"Insider": "Jane Doe", "Position": "CEO", "Shares": 1000,
                     "Value": 200000, "Start Date": "2026-06-01", "Text": "Sale at price"},
                    {"Insider": "John Roe", "Position": "CFO", "Shares": 500,
                     "Value": 100000, "Start Date": "2026-05-01", "Text": "Purchase at price"},
                ]
            )

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=_InsiderTicker))
    provider = YFinanceFinancialProvider()
    activity = asyncio.run(provider.get_insider_activity("aapl"))

    assert len(activity.transactions) == 2
    assert activity.transactions[0].transaction_type == "sell"
    assert activity.transactions[1].transaction_type == "buy"
    # 500 bought - 1000 sold = -500 net.
    assert activity.net_shares == -500


def test_yfinance_info_is_cached(fake_yfinance_info):
    provider = YFinanceFinancialProvider()
    calls = {"n": 0}
    original = _FakeInfoTicker.info.fget

    def counting_info(self):
        calls["n"] += 1
        return original(self)

    # profile + metrics + ratios all read .info; with caching, one fetch.
    _FakeInfoTicker.info = property(counting_info)
    try:
        asyncio.run(provider.get_company_profile("aapl"))
        asyncio.run(provider.get_company_metrics("aapl"))
        asyncio.run(provider.get_ratios("aapl"))
    finally:
        _FakeInfoTicker.info = property(original)
    assert calls["n"] == 1


# --------------------------------------------------------------------------
# Finnhub new datasets
# --------------------------------------------------------------------------


class _Transport:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        status, payload = self._responses.pop(0)
        return httpx.Response(status, json=payload)


def _finnhub(transport):
    return FinnhubFinancialProvider(
        api_key="k", client=httpx.AsyncClient(transport=httpx.MockTransport(transport))
    )


def test_finnhub_earnings_history():
    transport = _Transport(
        (200, [
            {"period": "2026-03-31", "actual": 1.5, "estimate": 1.4, "surprise": 0.1, "surprisePercent": 7.1},
            {"period": "2025-12-31", "actual": 2.1, "estimate": 2.0, "surprise": 0.1, "surprisePercent": 5.0},
        ])
    )
    history = asyncio.run(_finnhub(transport).get_earnings_history("aapl"))
    assert len(history.surprises) == 2
    assert history.surprises[0].actual_eps == 1.5
    assert history.surprises[0].surprise_percent == pytest.approx(0.071)


def test_finnhub_insider_activity_maps_codes():
    transport = _Transport(
        (200, {"data": [
            {"name": "Jane Doe", "share": 1000, "transactionPrice": 200, "transactionCode": "S",
             "filingDate": "2026-06-02"},
            {"name": "John Roe", "share": 500, "transactionPrice": 200, "transactionCode": "P",
             "filingDate": "2026-05-02"},
        ]})
    )
    activity = asyncio.run(_finnhub(transport).get_insider_activity("aapl"))
    assert activity.transactions[0].transaction_type == "sell"
    assert activity.transactions[0].value == 200_000
    assert activity.net_shares == -500  # 500 buy - 1000 sell


def test_finnhub_earnings_calendar_picks_upcoming():
    future = (date.today().replace(year=date.today().year + 1)).isoformat()
    transport = _Transport(
        (200, {"earningsCalendar": [
            {"date": future, "epsEstimate": 1.6, "revenueEstimate": 1.0e11},
        ]})
    )
    calendar = asyncio.run(_finnhub(transport).get_earnings_calendar("aapl"))
    assert calendar.next_date is not None
    assert calendar.eps_estimate == 1.6


# --------------------------------------------------------------------------
# merge router + snapshot
# --------------------------------------------------------------------------


class _StubProvider(FinancialProvider):
    def __init__(self, datasets, profile=None, metrics=None, earnings=None):
        self._datasets = frozenset(datasets)
        self._profile = profile
        self._metrics = metrics
        self._earnings = earnings

    @property
    def supported_datasets(self):
        return self._datasets

    async def get_company_profile(self, ticker):
        if self._profile is None:
            raise FinancialProviderRateLimited("no profile")
        return self._profile

    async def get_financial_statements(self, ticker):
        raise FinancialProviderError("n/a")

    async def get_ratios(self, ticker):
        raise FinancialProviderError("n/a")

    async def get_analyst_estimates(self, ticker):
        raise FinancialProviderError("n/a")

    async def get_company_metrics(self, ticker):
        if self._metrics is None:
            raise FinancialProviderRateLimited("no metrics")
        return self._metrics

    async def get_earnings_history(self, ticker):
        return self._earnings

    async def health(self):
        raise NotImplementedError


def test_router_merges_profile_across_providers():
    a = _StubProvider(
        {PROFILE}, profile=CompanyProfile(ticker="AAPL", name="Apple", industry="Electronics")
    )
    b = _StubProvider(
        {PROFILE}, profile=CompanyProfile(ticker="AAPL", name="Apple Inc", sector="Technology")
    )
    router = FinancialProviderRouter([("a", a), ("b", b)])

    merged = asyncio.run(router.get_company_profile("AAPL"))
    assert merged.name == "Apple"          # a first
    assert merged.industry == "Electronics"  # only a
    assert merged.sector == "Technology"     # filled from b


def test_router_merges_earnings_history_lists():
    a = _StubProvider(
        {EARNINGS}, earnings=EarningsHistory(
            ticker="AAPL", surprises=[EarningsSurprise(period="2026-03-31", actual_eps=1.5)]
        )
    )
    b = _StubProvider(
        {EARNINGS}, earnings=EarningsHistory(
            ticker="AAPL",
            surprises=[
                EarningsSurprise(period="2026-03-31", actual_eps=1.5),  # dup
                EarningsSurprise(period="2025-12-31", actual_eps=2.1),
            ],
        )
    )
    router = FinancialProviderRouter([("a", a), ("b", b)])

    merged = asyncio.run(router.get_earnings_history("AAPL"))
    periods = {s.period for s in merged.surprises}
    assert periods == {"2026-03-31", "2025-12-31"}  # deduped union


def test_router_metrics_survives_one_provider_failing():
    good = _StubProvider({METRICS}, metrics=CompanyMetrics(ticker="AAPL", pe_ratio=30.0))
    bad = _StubProvider({METRICS})  # metrics None -> raises
    router = FinancialProviderRouter([("bad", bad), ("good", good)])

    merged = asyncio.run(router.get_company_metrics("AAPL"))
    assert merged.pe_ratio == 30.0


def test_gather_fundamentals_collects_supported_and_warns_on_failure():
    provider = _StubProvider(
        {PROFILE, METRICS},
        profile=CompanyProfile(ticker="AAPL", name="Apple"),
        metrics=None,  # advertises metrics but fails
    )
    snapshot, warnings = asyncio.run(gather_fundamentals(provider, "AAPL"))

    assert isinstance(snapshot, FundamentalsSnapshot)
    assert snapshot.profile is not None and snapshot.profile.name == "Apple"
    assert snapshot.metrics is None            # failed -> None
    assert snapshot.statements is None         # not advertised -> None
    assert any("metrics" in w for w in warnings)


# --------------------------------------------------------------------------
# Yahoo news adapter
# --------------------------------------------------------------------------


def test_yfinance_news_search_parses_new_and_legacy_shapes(monkeypatch):
    from agentic_options_reporter.data.news import YFinanceNewsProvider

    class _NewsTicker:
        def __init__(self, symbol):
            pass

        @property
        def news(self):
            return [
                {  # new nested "content" shape
                    "content": {
                        "title": "Apple beats earnings",
                        "canonicalUrl": {"url": "https://ex.com/a"},
                        "provider": {"displayName": "Reuters"},
                        "pubDate": "2026-06-01T00:00:00Z",
                        "summary": "Solid quarter.",
                    }
                },
                {  # legacy flat shape
                    "title": "Apple unveils product",
                    "link": "https://ex.com/b",
                    "publisher": "Bloomberg",
                    "providerPublishTime": 1_780_000_000,
                },
                {"content": {"title": "", "canonicalUrl": {"url": ""}}},  # dropped (no title/url)
            ]

    monkeypatch.setitem(sys.modules, "yfinance", types.SimpleNamespace(Ticker=_NewsTicker))

    articles = asyncio.run(YFinanceNewsProvider().search("AAPL"))

    assert len(articles) == 2
    assert articles[0].headline == "Apple beats earnings"
    assert articles[0].source == "Reuters"
    assert articles[0].url == "https://ex.com/a"
    assert articles[1].source == "Bloomberg"


def test_yfinance_news_is_company_news_capability():
    from agentic_options_reporter.data.news import YFinanceNewsProvider

    provider = YFinanceNewsProvider()
    from agentic_options_reporter.data.news import COMPANY_NEWS, TOP_HEADLINES

    assert provider.supports(COMPANY_NEWS)
    assert not provider.supports(TOP_HEADLINES)
