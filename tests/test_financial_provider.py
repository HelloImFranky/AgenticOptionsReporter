import asyncio
from datetime import datetime, timezone

import httpx
import pytest

from agentic_options_reporter.data.async_http import AsyncHttpProviderBase
from agentic_options_reporter.data.financial import (
    AlphaVantageFinancialProvider,
    FinancialProvider,
    FinancialProviderError,
    FinancialProviderRateLimited,
    FinancialProviderRouter,
    FinancialProviderUnavailable,
    FinancialProviderUnsupported,
    FinnhubFinancialProvider,
    FmpFinancialProvider,
    build_financial_provider,
)
from agentic_options_reporter.models.schemas import CompanyProfile, FinancialStatementSummary


@pytest.fixture(autouse=True)
def _reset_provider_cache():
    """The response cache is class-level on purpose (free tiers meter by
    the day, and main.py rebuilds providers per request) — reset it so
    tests stay independent."""
    AsyncHttpProviderBase.clear_shared_cache()
    yield
    AsyncHttpProviderBase.clear_shared_cache()


@pytest.fixture(autouse=True)
def _clear_key_env_vars(monkeypatch):
    for var in ("FMP_API_KEY", "FINNHUB_API_KEY", "ALPHA_VANTAGE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("AOR_FINANCIAL_PROVIDER_FALLBACK_ORDER", raising=False)
    for dataset in ("PROFILE", "STATEMENTS", "RATIOS", "ANALYST_ESTIMATES"):
        monkeypatch.delenv(f"AOR_FINANCIAL_PRIORITY_{dataset}", raising=False)


class RecordingTransport:
    """httpx.MockTransport handler that queues responses and records
    every request for assertions."""

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


_ALL_PROVIDERS = [FmpFinancialProvider, FinnhubFinancialProvider, AlphaVantageFinancialProvider]


@pytest.mark.parametrize("provider_cls", _ALL_PROVIDERS)
def test_provider_requires_api_key(provider_cls):
    with pytest.raises(FinancialProviderError):
        provider_cls()


@pytest.mark.parametrize("provider_cls", _ALL_PROVIDERS)
def test_provider_accepts_explicit_api_key(provider_cls):
    assert isinstance(provider_cls(api_key="test-key"), FinancialProvider)


# -- Financial Modeling Prep --


def test_fmp_get_company_profile():
    transport = RecordingTransport(
        (200, [
            {
                "companyName": "Apple Inc.",
                "sector": "Technology",
                "industry": "Consumer Electronics",
                "mktCap": 3_000_000_000_000,
                "description": "Makes phones.",
            }
        ])
    )
    provider = FmpFinancialProvider(api_key="test-key", client=_client(transport))

    profile = asyncio.run(provider.get_company_profile("aapl"))

    assert profile.ticker == "AAPL"
    assert profile.name == "Apple Inc."
    assert profile.market_cap == 3_000_000_000_000
    assert "apikey" in dict(transport.requests[0].url.params)


def test_fmp_get_financial_statements():
    transport = RecordingTransport(
        (200, [{"calendarYear": "2025", "revenue": 400_000_000_000, "netIncome": 100_000_000_000}]),
        (200, [{"operatingCashFlow": 120_000_000_000, "freeCashFlow": 100_000_000_000}]),
    )
    provider = FmpFinancialProvider(api_key="test-key", client=_client(transport))

    summary = asyncio.run(provider.get_financial_statements("AAPL"))

    assert summary.period == "2025"
    assert summary.revenue == 400_000_000_000
    assert summary.free_cash_flow == 100_000_000_000


def test_fmp_get_ratios():
    transport = RecordingTransport(
        (200, [{"priceEarningsRatio": 28.5, "returnOnEquity": 0.6, "debtEquityRatio": 1.5}])
    )
    provider = FmpFinancialProvider(api_key="test-key", client=_client(transport))

    ratios = asyncio.run(provider.get_ratios("AAPL"))

    assert ratios.pe_ratio == 28.5
    assert ratios.debt_to_equity == 1.5


def test_fmp_get_analyst_estimates():
    transport = RecordingTransport(
        (200, [
            {
                "consensusRating": "Buy",
                "estimatedPriceTargetAvg": 250.0,
                "numberAnalystEstimatedRevenue": 30,
            }
        ])
    )
    provider = FmpFinancialProvider(api_key="test-key", client=_client(transport))

    estimates = asyncio.run(provider.get_analyst_estimates("AAPL"))

    assert estimates.consensus_rating == "Buy"
    assert estimates.num_analysts == 30


def test_fmp_empty_response_falls_back_to_defaults():
    transport = RecordingTransport((200, []))
    provider = FmpFinancialProvider(api_key="test-key", client=_client(transport))

    profile = asyncio.run(provider.get_company_profile("AAPL"))

    assert profile.name == ""
    assert profile.market_cap is None


def test_fmp_rate_limit_error_does_not_leak_api_key():
    """Regression: httpx.HTTPStatusError.__str__ embeds the FULL request
    URL — including the real ?apikey=... — so a naive `f"...: {exc}"` would
    leak the key into whatever consumes this exception's message (logs,
    provider-router failure text, workflow data_warnings, and the
    HTTPException `detail` an API caller sees). _classify_httpx_error must
    scrub it before the message is ever built."""
    transport = RecordingTransport((429, {"error": "slow down"}))
    provider = FmpFinancialProvider(api_key="SUPERSECRET999", client=_client(transport))

    with pytest.raises(FinancialProviderRateLimited) as exc_info:
        asyncio.run(provider.get_company_profile("AAPL"))

    assert "SUPERSECRET999" not in str(exc_info.value)
    assert "apikey=***" in str(exc_info.value)


# -- Finnhub --


def test_finnhub_get_company_profile_scales_market_cap_from_millions():
    transport = RecordingTransport(
        (200, {"name": "Apple Inc", "finnhubIndustry": "Technology", "marketCapitalization": 3_000_000})
    )
    provider = FinnhubFinancialProvider(api_key="test-key", client=_client(transport))

    profile = asyncio.run(provider.get_company_profile("aapl"))

    assert profile.ticker == "AAPL"
    assert profile.industry == "Technology"
    assert profile.market_cap == 3_000_000_000_000


def test_finnhub_get_ratios_converts_percentage_metrics_to_fractions():
    transport = RecordingTransport(
        (200, {
            "metric": {
                "peTTM": 28.5,
                "pb": 40.0,
                "totalDebt/totalEquityQuarterly": 1.5,
                "currentRatioQuarterly": 1.1,
                "roeTTM": 60.0,
                "grossMarginTTM": 45.0,
                "netProfitMarginTTM": 25.0,
            }
        })
    )
    provider = FinnhubFinancialProvider(api_key="test-key", client=_client(transport))

    ratios = asyncio.run(provider.get_ratios("AAPL"))

    assert ratios.pe_ratio == 28.5
    assert ratios.return_on_equity == pytest.approx(0.6)
    assert ratios.gross_margin == pytest.approx(0.45)
    assert ratios.net_margin == pytest.approx(0.25)


def test_finnhub_get_analyst_estimates_derives_consensus_from_counts():
    transport = RecordingTransport(
        (200, [{"strongBuy": 10, "buy": 15, "hold": 5, "sell": 1, "strongSell": 0}])
    )
    provider = FinnhubFinancialProvider(api_key="test-key", client=_client(transport))

    estimates = asyncio.run(provider.get_analyst_estimates("AAPL"))

    assert estimates.consensus_rating == "Buy"
    assert estimates.num_analysts == 31
    assert estimates.price_target_mean is None  # premium endpoint, never guessed


def test_finnhub_get_analyst_estimates_handles_empty_response():
    transport = RecordingTransport((200, []))
    provider = FinnhubFinancialProvider(api_key="test-key", client=_client(transport))

    estimates = asyncio.run(provider.get_analyst_estimates("AAPL"))

    assert estimates.consensus_rating == "N/A"
    assert estimates.num_analysts == 0


def test_finnhub_does_not_advertise_statements():
    provider = FinnhubFinancialProvider(api_key="test-key")
    # Still no statements on the free tier, but Finnhub now also serves the
    # metrics/earnings/earnings_calendar/insider datasets.
    assert provider.supported_datasets == frozenset(
        {"profile", "ratios", "analyst_estimates", "metrics", "earnings", "earnings_calendar", "insider"}
    )
    assert provider.supports("statements") is False
    # The method is still a defensive guard for a direct call.
    with pytest.raises(FinancialProviderUnsupported):
        asyncio.run(provider.get_financial_statements("AAPL"))


@pytest.mark.parametrize(
    "provider_cls,expected",
    [
        (FmpFinancialProvider, {"profile", "statements", "ratios", "analyst_estimates"}),
        (AlphaVantageFinancialProvider, {"profile", "statements", "ratios", "analyst_estimates"}),
    ],
)
def test_full_coverage_providers_advertise_the_core_datasets(provider_cls, expected):
    # FMP and Alpha Vantage cover the original four datasets; the newer
    # metrics/earnings/insider datasets are served by Finnhub and Yahoo.
    provider = provider_cls(api_key="test-key")
    assert provider.supported_datasets == frozenset(expected)


# -- Alpha Vantage --


def test_alpha_vantage_get_company_profile():
    transport = RecordingTransport(
        (200, {
            "Name": "Apple Inc.",
            "Sector": "TECHNOLOGY",
            "Industry": "CONSUMER ELECTRONICS",
            "MarketCapitalization": "3000000000000",
            "Description": "Makes phones.",
        })
    )
    provider = AlphaVantageFinancialProvider(api_key="test-key", client=_client(transport))

    profile = asyncio.run(provider.get_company_profile("aapl"))

    assert profile.ticker == "AAPL"
    assert profile.market_cap == 3_000_000_000_000


def test_alpha_vantage_statements_compute_free_cash_flow():
    transport = RecordingTransport(
        (200, {"annualReports": [{"fiscalDateEnding": "2025-09-30", "totalRevenue": "400000000000", "netIncome": "100000000000"}]}),
        (200, {"annualReports": [{"operatingCashflow": "120000000000", "capitalExpenditures": "20000000000"}]}),
    )
    provider = AlphaVantageFinancialProvider(api_key="test-key", client=_client(transport))

    summary = asyncio.run(provider.get_financial_statements("AAPL"))

    assert summary.free_cash_flow == 100_000_000_000


def test_alpha_vantage_ratios_leave_unavailable_fields_null():
    transport = RecordingTransport(
        (200, {
            "PERatio": "28.5",
            "ReturnOnEquityTTM": "0.6",
            "RevenueTTM": "400000000000",
            "GrossProfitTTM": "180000000000",
            "ProfitMargin": "0.25",
        })
    )
    provider = AlphaVantageFinancialProvider(api_key="test-key", client=_client(transport))

    ratios = asyncio.run(provider.get_ratios("AAPL"))

    assert ratios.debt_to_equity is None
    assert ratios.current_ratio is None
    assert ratios.gross_margin == pytest.approx(0.45)


def test_alpha_vantage_information_note_raises_rate_limited():
    transport = RecordingTransport((200, {"Information": "rate limit reached"}))
    provider = AlphaVantageFinancialProvider(api_key="test-key", client=_client(transport))

    with pytest.raises(FinancialProviderRateLimited):
        asyncio.run(provider.get_company_profile("AAPL"))


# -- Shared adapter behavior --


def test_http_429_raises_rate_limited():
    transport = RecordingTransport((429, {}))
    provider = FmpFinancialProvider(api_key="test-key", client=_client(transport))
    with pytest.raises(FinancialProviderRateLimited):
        asyncio.run(provider.get_company_profile("AAPL"))


def test_http_5xx_raises_unavailable():
    transport = RecordingTransport((503, {}))
    provider = FmpFinancialProvider(api_key="test-key", client=_client(transport))
    with pytest.raises(FinancialProviderUnavailable):
        asyncio.run(provider.get_company_profile("AAPL"))


def test_identical_requests_are_served_from_cache_across_instances():
    transport = RecordingTransport((200, []))

    first = FmpFinancialProvider(api_key="test-key", client=_client(transport))
    asyncio.run(first.get_company_profile("AAPL"))
    second = FmpFinancialProvider(api_key="test-key", client=_client(transport))
    asyncio.run(second.get_company_profile("AAPL"))

    assert len(transport.requests) == 1


def test_health_reports_healthy_with_latency():
    transport = RecordingTransport((200, [{"companyName": "Apple Inc."}]))
    provider = FmpFinancialProvider(api_key="test-key", client=_client(transport))

    health = asyncio.run(provider.health())

    assert health.healthy is True
    assert health.provider == "Financial Modeling Prep"
    assert health.latency_ms is not None


def test_health_reports_unhealthy_instead_of_raising():
    transport = RecordingTransport((503, {}))
    provider = FmpFinancialProvider(api_key="test-key", client=_client(transport))

    health = asyncio.run(provider.health())

    assert health.healthy is False
    assert "unavailable" in health.detail.lower()


# -- FinancialProviderRouter --


class _StubFinancialProvider(FinancialProvider):
    def __init__(self, datasets, profile=None, statements=None, error=None, name="stub"):
        self._datasets = frozenset(datasets)
        self._profile = profile
        self._statements = statements
        self._error = error
        self._name = name
        self.calls: list[str] = []

    @property
    def supported_datasets(self):
        return self._datasets

    async def get_company_profile(self, ticker):
        self.calls.append("get_company_profile")
        if self._error is not None:
            raise self._error
        return self._profile

    async def get_financial_statements(self, ticker):
        self.calls.append("get_financial_statements")
        if self._error is not None:
            raise self._error
        return self._statements

    async def get_ratios(self, ticker):
        raise NotImplementedError

    async def get_analyst_estimates(self, ticker):
        raise NotImplementedError

    async def health(self):
        from agentic_options_reporter.data.financial import ProviderHealth

        return ProviderHealth(
            provider=self._name,
            healthy=self._error is None,
            detail="" if self._error is None else str(self._error),
            checked_at=datetime.now(timezone.utc),
        )


def test_router_rejects_empty_client_list():
    with pytest.raises(FinancialProviderError):
        FinancialProviderRouter([])


def test_router_only_queries_providers_that_advertise_the_dataset():
    """The Finnhub case: it doesn't advertise statements, so it's never
    asked for them — only FMP (which does) is queried."""
    finnhub = _StubFinancialProvider(
        {"profile", "ratios", "analyst_estimates"},
        profile=CompanyProfile(ticker="AAPL", name="From Finnhub"),
        name="finnhub",
    )
    fmp = _StubFinancialProvider(
        {"profile", "statements"},
        profile=CompanyProfile(ticker="AAPL", name="From FMP"),
        statements=FinancialStatementSummary(ticker="AAPL", period="2025"),
        name="fmp",
    )
    router = FinancialProviderRouter([("finnhub", finnhub), ("fmp", fmp)])

    statements = asyncio.run(router.get_financial_statements("AAPL"))

    assert statements.period == "2025"
    assert finnhub.calls == []  # never asked — it doesn't advertise statements


def test_router_raises_unsupported_when_no_provider_serves_dataset():
    finnhub = _StubFinancialProvider({"profile", "ratios"}, name="finnhub")
    router = FinancialProviderRouter([("finnhub", finnhub)])

    with pytest.raises(FinancialProviderUnsupported):
        asyncio.run(router.get_financial_statements("AAPL"))


def test_router_supported_datasets_is_union():
    finnhub = _StubFinancialProvider({"profile", "ratios", "analyst_estimates"})
    fmp = _StubFinancialProvider({"profile", "statements"})
    router = FinancialProviderRouter([("finnhub", finnhub), ("fmp", fmp)])
    assert router.supported_datasets == frozenset(
        {"profile", "statements", "ratios", "analyst_estimates"}
    )


def test_router_falls_through_on_rate_limit():
    limited = _StubFinancialProvider(
        {"profile"}, error=FinancialProviderRateLimited("429"), name="limited"
    )
    healthy = _StubFinancialProvider(
        {"profile"}, profile=CompanyProfile(ticker="AAPL", name="Apple Inc."), name="healthy"
    )
    router = FinancialProviderRouter([("limited", limited), ("healthy", healthy)])

    profile = asyncio.run(router.get_company_profile("AAPL"))

    assert profile.name == "Apple Inc."


def test_router_applies_per_dataset_priority_override(monkeypatch):
    monkeypatch.setenv("AOR_FINANCIAL_PRIORITY_PROFILE", "second,first")
    first = _StubFinancialProvider(
        {"profile"}, profile=CompanyProfile(ticker="AAPL", name="First"), name="first"
    )
    second = _StubFinancialProvider(
        {"profile"}, profile=CompanyProfile(ticker="AAPL", name="Second"), name="second"
    )
    router = FinancialProviderRouter([("first", first), ("second", second)])

    profile = asyncio.run(router.get_company_profile("AAPL"))

    assert profile.name == "Second"


def test_router_health_aggregates():
    healthy = _StubFinancialProvider({"profile"}, name="up")
    unhealthy = _StubFinancialProvider(
        {"profile"}, error=FinancialProviderUnavailable("down"), name="down"
    )
    router = FinancialProviderRouter([("up", healthy), ("down", unhealthy)])

    health = asyncio.run(router.health())

    assert health.healthy is True
    assert "up: ok" in health.detail


# -- build_financial_provider --


def test_build_financial_provider_defaults_to_keyless_yahoo():
    # Yahoo Finance is keyless, so fundamentals are always available even
    # with no API keys configured — the router is never empty.
    provider = build_financial_provider()
    assert provider.provider_names == ["yfinance"]


def test_build_financial_provider_orders_configured_providers(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")

    provider = build_financial_provider()

    # Keyless Yahoo joins after the key-gated providers in the default order.
    assert provider.provider_names == ["fmp", "finnhub", "yfinance"]


def test_build_financial_provider_respects_fallback_order_env_var(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key")
    monkeypatch.setenv("AOR_FINANCIAL_PROVIDER_FALLBACK_ORDER", "alphavantage,fmp")

    provider = build_financial_provider()

    assert provider.provider_names == ["alphavantage", "fmp"]
