import pytest

from agentic_options_reporter.data.financial_provider import (
    AlphaVantageFinancialProvider,
    FinancialProvider,
    FinancialProviderError,
    FinancialProviderRateLimited,
    FinancialProviderRouter,
    FinancialProviderUnsupported,
    FmpFinancialProvider,
    build_financial_provider,
)

from conftest import FakeHttpResponse, FakeRequestsGet


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    with pytest.raises(FinancialProviderError):
        FmpFinancialProvider()


def test_accepts_explicit_api_key(monkeypatch):
    monkeypatch.delenv("FMP_API_KEY", raising=False)
    provider = FmpFinancialProvider(api_key="test-key")
    assert isinstance(provider, FinancialProvider)


def test_get_company_profile(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        FakeHttpResponse(
            [
                {
                    "companyName": "Apple Inc.",
                    "sector": "Technology",
                    "industry": "Consumer Electronics",
                    "mktCap": 3_000_000_000_000,
                    "description": "Makes phones.",
                }
            ]
        )
    )
    provider = FmpFinancialProvider(api_key="test-key")
    profile = provider.get_company_profile("aapl")
    assert profile.ticker == "AAPL"
    assert profile.name == "Apple Inc."
    assert profile.sector == "Technology"
    assert profile.market_cap == 3_000_000_000_000


def test_get_company_profile_handles_empty_response(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(FakeHttpResponse([]))
    provider = FmpFinancialProvider(api_key="test-key")
    profile = provider.get_company_profile("AAPL")
    assert profile.name == ""
    assert profile.market_cap is None


def test_get_financial_statements(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        FakeHttpResponse([{"calendarYear": "2025", "revenue": 400_000_000_000, "netIncome": 100_000_000_000}]),
        FakeHttpResponse([{"operatingCashFlow": 120_000_000_000, "freeCashFlow": 100_000_000_000}]),
    )
    provider = FmpFinancialProvider(api_key="test-key")
    summary = provider.get_financial_statements("AAPL")
    assert summary.period == "2025"
    assert summary.revenue == 400_000_000_000
    assert summary.operating_cash_flow == 120_000_000_000
    assert summary.free_cash_flow == 100_000_000_000


def test_get_ratios(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        FakeHttpResponse(
            [
                {
                    "priceEarningsRatio": 28.5,
                    "priceToBookRatio": 40.1,
                    "debtEquityRatio": 1.5,
                    "currentRatio": 1.1,
                    "returnOnEquity": 0.6,
                    "grossProfitMargin": 0.45,
                    "netProfitMargin": 0.25,
                }
            ]
        )
    )
    provider = FmpFinancialProvider(api_key="test-key")
    ratios = provider.get_ratios("AAPL")
    assert ratios.pe_ratio == 28.5
    assert ratios.return_on_equity == 0.6


def test_get_analyst_estimates(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        FakeHttpResponse(
            [
                {
                    "consensusRating": "Buy",
                    "estimatedPriceTargetAvg": 250.0,
                    "estimatedPriceTargetHigh": 300.0,
                    "estimatedPriceTargetLow": 200.0,
                    "numberAnalystEstimatedRevenue": 30,
                }
            ]
        )
    )
    provider = FmpFinancialProvider(api_key="test-key")
    estimates = provider.get_analyst_estimates("AAPL")
    assert estimates.consensus_rating == "Buy"
    assert estimates.num_analysts == 30


def test_get_analyst_estimates_defaults_when_empty(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(FakeHttpResponse([]))
    provider = FmpFinancialProvider(api_key="test-key")
    estimates = provider.get_analyst_estimates("AAPL")
    assert estimates.consensus_rating == "N/A"
    assert estimates.num_analysts == 0


def test_http_failure_raises_financial_provider_error(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        FakeHttpResponse(None, raise_exc=fake_requests_module.exceptions.RequestException("boom"))
    )
    provider = FmpFinancialProvider(api_key="test-key")
    with pytest.raises(FinancialProviderError):
        provider.get_company_profile("AAPL")


def test_rate_limit_status_raises_financial_provider_rate_limited(fake_requests_module):
    exc = fake_requests_module.exceptions.RequestException("too many requests")
    fake_requests_module.get = FakeRequestsGet(FakeHttpResponse(None, status_code=429, raise_exc=exc))
    provider = FmpFinancialProvider(api_key="test-key")
    with pytest.raises(FinancialProviderRateLimited):
        provider.get_company_profile("AAPL")


# -- AlphaVantageFinancialProvider --


def test_alpha_vantage_requires_api_key(monkeypatch):
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    with pytest.raises(FinancialProviderError):
        AlphaVantageFinancialProvider()


def test_alpha_vantage_get_company_profile(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        FakeHttpResponse(
            {
                "Name": "Apple Inc.",
                "Sector": "TECHNOLOGY",
                "Industry": "CONSUMER ELECTRONICS",
                "MarketCapitalization": "3000000000000",
                "Description": "Makes phones.",
            }
        )
    )
    provider = AlphaVantageFinancialProvider(api_key="test-key")
    profile = provider.get_company_profile("aapl")
    assert profile.ticker == "AAPL"
    assert profile.name == "Apple Inc."
    assert profile.market_cap == 3_000_000_000_000


def test_alpha_vantage_get_financial_statements_computes_free_cash_flow(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        FakeHttpResponse(
            {"annualReports": [{"fiscalDateEnding": "2025-09-30", "totalRevenue": "400000000000", "netIncome": "100000000000"}]}
        ),
        FakeHttpResponse(
            {"annualReports": [{"operatingCashflow": "120000000000", "capitalExpenditures": "20000000000"}]}
        ),
    )
    provider = AlphaVantageFinancialProvider(api_key="test-key")
    summary = provider.get_financial_statements("AAPL")
    assert summary.period == "2025-09-30"
    assert summary.revenue == 400_000_000_000
    assert summary.operating_cash_flow == 120_000_000_000
    assert summary.free_cash_flow == 100_000_000_000


def test_alpha_vantage_get_financial_statements_handles_missing_reports(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        FakeHttpResponse({"annualReports": []}),
        FakeHttpResponse({"annualReports": []}),
    )
    provider = AlphaVantageFinancialProvider(api_key="test-key")
    summary = provider.get_financial_statements("AAPL")
    assert summary.revenue is None
    assert summary.free_cash_flow is None


def test_alpha_vantage_get_ratios_computes_gross_margin(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        FakeHttpResponse(
            {
                "PERatio": "28.5",
                "PriceToBookRatio": "40.1",
                "ReturnOnEquityTTM": "0.6",
                "RevenueTTM": "400000000000",
                "GrossProfitTTM": "180000000000",
                "ProfitMargin": "0.25",
            }
        )
    )
    provider = AlphaVantageFinancialProvider(api_key="test-key")
    ratios = provider.get_ratios("AAPL")
    assert ratios.pe_ratio == 28.5
    assert ratios.debt_to_equity is None
    assert ratios.current_ratio is None
    assert ratios.gross_margin == pytest.approx(0.45)
    assert ratios.net_margin == 0.25


def test_alpha_vantage_get_analyst_estimates_uses_target_price_only(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(FakeHttpResponse({"AnalystTargetPrice": "250.0"}))
    provider = AlphaVantageFinancialProvider(api_key="test-key")
    estimates = provider.get_analyst_estimates("AAPL")
    assert estimates.consensus_rating == "N/A"
    assert estimates.price_target_mean == 250.0
    assert estimates.price_target_high is None
    assert estimates.num_analysts == 0


def test_alpha_vantage_information_field_raises_rate_limited(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        FakeHttpResponse({"Note": "Thank you for using Alpha Vantage! Our standard API rate limit is 25 requests per day."})
    )
    provider = AlphaVantageFinancialProvider(api_key="test-key")
    with pytest.raises(FinancialProviderRateLimited):
        provider.get_company_profile("AAPL")


# -- FinancialProviderRouter --


class _FakeFinancialClient(FinancialProvider):
    def __init__(self, profile=None, error=None):
        self._profile = profile
        self._error = error

    def get_company_profile(self, ticker):
        if self._error is not None:
            raise self._error
        return self._profile

    def get_financial_statements(self, ticker):
        raise NotImplementedError

    def get_ratios(self, ticker):
        raise NotImplementedError

    def get_analyst_estimates(self, ticker):
        raise NotImplementedError


def test_financial_provider_router_rejects_empty_client_list():
    with pytest.raises(FinancialProviderError):
        FinancialProviderRouter([])


def test_financial_provider_router_falls_through_on_retryable_error():
    from agentic_options_reporter.models.schemas import CompanyProfile

    first = _FakeFinancialClient(error=FinancialProviderUnsupported("no profile"))
    second = _FakeFinancialClient(profile=CompanyProfile(ticker="AAPL", name="Apple Inc."))
    router = FinancialProviderRouter([("first", first), ("second", second)])

    result = router.get_company_profile("AAPL")

    assert result.name == "Apple Inc."


def test_build_financial_provider_returns_none_equivalent_when_unconfigured(monkeypatch):
    for var in ("FMP_API_KEY", "ALPHA_VANTAGE_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(FinancialProviderError):
        build_financial_provider()


def test_build_financial_provider_respects_fallback_order_env_var(monkeypatch):
    monkeypatch.setenv("FMP_API_KEY", "test-key")
    monkeypatch.setenv("ALPHA_VANTAGE_API_KEY", "test-key")
    monkeypatch.setenv("AOR_FINANCIAL_PROVIDER_FALLBACK_ORDER", "alphavantage,fmp")

    provider = build_financial_provider()

    assert provider.provider_names == ["alphavantage", "fmp"]
