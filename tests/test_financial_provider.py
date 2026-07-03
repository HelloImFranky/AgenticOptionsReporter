import pytest

from agentic_options_reporter.data.financial_provider import (
    FinancialProvider,
    FinancialProviderError,
    FmpFinancialProvider,
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
