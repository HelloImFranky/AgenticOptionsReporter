import pytest

from agentic_options_reporter.data.macro_provider import (
    FredMacroProvider,
    MacroProvider,
    MacroProviderError,
)

from conftest import FakeHttpResponse, FakeRequestsGet


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    with pytest.raises(MacroProviderError):
        FredMacroProvider()


def test_accepts_explicit_api_key(monkeypatch):
    monkeypatch.delenv("FRED_API_KEY", raising=False)
    provider = FredMacroProvider(api_key="test-key")
    assert isinstance(provider, MacroProvider)


def _observations(*values_and_dates):
    return {"observations": [{"value": v, "date": d} for v, d in values_and_dates]}


def test_get_interest_rates(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        FakeHttpResponse(_observations(("5.25", "2026-06-01"))),  # FEDFUNDS
        FakeHttpResponse(_observations(("4.30", "2026-06-01"))),  # DGS10
        FakeHttpResponse(_observations(("4.10", "2026-06-01"))),  # DGS2
    )
    provider = FredMacroProvider(api_key="test-key")
    rates = provider.get_interest_rates()
    assert rates.fed_funds_rate == 5.25
    assert rates.ten_year_yield == 4.30
    assert rates.two_year_yield == 4.10


def test_get_interest_rates_handles_missing_series(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        FakeHttpResponse(_observations()),
        FakeHttpResponse(_observations()),
        FakeHttpResponse(_observations()),
    )
    provider = FredMacroProvider(api_key="test-key")
    rates = provider.get_interest_rates()
    assert rates.fed_funds_rate is None
    assert rates.ten_year_yield is None


def test_get_cpi_computes_yoy_change(fake_requests_module):
    observations = [("310.0", "2026-06-01")] + [("300.0", f"2025-0{i}-01") for i in range(1, 10)]
    # 13 observations requested (limit=periods_per_year+1=13); pad to 13.
    while len(observations) < 13:
        observations.append(("300.0", "2025-01-01"))
    fake_requests_module.get = FakeRequestsGet(FakeHttpResponse(_observations(*observations)))
    provider = FredMacroProvider(api_key="test-key")
    cpi = provider.get_cpi()
    assert cpi.value == 310.0
    assert cpi.yoy_change_pct == pytest.approx((310.0 - 300.0) / 300.0 * 100)


def test_get_cpi_raises_when_no_observations(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(FakeHttpResponse(_observations()))
    provider = FredMacroProvider(api_key="test-key")
    with pytest.raises(MacroProviderError):
        provider.get_cpi()


def test_get_gdp_computes_yoy_growth(fake_requests_module):
    observations = [("23000.0", "2026-04-01"), ("22000.0", "2025-04-01")]
    while len(observations) < 5:
        observations.append(("22000.0", "2024-04-01"))
    fake_requests_module.get = FakeRequestsGet(FakeHttpResponse(_observations(*observations)))
    provider = FredMacroProvider(api_key="test-key")
    gdp = provider.get_gdp()
    assert gdp.value == 23000.0
    assert gdp.yoy_growth_pct == pytest.approx((23000.0 - 22000.0) / 22000.0 * 100)


def test_get_gdp_raises_when_no_observations(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(FakeHttpResponse(_observations()))
    provider = FredMacroProvider(api_key="test-key")
    with pytest.raises(MacroProviderError):
        provider.get_gdp()


def test_get_macro_calendar_returns_empty_list(fake_requests_module):
    provider = FredMacroProvider(api_key="test-key")
    assert provider.get_macro_calendar() == []


def test_fred_drops_missing_value_marker(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        FakeHttpResponse(_observations((".", "2026-06-01"), ("5.25", "2026-05-01")))
    )
    provider = FredMacroProvider(api_key="test-key")
    rate = provider._latest_value("FEDFUNDS")
    assert rate[0] == 5.25


def test_http_failure_raises_macro_provider_error(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        FakeHttpResponse(None, raise_exc=fake_requests_module.exceptions.RequestException("boom"))
    )
    provider = FredMacroProvider(api_key="test-key")
    with pytest.raises(MacroProviderError):
        provider.get_interest_rates()
