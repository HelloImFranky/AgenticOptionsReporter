import pytest

from agentic_options_reporter.data.sec_provider import SecEdgarProvider, SECProvider, SecProviderError

from conftest import FakeHttpResponse, FakeRequestsGet


def test_no_api_key_required():
    provider = SecEdgarProvider()
    assert isinstance(provider, SECProvider)


def test_uses_default_user_agent_when_unset(monkeypatch):
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
    provider = SecEdgarProvider()
    assert provider._user_agent == SecEdgarProvider.DEFAULT_USER_AGENT


def _ticker_map_response():
    return FakeHttpResponse({"0": {"ticker": "AAPL", "cik_str": 320193}})


def _submissions_response():
    return FakeHttpResponse(
        {
            "filings": {
                "recent": {
                    "form": ["10-K", "8-K", "10-Q"],
                    "filingDate": ["2026-01-01", "2026-02-01", "2025-11-01"],
                    "accessionNumber": ["0000320193-26-000001", "0000320193-26-000002", "0000320193-25-000003"],
                    "primaryDocument": ["10k.htm", "8k.htm", "10q.htm"],
                }
            }
        }
    )


def test_get_recent_filings(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(_ticker_map_response(), _submissions_response())
    provider = SecEdgarProvider(user_agent="test-agent")
    filings = provider.get_recent_filings("AAPL", limit=10)

    assert len(filings) == 3
    assert filings[0].form_type == "10-K"
    assert filings[0].ticker == "AAPL"
    assert "0000320193260000" in filings[0].url or "320193" in filings[0].url


def test_get_recent_filings_respects_limit(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(_ticker_map_response(), _submissions_response())
    provider = SecEdgarProvider(user_agent="test-agent")
    filings = provider.get_recent_filings("AAPL", limit=1)
    assert len(filings) == 1


def test_ticker_map_is_cached(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        _ticker_map_response(), _submissions_response(), _submissions_response()
    )
    provider = SecEdgarProvider(user_agent="test-agent")
    provider.get_recent_filings("AAPL")
    provider.get_recent_filings("AAPL")
    # Ticker map endpoint should only be hit once across both calls.
    ticker_map_calls = [c for c in fake_requests_module.get.calls if "company_tickers" in c["url"]]
    assert len(ticker_map_calls) == 1


def test_unknown_ticker_raises(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(_ticker_map_response())
    provider = SecEdgarProvider(user_agent="test-agent")
    with pytest.raises(SecProviderError):
        provider.get_recent_filings("UNKNOWN")


def test_get_10k_returns_matching_filing(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(_ticker_map_response(), _submissions_response())
    provider = SecEdgarProvider(user_agent="test-agent")
    filing = provider.get_10k("AAPL")
    assert filing is not None
    assert filing.form_type == "10-K"


def test_get_8k_returns_none_when_absent(fake_requests_module):
    response = FakeHttpResponse(
        {
            "filings": {
                "recent": {
                    "form": ["10-K"],
                    "filingDate": ["2026-01-01"],
                    "accessionNumber": ["0000320193-26-000001"],
                    "primaryDocument": ["10k.htm"],
                }
            }
        }
    )
    fake_requests_module.get = FakeRequestsGet(_ticker_map_response(), response)
    provider = SecEdgarProvider(user_agent="test-agent")
    assert provider.get_8k("AAPL") is None


def test_http_failure_raises_sec_provider_error(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        FakeHttpResponse(None, raise_exc=fake_requests_module.exceptions.RequestException("boom"))
    )
    provider = SecEdgarProvider(user_agent="test-agent")
    with pytest.raises(SecProviderError):
        provider.get_recent_filings("AAPL")
