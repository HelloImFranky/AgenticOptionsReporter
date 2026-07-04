import asyncio

import httpx
import pytest

from agentic_options_reporter.data.sec_provider import (
    SecEdgarProvider,
    SECProvider,
    SecProviderError,
    SecProviderRateLimited,
    SecProviderTimeout,
    SecProviderUnavailable,
)


@pytest.fixture(autouse=True)
def _reset_sec_cache():
    """The response cache is class-level (shared across the async HTTP
    adapters) — reset it so tests stay independent."""
    SecEdgarProvider.clear_shared_cache()
    yield
    SecEdgarProvider.clear_shared_cache()


class RecordingTransport:
    """httpx.MockTransport handler that queues responses and records every
    request for assertions."""

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


def _ticker_map():
    return (200, {"0": {"ticker": "AAPL", "cik_str": 320193}})


def _submissions():
    return (
        200,
        {
            "filings": {
                "recent": {
                    "form": ["10-K", "8-K", "10-Q"],
                    "filingDate": ["2026-01-01", "2026-02-01", "2025-11-01"],
                    "accessionNumber": [
                        "0000320193-26-000001",
                        "0000320193-26-000002",
                        "0000320193-25-000003",
                    ],
                    "primaryDocument": ["10k.htm", "8k.htm", "10q.htm"],
                }
            }
        },
    )


def test_no_api_key_required():
    provider = SecEdgarProvider()
    assert isinstance(provider, SECProvider)


def test_uses_default_user_agent_when_unset(monkeypatch):
    monkeypatch.delenv("SEC_EDGAR_USER_AGENT", raising=False)
    provider = SecEdgarProvider()
    assert provider._user_agent == SecEdgarProvider.DEFAULT_USER_AGENT


def test_user_agent_header_is_sent():
    transport = RecordingTransport(_ticker_map(), _submissions())
    provider = SecEdgarProvider(user_agent="test-agent", client=_client(transport))

    asyncio.run(provider.get_recent_filings("AAPL"))

    assert transport.requests[0].headers["user-agent"] == "test-agent"


def test_get_recent_filings():
    transport = RecordingTransport(_ticker_map(), _submissions())
    provider = SecEdgarProvider(user_agent="test-agent", client=_client(transport))

    filings = asyncio.run(provider.get_recent_filings("AAPL", limit=10))

    assert len(filings) == 3
    assert filings[0].form_type == "10-K"
    assert filings[0].ticker == "AAPL"
    assert "0000320193260000" in filings[0].url or "320193" in filings[0].url


def test_get_recent_filings_respects_limit():
    transport = RecordingTransport(_ticker_map(), _submissions())
    provider = SecEdgarProvider(user_agent="test-agent", client=_client(transport))

    filings = asyncio.run(provider.get_recent_filings("AAPL", limit=1))

    assert len(filings) == 1


def test_ticker_map_is_cached_on_the_instance():
    transport = RecordingTransport(_ticker_map(), _submissions())
    provider = SecEdgarProvider(user_agent="test-agent", client=_client(transport))

    asyncio.run(provider.get_recent_filings("AAPL"))
    asyncio.run(provider.get_recent_filings("AAPL"))

    # Ticker map endpoint should only be hit once across both calls (the
    # instance memoizes the parsed map; the shared cache covers the rest).
    ticker_map_calls = [r for r in transport.requests if "company_tickers" in str(r.url)]
    assert len(ticker_map_calls) == 1


def test_unknown_ticker_raises():
    transport = RecordingTransport(_ticker_map())
    provider = SecEdgarProvider(user_agent="test-agent", client=_client(transport))

    with pytest.raises(SecProviderError):
        asyncio.run(provider.get_recent_filings("UNKNOWN"))


def test_get_10k_returns_matching_filing():
    transport = RecordingTransport(_ticker_map(), _submissions())
    provider = SecEdgarProvider(user_agent="test-agent", client=_client(transport))

    filing = asyncio.run(provider.get_10k("AAPL"))

    assert filing is not None
    assert filing.form_type == "10-K"


def test_get_8k_returns_none_when_absent():
    submissions = (
        200,
        {
            "filings": {
                "recent": {
                    "form": ["10-K"],
                    "filingDate": ["2026-01-01"],
                    "accessionNumber": ["0000320193-26-000001"],
                    "primaryDocument": ["10k.htm"],
                }
            }
        },
    )
    transport = RecordingTransport(_ticker_map(), submissions)
    provider = SecEdgarProvider(user_agent="test-agent", client=_client(transport))

    assert asyncio.run(provider.get_8k("AAPL")) is None


def test_http_429_raises_rate_limited():
    transport = RecordingTransport((429, {}))
    provider = SecEdgarProvider(user_agent="test-agent", client=_client(transport))
    with pytest.raises(SecProviderRateLimited):
        asyncio.run(provider.get_recent_filings("AAPL"))


def test_http_5xx_raises_unavailable():
    transport = RecordingTransport((503, {}))
    provider = SecEdgarProvider(user_agent="test-agent", client=_client(transport))
    with pytest.raises(SecProviderUnavailable):
        asyncio.run(provider.get_recent_filings("AAPL"))


def test_timeout_raises_sec_provider_timeout():
    transport = RecordingTransport(httpx.ReadTimeout("timed out"))
    provider = SecEdgarProvider(user_agent="test-agent", client=_client(transport))
    with pytest.raises(SecProviderTimeout):
        asyncio.run(provider.get_recent_filings("AAPL"))


def test_connect_error_raises_unavailable():
    transport = RecordingTransport(httpx.ConnectError("refused"))
    provider = SecEdgarProvider(user_agent="test-agent", client=_client(transport))
    with pytest.raises(SecProviderUnavailable):
        asyncio.run(provider.get_recent_filings("AAPL"))


# -- health() --


def test_health_reports_healthy_with_latency():
    transport = RecordingTransport(_ticker_map())
    provider = SecEdgarProvider(user_agent="test-agent", client=_client(transport))

    health = asyncio.run(provider.health())

    assert health.healthy is True
    assert health.provider == "SEC EDGAR"
    assert health.latency_ms is not None


def test_health_reports_unhealthy_instead_of_raising():
    transport = RecordingTransport((503, {}))
    provider = SecEdgarProvider(user_agent="test-agent", client=_client(transport))

    health = asyncio.run(provider.health())

    assert health.healthy is False
    assert "unavailable" in health.detail.lower()
