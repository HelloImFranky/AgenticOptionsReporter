import asyncio
import json
from datetime import datetime, timezone

import httpx
import pytest

from agentic_options_reporter.data.news import (
    COMPANY_NEWS,
    GENERAL_NEWS,
    TOP_HEADLINES,
    AlphaVantageNewsProvider,
    FinnhubNewsProvider,
    GNewsProvider,
    GuardianNewsProvider,
    HackerNewsProvider,
    NewsApiOrgProvider,
    NewsDataProvider,
    NewsProvider,
    NewsProviderError,
    NewsProviderRateLimited,
    NewsProviderRouter,
    NewsProviderTimeout,
    NewsProviderUnavailable,
    build_news_provider,
)
from agentic_options_reporter.data.news.base import _HttpNewsProvider
from agentic_options_reporter.models.schemas import NewsArticle


@pytest.fixture(autouse=True)
def _reset_news_cache():
    """The response cache is class-level on purpose (free tiers meter by
    the day, and main.py rebuilds providers per request) — reset it so
    tests stay independent."""
    _HttpNewsProvider.clear_shared_cache()
    yield
    _HttpNewsProvider.clear_shared_cache()


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


_ALL_KEY_ENV_VARS = (
    "FINNHUB_API_KEY",
    "ALPHA_VANTAGE_API_KEY",
    "NEWSAPI_API_KEY",
    "NEWSDATA_API_KEY",
    "GUARDIAN_API_KEY",
    "GNEWS_API_KEY",
)


@pytest.fixture(autouse=True)
def _clear_news_key_env_vars(monkeypatch):
    for var in _ALL_KEY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("AOR_NEWS_PROVIDER_FALLBACK_ORDER", raising=False)


_KEYED_PROVIDERS = [
    FinnhubNewsProvider,
    AlphaVantageNewsProvider,
    NewsApiOrgProvider,
    NewsDataProvider,
    GuardianNewsProvider,
    GNewsProvider,
]


@pytest.mark.parametrize("provider_cls", _KEYED_PROVIDERS)
def test_keyed_provider_requires_api_key(provider_cls):
    with pytest.raises(NewsProviderError):
        provider_cls()


@pytest.mark.parametrize("provider_cls", _KEYED_PROVIDERS)
def test_keyed_provider_accepts_explicit_api_key(provider_cls):
    assert isinstance(provider_cls(api_key="test-key"), NewsProvider)


def test_hackernews_needs_no_api_key():
    assert isinstance(HackerNewsProvider(), NewsProvider)


# -- Finnhub --


def test_finnhub_search_maps_articles():
    transport = RecordingTransport(
        (200, [
            {
                "headline": "Company beats earnings",
                "source": "Reuters",
                "url": "https://example.com/a",
                "datetime": 1700000000,
                "summary": "Solid quarter.",
            }
        ])
    )
    provider = FinnhubNewsProvider(api_key="test-key", client=_client(transport))

    articles = asyncio.run(provider.search("aapl", limit=5))

    assert len(articles) == 1
    assert articles[0].headline == "Company beats earnings"
    assert articles[0].source == "Reuters"
    params = transport.requests[0].url.params
    assert params["symbol"] == "AAPL"
    assert "from" in params and "to" in params


def test_finnhub_search_respects_date_range():
    transport = RecordingTransport((200, []))
    provider = FinnhubNewsProvider(api_key="test-key", client=_client(transport))

    asyncio.run(
        provider.search(
            "AAPL",
            start_date=datetime(2026, 6, 1, tzinfo=timezone.utc),
            end_date=datetime(2026, 6, 30, tzinfo=timezone.utc),
        )
    )

    params = transport.requests[0].url.params
    assert params["from"] == "2026-06-01"
    assert params["to"] == "2026-06-30"


def test_finnhub_top_headlines_maps_unknown_category_to_general():
    transport = RecordingTransport((200, []))
    provider = FinnhubNewsProvider(api_key="test-key", client=_client(transport))

    asyncio.run(provider.top_headlines(category="business"))

    assert transport.requests[0].url.params["category"] == "general"


# -- Alpha Vantage --


def test_alpha_vantage_search_maps_articles():
    transport = RecordingTransport(
        (200, {
            "feed": [
                {
                    "title": "Company beats earnings",
                    "source": "Reuters",
                    "url": "https://example.com/a",
                    "time_published": "20260703T120000",
                    "summary": "Solid quarter.",
                }
            ]
        })
    )
    provider = AlphaVantageNewsProvider(api_key="test-key", client=_client(transport))

    articles = asyncio.run(provider.search("aapl"))

    assert articles[0].headline == "Company beats earnings"
    assert transport.requests[0].url.params["tickers"] == "AAPL"


def test_alpha_vantage_information_note_raises_rate_limited():
    transport = RecordingTransport((200, {"Information": "rate limit reached"}))
    provider = AlphaVantageNewsProvider(api_key="test-key", client=_client(transport))

    with pytest.raises(NewsProviderRateLimited):
        asyncio.run(provider.search("AAPL"))


# -- NewsAPI --


def test_newsapi_search_maps_articles():
    transport = RecordingTransport(
        (200, {
            "articles": [
                {
                    "title": "Company beats earnings",
                    "source": {"name": "Reuters"},
                    "url": "https://example.com/a",
                    "publishedAt": "2026-07-03T12:00:00Z",
                    "description": "Solid quarter.",
                }
            ]
        })
    )
    provider = NewsApiOrgProvider(api_key="test-key", client=_client(transport))

    articles = asyncio.run(provider.search("AAPL", language="en"))

    assert articles[0].source == "Reuters"
    request = transport.requests[0]
    assert request.headers["X-Api-Key"] == "test-key"
    assert request.url.params["language"] == "en"


def test_newsapi_top_headlines_defaults_to_business():
    transport = RecordingTransport((200, {"articles": []}))
    provider = NewsApiOrgProvider(api_key="test-key", client=_client(transport))

    asyncio.run(provider.top_headlines())

    assert transport.requests[0].url.params["category"] == "business"


# -- NewsData.io --


def test_newsdata_search_maps_articles():
    transport = RecordingTransport(
        (200, {
            "status": "success",
            "results": [
                {
                    "title": "Company beats earnings",
                    "source_id": "reuters",
                    "link": "https://example.com/a",
                    "pubDate": "2026-07-03 12:00:00",
                    "description": "Solid quarter.",
                }
            ],
        })
    )
    provider = NewsDataProvider(api_key="test-key", client=_client(transport))

    articles = asyncio.run(provider.search("AAPL"))

    assert articles[0].headline == "Company beats earnings"
    assert articles[0].source == "reuters"
    assert transport.requests[0].url.params["q"] == "AAPL"


def test_newsdata_search_ignores_dates_on_free_tier():
    transport = RecordingTransport((200, {"results": []}))
    provider = NewsDataProvider(api_key="test-key", client=_client(transport))

    asyncio.run(
        provider.search("AAPL", start_date=datetime(2026, 6, 1, tzinfo=timezone.utc))
    )

    params = transport.requests[0].url.params
    assert "from_date" not in params and "from" not in params


# -- The Guardian --


def test_guardian_search_maps_articles():
    transport = RecordingTransport(
        (200, {
            "response": {
                "results": [
                    {
                        "webTitle": "Company beats earnings",
                        "webUrl": "https://theguardian.com/a",
                        "webPublicationDate": "2026-07-03T12:00:00Z",
                        "fields": {"trailText": "Solid quarter."},
                    }
                ]
            }
        })
    )
    provider = GuardianNewsProvider(api_key="test-key", client=_client(transport))

    articles = asyncio.run(
        provider.search(
            "AAPL",
            start_date=datetime(2026, 6, 1, tzinfo=timezone.utc),
            end_date=datetime(2026, 6, 30, tzinfo=timezone.utc),
        )
    )

    assert articles[0].source == "The Guardian"
    assert articles[0].summary == "Solid quarter."
    params = transport.requests[0].url.params
    assert params["from-date"] == "2026-06-01"
    assert params["to-date"] == "2026-06-30"


def test_guardian_top_headlines_uses_section():
    transport = RecordingTransport((200, {"response": {"results": []}}))
    provider = GuardianNewsProvider(api_key="test-key", client=_client(transport))

    asyncio.run(provider.top_headlines(category="technology"))

    assert transport.requests[0].url.params["section"] == "technology"


# -- GNews --


def test_gnews_search_maps_articles():
    transport = RecordingTransport(
        (200, {
            "articles": [
                {
                    "title": "Company beats earnings",
                    "description": "Solid quarter.",
                    "url": "https://example.com/a",
                    "publishedAt": "2026-07-03T12:00:00Z",
                    "source": {"name": "Reuters"},
                }
            ]
        })
    )
    provider = GNewsProvider(api_key="test-key", client=_client(transport))

    articles = asyncio.run(provider.search("AAPL", language="en", limit=5))

    assert articles[0].headline == "Company beats earnings"
    params = transport.requests[0].url.params
    assert params["lang"] == "en"
    assert params["max"] == "5"


# -- Hacker News --


def test_hackernews_search_maps_hits():
    transport = RecordingTransport(
        (200, {
            "hits": [
                {
                    "title": "Company open-sources model",
                    "url": "https://example.com/a",
                    "created_at": "2026-07-03T12:00:00Z",
                    "objectID": "1",
                }
            ]
        })
    )
    provider = HackerNewsProvider(client=_client(transport))

    articles = asyncio.run(provider.search("AAPL"))

    assert articles[0].source == "Hacker News"
    assert transport.requests[0].url.params["tags"] == "story"


def test_hackernews_linkless_story_falls_back_to_thread_url():
    transport = RecordingTransport(
        (200, {"hits": [{"title": "Ask HN: thoughts?", "url": None, "created_at": "", "objectID": "42"}]})
    )
    provider = HackerNewsProvider(client=_client(transport))

    articles = asyncio.run(provider.search("AAPL"))

    assert articles[0].url == "https://news.ycombinator.com/item?id=42"


def test_hackernews_search_applies_date_filters():
    transport = RecordingTransport((200, {"hits": []}))
    provider = HackerNewsProvider(client=_client(transport))

    start = datetime(2026, 6, 1, tzinfo=timezone.utc)
    asyncio.run(provider.search("AAPL", start_date=start))

    assert f"created_at_i>{int(start.timestamp())}" in transport.requests[0].url.params["numericFilters"]


def test_hackernews_top_headlines_uses_front_page():
    transport = RecordingTransport((200, {"hits": []}))
    provider = HackerNewsProvider(client=_client(transport))

    asyncio.run(provider.top_headlines())

    assert transport.requests[0].url.params["tags"] == "front_page"


# -- Error classification (shared _HttpNewsProvider behavior) --


def test_http_429_raises_rate_limited():
    transport = RecordingTransport((429, {}))
    provider = HackerNewsProvider(client=_client(transport))
    with pytest.raises(NewsProviderRateLimited):
        asyncio.run(provider.search("AAPL"))


def test_http_5xx_raises_unavailable():
    transport = RecordingTransport((503, {}))
    provider = HackerNewsProvider(client=_client(transport))
    with pytest.raises(NewsProviderUnavailable):
        asyncio.run(provider.search("AAPL"))


def test_timeout_raises_news_provider_timeout():
    transport = RecordingTransport(httpx.ReadTimeout("timed out"))
    provider = HackerNewsProvider(client=_client(transport))
    with pytest.raises(NewsProviderTimeout):
        asyncio.run(provider.search("AAPL"))


def test_connect_error_raises_unavailable():
    transport = RecordingTransport(httpx.ConnectError("refused"))
    provider = HackerNewsProvider(client=_client(transport))
    with pytest.raises(NewsProviderUnavailable):
        asyncio.run(provider.search("AAPL"))


def test_http_4xx_raises_plain_provider_error():
    transport = RecordingTransport((404, {}))
    provider = HackerNewsProvider(client=_client(transport))
    with pytest.raises(NewsProviderError) as excinfo:
        asyncio.run(provider.search("AAPL"))
    assert not isinstance(excinfo.value, (NewsProviderRateLimited, NewsProviderUnavailable))


# -- Response cache --


def test_identical_requests_are_served_from_cache_across_instances():
    """Free tiers meter by the day and main.py rebuilds providers per
    request, so a regenerate click must not re-spend quota."""
    transport = RecordingTransport((200, {"hits": []}))

    first = HackerNewsProvider(client=_client(transport))
    asyncio.run(first.search("AAPL"))
    second = HackerNewsProvider(client=_client(transport))
    asyncio.run(second.search("AAPL"))

    assert len(transport.requests) == 1


def test_different_requests_are_not_cache_collided():
    transport = RecordingTransport((200, {"hits": []}), (200, {"hits": []}))
    provider = HackerNewsProvider(client=_client(transport))

    asyncio.run(provider.search("AAPL"))
    asyncio.run(provider.search("MSFT"))

    assert len(transport.requests) == 2


# -- health() --


def test_health_reports_healthy_with_latency():
    transport = RecordingTransport((200, {"hits": []}))
    provider = HackerNewsProvider(client=_client(transport))

    health = asyncio.run(provider.health())

    assert health.healthy is True
    assert health.provider == "Hacker News"
    assert health.latency_ms is not None


def test_health_reports_unhealthy_instead_of_raising():
    transport = RecordingTransport((503, {}))
    provider = HackerNewsProvider(client=_client(transport))

    health = asyncio.run(provider.health())

    assert health.healthy is False
    assert "unavailable" in health.detail.lower()


# -- NewsProviderRouter --


class _StubNewsProvider(NewsProvider):
    def __init__(self, articles=None, error=None, name="stub", capabilities=frozenset({GENERAL_NEWS, TOP_HEADLINES})):
        self._articles = articles or []
        self._error = error
        self._name = name
        self._capabilities = capabilities

    @property
    def capabilities(self):
        return self._capabilities

    async def search(self, query, start_date=None, end_date=None, language="en", limit=20):
        if self._error is not None:
            raise self._error
        return self._articles

    async def top_headlines(self, category=None, limit=20):
        if self._error is not None:
            raise self._error
        return self._articles

    async def health(self):
        from agentic_options_reporter.data.news import ProviderHealth

        return ProviderHealth(
            provider=self._name,
            healthy=self._error is None,
            detail="" if self._error is None else str(self._error),
            checked_at=datetime.now(timezone.utc),
        )


def _article() -> NewsArticle:
    return NewsArticle(
        headline="x", source="s", url="u", published_at=datetime.now(timezone.utc)
    )


def test_router_rejects_empty_client_list():
    with pytest.raises(NewsProviderError):
        NewsProviderRouter([])


def test_router_returns_first_success():
    first = _StubNewsProvider(articles=[_article()])
    second = _StubNewsProvider(articles=[])
    router = NewsProviderRouter([("first", first), ("second", second)])

    articles = asyncio.run(router.search("AAPL"))

    assert len(articles) == 1


def test_router_falls_through_on_retryable_error():
    first = _StubNewsProvider(error=NewsProviderRateLimited("429"))
    second = _StubNewsProvider(articles=[_article()])
    router = NewsProviderRouter([("first", first), ("second", second)])

    articles = asyncio.run(router.search("AAPL"))

    assert len(articles) == 1


def test_router_raises_with_all_failures_when_every_provider_fails():
    first = _StubNewsProvider(error=NewsProviderRateLimited("429"))
    second = _StubNewsProvider(error=NewsProviderUnavailable("down"))
    router = NewsProviderRouter([("first", first), ("second", second)])

    with pytest.raises(NewsProviderError, match="first:.*429.*second:.*down"):
        asyncio.run(router.search("AAPL"))


def test_ticker_specialists_advertise_company_news():
    assert FinnhubNewsProvider(api_key="k").supports(COMPANY_NEWS)
    assert AlphaVantageNewsProvider(api_key="k").supports(COMPANY_NEWS)


def test_general_providers_do_not_advertise_company_news():
    assert not HackerNewsProvider().supports(COMPANY_NEWS)
    assert HackerNewsProvider().supports(GENERAL_NEWS)


def test_router_search_prefers_company_news_providers():
    """A ticker search prioritizes COMPANY_NEWS providers even when a
    general-news provider is configured earlier in the fallback order."""
    called: list[str] = []

    class _RecordingStub(_StubNewsProvider):
        async def search(self, query, start_date=None, end_date=None, language="en", limit=20):
            called.append(self._name)
            return await super().search(query, start_date, end_date, language, limit)

    general = _RecordingStub(
        articles=[_article()], name="general", capabilities=frozenset({GENERAL_NEWS, TOP_HEADLINES})
    )
    specialist = _RecordingStub(
        articles=[_article()], name="specialist", capabilities=frozenset({COMPANY_NEWS, TOP_HEADLINES})
    )
    router = NewsProviderRouter([("general", general), ("specialist", specialist)])

    asyncio.run(router.search("AAPL"))

    assert called == ["specialist"]  # specialist tried first despite later order


def test_router_falls_back_to_general_when_specialist_fails():
    general = _StubNewsProvider(
        articles=[_article()], name="general", capabilities=frozenset({GENERAL_NEWS, TOP_HEADLINES})
    )
    specialist = _StubNewsProvider(
        error=NewsProviderRateLimited("429"),
        name="specialist",
        capabilities=frozenset({COMPANY_NEWS, TOP_HEADLINES}),
    )
    router = NewsProviderRouter([("general", general), ("specialist", specialist)])

    articles = asyncio.run(router.search("AAPL"))

    assert len(articles) == 1  # general-news provider still answered


def test_router_capabilities_is_union_of_clients():
    general = _StubNewsProvider(capabilities=frozenset({GENERAL_NEWS, TOP_HEADLINES}))
    specialist = _StubNewsProvider(capabilities=frozenset({COMPANY_NEWS, TOP_HEADLINES}))
    router = NewsProviderRouter([("general", general), ("specialist", specialist)])

    assert router.capabilities == frozenset({COMPANY_NEWS, GENERAL_NEWS, TOP_HEADLINES})
    assert router.supports(COMPANY_NEWS)


def test_router_health_aggregates_and_is_healthy_if_any_provider_is():
    healthy = _StubNewsProvider(name="up")
    unhealthy = _StubNewsProvider(error=NewsProviderUnavailable("down"), name="down")
    router = NewsProviderRouter([("up", healthy), ("down", unhealthy)])

    health = asyncio.run(router.health())

    assert health.healthy is True
    assert "up: ok" in health.detail
    assert "down" in health.detail


# -- build_news_provider --


def test_build_news_provider_includes_only_configured_plus_keyless():
    provider = build_news_provider()
    assert provider.provider_names == ["hackernews"]


def test_build_news_provider_orders_configured_providers(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
    monkeypatch.setenv("GUARDIAN_API_KEY", "test-key")

    provider = build_news_provider()

    assert provider.provider_names == ["finnhub", "guardian", "hackernews"]


def test_build_news_provider_respects_fallback_order_env_var(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
    monkeypatch.setenv("GNEWS_API_KEY", "test-key")
    monkeypatch.setenv("AOR_NEWS_PROVIDER_FALLBACK_ORDER", "gnews,finnhub")

    provider = build_news_provider()

    assert provider.provider_names == ["gnews", "finnhub"]
