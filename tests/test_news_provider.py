import pytest

from agentic_options_reporter.data.news_provider import (
    AlphaVantageNewsProvider,
    FinnhubNewsProvider,
    GdeltNewsProvider,
    NewsApiOrgProvider,
    NewsProvider,
    NewsProviderError,
    NewsProviderRateLimited,
    NewsProviderRouter,
    NewsProviderUnsupported,
    build_news_provider,
)

from conftest import FakeHttpResponse, FakeRequestsGet


def test_requires_api_key(monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    with pytest.raises(NewsProviderError):
        FinnhubNewsProvider()


def test_accepts_explicit_api_key(monkeypatch):
    monkeypatch.delenv("FINNHUB_API_KEY", raising=False)
    provider = FinnhubNewsProvider(api_key="test-key")
    assert isinstance(provider, NewsProvider)


def test_get_company_news_maps_articles(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        FakeHttpResponse(
            [
                {
                    "headline": "Company beats earnings",
                    "source": "Reuters",
                    "url": "https://example.com/a",
                    "datetime": 1700000000,
                    "summary": "Solid quarter.",
                }
            ]
        )
    )
    provider = FinnhubNewsProvider(api_key="test-key")
    articles = provider.get_company_news("AAPL", limit=5)

    assert len(articles) == 1
    assert articles[0].headline == "Company beats earnings"
    assert articles[0].source == "Reuters"
    assert articles[0].summary == "Solid quarter."


def test_get_company_news_respects_limit(fake_requests_module):
    items = [
        {"headline": f"Story {i}", "source": "s", "url": "u", "datetime": 1700000000, "summary": ""}
        for i in range(5)
    ]
    fake_requests_module.get = FakeRequestsGet(FakeHttpResponse(items))
    provider = FinnhubNewsProvider(api_key="test-key")
    articles = provider.get_company_news("AAPL", limit=2)
    assert len(articles) == 2


def test_get_market_news_maps_articles(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        FakeHttpResponse(
            [{"headline": "Fed holds rates", "source": "AP", "url": "u", "datetime": 1700000000, "summary": ""}]
        )
    )
    provider = FinnhubNewsProvider(api_key="test-key")
    articles = provider.get_market_news(limit=5)
    assert articles[0].headline == "Fed holds rates"


def test_get_sentiment_classifies_bullish(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        FakeHttpResponse(
            {
                "sentiment": {"bullishPercent": 0.8, "bearishPercent": 0.2},
                "buzz": {"articlesInLastWeek": 12},
            }
        )
    )
    provider = FinnhubNewsProvider(api_key="test-key")
    snapshot = provider.get_sentiment("AAPL")
    assert snapshot.label == "bullish"
    assert snapshot.article_count == 12
    assert snapshot.ticker == "AAPL"


def test_get_sentiment_classifies_bearish(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        FakeHttpResponse({"sentiment": {"bullishPercent": 0.1, "bearishPercent": 0.9}, "buzz": {}})
    )
    provider = FinnhubNewsProvider(api_key="test-key")
    snapshot = provider.get_sentiment("AAPL")
    assert snapshot.label == "bearish"


def test_get_sentiment_classifies_neutral_when_balanced(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        FakeHttpResponse({"sentiment": {"bullishPercent": 0.5, "bearishPercent": 0.5}, "buzz": {}})
    )
    provider = FinnhubNewsProvider(api_key="test-key")
    snapshot = provider.get_sentiment("AAPL")
    assert snapshot.label == "neutral"


def test_http_failure_raises_news_provider_error(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        FakeHttpResponse(None, raise_exc=fake_requests_module.exceptions.RequestException("boom"))
    )
    provider = FinnhubNewsProvider(api_key="test-key")
    with pytest.raises(NewsProviderError):
        provider.get_market_news()


def test_rate_limit_status_raises_news_provider_rate_limited(fake_requests_module):
    exc = fake_requests_module.exceptions.RequestException("too many requests")
    fake_requests_module.get = FakeRequestsGet(FakeHttpResponse(None, status_code=429, raise_exc=exc))
    provider = FinnhubNewsProvider(api_key="test-key")
    with pytest.raises(NewsProviderRateLimited):
        provider.get_market_news()


# -- AlphaVantageNewsProvider --


def test_alpha_vantage_requires_api_key(monkeypatch):
    monkeypatch.delenv("ALPHA_VANTAGE_API_KEY", raising=False)
    with pytest.raises(NewsProviderError):
        AlphaVantageNewsProvider()


def test_alpha_vantage_get_company_news(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        FakeHttpResponse(
            {
                "feed": [
                    {
                        "title": "Company beats earnings",
                        "source": "Reuters",
                        "url": "https://example.com/a",
                        "time_published": "20260703T120000",
                        "summary": "Solid quarter.",
                    }
                ]
            }
        )
    )
    provider = AlphaVantageNewsProvider(api_key="test-key")
    articles = provider.get_company_news("AAPL", limit=5)
    assert articles[0].headline == "Company beats earnings"
    assert articles[0].summary == "Solid quarter."


def test_alpha_vantage_get_sentiment_uses_ticker_sentiment_when_present(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        FakeHttpResponse(
            {
                "feed": [
                    {
                        "overall_sentiment_score": 0.05,
                        "ticker_sentiment": [
                            {"ticker": "AAPL", "ticker_sentiment_score": "0.4"},
                        ],
                    }
                ]
            }
        )
    )
    provider = AlphaVantageNewsProvider(api_key="test-key")
    snapshot = provider.get_sentiment("AAPL")
    assert snapshot.label == "bullish"
    assert snapshot.article_count == 1


def test_alpha_vantage_get_sentiment_falls_back_to_overall_score(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        FakeHttpResponse({"feed": [{"overall_sentiment_score": -0.3, "ticker_sentiment": []}]})
    )
    provider = AlphaVantageNewsProvider(api_key="test-key")
    snapshot = provider.get_sentiment("AAPL")
    assert snapshot.label == "bearish"


def test_alpha_vantage_information_field_raises_rate_limited(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        FakeHttpResponse({"Information": "Thank you for using Alpha Vantage! Our standard API rate limit is 25 requests per day."})
    )
    provider = AlphaVantageNewsProvider(api_key="test-key")
    with pytest.raises(NewsProviderRateLimited):
        provider.get_company_news("AAPL")


# -- NewsApiOrgProvider --


def test_newsapi_requires_api_key(monkeypatch):
    monkeypatch.delenv("NEWSAPI_API_KEY", raising=False)
    with pytest.raises(NewsProviderError):
        NewsApiOrgProvider()


def test_newsapi_get_company_news(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        FakeHttpResponse(
            {
                "articles": [
                    {
                        "title": "Company beats earnings",
                        "source": {"name": "Reuters"},
                        "url": "https://example.com/a",
                        "publishedAt": "2026-07-03T12:00:00Z",
                        "description": "Solid quarter.",
                    }
                ]
            }
        )
    )
    provider = NewsApiOrgProvider(api_key="test-key")
    articles = provider.get_company_news("AAPL", limit=5)
    assert articles[0].headline == "Company beats earnings"
    assert articles[0].source == "Reuters"


def test_newsapi_get_sentiment_is_unsupported(fake_requests_module):
    provider = NewsApiOrgProvider(api_key="test-key")
    with pytest.raises(NewsProviderUnsupported):
        provider.get_sentiment("AAPL")


# -- GdeltNewsProvider --


def test_gdelt_needs_no_api_key():
    provider = GdeltNewsProvider()
    assert isinstance(provider, NewsProvider)


def test_gdelt_get_company_news(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        FakeHttpResponse(
            {
                "articles": [
                    {
                        "title": "Company beats earnings",
                        "domain": "reuters.com",
                        "url": "https://example.com/a",
                        "seendate": "20260703T120000Z",
                    }
                ]
            }
        )
    )
    provider = GdeltNewsProvider()
    articles = provider.get_company_news("AAPL", limit=5)
    assert articles[0].headline == "Company beats earnings"
    assert articles[0].source == "reuters.com"


def test_gdelt_get_sentiment_uses_tone_timeline(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        FakeHttpResponse({"articles": [{"title": "x", "domain": "d", "url": "u", "seendate": "20260703T120000Z"}]}),
        FakeHttpResponse({"timeline": [{"series": "tone", "data": [{"date": "20260701", "value": 2.0}, {"date": "20260702", "value": 5.0}]}]}),
    )
    provider = GdeltNewsProvider()
    snapshot = provider.get_sentiment("AAPL")
    assert snapshot.label == "bullish"
    assert snapshot.score == pytest.approx(0.5)
    assert snapshot.article_count == 1


def test_gdelt_get_sentiment_handles_missing_timeline(fake_requests_module):
    fake_requests_module.get = FakeRequestsGet(
        FakeHttpResponse({"articles": []}),
        FakeHttpResponse({"timeline": []}),
    )
    provider = GdeltNewsProvider()
    snapshot = provider.get_sentiment("AAPL")
    assert snapshot.label == "neutral"
    assert snapshot.score == 0.0


# -- NewsProviderRouter --


class _FakeNewsClient(NewsProvider):
    def __init__(self, articles=None, sentiment=None, error=None):
        self._articles = articles
        self._sentiment = sentiment
        self._error = error

    def get_company_news(self, ticker, limit=20):
        if self._error is not None:
            raise self._error
        return self._articles or []

    def get_market_news(self, limit=20):
        if self._error is not None:
            raise self._error
        return self._articles or []

    def get_sentiment(self, ticker):
        if self._error is not None:
            raise self._error
        return self._sentiment


def test_news_provider_router_rejects_empty_client_list():
    with pytest.raises(NewsProviderError):
        NewsProviderRouter([])


def test_news_provider_router_falls_through_on_unsupported_sentiment():
    from agentic_options_reporter.models.schemas import SentimentSnapshot

    unsupported = _FakeNewsClient(error=NewsProviderUnsupported("no sentiment"))
    supported = _FakeNewsClient(
        sentiment=SentimentSnapshot(ticker="AAPL", score=0.5, label="bullish", article_count=3)
    )
    router = NewsProviderRouter([("first", unsupported), ("second", supported)])

    result = router.get_sentiment("AAPL")

    assert result.label == "bullish"


def test_news_provider_router_provider_names():
    router = NewsProviderRouter([("a", _FakeNewsClient()), ("b", _FakeNewsClient())])
    assert router.provider_names == ["a", "b"]


def test_build_news_provider_skips_unconfigured_and_includes_gdelt(monkeypatch):
    for var in ("FINNHUB_API_KEY", "ALPHA_VANTAGE_API_KEY", "NEWSAPI_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("AOR_NEWS_PROVIDER_FALLBACK_ORDER", raising=False)

    provider = build_news_provider()

    assert isinstance(provider, NewsProviderRouter)
    assert provider.provider_names == ["gdelt"]


def test_build_news_provider_respects_fallback_order_env_var(monkeypatch):
    monkeypatch.setenv("FINNHUB_API_KEY", "test-key")
    monkeypatch.setenv("NEWSAPI_API_KEY", "test-key")
    monkeypatch.setenv("AOR_NEWS_PROVIDER_FALLBACK_ORDER", "newsapi,finnhub")

    provider = build_news_provider()

    assert provider.provider_names == ["newsapi", "finnhub"]
