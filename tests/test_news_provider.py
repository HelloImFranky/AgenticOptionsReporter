import pytest

from agentic_options_reporter.data.news_provider import (
    FinnhubNewsProvider,
    NewsProvider,
    NewsProviderError,
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
