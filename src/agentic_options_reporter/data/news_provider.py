"""News data access.

`NewsProvider` is the interface the news_research agent depends on
(dependency injection — the same pattern as `market_data.MarketDataProvider`).
Four concrete implementations exist (Finnhub, Alpha Vantage, NewsAPI,
GDELT — see specs/providers.yaml); `build_news_provider()` composes
whichever are currently configured into a `NewsProviderRouter` that fails
over between them per method call, the data-provider analog of
`thesis.llm_client.LlmRouter`.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta, timezone
from typing import Any

from agentic_options_reporter.data.provider_errors import (
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
    ProviderUnsupported,
)
from agentic_options_reporter.data.provider_router import call_with_fallback, classify_requests_error
from agentic_options_reporter.models.schemas import NewsArticle, SentimentSnapshot


class NewsProviderError(RuntimeError):
    """Raised when a NewsProvider cannot return the requested data."""


class NewsProviderRateLimited(NewsProviderError, ProviderRateLimited):
    """The provider rejected the request for exceeding its rate limit (HTTP 429)."""


class NewsProviderTimeout(NewsProviderError, ProviderTimeout):
    """The request to the provider timed out."""


class NewsProviderUnavailable(NewsProviderError, ProviderUnavailable):
    """The provider is unreachable or returned a server error (5xx / network failure)."""


class NewsProviderUnsupported(NewsProviderError, ProviderUnsupported):
    """This provider doesn't offer the requested data at all (e.g. NewsAPI has no sentiment endpoint)."""


class NewsProvider(ABC):
    """Interface implemented by all news providers."""

    @abstractmethod
    def get_company_news(self, ticker: str, limit: int = 20) -> list[NewsArticle]:
        raise NotImplementedError

    @abstractmethod
    def get_market_news(self, limit: int = 20) -> list[NewsArticle]:
        raise NotImplementedError

    @abstractmethod
    def get_sentiment(self, ticker: str) -> SentimentSnapshot:
        raise NotImplementedError


class FinnhubNewsProvider(NewsProvider):
    """NewsProvider implementation backed by the Finnhub API."""

    BASE_URL = "https://finnhub.io/api/v1"
    PROVIDER_LABEL = "Finnhub"

    def __init__(self, api_key: str | None = None, timeout_seconds: int = 15) -> None:
        self._api_key = api_key or os.environ.get("FINNHUB_API_KEY")
        if not self._api_key:
            raise NewsProviderError(
                "No Finnhub API key configured. Set FINNHUB_API_KEY, or supply one explicitly."
            )
        self._timeout = timeout_seconds

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        import requests

        url = f"{self.BASE_URL}{path}"
        try:
            response = requests.get(url, params=params, timeout=self._timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise classify_requests_error(
                exc,
                self.PROVIDER_LABEL,
                base_error_cls=NewsProviderError,
                rate_limited_cls=NewsProviderRateLimited,
                timeout_cls=NewsProviderTimeout,
                unavailable_cls=NewsProviderUnavailable,
            ) from exc
        return response.json()

    def _to_article(self, item: dict[str, Any]) -> NewsArticle:
        return NewsArticle(
            headline=item.get("headline", ""),
            source=item.get("source", ""),
            url=item.get("url", ""),
            published_at=datetime.fromtimestamp(item.get("datetime", 0), tz=timezone.utc),
            summary=item.get("summary", ""),
        )

    def get_company_news(self, ticker: str, limit: int = 20) -> list[NewsArticle]:
        to_date = date.today()
        from_date = to_date - timedelta(days=14)
        data = self._get(
            "/company-news",
            {
                "symbol": ticker,
                "from": from_date.isoformat(),
                "to": to_date.isoformat(),
                "token": self._api_key,
            },
        )
        return [self._to_article(item) for item in data[:limit]]

    def get_market_news(self, limit: int = 20) -> list[NewsArticle]:
        data = self._get("/news", {"category": "general", "token": self._api_key})
        return [self._to_article(item) for item in data[:limit]]

    def get_sentiment(self, ticker: str) -> SentimentSnapshot:
        data = self._get("/news-sentiment", {"symbol": ticker, "token": self._api_key})
        sentiment = data.get("sentiment") or {}
        bullish = float(sentiment.get("bullishPercent") or 0.0)
        bearish = float(sentiment.get("bearishPercent") or 0.0)
        score = bullish - bearish

        if score > 0.1:
            label = "bullish"
        elif score < -0.1:
            label = "bearish"
        else:
            label = "neutral"

        buzz = data.get("buzz") or {}
        article_count = int(buzz.get("articlesInLastWeek") or 0)

        return SentimentSnapshot(
            ticker=ticker.upper(), score=score, label=label, article_count=article_count
        )


class AlphaVantageNewsProvider(NewsProvider):
    """NewsProvider implementation backed by Alpha Vantage's NEWS_SENTIMENT
    endpoint, which — unusually — covers both headlines and sentiment in
    one response. Alpha Vantage's free tier returns HTTP 200 with an
    "Information"/"Note" field instead of a proper 429 when rate limited;
    `_get` treats that the same as a real 429.
    """

    BASE_URL = "https://www.alphavantage.co/query"
    PROVIDER_LABEL = "Alpha Vantage"

    def __init__(self, api_key: str | None = None, timeout_seconds: int = 15) -> None:
        self._api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY")
        if not self._api_key:
            raise NewsProviderError(
                "No Alpha Vantage API key configured. Set ALPHA_VANTAGE_API_KEY, "
                "or supply one explicitly."
            )
        self._timeout = timeout_seconds

    def _get(self, params: dict[str, Any]) -> Any:
        import requests

        query = dict(params)
        query["apikey"] = self._api_key
        try:
            response = requests.get(self.BASE_URL, params=query, timeout=self._timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise classify_requests_error(
                exc,
                self.PROVIDER_LABEL,
                base_error_cls=NewsProviderError,
                rate_limited_cls=NewsProviderRateLimited,
                timeout_cls=NewsProviderTimeout,
                unavailable_cls=NewsProviderUnavailable,
            ) from exc

        data = response.json()
        if "Information" in data or "Note" in data:
            raise NewsProviderRateLimited(
                f"{self.PROVIDER_LABEL} rate limited or restricted: "
                f"{data.get('Information') or data.get('Note')}"
            )
        return data

    @staticmethod
    def _parse_time_published(value: str) -> datetime:
        if not value:
            return datetime.now(timezone.utc)
        return datetime.strptime(value, "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)

    def _to_article(self, item: dict[str, Any]) -> NewsArticle:
        return NewsArticle(
            headline=item.get("title", ""),
            source=item.get("source", ""),
            url=item.get("url", ""),
            published_at=self._parse_time_published(item.get("time_published", "")),
            summary=item.get("summary", ""),
        )

    def get_company_news(self, ticker: str, limit: int = 20) -> list[NewsArticle]:
        data = self._get({"function": "NEWS_SENTIMENT", "tickers": ticker.upper(), "limit": limit})
        feed = data.get("feed") or []
        return [self._to_article(item) for item in feed[:limit]]

    def get_market_news(self, limit: int = 20) -> list[NewsArticle]:
        data = self._get(
            {"function": "NEWS_SENTIMENT", "topics": "financial_markets", "limit": limit}
        )
        feed = data.get("feed") or []
        return [self._to_article(item) for item in feed[:limit]]

    def get_sentiment(self, ticker: str) -> SentimentSnapshot:
        data = self._get({"function": "NEWS_SENTIMENT", "tickers": ticker.upper()})
        feed = data.get("feed") or []
        scores: list[float] = []
        for item in feed:
            ticker_sentiments = item.get("ticker_sentiment") or []
            match = next(
                (t for t in ticker_sentiments if str(t.get("ticker", "")).upper() == ticker.upper()),
                None,
            )
            if match is not None:
                scores.append(float(match.get("ticker_sentiment_score") or 0.0))
            elif "overall_sentiment_score" in item:
                scores.append(float(item.get("overall_sentiment_score") or 0.0))

        score = sum(scores) / len(scores) if scores else 0.0
        if score > 0.15:
            label = "bullish"
        elif score < -0.15:
            label = "bearish"
        else:
            label = "neutral"
        return SentimentSnapshot(
            ticker=ticker.upper(), score=score, label=label, article_count=len(feed)
        )


class NewsApiOrgProvider(NewsProvider):
    """NewsProvider implementation backed by NewsAPI.org.

    NewsAPI has no sentiment endpoint at any tier: `get_sentiment` always
    raises `NewsProviderUnsupported` so `NewsProviderRouter` falls through
    to a provider that can answer it.
    """

    BASE_URL = "https://newsapi.org/v2"
    PROVIDER_LABEL = "NewsAPI"

    def __init__(self, api_key: str | None = None, timeout_seconds: int = 15) -> None:
        self._api_key = api_key or os.environ.get("NEWSAPI_API_KEY")
        if not self._api_key:
            raise NewsProviderError(
                "No NewsAPI API key configured. Set NEWSAPI_API_KEY, or supply one explicitly."
            )
        self._timeout = timeout_seconds

    def _get(self, path: str, params: dict[str, Any]) -> Any:
        import requests

        url = f"{self.BASE_URL}{path}"
        try:
            response = requests.get(
                url, params=params, headers={"X-Api-Key": self._api_key}, timeout=self._timeout
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise classify_requests_error(
                exc,
                self.PROVIDER_LABEL,
                base_error_cls=NewsProviderError,
                rate_limited_cls=NewsProviderRateLimited,
                timeout_cls=NewsProviderTimeout,
                unavailable_cls=NewsProviderUnavailable,
            ) from exc
        return response.json()

    @staticmethod
    def _parse_published_at(value: str) -> datetime:
        if not value:
            return datetime.now(timezone.utc)
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    def _to_article(self, item: dict[str, Any]) -> NewsArticle:
        source = item.get("source") or {}
        return NewsArticle(
            headline=item.get("title", ""),
            source=source.get("name", ""),
            url=item.get("url", ""),
            published_at=self._parse_published_at(item.get("publishedAt", "")),
            summary=item.get("description") or "",
        )

    def get_company_news(self, ticker: str, limit: int = 20) -> list[NewsArticle]:
        data = self._get(
            "/everything",
            {"q": ticker, "sortBy": "publishedAt", "pageSize": limit, "language": "en"},
        )
        articles = data.get("articles") or []
        return [self._to_article(item) for item in articles[:limit]]

    def get_market_news(self, limit: int = 20) -> list[NewsArticle]:
        data = self._get(
            "/top-headlines", {"category": "business", "pageSize": limit, "language": "en"}
        )
        articles = data.get("articles") or []
        return [self._to_article(item) for item in articles[:limit]]

    def get_sentiment(self, ticker: str) -> SentimentSnapshot:
        raise NewsProviderUnsupported("NewsAPI does not provide sentiment scores.")


class GdeltNewsProvider(NewsProvider):
    """NewsProvider implementation backed by the GDELT DOC 2.0 API — free
    and keyless. Sentiment is derived from GDELT's own "tone" timeline
    (which in practice ranges roughly -10..+10), rescaled by /10 and
    clipped to align with this project's -1..1 SentimentSnapshot.score
    convention — a unit conversion of a real provider-supplied signal, not
    a fabricated value.
    """

    BASE_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
    PROVIDER_LABEL = "GDELT"

    def __init__(self, timeout_seconds: int = 15) -> None:
        self._timeout = timeout_seconds

    def _get(self, params: dict[str, Any]) -> Any:
        import requests

        try:
            response = requests.get(self.BASE_URL, params=params, timeout=self._timeout)
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise classify_requests_error(
                exc,
                self.PROVIDER_LABEL,
                base_error_cls=NewsProviderError,
                rate_limited_cls=NewsProviderRateLimited,
                timeout_cls=NewsProviderTimeout,
                unavailable_cls=NewsProviderUnavailable,
            ) from exc
        return response.json()

    @staticmethod
    def _parse_seendate(value: str) -> datetime:
        if not value:
            return datetime.now(timezone.utc)
        return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)

    def _to_article(self, item: dict[str, Any]) -> NewsArticle:
        return NewsArticle(
            headline=item.get("title", ""),
            source=item.get("domain", ""),
            url=item.get("url", ""),
            published_at=self._parse_seendate(item.get("seendate", "")),
            summary="",  # GDELT's artlist mode doesn't return article summaries
        )

    def get_company_news(self, ticker: str, limit: int = 20) -> list[NewsArticle]:
        data = self._get(
            {"query": ticker, "mode": "artlist", "maxrecords": limit, "format": "json", "sort": "hybridrel"}
        )
        articles = data.get("articles") or []
        return [self._to_article(item) for item in articles[:limit]]

    def get_market_news(self, limit: int = 20) -> list[NewsArticle]:
        data = self._get(
            {"query": "financial markets", "mode": "artlist", "maxrecords": limit, "format": "json"}
        )
        articles = data.get("articles") or []
        return [self._to_article(item) for item in articles[:limit]]

    # Article sample used for get_sentiment's article_count. Kept modest:
    # GDELT rate-limits aggressively, and a larger sample only sharpens a
    # count while doubling the odds of a 429 killing the whole call.
    SENTIMENT_ARTICLE_SAMPLE = 50

    def get_sentiment(self, ticker: str) -> SentimentSnapshot:
        articles = self.get_company_news(ticker, limit=self.SENTIMENT_ARTICLE_SAMPLE)
        data = self._get({"query": ticker, "mode": "timelinetone", "format": "json"})
        timeline = data.get("timeline") or []
        latest_value = 0.0
        if timeline:
            points = timeline[0].get("data") or []
            if points:
                latest_value = float(points[-1].get("value") or 0.0)

        score = max(-1.0, min(1.0, latest_value / 10.0))
        if score > 0.15:
            label = "bullish"
        elif score < -0.15:
            label = "bearish"
        else:
            label = "neutral"
        return SentimentSnapshot(
            ticker=ticker.upper(), score=score, label=label, article_count=len(articles)
        )


class NewsProviderRouter(NewsProvider):
    """Tries a priority-ordered list of already-constructed NewsProvider
    clients per method call, advancing to the next on a retryable failure
    (see data.provider_router). Implements NewsProvider itself, so
    news_research.run() can't tell whether it's talking to a single
    provider or a router.
    """

    def __init__(self, clients: list[tuple[str, NewsProvider]]) -> None:
        if not clients:
            raise NewsProviderError(
                "No news providers are configured for automatic failover. Set at least "
                f"one provider's API key (supported: {', '.join(sorted(_PROVIDERS))})."
            )
        self._clients = clients

    @property
    def provider_names(self) -> list[str]:
        return [name for name, _ in self._clients]

    def get_company_news(self, ticker: str, limit: int = 20) -> list[NewsArticle]:
        return call_with_fallback(self._clients, "get_company_news", NewsProviderError, ticker, limit=limit)

    def get_market_news(self, limit: int = 20) -> list[NewsArticle]:
        return call_with_fallback(self._clients, "get_market_news", NewsProviderError, limit=limit)

    def get_sentiment(self, ticker: str) -> SentimentSnapshot:
        return call_with_fallback(self._clients, "get_sentiment", NewsProviderError, ticker)


_PROVIDERS: dict[str, type[NewsProvider]] = {
    "finnhub": FinnhubNewsProvider,
    "alphavantage": AlphaVantageNewsProvider,
    "newsapi": NewsApiOrgProvider,
    "gdelt": GdeltNewsProvider,
}

# GDELT needs no API key, so it's effectively always available — placed
# last since Finnhub/Alpha Vantage's dedicated sentiment scoring is more
# precise than GDELT's tone-timeline approximation.
_DEFAULT_FALLBACK_ORDER = ["finnhub", "alphavantage", "newsapi", "gdelt"]


def _fallback_order() -> list[str]:
    raw = os.environ.get("AOR_NEWS_PROVIDER_FALLBACK_ORDER", ",".join(_DEFAULT_FALLBACK_ORDER))
    return [name.strip().lower() for name in raw.split(",") if name.strip()]


def build_news_provider() -> NewsProvider:
    """Build a NewsProviderRouter from AOR_NEWS_PROVIDER_FALLBACK_ORDER,
    skipping any provider without a configured API key. Raises
    NewsProviderError if the resulting router would have zero clients."""
    clients: list[tuple[str, NewsProvider]] = []
    for name in _fallback_order():
        provider_cls = _PROVIDERS.get(name)
        if provider_cls is None:
            continue
        try:
            clients.append((name, provider_cls()))
        except NewsProviderError:
            continue  # not configured (missing API key) — skip, don't fail the request
    return NewsProviderRouter(clients)
