"""News data access.

`NewsProvider` is the interface the news_research agent depends on
(dependency injection — the same pattern as `market_data.MarketDataProvider`).
`FinnhubNewsProvider` is the phase-2a implementation (see
specs/providers.yaml); additional providers (Alpha Vantage, NewsAPI,
GDELT, RSS feeds) can be added later by implementing the same interface.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import date, datetime, timedelta, timezone
from typing import Any

from agentic_options_reporter.models.schemas import NewsArticle, SentimentSnapshot


class NewsProviderError(RuntimeError):
    """Raised when a NewsProvider cannot return the requested data."""


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
            raise NewsProviderError(f"Finnhub request to {path} failed: {exc}") from exc
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
