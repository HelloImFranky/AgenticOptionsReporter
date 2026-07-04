"""Finnhub adapter (finnhub.io — free tier ~60 requests/min).

Financial-news specialist: `search` treats the query as a ticker symbol
against /company-news. English-only source, so `language` is ignored.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from typing import Any

from agentic_options_reporter.data.news.base import COMPANY_NEWS, TOP_HEADLINES, _HttpNewsProvider
from agentic_options_reporter.models.schemas import NewsArticle

_DEFAULT_SEARCH_WINDOW_DAYS = 14

# Finnhub's /news categories; anything else falls back to "general".
_CATEGORIES = {"general", "forex", "crypto", "merger"}


class FinnhubNewsProvider(_HttpNewsProvider):
    BASE_URL = "https://finnhub.io/api/v1"
    PROVIDER_LABEL = "Finnhub"
    API_KEY_ENV_VAR = "FINNHUB_API_KEY"

    # search is ticker-aware (/company-news), not general keyword search.
    CAPABILITIES = frozenset({COMPANY_NEWS, TOP_HEADLINES})

    def _to_article(self, item: dict[str, Any]) -> NewsArticle:
        return NewsArticle(
            headline=item.get("headline", ""),
            source=item.get("source", ""),
            url=item.get("url", ""),
            published_at=datetime.fromtimestamp(item.get("datetime", 0), tz=timezone.utc),
            summary=item.get("summary", ""),
        )

    async def search(
        self,
        query: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        language: str = "en",
        limit: int = 20,
    ) -> list[NewsArticle]:
        to_date = end_date.date() if end_date else date.today()
        from_date = (
            start_date.date() if start_date else to_date - timedelta(days=_DEFAULT_SEARCH_WINDOW_DAYS)
        )
        data = await self._get_json(
            f"{self.BASE_URL}/company-news",
            {
                "symbol": query.upper(),
                "from": from_date.isoformat(),
                "to": to_date.isoformat(),
                "token": self._api_key,
            },
        )
        return [self._to_article(item) for item in (data or [])[:limit]]

    async def top_headlines(
        self, category: str | None = None, limit: int = 20
    ) -> list[NewsArticle]:
        finnhub_category = category if category in _CATEGORIES else "general"
        data = await self._get_json(
            f"{self.BASE_URL}/news", {"category": finnhub_category, "token": self._api_key}
        )
        return [self._to_article(item) for item in (data or [])[:limit]]
