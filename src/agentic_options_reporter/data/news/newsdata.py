"""NewsData.io adapter (newsdata.io — free tier ~200 requests/day).

General-news aggregator. The free tier only exposes the latest-news
window (the /archive endpoint is paid), so `start_date`/`end_date` are
accepted but not forwarded — a documented free-tier gap, not silently
different results.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agentic_options_reporter.data.news.base import _HttpNewsProvider
from agentic_options_reporter.models.schemas import NewsArticle


class NewsDataProvider(_HttpNewsProvider):
    BASE_URL = "https://newsdata.io/api/1/latest"
    PROVIDER_LABEL = "NewsData.io"
    API_KEY_ENV_VAR = "NEWSDATA_API_KEY"

    @staticmethod
    def _parse_pub_date(value: str) -> datetime:
        if not value:
            return datetime.now(timezone.utc)
        return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)

    def _to_article(self, item: dict[str, Any]) -> NewsArticle:
        return NewsArticle(
            headline=item.get("title", ""),
            source=item.get("source_id", ""),
            url=item.get("link", ""),
            published_at=self._parse_pub_date(item.get("pubDate", "")),
            summary=item.get("description") or "",
        )

    async def _results(self, params: dict[str, Any], limit: int) -> list[NewsArticle]:
        params = {"apikey": self._api_key, **params}
        data = await self._get_json(self.BASE_URL, params)
        results = data.get("results") or []
        return [self._to_article(item) for item in results[:limit]]

    async def search(
        self,
        query: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        language: str = "en",
        limit: int = 20,
    ) -> list[NewsArticle]:
        # start_date/end_date intentionally unused: see module docstring.
        return await self._results({"q": query, "language": language}, limit)

    async def top_headlines(
        self, category: str | None = None, limit: int = 20
    ) -> list[NewsArticle]:
        return await self._results({"category": category or "business", "language": "en"}, limit)
