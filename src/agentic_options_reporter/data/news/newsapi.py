"""NewsAPI adapter (newsapi.org — free tier ~100 requests/day, dev use only).

General-news aggregator: /v2/everything for search, /v2/top-headlines
for headlines.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agentic_options_reporter.data.news.base import _HttpNewsProvider
from agentic_options_reporter.models.schemas import NewsArticle


class NewsApiOrgProvider(_HttpNewsProvider):
    BASE_URL = "https://newsapi.org/v2"
    PROVIDER_LABEL = "NewsAPI"
    API_KEY_ENV_VAR = "NEWSAPI_API_KEY"

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

    async def _articles(self, path: str, params: dict[str, Any], limit: int) -> list[NewsArticle]:
        data = await self._get_json(
            f"{self.BASE_URL}{path}", params, headers={"X-Api-Key": self._api_key}
        )
        articles = data.get("articles") or []
        return [self._to_article(item) for item in articles[:limit]]

    async def search(
        self,
        query: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        language: str = "en",
        limit: int = 20,
    ) -> list[NewsArticle]:
        params: dict[str, Any] = {
            "q": query,
            "sortBy": "publishedAt",
            "pageSize": limit,
            "language": language,
        }
        if start_date is not None:
            params["from"] = start_date.date().isoformat()
        if end_date is not None:
            params["to"] = end_date.date().isoformat()
        return await self._articles("/everything", params, limit)

    async def top_headlines(
        self, category: str | None = None, limit: int = 20
    ) -> list[NewsArticle]:
        return await self._articles(
            "/top-headlines",
            {"category": category or "business", "pageSize": limit, "language": "en"},
            limit,
        )
