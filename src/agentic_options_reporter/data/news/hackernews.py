"""Hacker News adapter (hn.algolia.com — free, keyless, unlimited).

Technology/community news via Algolia's public HN search API. Keyless,
so it's always available — the router's last-resort fallback — but it's
community discussion, not journalism: placed last in the default
fallback order and best treated as supplemental signal. English-only in
practice, so `language` is ignored.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agentic_options_reporter.data.news.base import _HttpNewsProvider
from agentic_options_reporter.models.schemas import NewsArticle


class HackerNewsProvider(_HttpNewsProvider):
    BASE_URL = "https://hn.algolia.com/api/v1/search"
    PROVIDER_LABEL = "Hacker News"
    API_KEY_ENV_VAR = None  # keyless

    @staticmethod
    def _parse_created_at(value: str) -> datetime:
        if not value:
            return datetime.now(timezone.utc)
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    def _to_article(self, item: dict[str, Any]) -> NewsArticle:
        object_id = item.get("objectID", "")
        return NewsArticle(
            headline=item.get("title") or "",
            source="Hacker News",
            # Link-less stories (Ask HN etc.) fall back to the HN thread.
            url=item.get("url") or f"https://news.ycombinator.com/item?id={object_id}",
            published_at=self._parse_created_at(item.get("created_at", "")),
            summary="",
        )

    async def _hits(self, params: dict[str, Any], limit: int) -> list[NewsArticle]:
        params = {"tags": "story", "hitsPerPage": limit, **params}
        data = await self._get_json(self.BASE_URL, params)
        hits = data.get("hits") or []
        return [self._to_article(item) for item in hits[:limit]]

    async def search(
        self,
        query: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        language: str = "en",
        limit: int = 20,
    ) -> list[NewsArticle]:
        params: dict[str, Any] = {"query": query}
        filters = []
        if start_date is not None:
            filters.append(f"created_at_i>{int(start_date.timestamp())}")
        if end_date is not None:
            filters.append(f"created_at_i<{int(end_date.timestamp())}")
        if filters:
            params["numericFilters"] = ",".join(filters)
        return await self._hits(params, limit)

    async def top_headlines(
        self, category: str | None = None, limit: int = 20
    ) -> list[NewsArticle]:
        # HN has no category taxonomy; the front page is the headline set.
        return await self._hits({"tags": "front_page"}, limit)
