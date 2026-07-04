"""The Guardian Open Platform adapter (open-platform.theguardian.com —
free tier with generous limits).

High-quality English-language journalism; `language` is ignored (the
Guardian publishes in English). `top_headlines`' category maps to a
Guardian section (business, technology, ...).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agentic_options_reporter.data.news.base import _HttpNewsProvider
from agentic_options_reporter.models.schemas import NewsArticle


class GuardianNewsProvider(_HttpNewsProvider):
    BASE_URL = "https://content.guardianapis.com/search"
    PROVIDER_LABEL = "The Guardian"
    API_KEY_ENV_VAR = "GUARDIAN_API_KEY"

    @staticmethod
    def _parse_publication_date(value: str) -> datetime:
        if not value:
            return datetime.now(timezone.utc)
        return datetime.fromisoformat(value.replace("Z", "+00:00"))

    def _to_article(self, item: dict[str, Any]) -> NewsArticle:
        fields = item.get("fields") or {}
        return NewsArticle(
            headline=item.get("webTitle", ""),
            source="The Guardian",
            url=item.get("webUrl", ""),
            published_at=self._parse_publication_date(item.get("webPublicationDate", "")),
            summary=fields.get("trailText") or "",
        )

    async def _results(self, params: dict[str, Any], limit: int) -> list[NewsArticle]:
        params = {
            "api-key": self._api_key,
            "page-size": limit,
            "show-fields": "trailText",
            "order-by": "newest",
            **params,
        }
        data = await self._get_json(self.BASE_URL, params)
        results = (data.get("response") or {}).get("results") or []
        return [self._to_article(item) for item in results[:limit]]

    async def search(
        self,
        query: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        language: str = "en",
        limit: int = 20,
    ) -> list[NewsArticle]:
        params: dict[str, Any] = {"q": query}
        if start_date is not None:
            params["from-date"] = start_date.date().isoformat()
        if end_date is not None:
            params["to-date"] = end_date.date().isoformat()
        return await self._results(params, limit)

    async def top_headlines(
        self, category: str | None = None, limit: int = 20
    ) -> list[NewsArticle]:
        return await self._results({"section": category or "business"}, limit)
