"""Alpha Vantage adapter (alphavantage.co — free tier ~25 requests/day).

Uses the NEWS_SENTIMENT endpoint; `search` treats the query as a ticker.
English-only source, so `language` is ignored. The free tier returns
HTTP 200 with an "Information"/"Note" field instead of a proper 429 when
rate limited; `_check_payload` treats that the same as a real 429.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from agentic_options_reporter.data.news.base import NewsProviderRateLimited, _HttpNewsProvider
from agentic_options_reporter.models.schemas import NewsArticle


class AlphaVantageNewsProvider(_HttpNewsProvider):
    BASE_URL = "https://www.alphavantage.co/query"
    PROVIDER_LABEL = "Alpha Vantage"
    API_KEY_ENV_VAR = "ALPHA_VANTAGE_API_KEY"

    def _check_payload(self, payload: Any) -> None:
        if isinstance(payload, dict) and ("Information" in payload or "Note" in payload):
            raise NewsProviderRateLimited(
                f"{self.PROVIDER_LABEL} rate limited or restricted: "
                f"{payload.get('Information') or payload.get('Note')}"
            )

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

    async def _news_sentiment(self, params: dict[str, Any], limit: int) -> list[NewsArticle]:
        params = {"function": "NEWS_SENTIMENT", "limit": limit, "apikey": self._api_key, **params}
        data = await self._get_json(self.BASE_URL, params)
        feed = data.get("feed") or []
        return [self._to_article(item) for item in feed[:limit]]

    async def search(
        self,
        query: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        language: str = "en",
        limit: int = 20,
    ) -> list[NewsArticle]:
        params: dict[str, Any] = {"tickers": query.upper()}
        if start_date is not None:
            params["time_from"] = start_date.strftime("%Y%m%dT%H%M")
        if end_date is not None:
            params["time_to"] = end_date.strftime("%Y%m%dT%H%M")
        return await self._news_sentiment(params, limit)

    async def top_headlines(
        self, category: str | None = None, limit: int = 20
    ) -> list[NewsArticle]:
        return await self._news_sentiment({"topics": category or "financial_markets"}, limit)
