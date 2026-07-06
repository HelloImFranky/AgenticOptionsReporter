"""Yahoo Finance news adapter, backed by the synchronous `yfinance`
package.

Like the market-data and fundamentals Yahoo adapters, `yfinance` has no
async API, so this implements the async `NewsProvider` interface directly
and offloads the blocking `.news` fetch to a worker thread. Keyless and
ticker-aware: `search(query)` treats the query as a symbol and returns
Yahoo's curated company-news feed (COMPANY_NEWS). Yahoo has no general
headline endpoint, so `top_headlines` raises Unsupported (retryable), and
the news router simply routes headline requests elsewhere.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone
from typing import Any

from agentic_options_reporter.data.async_http import ProviderHealth
from agentic_options_reporter.data.news.base import (
    COMPANY_NEWS,
    NewsProvider,
    NewsProviderUnavailable,
    NewsProviderUnsupported,
)
from agentic_options_reporter.models.schemas import NewsArticle

_HEALTH_PROBE_TICKER = "AAPL"


class YFinanceNewsProvider(NewsProvider):
    """Yahoo Finance company news via the `yfinance` package. Keyless."""

    PROVIDER_LABEL = "Yahoo Finance"

    @property
    def capabilities(self) -> frozenset[str]:
        return frozenset({COMPANY_NEWS})

    async def search(
        self,
        query: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        language: str = "en",
        limit: int = 20,
    ) -> list[NewsArticle]:
        items = await asyncio.to_thread(self._news_sync, query)
        articles = [a for a in (_to_article(item) for item in items) if a is not None]
        return articles[:limit]

    def _news_sync(self, ticker: str) -> list[dict[str, Any]]:
        import yfinance as yf

        try:
            news = yf.Ticker(ticker).news
        except NewsProviderUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 — normalize yfinance/network errors for the router
            raise NewsProviderUnavailable(
                f"Yahoo Finance news request failed for {ticker!r}: {exc}"
            ) from exc
        return news or []

    async def top_headlines(
        self, category: str | None = None, limit: int = 20
    ) -> list[NewsArticle]:
        raise NewsProviderUnsupported("Yahoo Finance has no general headline feed.")

    async def health(self) -> ProviderHealth:
        started = time.monotonic()
        try:
            await self.search(_HEALTH_PROBE_TICKER, limit=1)
        except Exception as exc:  # noqa: BLE001 — health() reports, never raises
            return ProviderHealth(
                provider=self.PROVIDER_LABEL,
                healthy=False,
                latency_ms=(time.monotonic() - started) * 1000,
                detail=str(exc),
                checked_at=datetime.now(timezone.utc),
            )
        return ProviderHealth(
            provider=self.PROVIDER_LABEL,
            healthy=True,
            latency_ms=(time.monotonic() - started) * 1000,
            checked_at=datetime.now(timezone.utc),
        )


def _to_article(item: dict[str, Any]) -> NewsArticle | None:
    """Normalize one yfinance news item. yfinance has used two shapes over
    versions: the legacy flat dict (title/link/publisher/providerPublishTime)
    and the newer nested `content` object — handle both."""
    content = item.get("content") if isinstance(item.get("content"), dict) else None
    if content is not None:
        title = content.get("title") or ""
        url = (
            (content.get("canonicalUrl") or {}).get("url")
            or (content.get("clickThroughUrl") or {}).get("url")
            or ""
        )
        source = (content.get("provider") or {}).get("displayName") or "Yahoo Finance"
        published_at = _parse_ts(content.get("pubDate") or content.get("displayTime"))
        summary = content.get("summary") or ""
    else:
        title = item.get("title") or ""
        url = item.get("link") or ""
        source = item.get("publisher") or "Yahoo Finance"
        published_at = _parse_ts(item.get("providerPublishTime"))
        summary = ""
    if not title or not url:
        return None
    return NewsArticle(
        headline=title, source=source, url=url, published_at=published_at, summary=summary
    )


def _parse_ts(value: Any) -> datetime:
    if value is None:
        return datetime.now(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=timezone.utc)
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return datetime.now(timezone.utc)
