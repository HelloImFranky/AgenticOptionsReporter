"""News provider failover router and configuration-driven factory."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

from agentic_options_reporter.data.news.alphavantage import AlphaVantageNewsProvider
from agentic_options_reporter.data.news.base import (
    COMPANY_NEWS,
    NewsProvider,
    NewsProviderError,
    ProviderHealth,
)
from agentic_options_reporter.data.news.finnhub import FinnhubNewsProvider
from agentic_options_reporter.data.news.gnews import GNewsProvider
from agentic_options_reporter.data.news.guardian import GuardianNewsProvider
from agentic_options_reporter.data.news.hackernews import HackerNewsProvider
from agentic_options_reporter.data.news.newsapi import NewsApiOrgProvider
from agentic_options_reporter.data.news.newsdata import NewsDataProvider
from agentic_options_reporter.data.news.yfinance_provider import YFinanceNewsProvider
from agentic_options_reporter.data.provider_router import (
    acall_and_merge,
    acall_with_fallback,
    merge_lists,
    prioritize_supporting,
)
from agentic_options_reporter.models.schemas import NewsArticle


class NewsProviderRouter(NewsProvider):
    """Tries a priority-ordered list of already-constructed NewsProvider
    adapters per method call, advancing to the next on a retryable failure
    (see data.provider_router). Implements NewsProvider itself, so
    news_research can't tell whether it's talking to one adapter or many.
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

    @property
    def capabilities(self) -> frozenset[str]:
        """Union of every configured adapter's capabilities."""
        return frozenset().union(*(client.capabilities for _, client in self._clients))

    async def search(
        self,
        query: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        language: str = "en",
        limit: int = 20,
    ) -> list[NewsArticle]:
        # `search` treats the query as a ticker/company. Prefer providers
        # that advertise COMPANY_NEWS, but keep general-news providers too
        # (they can still surface a keyword match) — a SOFT priority, not a
        # hard filter (see data.provider_router). Then FAN OUT across all of
        # them and merge, so coverage is the union of every source (Yahoo +
        # Finnhub + …) de-duplicated by URL, not just the first that
        # answers. Results are trimmed to `limit` after merging.
        candidates = prioritize_supporting(self._clients, COMPANY_NEWS)

        def _combine(results: list[list[NewsArticle]]) -> list[NewsArticle]:
            merged = merge_lists(results, key=lambda a: a.url or a.headline)
            merged.sort(key=lambda a: a.published_at, reverse=True)
            return merged[:limit]

        return await acall_and_merge(
            candidates,
            "search",
            NewsProviderError,
            _combine,
            query,
            start_date=start_date,
            end_date=end_date,
            language=language,
            limit=limit,
        )

    async def top_headlines(
        self, category: str | None = None, limit: int = 20
    ) -> list[NewsArticle]:
        return await acall_with_fallback(
            self._clients, "top_headlines", NewsProviderError, category=category, limit=limit
        )

    async def health(self) -> ProviderHealth:
        """Probe every adapter concurrently; the router is healthy if any
        adapter is. `detail` carries the per-adapter breakdown."""
        results = await asyncio.gather(*(client.health() for _, client in self._clients))
        healthy = any(result.healthy for result in results)
        detail = "; ".join(
            f"{result.provider}: {'ok' if result.healthy else result.detail or 'unhealthy'}"
            for result in results
        )
        return ProviderHealth(
            provider="router",
            healthy=healthy,
            latency_ms=max((r.latency_ms or 0.0) for r in results) if results else None,
            detail=detail,
            checked_at=datetime.now(timezone.utc),
        )


_PROVIDERS: dict[str, type[NewsProvider]] = {
    "finnhub": FinnhubNewsProvider,
    "yfinance": YFinanceNewsProvider,
    "newsdata": NewsDataProvider,
    "guardian": GuardianNewsProvider,
    "gnews": GNewsProvider,
    "alphavantage": AlphaVantageNewsProvider,
    "newsapi": NewsApiOrgProvider,
    "hackernews": HackerNewsProvider,
}

# Financial-news specialists first (Finnhub, then keyless Yahoo company
# news), general journalism next, Alpha Vantage late (25 requests/day),
# Hacker News last (keyless and always available, but community discussion
# rather than journalism). With merge routing, `search` fans out across all
# of these; order sets the de-dup tie-break priority.
_DEFAULT_FALLBACK_ORDER = [
    "finnhub",
    "yfinance",
    "newsdata",
    "guardian",
    "gnews",
    "alphavantage",
    "newsapi",
    "hackernews",
]


def _fallback_order() -> list[str]:
    raw = os.environ.get("AOR_NEWS_PROVIDER_FALLBACK_ORDER", ",".join(_DEFAULT_FALLBACK_ORDER))
    return [name.strip().lower() for name in raw.split(",") if name.strip()]


def build_news_provider() -> NewsProviderRouter:
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
