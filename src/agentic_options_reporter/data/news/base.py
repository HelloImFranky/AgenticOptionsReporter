"""News provider interface and adapter base.

`NewsProvider` is the async interface the news_research agent depends on
(dependency injection — the same pattern as
`market_data.MarketDataProvider`, but async so adapters can later be
fanned out concurrently). One adapter per source lives in this package
(see specs/providers.yaml); `router.build_news_provider()` composes
whichever are configured into a failover router.

`_HttpNewsProvider` binds the shared async-HTTP infrastructure
(data.async_http: key handling, error normalization, class-level TTL
response cache, health probe) to this interface's error hierarchy.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from agentic_options_reporter.data.async_http import AsyncHttpProviderBase, ProviderHealth
from agentic_options_reporter.data.provider_errors import (
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
    ProviderUnsupported,
)
from agentic_options_reporter.models.schemas import NewsArticle

__all__ = [
    "NewsProvider",
    "NewsProviderError",
    "NewsProviderRateLimited",
    "NewsProviderTimeout",
    "NewsProviderUnavailable",
    "NewsProviderUnsupported",
    "ProviderHealth",
]


class NewsProviderError(RuntimeError):
    """Raised when a NewsProvider cannot return the requested data."""


class NewsProviderRateLimited(NewsProviderError, ProviderRateLimited):
    """The provider rejected the request for exceeding its rate limit (HTTP 429)."""


class NewsProviderTimeout(NewsProviderError, ProviderTimeout):
    """The request to the provider timed out."""


class NewsProviderUnavailable(NewsProviderError, ProviderUnavailable):
    """The provider is unreachable or returned a server error (5xx / network failure)."""


class NewsProviderUnsupported(NewsProviderError, ProviderUnsupported):
    """This provider doesn't offer the requested data at all."""


class NewsProvider(ABC):
    """Interface implemented by all news providers."""

    @abstractmethod
    async def search(
        self,
        query: str,
        start_date: datetime | None = None,
        end_date: datetime | None = None,
        language: str = "en",
        limit: int = 20,
    ) -> list[NewsArticle]:
        raise NotImplementedError

    @abstractmethod
    async def top_headlines(
        self,
        category: str | None = None,
        limit: int = 20,
    ) -> list[NewsArticle]:
        raise NotImplementedError

    @abstractmethod
    async def health(self) -> ProviderHealth:
        raise NotImplementedError


class _HttpNewsProvider(AsyncHttpProviderBase, NewsProvider):
    """Base for HTTP-backed news adapters. Subclasses set PROVIDER_LABEL
    and API_KEY_ENV_VAR (None for keyless sources) and implement
    search/top_headlines on top of `_get_json`."""

    ERROR_CLS = NewsProviderError
    RATE_LIMITED_CLS = NewsProviderRateLimited
    TIMEOUT_CLS = NewsProviderTimeout
    UNAVAILABLE_CLS = NewsProviderUnavailable

    async def _health_probe(self) -> None:
        await self.top_headlines(limit=1)
