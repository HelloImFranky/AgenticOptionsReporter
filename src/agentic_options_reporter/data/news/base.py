"""News provider interface and shared adapter infrastructure.

`NewsProvider` is the async interface the news_research agent depends on
(dependency injection — the same pattern as
`market_data.MarketDataProvider`, but async so adapters can later be
fanned out concurrently). One adapter per source lives in this package
(see specs/providers.yaml); `router.build_news_provider()` composes
whichever are configured into a failover router.

`_HttpNewsProvider` is the common base for HTTP-backed adapters: API-key
handling, a shared httpx GET with error normalization into the
`NewsProviderError` hierarchy, and a class-level TTL response cache.
The cache is process-wide on purpose — main.py builds a fresh provider
per request, and free tiers meter by the day (Alpha Vantage: 25/day),
so a "Regenerate" click seconds later must not re-spend quota.
"""

from __future__ import annotations

import os
import threading
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

from agentic_options_reporter.data.provider_errors import (
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
    ProviderUnsupported,
)
from agentic_options_reporter.models.schemas import NewsArticle


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


class ProviderHealth(BaseModel):
    """Result of a NewsProvider.health() probe."""

    provider: str
    healthy: bool
    latency_ms: float | None = None
    detail: str = ""
    checked_at: datetime


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


def classify_httpx_error(exc: Exception, provider_label: str) -> NewsProviderError:
    """Normalize an httpx exception into the NewsProviderError hierarchy
    (the async analog of data.provider_router.classify_requests_error)."""
    import httpx

    if isinstance(exc, httpx.TimeoutException):
        return NewsProviderTimeout(f"{provider_label} request timed out: {exc}")
    if isinstance(exc, httpx.HTTPStatusError):
        status_code = exc.response.status_code
        if status_code == 429:
            return NewsProviderRateLimited(f"{provider_label} rate limited: {exc}")
        if status_code >= 500:
            return NewsProviderUnavailable(f"{provider_label} is unavailable: {exc}")
        return NewsProviderError(f"{provider_label} request failed: {exc}")
    if isinstance(exc, httpx.TransportError):
        return NewsProviderUnavailable(f"{provider_label} is unreachable: {exc}")
    return NewsProviderError(f"{provider_label} request failed: {exc}")


class _HttpNewsProvider(NewsProvider):
    """Base for HTTP-backed adapters. Subclasses set PROVIDER_LABEL and
    API_KEY_ENV_VAR (None for keyless sources) and implement the three
    interface methods on top of `_get_json`."""

    PROVIDER_LABEL: str
    API_KEY_ENV_VAR: str | None = None

    CACHE_TTL_SECONDS = 300.0
    _cache_lock = threading.Lock()
    _shared_cache: dict[tuple, tuple[float, Any]] = {}

    def __init__(
        self,
        api_key: str | None = None,
        timeout_seconds: float = 15.0,
        client: Any | None = None,  # injectable httpx.AsyncClient for tests
    ) -> None:
        if self.API_KEY_ENV_VAR is not None:
            self._api_key = api_key or os.environ.get(self.API_KEY_ENV_VAR)
            if not self._api_key:
                raise NewsProviderError(
                    f"No {self.PROVIDER_LABEL} API key configured. Set "
                    f"{self.API_KEY_ENV_VAR}, or supply one explicitly."
                )
        else:
            self._api_key = None
        self._timeout = timeout_seconds
        self._client = client

    @classmethod
    def clear_shared_cache(cls) -> None:
        with _HttpNewsProvider._cache_lock:
            _HttpNewsProvider._shared_cache.clear()

    def _check_payload(self, payload: Any) -> None:
        """Hook for providers whose errors arrive as HTTP-200 bodies
        (e.g. Alpha Vantage's rate-limit "Information" note)."""

    async def _get_json(
        self, url: str, params: dict[str, Any], headers: dict[str, str] | None = None
    ) -> Any:
        import httpx

        cache_key = (type(self).__name__, url, tuple(sorted(params.items())))
        with _HttpNewsProvider._cache_lock:
            cached = _HttpNewsProvider._shared_cache.get(cache_key)
            if cached is not None:
                cached_at, payload = cached
                if time.monotonic() - cached_at < self.CACHE_TTL_SECONDS:
                    return payload

        try:
            if self._client is not None:
                response = await self._client.get(url, params=params, headers=headers)
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise classify_httpx_error(exc, self.PROVIDER_LABEL) from exc

        payload = response.json()
        self._check_payload(payload)
        with _HttpNewsProvider._cache_lock:
            _HttpNewsProvider._shared_cache[cache_key] = (time.monotonic(), payload)
        return payload

    async def health(self) -> ProviderHealth:
        """Probe the source with a minimal top_headlines call. Never
        raises — an unhealthy provider is a result, not an error."""
        started = time.monotonic()
        try:
            await self.top_headlines(limit=1)
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
