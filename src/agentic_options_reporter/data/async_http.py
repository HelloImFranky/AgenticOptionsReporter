"""Shared infrastructure for async HTTP-backed provider adapters.

`AsyncHttpProviderBase` carries everything the news and financial
adapter packages have in common: API-key handling, httpx-error
normalization into the owning interface's error hierarchy (subclasses
point ERROR_CLS/RATE_LIMITED_CLS/TIMEOUT_CLS/UNAVAILABLE_CLS at their
own exception types, mirroring classify_requests_error for the sync
providers), a CLASS-level TTL response cache — process-wide because
main.py rebuilds providers per request and free tiers meter by the day,
so a "Regenerate" click seconds later must not re-spend quota — and a
`health()` probe timed around each adapter's `_health_probe()`.
"""

from __future__ import annotations

import os
import threading
import time
from abc import abstractmethod
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel


class ProviderHealth(BaseModel):
    """Result of a provider health() probe."""

    provider: str
    healthy: bool
    latency_ms: float | None = None
    detail: str = ""
    checked_at: datetime


class AsyncHttpProviderBase:
    """Common base for async HTTP adapters (see module docstring)."""

    PROVIDER_LABEL: str
    API_KEY_ENV_VAR: str | None = None

    # Subclass packages point these at their interface's error hierarchy.
    ERROR_CLS: type[Exception]
    RATE_LIMITED_CLS: type[Exception]
    TIMEOUT_CLS: type[Exception]
    UNAVAILABLE_CLS: type[Exception]

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
                raise self.ERROR_CLS(
                    f"No {self.PROVIDER_LABEL} API key configured. Set "
                    f"{self.API_KEY_ENV_VAR}, or supply one explicitly."
                )
        else:
            self._api_key = None
        self._timeout = timeout_seconds
        self._client = client

    @classmethod
    def clear_shared_cache(cls) -> None:
        with AsyncHttpProviderBase._cache_lock:
            AsyncHttpProviderBase._shared_cache.clear()

    def _classify_httpx_error(self, exc: Exception) -> Exception:
        import httpx

        label = self.PROVIDER_LABEL
        if isinstance(exc, httpx.TimeoutException):
            return self.TIMEOUT_CLS(f"{label} request timed out: {exc}")
        if isinstance(exc, httpx.HTTPStatusError):
            status_code = exc.response.status_code
            if status_code == 429:
                return self.RATE_LIMITED_CLS(f"{label} rate limited: {exc}")
            if status_code >= 500:
                return self.UNAVAILABLE_CLS(f"{label} is unavailable: {exc}")
            return self.ERROR_CLS(f"{label} request failed: {exc}")
        if isinstance(exc, httpx.TransportError):
            return self.UNAVAILABLE_CLS(f"{label} is unreachable: {exc}")
        return self.ERROR_CLS(f"{label} request failed: {exc}")

    def _check_payload(self, payload: Any) -> None:
        """Hook for providers whose errors arrive as HTTP-200 bodies
        (e.g. Alpha Vantage's rate-limit "Information" note)."""

    async def _get_json(
        self, url: str, params: dict[str, Any], headers: dict[str, str] | None = None
    ) -> Any:
        import httpx

        cache_key = (type(self).__name__, url, tuple(sorted(params.items())))
        with AsyncHttpProviderBase._cache_lock:
            cached = AsyncHttpProviderBase._shared_cache.get(cache_key)
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
            raise self._classify_httpx_error(exc) from exc

        payload = response.json()
        self._check_payload(payload)
        with AsyncHttpProviderBase._cache_lock:
            AsyncHttpProviderBase._shared_cache[cache_key] = (time.monotonic(), payload)
        return payload

    @abstractmethod
    async def _health_probe(self) -> None:
        """Cheapest real request this source supports; raises on failure."""
        raise NotImplementedError

    async def health(self) -> ProviderHealth:
        """Probe the source. Never raises — an unhealthy provider is a
        result, not an error."""
        started = time.monotonic()
        try:
            await self._health_probe()
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
