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

import logging
import os
import re
import threading
import time
from abc import abstractmethod
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Query-param names that carry credentials — never written to the log
# buffer the frontend's Log tab reads from, and never returned in an API
# error response either (see scrub_secrets below).
_SENSITIVE_PARAM_NAMES = {"apikey", "api_key", "token", "access_key", "key", "secret"}

# Matches `?apikey=VALUE` / `&token=VALUE` query-string fragments — anchored
# on the preceding `?`/`&` so this only fires on an actual query parameter,
# not on an unrelated word that happens to contain "key".
_SECRET_QUERY_PATTERN = re.compile(
    r"([?&])(" + "|".join(re.escape(name) for name in _SENSITIVE_PARAM_NAMES) + r")=[^&\s'\"]+",
    re.IGNORECASE,
)


def redact_params(params: dict[str, Any]) -> dict[str, Any]:
    """Replace credential-shaped query params with a placeholder before a
    request is logged, so an API key never lands in the in-memory log
    buffer (or the console) that GET /logs and the Log tab expose."""
    return {
        name: ("***" if name.lower() in _SENSITIVE_PARAM_NAMES else value)
        for name, value in params.items()
    }


def scrub_secrets(text: str) -> str:
    """Mask credential-shaped query params inside arbitrary text — in
    particular an httpx exception's string form, which embeds the FULL
    request URL (including the real `?apikey=...`) via `response.url` or
    `request.url`. `redact_params` only covers the structured params dict
    on the request-log line; this covers the free-text exception messages
    that `_classify_httpx_error` builds, which is the one chokepoint every
    failure path funnels through — logged warnings/errors, provider-router
    failover messages, workflow data_warnings, and API error responses
    (main.py's `HTTPException(detail=str(exc))`) all consume the resulting
    exception's `str()`, so scrubbing here covers all of them at once."""
    return _SECRET_QUERY_PATTERN.sub(lambda m: f"{m.group(1)}{m.group(2)}=***", text)


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
        # httpx bakes the FULL request URL — including the real ?apikey=...
        # — into these exceptions' str(); scrub it here, at the one
        # chokepoint every failure path funnels through, so the key can
        # never surface downstream (logs, provider-router messages,
        # workflow warnings, or an API error response's `detail`).
        detail = scrub_secrets(str(exc))
        if isinstance(exc, httpx.TimeoutException):
            return self.TIMEOUT_CLS(f"{label} request timed out: {detail}")
        if isinstance(exc, httpx.HTTPStatusError):
            status_code = exc.response.status_code
            if status_code == 429:
                return self.RATE_LIMITED_CLS(f"{label} rate limited: {detail}")
            if status_code >= 500:
                return self.UNAVAILABLE_CLS(f"{label} is unavailable: {detail}")
            return self.ERROR_CLS(f"{label} request failed: {detail}")
        if isinstance(exc, httpx.TransportError):
            return self.UNAVAILABLE_CLS(f"{label} is unreachable: {detail}")
        return self.ERROR_CLS(f"{label} request failed: {detail}")

    def _check_payload(self, payload: Any) -> None:
        """Hook for providers whose errors arrive as HTTP-200 bodies
        (e.g. Alpha Vantage's rate-limit "Information" note)."""

    async def _get_json(
        self, url: str, params: dict[str, Any], headers: dict[str, str] | None = None
    ) -> Any:
        import httpx

        safe_params = redact_params(params)
        cache_key = (type(self).__name__, url, tuple(sorted(params.items())))
        with AsyncHttpProviderBase._cache_lock:
            cached = AsyncHttpProviderBase._shared_cache.get(cache_key)
            if cached is not None:
                cached_at, payload = cached
                if time.monotonic() - cached_at < self.CACHE_TTL_SECONDS:
                    logger.debug("%s cache hit: GET %s %s", self.PROVIDER_LABEL, url, safe_params)
                    return payload

        started = time.monotonic()
        logger.info("%s → GET %s %s", self.PROVIDER_LABEL, url, safe_params)
        try:
            if self._client is not None:
                response = await self._client.get(url, params=params, headers=headers)
            else:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.get(url, params=params, headers=headers)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            elapsed_ms = (time.monotonic() - started) * 1000
            # exc's str() embeds the full request URL (real apikey and
            # all) before it's classified below — scrub it here too, not
            # just inside _classify_httpx_error, since this line logs the
            # raw exception directly.
            logger.warning(
                "%s ✗ GET %s failed after %.0fms: %s",
                self.PROVIDER_LABEL, url, elapsed_ms, scrub_secrets(str(exc)),
            )
            raise self._classify_httpx_error(exc) from exc

        elapsed_ms = (time.monotonic() - started) * 1000
        logger.info(
            "%s ← %d GET %s (%.0fms)", self.PROVIDER_LABEL, response.status_code, url, elapsed_ms
        )
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
