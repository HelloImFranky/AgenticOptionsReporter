"""Generic per-method automatic failover across multiple configured data
providers of the same interface — the data-provider analog of
`thesis.llm_client.LlmRouter`.

Routing happens per METHOD CALL rather than per whole provider instance:
each `<X>ProviderRouter` tries its configured providers in priority order
for one method (e.g. `get_cpi`), independently of which provider answered
the previous call. This matters because some providers only cover part of
an interface (e.g. BLS has no GDP data, NewsAPI has no sentiment scores) —
they raise `ProviderUnsupported` for the methods they don't offer, which
is retryable, so the router still uses them for the methods they do
support while falling through to another provider for the rest.

See specs/providers.yaml: provider_router for the full design.
"""

from __future__ import annotations

from typing import Any

from agentic_options_reporter.data.provider_errors import RetryableProviderError


def classify_requests_error(
    exc: Exception,
    provider_label: str,
    *,
    base_error_cls: type[Exception],
    rate_limited_cls: type[Exception],
    timeout_cls: type[Exception],
    unavailable_cls: type[Exception],
) -> Exception:
    """Normalize a `requests` exception into the calling module's own error
    hierarchy, shared across every HTTP-backed data provider (news,
    financial, macro all use `requests` the same way)."""
    import requests

    if isinstance(exc, requests.exceptions.Timeout):
        return timeout_cls(f"{provider_label} request timed out: {exc}")

    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code == 429:
        return rate_limited_cls(f"{provider_label} rate limited: {exc}")
    if status_code is not None and status_code >= 500:
        return unavailable_cls(f"{provider_label} is unavailable: {exc}")
    if isinstance(exc, requests.exceptions.ConnectionError):
        return unavailable_cls(f"{provider_label} is unreachable: {exc}")
    return base_error_cls(f"{provider_label} request failed: {exc}")


def call_with_fallback(
    clients: list[tuple[str, Any]],
    method_name: str,
    all_failed_error_cls: type[Exception],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Call `method_name(*args, **kwargs)` on each `(name, client)` pair in
    order, advancing to the next client on a `RetryableProviderError`, and
    returning the first success. Raises `all_failed_error_cls` (with every
    provider's failure message) if every client fails."""
    failures: list[str] = []
    for name, client in clients:
        try:
            return getattr(client, method_name)(*args, **kwargs)
        except RetryableProviderError as exc:
            failures.append(f"{name}: {exc}")
            continue
    raise all_failed_error_cls(
        f"All configured providers failed for {method_name}(): " + "; ".join(failures)
    )


async def acall_with_fallback(
    clients: list[tuple[str, Any]],
    method_name: str,
    all_failed_error_cls: type[Exception],
    *args: Any,
    **kwargs: Any,
) -> Any:
    """Async twin of `call_with_fallback`, for provider interfaces whose
    methods are coroutines (data.news)."""
    failures: list[str] = []
    for name, client in clients:
        try:
            return await getattr(client, method_name)(*args, **kwargs)
        except RetryableProviderError as exc:
            failures.append(f"{name}: {exc}")
            continue
    raise all_failed_error_cls(
        f"All configured providers failed for {method_name}(): " + "; ".join(failures)
    )
