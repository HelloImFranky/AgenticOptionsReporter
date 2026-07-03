"""Retryable-error markers shared by the data-provider routers.

`NewsProviderRouter`, `FinancialProviderRouter`, and `MacroProviderRouter`
(in news_provider.py, financial_provider.py, macro_provider.py) are the
data-provider analog of `thesis.llm_client.LlmRouter` — automatic
failover across multiple configured implementations of the same
interface. Each concrete provider error (e.g. `NewsProviderRateLimited`)
multiply inherits both its own `<X>ProviderError` base (so existing
`except NewsProviderError` call sites keep working unchanged) and one of
the markers below (so `data.provider_router.call_with_fallback` can
recognize a retryable failure generically, across all three provider
types, without knowing which concrete interface raised it).

See specs/providers.yaml: provider_router for the full design.
"""

from __future__ import annotations


class RetryableProviderError(Exception):
    """Marker: a *ProviderRouter should try the next configured provider."""


class ProviderRateLimited(RetryableProviderError):
    """The provider rejected the request for exceeding its rate limit (HTTP 429)."""


class ProviderTimeout(RetryableProviderError):
    """The request to the provider timed out."""


class ProviderUnavailable(RetryableProviderError):
    """The provider is unreachable or returned a server error (5xx / network failure)."""


class ProviderUnsupported(RetryableProviderError):
    """This provider doesn't offer the requested data at all (e.g. NewsAPI has
    no sentiment endpoint, BLS doesn't publish GDP, BEA doesn't publish CPI).
    Retryable because another configured provider may genuinely support it.
    """
