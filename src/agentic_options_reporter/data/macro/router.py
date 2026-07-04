"""Macro provider failover router and configuration-driven factory."""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

from agentic_options_reporter.data.macro.base import (
    MacroProvider,
    MacroProviderError,
    ProviderHealth,
)
from agentic_options_reporter.data.macro.bea import BeaMacroProvider
from agentic_options_reporter.data.macro.bls import BlsMacroProvider
from agentic_options_reporter.data.macro.fred import FredMacroProvider
from agentic_options_reporter.data.macro.imf import ImfMacroProvider
from agentic_options_reporter.data.macro.worldbank import WorldBankMacroProvider
from agentic_options_reporter.data.provider_router import acall_with_fallback
from agentic_options_reporter.models.schemas import (
    CpiSnapshot,
    GdpSnapshot,
    InterestRates,
    MacroEvent,
)


class MacroProviderRouter(MacroProvider):
    """Tries a priority-ordered list of already-constructed MacroProvider
    adapters per method call, advancing to the next on a retryable
    failure (see data.provider_router) — essential here, since most macro
    sources are specialists that raise Unsupported outside their domain."""

    def __init__(self, clients: list[tuple[str, MacroProvider]]) -> None:
        if not clients:
            raise MacroProviderError(
                "No macro providers are configured for automatic failover. Set at least "
                f"one provider's API key (supported: {', '.join(sorted(_PROVIDERS))})."
            )
        self._clients = clients

    @property
    def provider_names(self) -> list[str]:
        return [name for name, _ in self._clients]

    async def get_interest_rates(self) -> InterestRates:
        return await acall_with_fallback(self._clients, "get_interest_rates", MacroProviderError)

    async def get_cpi(self) -> CpiSnapshot:
        return await acall_with_fallback(self._clients, "get_cpi", MacroProviderError)

    async def get_gdp(self) -> GdpSnapshot:
        return await acall_with_fallback(self._clients, "get_gdp", MacroProviderError)

    async def get_macro_calendar(self) -> list[MacroEvent]:
        return await acall_with_fallback(self._clients, "get_macro_calendar", MacroProviderError)

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


_PROVIDERS: dict[str, type[MacroProvider]] = {
    "fred": FredMacroProvider,
    "bls": BlsMacroProvider,
    "bea": BeaMacroProvider,
    "imf": ImfMacroProvider,
    "worldbank": WorldBankMacroProvider,
}

# FRED first (broadest coverage, freshest data), then the primary US
# agencies for their specialties, then the keyless-but-lagging
# international sources as last resorts.
_DEFAULT_FALLBACK_ORDER = ["fred", "bls", "bea", "imf", "worldbank"]


def _fallback_order() -> list[str]:
    raw = os.environ.get("AOR_MACRO_PROVIDER_FALLBACK_ORDER", ",".join(_DEFAULT_FALLBACK_ORDER))
    return [name.strip().lower() for name in raw.split(",") if name.strip()]


def build_macro_provider() -> MacroProviderRouter:
    """Build a MacroProviderRouter from AOR_MACRO_PROVIDER_FALLBACK_ORDER,
    skipping any provider without a configured API key. IMF and the World
    Bank are keyless, so with the default order the router is never
    empty — macro research is effectively always available."""
    clients: list[tuple[str, MacroProvider]] = []
    for name in _fallback_order():
        provider_cls = _PROVIDERS.get(name)
        if provider_cls is None:
            continue
        try:
            clients.append((name, provider_cls()))
        except MacroProviderError:
            continue  # not configured (missing API key) — skip, don't fail the request
    return MacroProviderRouter(clients)
