"""Macro provider capability-filtering router and factory.

Unlike the news/financial routers (which try every provider per method
and catch Unsupported mid-call), this router SELECTS providers by
declared capability before calling: for a metric id, it narrows to the
providers that advertise it, applies any per-metric priority override,
then fails over among just those on transient errors. A provider is
never asked for a metric it doesn't publish — the fix for "World Bank
has no US policy rate."
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

from agentic_options_reporter.data.macro.base import (
    MacroProvider,
    MacroProviderError,
    MacroProviderUnsupported,
    ProviderHealth,
)
from agentic_options_reporter.data.macro.bea import BeaMacroProvider
from agentic_options_reporter.data.macro.bls import BlsMacroProvider
from agentic_options_reporter.data.macro.fred import FredMacroProvider
from agentic_options_reporter.data.macro.imf import ImfMacroProvider
from agentic_options_reporter.data.macro.worldbank import WorldBankMacroProvider
from agentic_options_reporter.data.provider_router import acall_with_fallback, filter_supporting
from agentic_options_reporter.models.schemas import MacroObservation


class MacroProviderRouter(MacroProvider):
    """Capability-filtering failover router across configured macro
    adapters. Implements MacroProvider itself, so the macro_research
    consumer can't tell whether it's talking to one adapter or many."""

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

    @property
    def supported_metrics(self) -> frozenset[str]:
        return frozenset().union(*(client.supported_metrics for _, client in self._clients))

    def _candidates_for(self, metric_id: str) -> list[tuple[str, MacroProvider]]:
        candidates = filter_supporting(self._clients, metric_id)
        override = _metric_priority_override(metric_id)
        if override:
            rank = {name: i for i, name in enumerate(override)}
            candidates.sort(key=lambda nc: rank.get(nc[0], len(override)))
        return candidates

    async def fetch(self, metric_id: str) -> MacroObservation:
        candidates = self._candidates_for(metric_id)
        if not candidates:
            raise MacroProviderUnsupported(
                f"No configured macro provider serves metric '{metric_id}'."
            )
        return await acall_with_fallback(candidates, "fetch", MacroProviderError, metric_id)

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


def _metric_priority_override(metric_id: str) -> list[str]:
    """Optional per-metric provider priority, e.g.
    AOR_MACRO_PRIORITY_GDP="bea,fred,worldbank" to prefer BEA's GDP over
    FRED's mirror. Falls back to the global fallback order when unset
    (see specs/providers.yaml: configurable_priority)."""
    raw = os.environ.get(f"AOR_MACRO_PRIORITY_{metric_id.upper()}", "")
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
