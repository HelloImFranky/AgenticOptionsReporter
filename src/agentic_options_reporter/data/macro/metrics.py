"""Structured macro-metric registry.

The macro layer is capability-based (see specs/providers.yaml): a
provider declares which *metrics* it serves rather than implementing one
fixed method per metric. Each metric is structured data — the doc's
"better long-term" option over a growing capability enum — so a new
metric (unemployment, PPI, ...) is a registry entry plus the adapters
that happen to serve it, with no new interface methods and no provider
forced to stub out data it doesn't have.

Adding a metric here does NOT obligate every provider to serve it; a
provider only advertises the ids it actually publishes, and the router
queries just those (data.macro.router). That is exactly what fixes the
"World Bank has no US policy rate" case: World Bank never advertises
`policy_rate`, so it is never asked for it.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class MacroMetric(BaseModel):
    """A macroeconomic series the platform can request, described by
    structured metadata rather than an enum member."""

    model_config = ConfigDict(frozen=True)

    id: str
    label: str
    category: str  # "interest_rate" | "prices" | "output"
    country: str = "US"
    frequency: str  # "daily" | "monthly" | "quarterly" | "annual"
    unit: str  # "percent" | "index" | "usd"


POLICY_RATE = MacroMetric(
    id="policy_rate", label="Federal funds rate", category="interest_rate",
    frequency="monthly", unit="percent",
)
TREASURY_10Y = MacroMetric(
    id="treasury_10y", label="10-year Treasury yield", category="interest_rate",
    frequency="daily", unit="percent",
)
TREASURY_2Y = MacroMetric(
    id="treasury_2y", label="2-year Treasury yield", category="interest_rate",
    frequency="daily", unit="percent",
)
CPI = MacroMetric(
    id="cpi", label="Consumer Price Index", category="prices",
    frequency="monthly", unit="index",
)
GDP = MacroMetric(
    id="gdp", label="Gross domestic product (nominal)", category="output",
    frequency="quarterly", unit="usd",
)

MACRO_METRICS: dict[str, MacroMetric] = {
    m.id: m for m in (POLICY_RATE, TREASURY_10Y, TREASURY_2Y, CPI, GDP)
}

# The metric set macro_research requests by default. The router serves
# whatever of these some configured provider supports; the rest are
# simply absent (treated as expected, not an error).
DEFAULT_MACRO_METRICS: list[str] = [POLICY_RATE.id, TREASURY_10Y.id, TREASURY_2Y.id, CPI.id, GDP.id]


def get_metric(metric_id: str) -> MacroMetric:
    """Look up a metric's metadata. Raises KeyError for an unknown id —
    an internal programming error, since ids come from this registry."""
    return MACRO_METRICS[metric_id]
