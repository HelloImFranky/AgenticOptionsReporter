"""Macroeconomic data access — one adapter per source, per
specs/providers.yaml.

Public surface of the package; consumers import from here rather than
from individual adapter modules.
"""

from agentic_options_reporter.data.macro.base import (
    MacroProvider,
    MacroProviderError,
    MacroProviderRateLimited,
    MacroProviderTimeout,
    MacroProviderUnavailable,
    MacroProviderUnsupported,
    ProviderHealth,
)
from agentic_options_reporter.data.macro.bea import BeaMacroProvider
from agentic_options_reporter.data.macro.bls import BlsMacroProvider
from agentic_options_reporter.data.macro.fred import FredMacroProvider
from agentic_options_reporter.data.macro.imf import ImfMacroProvider
from agentic_options_reporter.data.macro.metrics import (
    DEFAULT_MACRO_METRICS,
    MACRO_METRICS,
    MacroMetric,
    get_metric,
)
from agentic_options_reporter.data.macro.router import MacroProviderRouter, build_macro_provider
from agentic_options_reporter.data.macro.worldbank import WorldBankMacroProvider

__all__ = [
    "DEFAULT_MACRO_METRICS",
    "MACRO_METRICS",
    "BeaMacroProvider",
    "BlsMacroProvider",
    "FredMacroProvider",
    "ImfMacroProvider",
    "MacroMetric",
    "MacroProvider",
    "MacroProviderError",
    "MacroProviderRateLimited",
    "MacroProviderRouter",
    "MacroProviderTimeout",
    "MacroProviderUnavailable",
    "MacroProviderUnsupported",
    "ProviderHealth",
    "WorldBankMacroProvider",
    "build_macro_provider",
    "get_metric",
]
