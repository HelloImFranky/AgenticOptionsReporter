"""Company-fundamentals data access — one adapter per source, per
specs/providers.yaml.

Public surface of the package; consumers import from here rather than
from individual adapter modules.
"""

from agentic_options_reporter.data.financial.alphavantage import AlphaVantageFinancialProvider
from agentic_options_reporter.data.financial.base import (
    ANALYST_ESTIMATES,
    EARNINGS,
    EARNINGS_CALENDAR,
    FINANCIAL_DATASETS,
    INSIDER,
    METRICS,
    PROFILE,
    RATIOS,
    STATEMENTS,
    FinancialProvider,
    FinancialProviderError,
    FinancialProviderRateLimited,
    FinancialProviderTimeout,
    FinancialProviderUnavailable,
    FinancialProviderUnsupported,
    ProviderHealth,
)
from agentic_options_reporter.data.financial.finnhub import FinnhubFinancialProvider
from agentic_options_reporter.data.financial.fmp import FmpFinancialProvider
from agentic_options_reporter.data.financial.router import (
    FinancialProviderRouter,
    build_financial_provider,
)
from agentic_options_reporter.data.financial.yfinance_provider import YFinanceFinancialProvider

__all__ = [
    "ANALYST_ESTIMATES",
    "EARNINGS",
    "EARNINGS_CALENDAR",
    "FINANCIAL_DATASETS",
    "INSIDER",
    "METRICS",
    "PROFILE",
    "RATIOS",
    "STATEMENTS",
    "AlphaVantageFinancialProvider",
    "FinancialProvider",
    "FinancialProviderError",
    "FinancialProviderRateLimited",
    "FinancialProviderRouter",
    "FinancialProviderTimeout",
    "FinancialProviderUnavailable",
    "FinancialProviderUnsupported",
    "FinnhubFinancialProvider",
    "FmpFinancialProvider",
    "ProviderHealth",
    "YFinanceFinancialProvider",
    "build_financial_provider",
]
