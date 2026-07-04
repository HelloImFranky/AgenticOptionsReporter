"""Market-data access — one adapter per source, per specs/providers.yaml.

Public surface of the package; consumers import from here rather than
from individual adapter modules.
"""

from agentic_options_reporter.data.market_data.alphavantage import AlphaVantageMarketDataProvider
from agentic_options_reporter.data.market_data.base import (
    MARKET_DATA_CAPABILITIES,
    OPTION_CHAIN,
    PRICE_HISTORY,
    MarketDataError,
    MarketDataProvider,
    MarketDataRateLimited,
    MarketDataTimeout,
    MarketDataUnavailable,
    MarketDataUnsupported,
    ProviderHealth,
)
from agentic_options_reporter.data.market_data.finnhub import FinnhubMarketDataProvider
from agentic_options_reporter.data.market_data.router import (
    MarketDataProviderRouter,
    build_market_data_provider,
)
from agentic_options_reporter.data.market_data.twelvedata import TwelveDataMarketDataProvider
from agentic_options_reporter.data.market_data.yfinance_provider import YFinanceProvider

__all__ = [
    "MARKET_DATA_CAPABILITIES",
    "OPTION_CHAIN",
    "PRICE_HISTORY",
    "AlphaVantageMarketDataProvider",
    "FinnhubMarketDataProvider",
    "MarketDataError",
    "MarketDataProvider",
    "MarketDataProviderRouter",
    "MarketDataRateLimited",
    "MarketDataTimeout",
    "MarketDataUnavailable",
    "MarketDataUnsupported",
    "ProviderHealth",
    "TwelveDataMarketDataProvider",
    "YFinanceProvider",
    "build_market_data_provider",
]
