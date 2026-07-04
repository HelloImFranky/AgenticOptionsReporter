"""News data access — one adapter per source, per specs/providers.yaml.

Public surface of the package; consumers import from here rather than
from individual adapter modules.
"""

from agentic_options_reporter.data.news.alphavantage import AlphaVantageNewsProvider
from agentic_options_reporter.data.news.base import (
    COMPANY_NEWS,
    GENERAL_NEWS,
    NEWS_CAPABILITIES,
    TOP_HEADLINES,
    NewsProvider,
    NewsProviderError,
    NewsProviderRateLimited,
    NewsProviderTimeout,
    NewsProviderUnavailable,
    NewsProviderUnsupported,
    ProviderHealth,
)
from agentic_options_reporter.data.news.finnhub import FinnhubNewsProvider
from agentic_options_reporter.data.news.gnews import GNewsProvider
from agentic_options_reporter.data.news.guardian import GuardianNewsProvider
from agentic_options_reporter.data.news.hackernews import HackerNewsProvider
from agentic_options_reporter.data.news.newsapi import NewsApiOrgProvider
from agentic_options_reporter.data.news.newsdata import NewsDataProvider
from agentic_options_reporter.data.news.router import NewsProviderRouter, build_news_provider

__all__ = [
    "COMPANY_NEWS",
    "GENERAL_NEWS",
    "NEWS_CAPABILITIES",
    "TOP_HEADLINES",
    "AlphaVantageNewsProvider",
    "FinnhubNewsProvider",
    "GNewsProvider",
    "GuardianNewsProvider",
    "HackerNewsProvider",
    "NewsApiOrgProvider",
    "NewsDataProvider",
    "NewsProvider",
    "NewsProviderError",
    "NewsProviderRateLimited",
    "NewsProviderRouter",
    "NewsProviderTimeout",
    "NewsProviderUnavailable",
    "NewsProviderUnsupported",
    "ProviderHealth",
    "build_news_provider",
]
