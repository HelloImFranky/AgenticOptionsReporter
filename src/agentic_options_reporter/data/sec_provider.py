"""SEC filings data access.

`SECProvider` is the async interface used by the catalyst/research agents
(dependency injection — the same pattern as the news/financial/macro
providers). `SecEdgarProvider` is the phase-2a implementation (see
specs/providers.yaml), backed by the free, keyless SEC EDGAR API and
built on the shared async-HTTP infrastructure (data.async_http: httpx
error normalization, class-level TTL response cache, health probe).
EDGAR's fair-access policy requires a descriptive User-Agent identifying
the requester; see SEC_EDGAR_USER_AGENT / DEFAULT_USER_AGENT below.

Unlike news/financial/macro there is one keyless source and no failover
router — EDGAR is the sole free provider of the full filings index — so
this stays a single adapter module rather than a package.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import date
from typing import Any

from agentic_options_reporter.data.async_http import AsyncHttpProviderBase, ProviderHealth
from agentic_options_reporter.data.provider_errors import (
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
)
from agentic_options_reporter.models.schemas import SecFiling


class SecProviderError(RuntimeError):
    """Raised when a SECProvider cannot return the requested data."""


class SecProviderRateLimited(SecProviderError, ProviderRateLimited):
    """The provider rejected the request for exceeding its rate limit (HTTP 429)."""


class SecProviderTimeout(SecProviderError, ProviderTimeout):
    """The request to the provider timed out."""


class SecProviderUnavailable(SecProviderError, ProviderUnavailable):
    """The provider is unreachable or returned a server error (5xx / network failure)."""


class SECProvider(ABC):
    """Interface implemented by all SEC filings providers."""

    @abstractmethod
    async def get_recent_filings(self, ticker: str, limit: int = 10) -> list[SecFiling]:
        raise NotImplementedError

    @abstractmethod
    async def get_10k(self, ticker: str) -> SecFiling | None:
        raise NotImplementedError

    @abstractmethod
    async def get_10q(self, ticker: str) -> SecFiling | None:
        raise NotImplementedError

    @abstractmethod
    async def get_8k(self, ticker: str) -> SecFiling | None:
        raise NotImplementedError

    @abstractmethod
    async def health(self) -> ProviderHealth:
        raise NotImplementedError


class SecEdgarProvider(AsyncHttpProviderBase, SECProvider):
    """SECProvider implementation backed by SEC EDGAR (free, keyless)."""

    TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
    DEFAULT_USER_AGENT = "AgenticOptionsReporter research (contact: set SEC_EDGAR_USER_AGENT)"

    PROVIDER_LABEL = "SEC EDGAR"
    API_KEY_ENV_VAR = None  # keyless

    ERROR_CLS = SecProviderError
    RATE_LIMITED_CLS = SecProviderRateLimited
    TIMEOUT_CLS = SecProviderTimeout
    UNAVAILABLE_CLS = SecProviderUnavailable

    def __init__(
        self,
        user_agent: str | None = None,
        timeout_seconds: float = 15.0,
        client: Any | None = None,
    ) -> None:
        super().__init__(timeout_seconds=timeout_seconds, client=client)
        self._user_agent = user_agent or os.environ.get(
            "SEC_EDGAR_USER_AGENT", self.DEFAULT_USER_AGENT
        )
        self._ticker_to_cik: dict[str, str] | None = None

    async def _edgar_get(self, url: str) -> Any:
        # EDGAR's fair-access policy requires an identifying User-Agent;
        # the endpoints take no query params.
        return await self._get_json(url, {}, headers={"User-Agent": self._user_agent})

    async def _load_ticker_map(self) -> dict[str, str]:
        if self._ticker_to_cik is not None:
            return self._ticker_to_cik

        data = await self._edgar_get(self.TICKER_MAP_URL)
        self._ticker_to_cik = {
            entry["ticker"].upper(): str(entry["cik_str"]).zfill(10) for entry in data.values()
        }
        return self._ticker_to_cik

    async def _cik_for(self, ticker: str) -> str:
        mapping = await self._load_ticker_map()
        cik = mapping.get(ticker.upper())
        if cik is None:
            raise SecProviderError(f"No CIK found for ticker {ticker!r}")
        return cik

    async def get_recent_filings(self, ticker: str, limit: int = 10) -> list[SecFiling]:
        cik = await self._cik_for(ticker)
        data = await self._edgar_get(self.SUBMISSIONS_URL.format(cik=cik))
        recent = (data.get("filings") or {}).get("recent") or {}

        forms = recent.get("form", [])
        filing_dates = recent.get("filingDate", [])
        accession_numbers = recent.get("accessionNumber", [])
        primary_documents = recent.get("primaryDocument", [])

        filings = []
        for i in range(min(limit, len(forms))):
            accession_no_dashes = accession_numbers[i].replace("-", "")
            doc_url = (
                f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                f"{accession_no_dashes}/{primary_documents[i]}"
            )
            filings.append(
                SecFiling(
                    ticker=ticker.upper(),
                    form_type=forms[i],
                    filed_at=date.fromisoformat(filing_dates[i]),
                    url=doc_url,
                    accession_number=accession_numbers[i],
                )
            )
        return filings

    async def _first_of_type(self, ticker: str, form_type: str) -> SecFiling | None:
        for filing in await self.get_recent_filings(ticker, limit=50):
            if filing.form_type == form_type:
                return filing
        return None

    async def get_10k(self, ticker: str) -> SecFiling | None:
        return await self._first_of_type(ticker, "10-K")

    async def get_10q(self, ticker: str) -> SecFiling | None:
        return await self._first_of_type(ticker, "10-Q")

    async def get_8k(self, ticker: str) -> SecFiling | None:
        return await self._first_of_type(ticker, "8-K")

    async def _health_probe(self) -> None:
        await self._load_ticker_map()
