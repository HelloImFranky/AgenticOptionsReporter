"""SEC filings data access.

`SECProvider` is the interface used by future catalyst/research agents
(dependency injection — the same pattern as
`market_data.MarketDataProvider`). `SecEdgarProvider` is the phase-2a
implementation (see specs/providers.yaml), backed by the free, keyless
SEC EDGAR API. EDGAR's fair-access policy requires a descriptive
User-Agent identifying the requester; see SEC_EDGAR_USER_AGENT below.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from datetime import date
from typing import Any

from agentic_options_reporter.models.schemas import SecFiling


class SecProviderError(RuntimeError):
    """Raised when a SECProvider cannot return the requested data."""


class SECProvider(ABC):
    """Interface implemented by all SEC filings providers."""

    @abstractmethod
    def get_recent_filings(self, ticker: str, limit: int = 10) -> list[SecFiling]:
        raise NotImplementedError

    @abstractmethod
    def get_10k(self, ticker: str) -> SecFiling | None:
        raise NotImplementedError

    @abstractmethod
    def get_10q(self, ticker: str) -> SecFiling | None:
        raise NotImplementedError

    @abstractmethod
    def get_8k(self, ticker: str) -> SecFiling | None:
        raise NotImplementedError


class SecEdgarProvider(SECProvider):
    """SECProvider implementation backed by SEC EDGAR (free, keyless)."""

    TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
    SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
    DEFAULT_USER_AGENT = "AgenticOptionsReporter research (contact: set SEC_EDGAR_USER_AGENT)"

    def __init__(self, user_agent: str | None = None, timeout_seconds: int = 15) -> None:
        self._user_agent = user_agent or os.environ.get(
            "SEC_EDGAR_USER_AGENT", self.DEFAULT_USER_AGENT
        )
        self._timeout = timeout_seconds
        self._ticker_to_cik: dict[str, str] | None = None

    def _get(self, url: str) -> Any:
        import requests

        try:
            response = requests.get(
                url, headers={"User-Agent": self._user_agent}, timeout=self._timeout
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as exc:
            raise SecProviderError(f"SEC EDGAR request to {url} failed: {exc}") from exc
        return response.json()

    def _load_ticker_map(self) -> dict[str, str]:
        if self._ticker_to_cik is not None:
            return self._ticker_to_cik

        data = self._get(self.TICKER_MAP_URL)
        self._ticker_to_cik = {
            entry["ticker"].upper(): str(entry["cik_str"]).zfill(10) for entry in data.values()
        }
        return self._ticker_to_cik

    def _cik_for(self, ticker: str) -> str:
        mapping = self._load_ticker_map()
        cik = mapping.get(ticker.upper())
        if cik is None:
            raise SecProviderError(f"No CIK found for ticker {ticker!r}")
        return cik

    def get_recent_filings(self, ticker: str, limit: int = 10) -> list[SecFiling]:
        cik = self._cik_for(ticker)
        data = self._get(self.SUBMISSIONS_URL.format(cik=cik))
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

    def _first_of_type(self, ticker: str, form_type: str) -> SecFiling | None:
        for filing in self.get_recent_filings(ticker, limit=50):
            if filing.form_type == form_type:
                return filing
        return None

    def get_10k(self, ticker: str) -> SecFiling | None:
        return self._first_of_type(ticker, "10-K")

    def get_10q(self, ticker: str) -> SecFiling | None:
        return self._first_of_type(ticker, "10-Q")

    def get_8k(self, ticker: str) -> SecFiling | None:
        return self._first_of_type(ticker, "8-K")
