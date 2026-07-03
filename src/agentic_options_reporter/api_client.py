"""Shared HTTP client for the AgenticOptionsReporter API.

Uses `requests` (not shell `curl`). Both `cli.py` (argparse) and
`frontend/app.py` (Flet) depend on this module rather than talking to
`requests` directly, so the two clients stay in sync with each other and
with the contract in specs/api.yaml.
"""

from __future__ import annotations

from typing import Any

import requests

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT_SECONDS = 30


class ApiError(RuntimeError):
    """Raised when the API cannot be reached or returns an error status."""


class ApiClient:
    def __init__(
        self, base_url: str = DEFAULT_BASE_URL, timeout: int = DEFAULT_TIMEOUT_SECONDS
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _request(
        self,
        method: str,
        path: str,
        params: dict[str, Any] | None = None,
        json_body: dict[str, Any] | None = None,
    ) -> Any:
        url = f"{self.base_url}{path}"
        try:
            response = requests.request(
                method, url, params=params, json=json_body, timeout=self.timeout
            )
        except requests.exceptions.RequestException as exc:
            raise ApiError(f"Request to {url} failed: {exc}") from exc

        if not response.ok:
            raise ApiError(f"{method} {url} returned {response.status_code}: {response.text}")
        return response.json()

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def analyze(
        self, symbol: str, lookback_days: int = 365, expiration: str | None = None
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"lookback_days": lookback_days}
        if expiration:
            params["expiration"] = expiration
        return self._request("GET", f"/analyze/{symbol}", params=params)

    def list_runs(self, symbol: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"limit": limit}
        if symbol:
            params["symbol"] = symbol
        return self._request("GET", "/runs", params=params)

    def get_run(self, run_id: int) -> dict[str, Any]:
        return self._request("GET", f"/runs/{run_id}")

    def generate_thesis(
        self,
        run_id: int,
        regenerate: bool = False,
        provider: str = "auto",
        api_key: str | None = None,
    ) -> dict[str, Any]:
        body = {"provider": provider, "api_key": api_key, "regenerate": regenerate}
        return self._request("POST", f"/runs/{run_id}/thesis", json_body=body)

    def get_thesis(self, run_id: int) -> dict[str, Any]:
        return self._request("GET", f"/runs/{run_id}/thesis")
