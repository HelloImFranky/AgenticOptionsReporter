"""Shared HTTP client for the AgenticOptionsReporter API.

Uses `requests` (not shell `curl`). Both `cli.py` (argparse) and
`frontend/app.py` (Flet) depend on this module rather than talking to
`requests` directly, so the two clients stay in sync with each other and
with the contract in specs/api.yaml.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Any

import requests

DEFAULT_BASE_URL = "http://localhost:8000"
DEFAULT_TIMEOUT_SECONDS = 30
# The streaming thesis endpoint stays open for the whole pipeline; between
# agents there can be tens of seconds of model latency, so allow a long
# per-read gap before giving up.
DEFAULT_STREAM_TIMEOUT_SECONDS = 300


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

    def get_logs(self, since_seq: int = 0, limit: int = 500) -> list[dict[str, Any]]:
        return self._request("GET", "/logs", params={"since_seq": since_seq, "limit": limit})

    def analyze(
        self,
        symbol: str,
        lookback_days: int = 365,
        expiration: str | None = None,
        weighting_profile: str = "swing",
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"lookback_days": lookback_days, "weighting_profile": weighting_profile}
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

    def stream_thesis(
        self,
        run_id: int,
        regenerate: bool = True,
        provider: str = "auto",
        api_key: str | None = None,
        stream_timeout: int = DEFAULT_STREAM_TIMEOUT_SECONDS,
    ) -> Iterator[dict[str, Any]]:
        """Yield live events from the Server-Sent Events thesis stream as
        each agent runs. Each yielded item is `{"event": name, "data": ...}`:
          - "agent"  → an AgentEvent (agent, phase, exchange, output, ...)
          - "result" → the final AgentThesisResult (also persisted server-side)
          - "error"  → {"detail": ...} when a required agent failed
        The pre-stream guards (404/409/422) surface as an ApiError, same as
        the blocking `generate_thesis`."""
        url = f"{self.base_url}/runs/{run_id}/thesis/stream"
        body = {"provider": provider, "api_key": api_key, "regenerate": regenerate}
        try:
            response = requests.post(url, json=body, stream=True, timeout=stream_timeout)
        except requests.exceptions.RequestException as exc:
            raise ApiError(f"Request to {url} failed: {exc}") from exc

        with response:
            if not response.ok:
                raise ApiError(f"POST {url} returned {response.status_code}: {response.text}")
            event_name: str | None = None
            data_lines: list[str] = []
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    if line.startswith("event:"):
                        event_name = line[len("event:") :].strip()
                    elif line.startswith("data:"):
                        data_lines.append(line[len("data:") :].lstrip())
                    continue
                # blank line terminates one SSE frame
                if event_name is not None and data_lines:
                    yield {"event": event_name, "data": json.loads("\n".join(data_lines))}
                event_name, data_lines = None, []
