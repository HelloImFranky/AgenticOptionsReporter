import json

import pytest
import requests as requests_module

from agentic_options_reporter.api_client import ApiClient, ApiError


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload=None, text: str = ""):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._payload = payload if payload is not None else {}
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload


def test_request_uses_requests_and_returns_json(monkeypatch):
    captured = {}

    def fake_request(method, url, params=None, timeout=None):
        captured["method"] = method
        captured["url"] = url
        captured["params"] = params
        return _FakeResponse(payload={"status": "ok"})

    monkeypatch.setattr(requests_module, "request", fake_request)

    client = ApiClient(base_url="http://localhost:8000")
    result = client.health()

    assert result == {"status": "ok"}
    assert captured["method"] == "GET"
    assert captured["url"] == "http://localhost:8000/health"


def test_base_url_trailing_slash_is_stripped(monkeypatch):
    captured = {}

    def fake_request(method, url, params=None, timeout=None):
        captured["url"] = url
        return _FakeResponse(payload={})

    monkeypatch.setattr(requests_module, "request", fake_request)

    client = ApiClient(base_url="http://localhost:8000/")
    client.health()

    assert captured["url"] == "http://localhost:8000/health"


def test_analyze_passes_expiration_only_when_set(monkeypatch):
    captured = {}

    def fake_request(method, url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return _FakeResponse(payload={"symbol": "AAPL"})

    monkeypatch.setattr(requests_module, "request", fake_request)
    client = ApiClient()

    client.analyze("AAPL", lookback_days=90)
    assert captured["params"] == {"lookback_days": 90}

    client.analyze("AAPL", lookback_days=90, expiration="2026-01-16")
    assert captured["params"] == {"lookback_days": 90, "expiration": "2026-01-16"}


def test_list_runs_passes_symbol_only_when_set(monkeypatch):
    captured = {}

    def fake_request(method, url, params=None, timeout=None):
        captured["params"] = params
        return _FakeResponse(payload=[])

    monkeypatch.setattr(requests_module, "request", fake_request)
    client = ApiClient()

    client.list_runs(limit=5)
    assert captured["params"] == {"limit": 5}

    client.list_runs(symbol="AAPL", limit=5)
    assert captured["params"] == {"limit": 5, "symbol": "AAPL"}


def test_get_run_builds_expected_path(monkeypatch):
    captured = {}

    def fake_request(method, url, params=None, timeout=None):
        captured["url"] = url
        return _FakeResponse(payload={"run_id": 42})

    monkeypatch.setattr(requests_module, "request", fake_request)
    client = ApiClient()

    client.get_run(42)
    assert captured["url"] == "http://localhost:8000/runs/42"


def test_request_raises_api_error_on_http_failure(monkeypatch):
    def fake_request(method, url, params=None, timeout=None):
        return _FakeResponse(status_code=404, text="not found")

    monkeypatch.setattr(requests_module, "request", fake_request)

    with pytest.raises(ApiError):
        ApiClient().get_run(999)


def test_request_raises_api_error_on_connection_failure(monkeypatch):
    def fake_request(method, url, params=None, timeout=None):
        raise requests_module.exceptions.ConnectionError("boom")

    monkeypatch.setattr(requests_module, "request", fake_request)

    with pytest.raises(ApiError):
        ApiClient().health()
