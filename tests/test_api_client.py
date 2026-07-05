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

    def fake_request(method, url, params=None, json=None, timeout=None):
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

    def fake_request(method, url, params=None, json=None, timeout=None):
        captured["url"] = url
        return _FakeResponse(payload={})

    monkeypatch.setattr(requests_module, "request", fake_request)

    client = ApiClient(base_url="http://localhost:8000/")
    client.health()

    assert captured["url"] == "http://localhost:8000/health"


def test_analyze_passes_expiration_only_when_set(monkeypatch):
    captured = {}

    def fake_request(method, url, params=None, json=None, timeout=None):
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

    def fake_request(method, url, params=None, json=None, timeout=None):
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

    def fake_request(method, url, params=None, json=None, timeout=None):
        captured["url"] = url
        return _FakeResponse(payload={"run_id": 42})

    monkeypatch.setattr(requests_module, "request", fake_request)
    client = ApiClient()

    client.get_run(42)
    assert captured["url"] == "http://localhost:8000/runs/42"


def test_generate_thesis_builds_expected_request(monkeypatch):
    captured = {}

    def fake_request(method, url, params=None, json=None, timeout=None):
        captured["method"] = method
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse(payload={"run_id": 42})

    monkeypatch.setattr(requests_module, "request", fake_request)
    client = ApiClient()

    client.generate_thesis(42)
    assert captured["method"] == "POST"
    assert captured["url"] == "http://localhost:8000/runs/42/thesis"
    assert captured["json"] == {"provider": "auto", "api_key": None, "regenerate": False}


def test_generate_thesis_passes_regenerate_flag(monkeypatch):
    captured = {}

    def fake_request(method, url, params=None, json=None, timeout=None):
        captured["json"] = json
        return _FakeResponse(payload={"run_id": 42})

    monkeypatch.setattr(requests_module, "request", fake_request)
    client = ApiClient()

    client.generate_thesis(42, regenerate=True)
    assert captured["json"]["regenerate"] is True


def test_generate_thesis_passes_provider_and_api_key(monkeypatch):
    captured = {}

    def fake_request(method, url, params=None, json=None, timeout=None):
        captured["json"] = json
        return _FakeResponse(payload={"run_id": 42})

    monkeypatch.setattr(requests_module, "request", fake_request)
    client = ApiClient()

    client.generate_thesis(42, provider="openai", api_key="sk-custom")
    assert captured["json"] == {"provider": "openai", "api_key": "sk-custom", "regenerate": False}


def test_get_thesis_builds_expected_path(monkeypatch):
    captured = {}

    def fake_request(method, url, params=None, json=None, timeout=None):
        captured["method"] = method
        captured["url"] = url
        return _FakeResponse(payload={"run_id": 42})

    monkeypatch.setattr(requests_module, "request", fake_request)
    client = ApiClient()

    client.get_thesis(42)
    assert captured["method"] == "GET"
    assert captured["url"] == "http://localhost:8000/runs/42/thesis"


def test_generate_thesis_raises_api_error_on_conflict(monkeypatch):
    def fake_request(method, url, params=None, json=None, timeout=None):
        return _FakeResponse(status_code=409, text="already exists")

    monkeypatch.setattr(requests_module, "request", fake_request)

    with pytest.raises(ApiError):
        ApiClient().generate_thesis(42)


def test_request_raises_api_error_on_http_failure(monkeypatch):
    def fake_request(method, url, params=None, json=None, timeout=None):
        return _FakeResponse(status_code=404, text="not found")

    monkeypatch.setattr(requests_module, "request", fake_request)

    with pytest.raises(ApiError):
        ApiClient().get_run(999)


def test_request_raises_api_error_on_connection_failure(monkeypatch):
    def fake_request(method, url, params=None, json=None, timeout=None):
        raise requests_module.exceptions.ConnectionError("boom")

    monkeypatch.setattr(requests_module, "request", fake_request)

    with pytest.raises(ApiError):
        ApiClient().health()


class _FakeStreamResponse:
    """Stand-in for a streaming requests.Response: yields SSE lines and
    supports the `with response:` context-manager protocol."""

    def __init__(self, lines, status_code: int = 200, text: str = ""):
        self._lines = lines
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def iter_lines(self, decode_unicode=False):
        yield from self._lines


def test_stream_thesis_parses_sse_frames(monkeypatch):
    lines = [
        "event: agent",
        'data: {"agent": "quant_interpreter", "phase": "started"}',
        "",
        "event: agent",
        'data: {"agent": "quant_interpreter", "phase": "completed", "output": {"x": 1}}',
        "",
        "event: result",
        'data: {"investment_thesis": {"consensus": "bullish"}}',
        "",
    ]
    captured = {}

    def fake_post(url, json=None, stream=None, timeout=None):
        captured["url"] = url
        captured["json"] = json
        captured["stream"] = stream
        return _FakeStreamResponse(lines)

    monkeypatch.setattr(requests_module, "post", fake_post)

    events = list(ApiClient().stream_thesis(7, provider="openai", api_key="sk-x"))

    assert captured["url"].endswith("/runs/7/thesis/stream")
    assert captured["stream"] is True
    assert captured["json"] == {"provider": "openai", "api_key": "sk-x", "regenerate": True}
    assert [e["event"] for e in events] == ["agent", "agent", "result"]
    assert events[0]["data"]["phase"] == "started"
    assert events[-1]["data"]["investment_thesis"]["consensus"] == "bullish"


def test_stream_thesis_multiline_data_frame(monkeypatch):
    """SSE allows a single frame's data to span multiple `data:` lines; they
    must be rejoined with newlines before JSON parsing."""
    lines = [
        "event: agent",
        'data: {"agent": "quant_interpreter",',
        'data: "phase": "started"}',
        "",
    ]

    def fake_post(url, json=None, stream=None, timeout=None):
        return _FakeStreamResponse(lines)

    monkeypatch.setattr(requests_module, "post", fake_post)

    events = list(ApiClient().stream_thesis(1))
    assert events == [{"event": "agent", "data": {"agent": "quant_interpreter", "phase": "started"}}]


def test_stream_thesis_raises_api_error_on_non_ok(monkeypatch):
    def fake_post(url, json=None, stream=None, timeout=None):
        return _FakeStreamResponse([], status_code=404, text="Run 9 not found")

    monkeypatch.setattr(requests_module, "post", fake_post)

    with pytest.raises(ApiError):
        list(ApiClient().stream_thesis(9))


def test_stream_thesis_raises_api_error_on_connection_failure(monkeypatch):
    def fake_post(url, json=None, stream=None, timeout=None):
        raise requests_module.exceptions.ConnectionError("boom")

    monkeypatch.setattr(requests_module, "post", fake_post)

    with pytest.raises(ApiError):
        list(ApiClient().stream_thesis(1))
