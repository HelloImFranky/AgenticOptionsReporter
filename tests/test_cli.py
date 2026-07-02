import json

import pytest
import requests as requests_module

from agentic_options_reporter import cli


class _FakeResponse:
    def __init__(self, status_code: int = 200, payload=None, text: str = ""):
        self.status_code = status_code
        self.ok = 200 <= status_code < 300
        self._payload = payload if payload is not None else {}
        self.text = text or json.dumps(self._payload)

    def json(self):
        return self._payload


def test_build_parser_requires_command():
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_analyze_parses_defaults():
    parser = cli.build_parser()
    args = parser.parse_args(["analyze", "AAPL"])
    assert args.symbol == "AAPL"
    assert args.lookback_days == 365
    assert args.expiration is None
    assert args.base_url == cli.DEFAULT_BASE_URL


def test_analyze_parses_overrides():
    parser = cli.build_parser()
    args = parser.parse_args(
        [
            "--base-url",
            "http://example.com",
            "analyze",
            "MSFT",
            "--lookback-days",
            "90",
            "--expiration",
            "2026-01-16",
        ]
    )
    assert args.base_url == "http://example.com"
    assert args.symbol == "MSFT"
    assert args.lookback_days == 90
    assert args.expiration == "2026-01-16"


def test_runs_and_run_parsers():
    parser = cli.build_parser()

    runs_args = parser.parse_args(["runs", "--symbol", "AAPL", "--limit", "5"])
    assert runs_args.symbol == "AAPL"
    assert runs_args.limit == 5

    run_args = parser.parse_args(["run", "42"])
    assert run_args.run_id == 42


def test_request_uses_requests_and_returns_json(monkeypatch):
    captured = {}

    def fake_request(method, url, params=None, timeout=None):
        captured["method"] = method
        captured["url"] = url
        captured["params"] = params
        return _FakeResponse(payload={"status": "ok"})

    monkeypatch.setattr(cli.requests, "request", fake_request)

    result = cli._request("GET", "http://localhost:8000", "/health")

    assert result == {"status": "ok"}
    assert captured["method"] == "GET"
    assert captured["url"] == "http://localhost:8000/health"


def test_request_raises_api_error_on_http_failure(monkeypatch):
    def fake_request(method, url, params=None, timeout=None):
        return _FakeResponse(status_code=404, text="not found")

    monkeypatch.setattr(cli.requests, "request", fake_request)

    with pytest.raises(cli.ApiError):
        cli._request("GET", "http://localhost:8000", "/runs/999")


def test_request_raises_api_error_on_connection_failure(monkeypatch):
    def fake_request(method, url, params=None, timeout=None):
        raise requests_module.exceptions.ConnectionError("boom")

    monkeypatch.setattr(cli.requests, "request", fake_request)

    with pytest.raises(cli.ApiError):
        cli._request("GET", "http://localhost:8000", "/health")


def test_main_prints_json_and_returns_zero(monkeypatch, capsys):
    def fake_request(method, url, params=None, timeout=None):
        return _FakeResponse(payload={"status": "ok"})

    monkeypatch.setattr(cli.requests, "request", fake_request)

    exit_code = cli.main(["health"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"status": "ok"}


def test_main_returns_one_on_api_error(monkeypatch, capsys):
    def fake_request(method, url, params=None, timeout=None):
        return _FakeResponse(status_code=500, text="boom")

    monkeypatch.setattr(cli.requests, "request", fake_request)

    exit_code = cli.main(["health"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "error" in captured.err
