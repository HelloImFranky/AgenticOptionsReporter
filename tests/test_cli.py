import json

import requests as requests_module

from agentic_options_reporter import cli
from agentic_options_reporter.api_client import DEFAULT_BASE_URL
import pytest


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
    assert args.base_url == DEFAULT_BASE_URL


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


def test_thesis_parser_defaults():
    parser = cli.build_parser()
    args = parser.parse_args(["thesis", "42"])
    assert args.run_id == 42
    assert args.regenerate is False
    assert args.fetch_only is False
    assert args.provider == "auto"
    assert args.api_key is None


def test_thesis_parser_flags():
    parser = cli.build_parser()
    args = parser.parse_args(["thesis", "42", "--regenerate"])
    assert args.regenerate is True

    args = parser.parse_args(["thesis", "42", "--fetch-only"])
    assert args.fetch_only is True

    args = parser.parse_args(["thesis", "42", "--provider", "openai", "--api-key", "sk-custom"])
    assert args.provider == "openai"
    assert args.api_key == "sk-custom"


def test_main_prints_json_and_returns_zero(monkeypatch, capsys):
    def fake_request(method, url, params=None, json=None, timeout=None):
        return _FakeResponse(payload={"status": "ok"})

    monkeypatch.setattr(requests_module, "request", fake_request)

    exit_code = cli.main(["health"])

    assert exit_code == 0
    captured = capsys.readouterr()
    assert json.loads(captured.out) == {"status": "ok"}


def test_main_returns_one_on_api_error(monkeypatch, capsys):
    def fake_request(method, url, params=None, json=None, timeout=None):
        return _FakeResponse(status_code=500, text="boom")

    monkeypatch.setattr(requests_module, "request", fake_request)

    exit_code = cli.main(["health"])

    assert exit_code == 1
    captured = capsys.readouterr()
    assert "error" in captured.err


def test_main_analyze_uses_base_url_and_symbol(monkeypatch, capsys):
    captured = {}

    def fake_request(method, url, params=None, json=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return _FakeResponse(payload={"symbol": "AAPL"})

    monkeypatch.setattr(requests_module, "request", fake_request)

    exit_code = cli.main(["--base-url", "http://example.com", "analyze", "AAPL"])

    assert exit_code == 0
    assert captured["url"] == "http://example.com/analyze/AAPL"
    assert captured["params"] == {"lookback_days": 365}


def test_main_thesis_generates_by_default(monkeypatch):
    captured = {}

    def fake_request(method, url, params=None, json=None, timeout=None):
        captured["method"] = method
        captured["url"] = url
        captured["json"] = json
        return _FakeResponse(payload={"run_id": 42})

    monkeypatch.setattr(requests_module, "request", fake_request)

    exit_code = cli.main(["thesis", "42"])

    assert exit_code == 0
    assert captured["method"] == "POST"
    assert captured["url"] == "http://localhost:8000/runs/42/thesis"
    assert captured["json"] == {"provider": "auto", "api_key": None, "regenerate": False}


def test_main_thesis_passes_provider_and_api_key(monkeypatch):
    captured = {}

    def fake_request(method, url, params=None, json=None, timeout=None):
        captured["json"] = json
        return _FakeResponse(payload={"run_id": 42})

    monkeypatch.setattr(requests_module, "request", fake_request)

    exit_code = cli.main(["thesis", "42", "--provider", "openai", "--api-key", "sk-custom", "--regenerate"])

    assert exit_code == 0
    assert captured["json"] == {"provider": "openai", "api_key": "sk-custom", "regenerate": True}


def test_main_thesis_fetch_only_uses_get(monkeypatch):
    captured = {}

    def fake_request(method, url, params=None, json=None, timeout=None):
        captured["method"] = method
        captured["url"] = url
        return _FakeResponse(payload={"run_id": 42})

    monkeypatch.setattr(requests_module, "request", fake_request)

    exit_code = cli.main(["thesis", "42", "--fetch-only"])

    assert exit_code == 0
    assert captured["method"] == "GET"
    assert captured["url"] == "http://localhost:8000/runs/42/thesis"
