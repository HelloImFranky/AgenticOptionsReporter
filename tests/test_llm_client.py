import sys
import types

import pytest

from agentic_options_reporter.thesis.llm_client import AnthropicLlmClient, LlmError


def test_anthropic_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LlmError):
        AnthropicLlmClient()


def test_anthropic_client_accepts_explicit_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = AnthropicLlmClient(api_key="test-key")
    assert client is not None


class _FakeTextBlock:
    def __init__(self, text):
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.content = [_FakeTextBlock(text)]


class _FakeMessages:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    def create(self, **kwargs):
        if self._error is not None:
            raise self._error
        return self._response


class _FakeAnthropicClient:
    def __init__(self, response=None, error=None, **kwargs):
        self.messages = _FakeMessages(response=response, error=error)


@pytest.fixture
def fake_anthropic_module(monkeypatch):
    fake_module = types.SimpleNamespace(
        Anthropic=None,
        APIError=RuntimeError,
    )
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    return fake_module


def test_complete_returns_text(fake_anthropic_module):
    fake_anthropic_module.Anthropic = lambda **kwargs: _FakeAnthropicClient(
        response=_FakeResponse('{"narrative": "ok"}')
    )
    client = AnthropicLlmClient(api_key="test-key")
    result = client.complete("system", "user")
    assert result == '{"narrative": "ok"}'


def test_complete_raises_llm_error_on_api_error(fake_anthropic_module):
    fake_anthropic_module.Anthropic = lambda **kwargs: _FakeAnthropicClient(
        error=fake_anthropic_module.APIError("boom")
    )
    client = AnthropicLlmClient(api_key="test-key")
    with pytest.raises(LlmError):
        client.complete("system", "user")


def test_complete_raises_llm_error_on_empty_text(fake_anthropic_module):
    fake_anthropic_module.Anthropic = lambda **kwargs: _FakeAnthropicClient(response=_FakeResponse(""))
    client = AnthropicLlmClient(api_key="test-key")
    with pytest.raises(LlmError):
        client.complete("system", "user")
