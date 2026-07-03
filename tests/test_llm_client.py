import sys
import types

import pytest

from agentic_options_reporter.thesis.llm_client import (
    AnthropicLlmClient,
    LlmError,
    OpenAiLlmClient,
    build_llm_client,
)


def test_anthropic_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(LlmError):
        AnthropicLlmClient()


def test_anthropic_client_accepts_explicit_api_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = AnthropicLlmClient(api_key="test-key")
    assert client is not None


def test_openai_client_requires_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(LlmError):
        OpenAiLlmClient()


def test_openai_client_accepts_explicit_api_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = OpenAiLlmClient(api_key="test-key")
    assert client is not None


def test_openai_client_uses_default_model_when_unset():
    client = OpenAiLlmClient(api_key="test-key")
    assert client._model == OpenAiLlmClient.DEFAULT_MODEL


def test_build_llm_client_dispatches_to_anthropic(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = build_llm_client("anthropic", api_key="test-key")
    assert isinstance(client, AnthropicLlmClient)


def test_build_llm_client_dispatches_to_openai(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = build_llm_client("openai", api_key="test-key")
    assert isinstance(client, OpenAiLlmClient)


def test_build_llm_client_defaults_to_anthropic(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = build_llm_client(api_key="test-key")
    assert isinstance(client, AnthropicLlmClient)


def test_build_llm_client_is_case_insensitive(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    client = build_llm_client("OpenAI", api_key="test-key")
    assert isinstance(client, OpenAiLlmClient)


def test_build_llm_client_rejects_unsupported_provider():
    with pytest.raises(LlmError, match="Unsupported LLM provider"):
        build_llm_client("made-up-provider", api_key="test-key")


def test_build_llm_client_passes_model_and_max_tokens(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    client = build_llm_client("anthropic", api_key="test-key", model="custom-model", max_tokens=2048)
    assert client._model == "custom-model"
    assert client._max_tokens == 2048


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


class _FakeOpenAIMessage:
    def __init__(self, content):
        self.content = content


class _FakeOpenAIChoice:
    def __init__(self, content):
        self.message = _FakeOpenAIMessage(content)


class _FakeOpenAIResponse:
    def __init__(self, content):
        self.choices = [_FakeOpenAIChoice(content)] if content is not None else []


class _FakeChatCompletions:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    def create(self, **kwargs):
        if self._error is not None:
            raise self._error
        return self._response


class _FakeChat:
    def __init__(self, response=None, error=None):
        self.completions = _FakeChatCompletions(response=response, error=error)


class _FakeOpenAIClient:
    def __init__(self, response=None, error=None, **kwargs):
        self.chat = _FakeChat(response=response, error=error)


@pytest.fixture
def fake_openai_module(monkeypatch):
    fake_module = types.SimpleNamespace(
        OpenAI=None,
        APIError=RuntimeError,
    )
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    return fake_module


def test_openai_complete_returns_text(fake_openai_module):
    fake_openai_module.OpenAI = lambda **kwargs: _FakeOpenAIClient(
        response=_FakeOpenAIResponse('{"narrative": "ok"}')
    )
    client = OpenAiLlmClient(api_key="test-key")
    result = client.complete("system", "user")
    assert result == '{"narrative": "ok"}'


def test_openai_complete_raises_llm_error_on_api_error(fake_openai_module):
    fake_openai_module.OpenAI = lambda **kwargs: _FakeOpenAIClient(
        error=fake_openai_module.APIError("boom")
    )
    client = OpenAiLlmClient(api_key="test-key")
    with pytest.raises(LlmError):
        client.complete("system", "user")


def test_openai_complete_raises_llm_error_on_empty_choices(fake_openai_module):
    fake_openai_module.OpenAI = lambda **kwargs: _FakeOpenAIClient(response=_FakeOpenAIResponse(None))
    client = OpenAiLlmClient(api_key="test-key")
    with pytest.raises(LlmError):
        client.complete("system", "user")
