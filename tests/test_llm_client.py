import sys
import types

import pytest

from agentic_options_reporter.thesis.llm_client import (
    AnthropicLlmClient,
    DeepSeekLlmClient,
    GeminiLlmClient,
    GroqLlmClient,
    LlmAuthenticationError,
    LlmBadRequest,
    LlmClient,
    LlmError,
    LlmQuotaExceeded,
    LlmRateLimited,
    LlmRouter,
    LlmTimeout,
    LlmUnavailable,
    OpenAiLlmClient,
    OpenRouterLlmClient,
    RecordingLlmClient,
    build_llm_client,
)


def test_recording_client_records_last_exchange_and_forwards():
    # _StubLlmClient (defined later in this module) counts calls and returns
    # a canned response; the recorder must forward to it and remember the
    # full exchange.
    inner = _StubLlmClient(response="model text")
    recorder = RecordingLlmClient(inner)

    result = recorder.complete("sys prompt", "user prompt")

    assert result == "model text"
    assert inner.calls == 1
    assert recorder.last_exchange == ("sys prompt", "user prompt", "model text")


def test_recording_client_overwrites_and_clears_exchange():
    recorder = RecordingLlmClient(_StubLlmClient(response="first"))
    recorder.complete("s1", "u1")
    assert recorder.last_exchange == ("s1", "u1", "first")

    # A caller resets between agents so each event carries only its own call.
    recorder.last_exchange = None
    assert recorder.last_exchange is None
    recorder.complete("s2", "u2")
    assert recorder.last_exchange == ("s2", "u2", "first")

_ALL_PROVIDER_ENV_VARS = (
    "ANTHROPIC_API_KEY",
    "OPENAI_API_KEY",
    "GROQ_API_KEY",
    "GEMINI_API_KEY",
    "DEEPSEEK_API_KEY",
    "OPENROUTER_API_KEY",
)


@pytest.fixture(autouse=True)
def _clear_provider_env_vars(monkeypatch):
    """Every test explicitly sets the env vars it needs; start from a
    clean slate so provider auto-discovery is deterministic regardless of
    the host environment or test order."""
    for var in _ALL_PROVIDER_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("AOR_LLM_FALLBACK_ORDER", raising=False)


def test_anthropic_client_requires_api_key():
    with pytest.raises(LlmError):
        AnthropicLlmClient()


def test_anthropic_client_accepts_explicit_api_key():
    client = AnthropicLlmClient(api_key="test-key")
    assert client is not None


def test_openai_client_requires_api_key():
    with pytest.raises(LlmError):
        OpenAiLlmClient()


def test_openai_client_accepts_explicit_api_key():
    client = OpenAiLlmClient(api_key="test-key")
    assert client is not None


def test_openai_client_uses_default_model_when_unset():
    client = OpenAiLlmClient(api_key="test-key")
    assert client._model == OpenAiLlmClient.DEFAULT_MODEL


def test_build_llm_client_dispatches_to_anthropic():
    client = build_llm_client("anthropic", api_key="test-key")
    assert isinstance(client, AnthropicLlmClient)


def test_build_llm_client_dispatches_to_openai():
    client = build_llm_client("openai", api_key="test-key")
    assert isinstance(client, OpenAiLlmClient)


def test_build_llm_client_is_case_insensitive():
    client = build_llm_client("OpenAI", api_key="test-key")
    assert isinstance(client, OpenAiLlmClient)


def test_build_llm_client_rejects_unsupported_provider():
    with pytest.raises(LlmError, match="Unsupported LLM provider"):
        build_llm_client("made-up-provider", api_key="test-key")


def test_build_llm_client_passes_model_and_max_tokens():
    client = build_llm_client("anthropic", api_key="test-key", model="custom-model", max_tokens=2048)
    assert client._model == "custom-model"
    assert client._max_tokens == 2048


# -- OpenAI-compatible providers (OpenAI, Groq, DeepSeek, OpenRouter) --

_OPENAI_COMPATIBLE_PROVIDERS = [
    (OpenAiLlmClient, "OPENAI_API_KEY", None, "gpt-4o-mini"),
    (GroqLlmClient, "GROQ_API_KEY", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    (DeepSeekLlmClient, "DEEPSEEK_API_KEY", "https://api.deepseek.com", "deepseek-reasoner"),
    (OpenRouterLlmClient, "OPENROUTER_API_KEY", "https://openrouter.ai/api/v1", "deepseek/deepseek-r1"),
]


@pytest.mark.parametrize("client_cls,env_var,base_url,default_model", _OPENAI_COMPATIBLE_PROVIDERS)
def test_openai_compatible_provider_requires_api_key(client_cls, env_var, base_url, default_model):
    with pytest.raises(LlmError):
        client_cls()


@pytest.mark.parametrize("client_cls,env_var,base_url,default_model", _OPENAI_COMPATIBLE_PROVIDERS)
def test_openai_compatible_provider_accepts_explicit_api_key(client_cls, env_var, base_url, default_model):
    client = client_cls(api_key="test-key")
    assert isinstance(client, LlmClient)
    assert client._model == default_model


@pytest.mark.parametrize("client_cls,env_var,base_url,default_model", _OPENAI_COMPATIBLE_PROVIDERS)
def test_openai_compatible_provider_calls_correct_base_url(
    fake_openai_module, client_cls, env_var, base_url, default_model
):
    captured = {}

    def fake_openai_ctor(**kwargs):
        captured.update(kwargs)
        return _FakeOpenAIClient(response=_FakeOpenAIResponse("ok"))

    fake_openai_module.OpenAI = fake_openai_ctor
    client = client_cls(api_key="test-key")
    result = client.complete("system", "user")

    assert result == "ok"
    assert captured["base_url"] == base_url


def test_build_llm_client_dispatches_to_groq():
    client = build_llm_client("groq", api_key="test-key")
    assert isinstance(client, GroqLlmClient)


def test_build_llm_client_dispatches_to_deepseek():
    client = build_llm_client("deepseek", api_key="test-key")
    assert isinstance(client, DeepSeekLlmClient)


def test_build_llm_client_dispatches_to_openrouter():
    client = build_llm_client("openrouter", api_key="test-key")
    assert isinstance(client, OpenRouterLlmClient)


def test_build_llm_client_dispatches_to_gemini():
    client = build_llm_client("gemini", api_key="test-key")
    assert isinstance(client, GeminiLlmClient)


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


class _FakeAPIError(Exception):
    pass


class _FakeRateLimitError(_FakeAPIError):
    pass


class _FakeAPITimeoutError(_FakeAPIError):
    pass


class _FakeAuthenticationError(_FakeAPIError):
    pass


class _FakeBadRequestError(_FakeAPIError):
    pass


class _FakeInternalServerError(_FakeAPIError):
    pass


class _FakeAPIConnectionError(_FakeAPIError):
    pass


def _fake_sdk_exceptions():
    return dict(
        APIError=_FakeAPIError,
        RateLimitError=_FakeRateLimitError,
        APITimeoutError=_FakeAPITimeoutError,
        AuthenticationError=_FakeAuthenticationError,
        BadRequestError=_FakeBadRequestError,
        InternalServerError=_FakeInternalServerError,
        APIConnectionError=_FakeAPIConnectionError,
    )


@pytest.fixture
def fake_anthropic_module(monkeypatch):
    fake_module = types.SimpleNamespace(Anthropic=None, **_fake_sdk_exceptions())
    monkeypatch.setitem(sys.modules, "anthropic", fake_module)
    return fake_module


def test_complete_returns_text(fake_anthropic_module):
    fake_anthropic_module.Anthropic = lambda **kwargs: _FakeAnthropicClient(
        response=_FakeResponse('{"narrative": "ok"}')
    )
    client = AnthropicLlmClient(api_key="test-key")
    result = client.complete("system", "user")
    assert result == '{"narrative": "ok"}'


def test_complete_raises_llm_error_on_generic_api_error(fake_anthropic_module):
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
    fake_module = types.SimpleNamespace(OpenAI=None, **_fake_sdk_exceptions())
    monkeypatch.setitem(sys.modules, "openai", fake_module)
    return fake_module


def test_openai_complete_returns_text(fake_openai_module):
    fake_openai_module.OpenAI = lambda **kwargs: _FakeOpenAIClient(
        response=_FakeOpenAIResponse('{"narrative": "ok"}')
    )
    client = OpenAiLlmClient(api_key="test-key")
    result = client.complete("system", "user")
    assert result == '{"narrative": "ok"}'


def test_openai_complete_raises_llm_error_on_generic_api_error(fake_openai_module):
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


# -- Error normalization (anthropic and openai share the same exception
# class names, so the same classification helper serves both, and
# Groq/DeepSeek/OpenRouter for free since they subclass the OpenAI client) --

_ERROR_CLASSIFICATION_CASES = [
    ("RateLimitError", LlmRateLimited),
    ("APITimeoutError", LlmTimeout),
    ("AuthenticationError", LlmAuthenticationError),
    ("BadRequestError", LlmBadRequest),
    ("InternalServerError", LlmUnavailable),
    ("APIConnectionError", LlmUnavailable),
]


@pytest.mark.parametrize("error_attr,expected_cls", _ERROR_CLASSIFICATION_CASES)
def test_anthropic_normalizes_sdk_errors(fake_anthropic_module, error_attr, expected_cls):
    error = getattr(fake_anthropic_module, error_attr)("boom")
    fake_anthropic_module.Anthropic = lambda **kwargs: _FakeAnthropicClient(error=error)
    client = AnthropicLlmClient(api_key="test-key")
    with pytest.raises(expected_cls):
        client.complete("system", "user")


@pytest.mark.parametrize("error_attr,expected_cls", _ERROR_CLASSIFICATION_CASES)
def test_openai_normalizes_sdk_errors(fake_openai_module, error_attr, expected_cls):
    error = getattr(fake_openai_module, error_attr)("boom")
    fake_openai_module.OpenAI = lambda **kwargs: _FakeOpenAIClient(error=error)
    client = OpenAiLlmClient(api_key="test-key")
    with pytest.raises(expected_cls):
        client.complete("system", "user")


def test_openai_classifies_rate_limit_with_quota_message_as_quota_exceeded(fake_openai_module):
    error = fake_openai_module.RateLimitError("insufficient_quota: you exceeded your current quota")
    fake_openai_module.OpenAI = lambda **kwargs: _FakeOpenAIClient(error=error)
    client = OpenAiLlmClient(api_key="test-key")
    with pytest.raises(LlmQuotaExceeded):
        client.complete("system", "user")


def test_openai_classifies_plain_rate_limit_without_quota_message(fake_openai_module):
    error = fake_openai_module.RateLimitError("too many requests")
    fake_openai_module.OpenAI = lambda **kwargs: _FakeOpenAIClient(error=error)
    client = OpenAiLlmClient(api_key="test-key")
    with pytest.raises(LlmRateLimited) as excinfo:
        client.complete("system", "user")
    assert not isinstance(excinfo.value, LlmQuotaExceeded)


# -- Gemini (google-genai SDK) --


class _FakeGenaiResponse:
    def __init__(self, text):
        self.text = text


class _FakeGenaiModels:
    def __init__(self, response=None, error=None):
        self._response = response
        self._error = error

    def generate_content(self, **kwargs):
        if self._error is not None:
            raise self._error
        return self._response


class _FakeGenaiClient:
    def __init__(self, response=None, error=None, **kwargs):
        self.models = _FakeGenaiModels(response=response, error=error)


class _FakeGenaiAPIError(Exception):
    def __init__(self, code, message):
        self.code = code
        super().__init__(message)


@pytest.fixture
def fake_genai_module(monkeypatch):
    fake_errors = types.SimpleNamespace(APIError=_FakeGenaiAPIError)
    fake_types = types.SimpleNamespace(GenerateContentConfig=lambda **kwargs: kwargs)
    fake_genai = types.SimpleNamespace(Client=None, errors=fake_errors, types=fake_types)
    fake_google = types.SimpleNamespace(genai=fake_genai)

    monkeypatch.setitem(sys.modules, "google", fake_google)
    monkeypatch.setitem(sys.modules, "google.genai", fake_genai)
    monkeypatch.setitem(sys.modules, "google.genai.errors", fake_errors)
    monkeypatch.setitem(sys.modules, "google.genai.types", fake_types)
    return fake_genai


def test_gemini_client_requires_api_key():
    with pytest.raises(LlmError):
        GeminiLlmClient()


def test_gemini_client_uses_default_model_when_unset():
    client = GeminiLlmClient(api_key="test-key")
    assert client._model == GeminiLlmClient.DEFAULT_MODEL


def test_gemini_complete_returns_text(fake_genai_module):
    fake_genai_module.Client = lambda **kwargs: _FakeGenaiClient(response=_FakeGenaiResponse('{"a": 1}'))
    client = GeminiLlmClient(api_key="test-key")
    assert client.complete("system", "user") == '{"a": 1}'


def test_gemini_raises_llm_error_on_empty_text(fake_genai_module):
    fake_genai_module.Client = lambda **kwargs: _FakeGenaiClient(response=_FakeGenaiResponse(""))
    client = GeminiLlmClient(api_key="test-key")
    with pytest.raises(LlmError):
        client.complete("system", "user")


@pytest.mark.parametrize(
    "code,message,expected_cls",
    [
        (429, "rate limit exceeded", LlmRateLimited),
        (429, "quota exceeded for this project", LlmQuotaExceeded),
        (401, "invalid API key", LlmAuthenticationError),
        (403, "permission denied", LlmAuthenticationError),
        (400, "invalid argument", LlmBadRequest),
        (500, "internal error", LlmUnavailable),
        (503, "service unavailable", LlmUnavailable),
    ],
)
def test_gemini_normalizes_api_errors(fake_genai_module, code, message, expected_cls):
    error = fake_genai_module.errors.APIError(code, message)
    fake_genai_module.Client = lambda **kwargs: _FakeGenaiClient(error=error)
    client = GeminiLlmClient(api_key="test-key")
    with pytest.raises(expected_cls):
        client.complete("system", "user")


def test_gemini_wraps_non_api_error_as_unavailable(fake_genai_module):
    fake_genai_module.Client = lambda **kwargs: _FakeGenaiClient(error=ConnectionError("network down"))
    client = GeminiLlmClient(api_key="test-key")
    with pytest.raises(LlmUnavailable):
        client.complete("system", "user")


# -- LlmRouter --


class _StubLlmClient(LlmClient):
    def __init__(self, response="ok", error=None):
        self.response = response
        self.error = error
        self.calls = 0

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.response


def test_llm_router_rejects_empty_client_list():
    with pytest.raises(LlmError):
        LlmRouter([])


def test_llm_router_returns_first_success():
    first = _StubLlmClient(response="from first")
    second = _StubLlmClient(response="from second")
    router = LlmRouter([("first", first), ("second", second)])

    assert router.complete("s", "u") == "from first"
    assert second.calls == 0


def test_llm_router_provider_names():
    router = LlmRouter([("a", _StubLlmClient()), ("b", _StubLlmClient())])
    assert router.provider_names == ["a", "b"]


@pytest.mark.parametrize(
    "error",
    [
        LlmRateLimited("rate limited"),
        LlmQuotaExceeded("quota exceeded"),
        LlmTimeout("timed out"),
        LlmUnavailable("unavailable"),
    ],
)
def test_llm_router_advances_to_next_provider_on_retryable_error(error):
    first = _StubLlmClient(error=error)
    second = _StubLlmClient(response="from second")
    router = LlmRouter([("first", first), ("second", second)])

    assert router.complete("s", "u") == "from second"
    assert second.calls == 1


@pytest.mark.parametrize("error_cls", [LlmBadRequest, LlmAuthenticationError])
def test_llm_router_does_not_advance_on_non_retryable_error(error_cls):
    first = _StubLlmClient(error=error_cls("boom"))
    second = _StubLlmClient(response="from second")
    router = LlmRouter([("first", first), ("second", second)])

    with pytest.raises(error_cls):
        router.complete("s", "u")
    assert second.calls == 0


def test_llm_router_raises_with_all_failure_details_when_every_provider_fails():
    first = _StubLlmClient(error=LlmTimeout("timed out"))
    second = _StubLlmClient(error=LlmUnavailable("down"))
    router = LlmRouter([("first", first), ("second", second)])

    with pytest.raises(LlmError, match="first:.*timed out.*second:.*down"):
        router.complete("s", "u")


# -- build_llm_client(provider="auto", ...) --


def test_build_llm_client_auto_builds_router_from_configured_providers(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    client = build_llm_client("auto")

    assert isinstance(client, LlmRouter)
    assert client.provider_names == ["anthropic"]


def test_build_llm_client_auto_includes_every_configured_provider_in_default_order(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    client = build_llm_client("auto")

    assert client.provider_names == ["openai", "groq"]


def test_build_llm_client_auto_respects_fallback_order_env_var(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setenv("AOR_LLM_FALLBACK_ORDER", "groq,openai")

    client = build_llm_client("auto")

    assert client.provider_names == ["groq", "openai"]


def test_build_llm_client_auto_ignores_unknown_provider_names_in_fallback_order(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    monkeypatch.setenv("AOR_LLM_FALLBACK_ORDER", "made-up-provider,anthropic")

    client = build_llm_client("auto")

    assert client.provider_names == ["anthropic"]


def test_build_llm_client_auto_raises_when_no_provider_configured():
    with pytest.raises(LlmError):
        build_llm_client("auto")


def test_build_llm_client_auto_rejects_explicit_api_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    with pytest.raises(LlmError, match="auto"):
        build_llm_client("auto", api_key="sk-custom-123")


def test_build_llm_client_auto_is_case_insensitive(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    client = build_llm_client("Auto")

    assert isinstance(client, LlmRouter)
