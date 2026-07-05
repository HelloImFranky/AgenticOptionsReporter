"""Provider-agnostic LLM access for the investment-thesis agent pipeline.

`LlmClient` is the interface agents depend on (dependency injection — the
same pattern as `data.market_data.MarketDataProvider`). Each concrete
client wraps one provider's SDK and normalizes its exceptions into the
`LlmError` hierarchy below, so callers never see an SDK-specific
exception type (see specs/llm_providers.yaml).

`LlmRouter` composes multiple clients into one `LlmClient` that tries them
in priority order, advancing to the next provider on a transient failure
(rate limit, quota exhaustion, timeout, or an unavailable/5xx backend)
instead of failing the whole request — relying on a single provider meant
one quota reset or outage could block every thesis generation.
`build_llm_client(provider="auto", ...)` returns a router built from
whichever providers currently have an API key configured; a named
provider still bypasses the router entirely for one-off use (e.g. the
Agents tab's explicit provider + custom API key fields).

An API key passed in explicitly (e.g. from a per-request UI field) is used
only for the duration of the call that constructs the client — it is never
logged or persisted (see main.py's ThesisGenerationRequest handling).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class LlmError(RuntimeError):
    """Raised when an LlmClient cannot produce a completion."""


class LlmRateLimited(LlmError):
    """The provider rejected the request for exceeding its rate limit (HTTP 429)."""


class LlmQuotaExceeded(LlmRateLimited):
    """The provider rejected the request because the account's quota/budget is exhausted."""


class LlmTimeout(LlmError):
    """The request to the provider timed out."""


class LlmUnavailable(LlmError):
    """The provider is unreachable or returned a server error (5xx / network failure)."""


class LlmBadRequest(LlmError):
    """The provider rejected the request itself (malformed prompt, unsupported model, ...).

    Not retried by `LlmRouter`: another provider would reject the same
    request for the same reason, so failing over would just hide a real
    bug behind a slower, more confusing error.
    """


class LlmAuthenticationError(LlmError):
    """The provider rejected the configured API key.

    Not retried by `LlmRouter`: a bad key is a configuration problem with
    that one provider, not the kind of transient blip failover exists for
    (see specs/llm_providers.yaml: retryable_errors).
    """


class LlmClient(ABC):
    """Interface implemented by all LLM providers used by the thesis pipeline."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's text response to a single-turn prompt."""
        raise NotImplementedError


class RecordingLlmClient(LlmClient):
    """Wraps another LlmClient and remembers the most recent
    (system_prompt, user_prompt, response) exchange, so the orchestrator
    can surface the raw prompt/response an agent sent and received for a
    live "under the hood" view. `last_exchange` is None until the wrapped
    client is called; the orchestrator clears it between agents so each
    agent's event carries only its own exchange."""

    def __init__(self, inner: LlmClient) -> None:
        self._inner = inner
        self.last_exchange: tuple[str, str, str] | None = None

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        response = self._inner.complete(system_prompt, user_prompt)
        self.last_exchange = (system_prompt, user_prompt, response)
        return response


def _classify_openai_style_error(exc: Exception, sdk: object, provider_label: str) -> LlmError:
    """Normalize an OpenAI-SDK-shaped exception (openai and anthropic both
    expose this same set of exception class names) into the LlmError
    hierarchy above."""
    if isinstance(exc, sdk.RateLimitError):
        message = str(exc)
        if "insufficient_quota" in message or "quota" in message.lower():
            return LlmQuotaExceeded(f"{provider_label} quota exceeded: {exc}")
        return LlmRateLimited(f"{provider_label} rate limited: {exc}")
    if isinstance(exc, sdk.APITimeoutError):
        return LlmTimeout(f"{provider_label} request timed out: {exc}")
    if isinstance(exc, sdk.AuthenticationError):
        return LlmAuthenticationError(f"{provider_label} authentication failed: {exc}")
    if isinstance(exc, sdk.BadRequestError):
        return LlmBadRequest(f"{provider_label} rejected the request: {exc}")
    if isinstance(exc, (sdk.InternalServerError, sdk.APIConnectionError)):
        return LlmUnavailable(f"{provider_label} is unavailable: {exc}")
    return LlmError(f"{provider_label} API call failed: {exc}")


class AnthropicLlmClient(LlmClient):
    """LlmClient implementation backed by the Claude API."""

    DEFAULT_MODEL = "claude-sonnet-5"
    ENV_VAR = "ANTHROPIC_API_KEY"
    PROVIDER_LABEL = "Anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        self._api_key = api_key or os.environ.get(self.ENV_VAR)
        if not self._api_key:
            raise LlmError(
                f"No Anthropic API key configured. Set {self.ENV_VAR}, or supply one explicitly."
            )
        self._model = model or self.DEFAULT_MODEL
        self._max_tokens = max_tokens

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        import anthropic

        client = anthropic.Anthropic(api_key=self._api_key)
        try:
            response = client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
            )
        except anthropic.APIError as exc:
            raise _classify_openai_style_error(exc, anthropic, self.PROVIDER_LABEL) from exc

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        if not text:
            raise LlmError("Anthropic API returned no text content")
        return text


class _OpenAiCompatibleLlmClient(LlmClient):
    """Base for any provider that speaks the OpenAI chat-completions API
    shape (OpenAI itself, Groq, DeepSeek, OpenRouter — see
    specs/llm_providers.yaml). Subclasses only need to set the four class
    attributes below."""

    BASE_URL: str | None = None  # None = OpenAI's own default endpoint
    DEFAULT_MODEL: str
    ENV_VAR: str
    PROVIDER_LABEL: str

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        self._api_key = api_key or os.environ.get(self.ENV_VAR)
        if not self._api_key:
            raise LlmError(
                f"No {self.PROVIDER_LABEL} API key configured. Set {self.ENV_VAR}, "
                "or supply one explicitly."
            )
        self._model = model or self.DEFAULT_MODEL
        self._max_tokens = max_tokens

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        import openai

        client = openai.OpenAI(api_key=self._api_key, base_url=self.BASE_URL)
        try:
            response = client.chat.completions.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except openai.APIError as exc:
            raise _classify_openai_style_error(exc, openai, self.PROVIDER_LABEL) from exc

        text = (response.choices[0].message.content or "").strip() if response.choices else ""
        if not text:
            raise LlmError(f"{self.PROVIDER_LABEL} API returned no text content")
        return text


class OpenAiLlmClient(_OpenAiCompatibleLlmClient):
    """LlmClient implementation backed by the OpenAI API."""

    BASE_URL = None
    DEFAULT_MODEL = "gpt-4o-mini"
    ENV_VAR = "OPENAI_API_KEY"
    PROVIDER_LABEL = "OpenAI"


class GroqLlmClient(_OpenAiCompatibleLlmClient):
    """LlmClient implementation backed by Groq's OpenAI-compatible API."""

    BASE_URL = "https://api.groq.com/openai/v1"
    DEFAULT_MODEL = "llama-3.3-70b-versatile"
    ENV_VAR = "GROQ_API_KEY"
    PROVIDER_LABEL = "Groq"


class DeepSeekLlmClient(_OpenAiCompatibleLlmClient):
    """LlmClient implementation backed by DeepSeek's OpenAI-compatible API."""

    BASE_URL = "https://api.deepseek.com"
    DEFAULT_MODEL = "deepseek-reasoner"
    ENV_VAR = "DEEPSEEK_API_KEY"
    PROVIDER_LABEL = "DeepSeek"


class OpenRouterLlmClient(_OpenAiCompatibleLlmClient):
    """LlmClient implementation backed by OpenRouter's OpenAI-compatible API."""

    BASE_URL = "https://openrouter.ai/api/v1"
    DEFAULT_MODEL = "deepseek/deepseek-r1"
    ENV_VAR = "OPENROUTER_API_KEY"
    PROVIDER_LABEL = "OpenRouter"


class GeminiLlmClient(LlmClient):
    """LlmClient implementation backed by Google's Gemini API (google-genai SDK)."""

    DEFAULT_MODEL = "gemini-2.5-pro"
    ENV_VAR = "GEMINI_API_KEY"
    PROVIDER_LABEL = "Gemini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        self._api_key = api_key or os.environ.get(self.ENV_VAR)
        if not self._api_key:
            raise LlmError(
                f"No Gemini API key configured. Set {self.ENV_VAR}, or supply one explicitly."
            )
        self._model = model or self.DEFAULT_MODEL
        self._max_tokens = max_tokens

    def _classify_error(self, exc: Exception) -> LlmError:
        from google.genai import errors as genai_errors

        if isinstance(exc, genai_errors.APIError):
            code = exc.code
            message = str(exc)
            if code == 429:
                if "quota" in message.lower():
                    return LlmQuotaExceeded(f"Gemini quota exceeded: {exc}")
                return LlmRateLimited(f"Gemini rate limited: {exc}")
            if code in (401, 403):
                return LlmAuthenticationError(f"Gemini authentication failed: {exc}")
            if code == 400:
                return LlmBadRequest(f"Gemini rejected the request: {exc}")
            if code in (500, 502, 503, 504):
                return LlmUnavailable(f"Gemini is unavailable: {exc}")
            return LlmError(f"Gemini API call failed: {exc}")
        return LlmUnavailable(f"Gemini request failed: {exc}")

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        from google import genai
        from google.genai import errors as genai_errors
        from google.genai import types as genai_types

        client = genai.Client(api_key=self._api_key)
        try:
            response = client.models.generate_content(
                model=self._model,
                contents=user_prompt,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    max_output_tokens=self._max_tokens,
                ),
            )
        except genai_errors.APIError as exc:
            raise self._classify_error(exc) from exc
        except Exception as exc:  # network/transport failures, not wrapped in APIError
            raise self._classify_error(exc) from exc

        text = (response.text or "").strip() if response is not None else ""
        if not text:
            raise LlmError("Gemini API returned no text content")
        return text


_PROVIDERS: dict[str, type[LlmClient]] = {
    "anthropic": AnthropicLlmClient,
    "openai": OpenAiLlmClient,
    "groq": GroqLlmClient,
    "gemini": GeminiLlmClient,
    "deepseek": DeepSeekLlmClient,
    "openrouter": OpenRouterLlmClient,
}

# Order used by provider="auto" when AOR_LLM_FALLBACK_ORDER isn't set.
# Configurable per specs/llm_providers.yaml so operators can reprioritize
# for cost or quality without a code change.
_DEFAULT_FALLBACK_ORDER = ["anthropic", "openai", "groq", "gemini", "deepseek", "openrouter"]

# Failing over to the next provider only makes sense for failures that are
# specific to *this* provider being momentarily unavailable — not for
# failures that would recur identically on every provider.
_RETRYABLE_ERRORS = (LlmRateLimited, LlmTimeout, LlmUnavailable)


class LlmRouter(LlmClient):
    """Tries a priority-ordered list of already-constructed clients,
    advancing to the next on a retryable failure. Implements `LlmClient`
    itself, so agents calling `.complete()` can't tell whether they're
    talking to a single provider or a router (see specs/llm_providers.yaml).
    """

    def __init__(self, clients: list[tuple[str, LlmClient]]) -> None:
        if not clients:
            raise LlmError(
                "No LLM providers are configured for automatic failover. Set at least "
                f"one provider's API key (supported: {', '.join(sorted(_PROVIDERS))})."
            )
        self._clients = clients

    @property
    def provider_names(self) -> list[str]:
        return [name for name, _ in self._clients]

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        failures: list[str] = []
        for name, client in self._clients:
            try:
                return client.complete(system_prompt, user_prompt)
            except _RETRYABLE_ERRORS as exc:
                failures.append(f"{name}: {exc}")
                continue
        raise LlmError(
            "All configured LLM providers failed: " + "; ".join(failures)
        )


def _fallback_order() -> list[str]:
    raw = os.environ.get("AOR_LLM_FALLBACK_ORDER", ",".join(_DEFAULT_FALLBACK_ORDER))
    return [name.strip().lower() for name in raw.split(",") if name.strip()]


def _build_router(max_tokens: int) -> LlmRouter:
    clients: list[tuple[str, LlmClient]] = []
    for name in _fallback_order():
        provider_cls = _PROVIDERS.get(name)
        if provider_cls is None:
            continue
        try:
            clients.append((name, provider_cls(max_tokens=max_tokens)))
        except LlmError:
            continue  # not configured (no API key) — skip, don't fail the request
    return LlmRouter(clients)


def build_llm_client(
    provider: str = "anthropic",
    api_key: str | None = None,
    model: str | None = None,
    max_tokens: int = 1024,
) -> LlmClient:
    """Construct an LlmClient for the named provider, or an `LlmRouter`
    across every configured provider when `provider == "auto"`.

    `api_key` overrides the provider's usual environment variable for this
    client only; pass None to fall back to the server's configured key.
    `api_key` cannot be combined with `provider="auto"` — there's no single
    provider for it to apply to.
    """
    key = (provider or "anthropic").strip().lower()
    if key == "auto":
        if api_key:
            raise LlmError(
                "api_key cannot be combined with provider='auto'; choose a specific "
                "provider to use a custom key."
            )
        return _build_router(max_tokens=max_tokens)
    provider_cls = _PROVIDERS.get(key)
    if provider_cls is None:
        supported = ", ".join(sorted(_PROVIDERS) + ["auto"])
        raise LlmError(f"Unsupported LLM provider {provider!r}. Supported providers: {supported}.")
    return provider_cls(api_key=api_key, model=model, max_tokens=max_tokens)
