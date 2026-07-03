"""Provider-agnostic LLM access for the investment-thesis agent pipeline.

`LlmClient` is the interface agents depend on (dependency injection — the
same pattern as `data.market_data.MarketDataProvider`). `AnthropicLlmClient`
and `OpenAiLlmClient` are the built-in implementations; `build_llm_client`
selects one by provider name. A different provider can be added later by
implementing the same interface and registering it in `_PROVIDERS` without
touching any agent module.

An API key passed in explicitly (e.g. from a per-request UI field) is used
only for the duration of the call that constructs the client — it is never
logged or persisted (see main.py's ThesisGenerationRequest handling).
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod


class LlmError(RuntimeError):
    """Raised when an LlmClient cannot produce a completion."""


class LlmClient(ABC):
    """Interface implemented by all LLM providers used by the thesis pipeline."""

    @abstractmethod
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return the model's text response to a single-turn prompt."""
        raise NotImplementedError


class AnthropicLlmClient(LlmClient):
    """LlmClient implementation backed by the Claude API."""

    DEFAULT_MODEL = "claude-sonnet-5"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise LlmError(
                "No Anthropic API key configured. Set ANTHROPIC_API_KEY, or supply one explicitly."
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
            raise LlmError(f"Anthropic API call failed: {exc}") from exc

        text = "".join(
            block.text for block in response.content if getattr(block, "type", None) == "text"
        )
        if not text:
            raise LlmError("Anthropic API returned no text content")
        return text


class OpenAiLlmClient(LlmClient):
    """LlmClient implementation backed by the OpenAI API."""

    DEFAULT_MODEL = "gpt-4o-mini"

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise LlmError(
                "No OpenAI API key configured. Set OPENAI_API_KEY, or supply one explicitly."
            )
        self._model = model or self.DEFAULT_MODEL
        self._max_tokens = max_tokens

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        import openai

        client = openai.OpenAI(api_key=self._api_key)
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
            raise LlmError(f"OpenAI API call failed: {exc}") from exc

        text = (response.choices[0].message.content or "").strip() if response.choices else ""
        if not text:
            raise LlmError("OpenAI API returned no text content")
        return text


_PROVIDERS: dict[str, type[LlmClient]] = {
    "anthropic": AnthropicLlmClient,
    "openai": OpenAiLlmClient,
}


def build_llm_client(
    provider: str = "anthropic",
    api_key: str | None = None,
    model: str | None = None,
    max_tokens: int = 1024,
) -> LlmClient:
    """Construct an LlmClient for the named provider.

    `api_key` overrides the provider's usual environment variable for this
    client only; pass None to fall back to the server's configured key.
    """
    key = (provider or "anthropic").strip().lower()
    provider_cls = _PROVIDERS.get(key)
    if provider_cls is None:
        supported = ", ".join(sorted(_PROVIDERS))
        raise LlmError(f"Unsupported LLM provider {provider!r}. Supported providers: {supported}.")
    return provider_cls(api_key=api_key, model=model, max_tokens=max_tokens)
