"""Provider-agnostic LLM access for the investment-thesis agent pipeline.

`LlmClient` is the interface agents depend on (dependency injection — the
same pattern as `data.market_data.MarketDataProvider`). `AnthropicLlmClient`
is the default implementation. A different provider can be added later by
implementing the same interface without touching any agent module.
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
    """Default LlmClient implementation, backed by the Claude API."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str = "claude-sonnet-5",
        max_tokens: int = 1024,
    ) -> None:
        self._api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise LlmError(
                "No Anthropic API key configured. Set ANTHROPIC_API_KEY or pass api_key= explicitly."
            )
        self._model = model
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
