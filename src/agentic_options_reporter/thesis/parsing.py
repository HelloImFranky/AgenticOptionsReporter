"""Shared response parsing for the investment-thesis agents.

Every agent instructs its LlmClient to respond with a single JSON object
and validates it against a Pydantic model. Centralized here so a parse
failure always raises the same error type instead of each agent handling
malformed LLM output differently.
"""

from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)


class ThesisGenerationError(RuntimeError):
    """Raised when an agent's LLM response cannot be parsed or validated."""


def parse_response(model_cls: type[ModelT], raw_text: str, agent_name: str) -> ModelT:
    text = raw_text.strip()
    if not text:
        raise ThesisGenerationError(f"{agent_name}: no JSON object found in LLM response: {raw_text!r}")

    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            data, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue

        try:
            return model_cls.model_validate(data)
        except ValidationError as exc:
            raise ThesisGenerationError(
                f"{agent_name}: LLM response did not match the expected schema: {exc}"
            ) from exc

    if "{" in text:
        raise ThesisGenerationError(
            f"{agent_name}: LLM response was not valid JSON: no complete JSON object could be parsed"
        )
    raise ThesisGenerationError(
        f"{agent_name}: no JSON object found in LLM response: {raw_text!r}"
    )
