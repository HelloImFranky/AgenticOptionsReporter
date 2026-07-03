"""Shared response parsing for the investment-thesis agents.

Every agent instructs its LlmClient to respond with a single JSON object
and validates it against a Pydantic model. Centralized here so a parse
failure always raises the same error type instead of each agent handling
malformed LLM output differently.
"""

from __future__ import annotations

import json
import re
from typing import TypeVar

from pydantic import BaseModel, ValidationError

ModelT = TypeVar("ModelT", bound=BaseModel)

_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


class ThesisGenerationError(RuntimeError):
    """Raised when an agent's LLM response cannot be parsed or validated."""


def parse_response(model_cls: type[ModelT], raw_text: str, agent_name: str) -> ModelT:
    match = _JSON_OBJECT_RE.search(raw_text.strip())
    if not match:
        raise ThesisGenerationError(
            f"{agent_name}: no JSON object found in LLM response: {raw_text!r}"
        )

    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ThesisGenerationError(f"{agent_name}: LLM response was not valid JSON: {exc}") from exc

    try:
        return model_cls.model_validate(data)
    except ValidationError as exc:
        raise ThesisGenerationError(
            f"{agent_name}: LLM response did not match the expected schema: {exc}"
        ) from exc
