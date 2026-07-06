import pytest
from pydantic import BaseModel

from agentic_options_reporter.thesis.parsing import ThesisGenerationError, parse_response


class _Simple(BaseModel):
    a: str
    b: int


def test_parse_response_valid_json():
    result = parse_response(_Simple, '{"a": "hello", "b": 1}', "agent")
    assert result.a == "hello"
    assert result.b == 1


def test_parse_response_extracts_json_from_surrounding_text():
    raw = 'Sure, here you go:\n```json\n{"a": "hi", "b": 2}\n```\nHope that helps!'
    result = parse_response(_Simple, raw, "agent")
    assert result.a == "hi"
    assert result.b == 2


def test_parse_response_no_json_object_raises():
    with pytest.raises(ThesisGenerationError):
        parse_response(_Simple, "no json here at all", "agent")


def test_parse_response_invalid_json_raises():
    with pytest.raises(ThesisGenerationError):
        parse_response(_Simple, "{not valid json}", "agent")


def test_parse_response_uses_first_complete_json_object_when_trailing_content_exists():
    raw = '{"a": "hello", "b": 1}\n\nSome trailing text after the object.'
    result = parse_response(_Simple, raw, "agent")
    assert result.a == "hello"
    assert result.b == 1


def test_parse_response_schema_mismatch_raises():
    with pytest.raises(ThesisGenerationError):
        parse_response(_Simple, '{"a": "hello"}', "agent")


def test_parse_response_error_message_includes_agent_name():
    with pytest.raises(ThesisGenerationError, match="my_agent"):
        parse_response(_Simple, "not json", "my_agent")
