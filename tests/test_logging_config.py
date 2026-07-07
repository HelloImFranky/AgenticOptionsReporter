import logging

import pytest

from agentic_options_reporter.data.async_http import redact_params
from agentic_options_reporter.logging_config import (
    _BUFFER_HANDLER,
    clear_log_entries,
    configure_logging,
    get_log_entries,
)

_LOGGER = logging.getLogger("agentic_options_reporter.test_logging_config")


@pytest.fixture(autouse=True)
def _reset_buffer():
    configure_logging()
    clear_log_entries()
    yield
    clear_log_entries()


def test_configure_logging_is_idempotent():
    configure_logging()
    configure_logging()
    root = logging.getLogger("agentic_options_reporter")
    assert sum(1 for h in root.handlers if h is _BUFFER_HANDLER) == 1


def test_log_entries_are_buffered_in_order():
    _LOGGER.info("first")
    _LOGGER.warning("second")

    entries = get_log_entries()

    assert [e.message for e in entries] == ["first", "second"]
    assert entries[0].level == "INFO"
    assert entries[1].level == "WARNING"
    assert entries[1].seq > entries[0].seq


def test_message_field_has_no_duplicated_prefix():
    """emit() must store the plain message, not a re-formatted line — the
    timestamp/level/logger are already their own LogEntry fields."""
    _LOGGER.info("plain message")

    entry = get_log_entries()[-1]

    assert entry.message == "plain message"
    assert "INFO" not in entry.message
    assert _LOGGER.name not in entry.message


def test_since_seq_only_returns_newer_entries():
    _LOGGER.info("one")
    first_seq = get_log_entries()[-1].seq
    _LOGGER.info("two")
    _LOGGER.info("three")

    newer = get_log_entries(since_seq=first_seq)

    assert [e.message for e in newer] == ["two", "three"]


def test_limit_caps_returned_entries():
    for i in range(10):
        _LOGGER.info("entry %d", i)

    entries = get_log_entries(limit=3)

    assert len(entries) == 3
    assert entries[-1].message == "entry 9"


def test_redact_params_masks_credential_shaped_keys():
    redacted = redact_params({"symbol": "AAPL", "apikey": "SECRET", "limit": 1})

    assert redacted == {"symbol": "AAPL", "apikey": "***", "limit": 1}


def test_redact_params_leaves_non_sensitive_keys_untouched():
    redacted = redact_params({"symbol": "AAPL", "period": "annual"})

    assert redacted == {"symbol": "AAPL", "period": "annual"}
