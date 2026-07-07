"""Process-wide logging setup.

Every module logs through the standard `logging` package (`logging.getLogger(__name__)`),
so log lines are attributable to the exact module that emitted them (a
provider adapter, the workflow pipeline, the thesis orchestrator, ...).
`configure_logging()` attaches two handlers to the `agentic_options_reporter`
logger: a console handler (for local/`docker logs` visibility) and
`_BUFFER_HANDLER`, which retains the most recent records in memory so the
frontend's Log tab can poll them via GET /logs without a separate log
shipper. Sensitive values (API keys) are never passed to the logging calls
in the first place — see `redact_params` in data/async_http.py — so nothing
here needs to scrub them after the fact.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from datetime import datetime, timezone
from itertools import count

from agentic_options_reporter.models.schemas import LogEntry

LOGGER_NAME = "agentic_options_reporter"
MAX_BUFFERED_ENTRIES = 2000

_LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


class _InMemoryLogHandler(logging.Handler):
    """Retains the last `capacity` log records as `LogEntry`s for the
    Log tab. Thread-safe: log calls can arrive from the FastAPI request
    thread, the asyncio event loops each request bridges into, and any
    background thread the Flet frontend spins up in-process."""

    def __init__(self, capacity: int = MAX_BUFFERED_ENTRIES) -> None:
        super().__init__()
        self._entries: deque[LogEntry] = deque(maxlen=capacity)
        self._lock = threading.Lock()
        self._seq = count(1)

    def emit(self, record: logging.LogRecord) -> None:
        # Just the formatted message (record.getMessage(), plus exception
        # text if any) — timestamp/level/logger are already their own
        # LogEntry fields, so re-running them through a Formatter here
        # would duplicate that prefix inside `message` itself.
        message = record.getMessage()
        if record.exc_info:
            formatter = self.formatter or logging.Formatter()
            message = f"{message}\n{formatter.formatException(record.exc_info)}"
        entry = LogEntry(
            seq=next(self._seq),
            timestamp=datetime.fromtimestamp(record.created, tz=timezone.utc).replace(tzinfo=None),
            level=record.levelname,
            logger=record.name,
            message=message,
        )
        with self._lock:
            self._entries.append(entry)

    def entries(self, since_seq: int = 0, limit: int = 500) -> list[LogEntry]:
        with self._lock:
            snapshot = [e for e in self._entries if e.seq > since_seq]
        return snapshot[-limit:]

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


_BUFFER_HANDLER = _InMemoryLogHandler()
_configured = False
_configure_lock = threading.Lock()


def configure_logging(level: int = logging.INFO) -> None:
    """Attach the console + in-memory handlers to the package logger.
    Idempotent — safe to call from both main.py (API process) and
    frontend/app.py (UI process), and safe under test re-imports."""
    global _configured
    with _configure_lock:
        if _configured:
            return
        formatter = logging.Formatter(_LOG_FORMAT)
        _BUFFER_HANDLER.setFormatter(formatter)

        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)

        logger = logging.getLogger(LOGGER_NAME)
        logger.setLevel(level)
        logger.addHandler(console_handler)
        logger.addHandler(_BUFFER_HANDLER)
        logger.propagate = False
        _configured = True


def get_log_entries(since_seq: int = 0, limit: int = 500) -> list[LogEntry]:
    """The `since_seq` entries newest-first callers haven't seen, for
    GET /logs polling; at most `limit` of them."""
    return _BUFFER_HANDLER.entries(since_seq=since_seq, limit=limit)


def clear_log_entries() -> None:
    """Test-only: reset the buffer between test cases."""
    _BUFFER_HANDLER.clear()
