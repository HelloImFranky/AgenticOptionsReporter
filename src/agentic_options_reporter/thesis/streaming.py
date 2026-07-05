"""Bridge the synchronous, blocking thesis pipeline to a live event stream.

`run_thesis_pipeline` (thesis.orchestrator) only returns once every agent
has finished, but it can call an `on_event` callback as each agent runs.
To surface those events live (for an SSE endpoint), we run the pipeline on
a worker thread that pushes each event onto a queue, and drain that queue
from the caller — yielding events as they happen and, finally, the result
or the error that ended the run.

HTTP-agnostic on purpose: it yields typed StreamItem tuples, and the API
layer (main.py) formats them as `text/event-stream` frames.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Iterator
from typing import Literal

from agentic_options_reporter.models.schemas import AgentEvent, AgentThesisResult

# ("event", AgentEvent) as each agent runs; then exactly one terminal item:
# ("result", AgentThesisResult) on success or ("error", Exception) on a
# fatal failure (an LlmError / ThesisGenerationError from a required agent).
StreamKind = Literal["event", "result", "error"]
StreamItem = tuple[StreamKind, object]

_DONE = object()


def run_thesis_streaming(
    run_pipeline: Callable[[Callable[[AgentEvent], None]], AgentThesisResult],
) -> Iterator[StreamItem]:
    """Run a pipeline that accepts an `on_event` callback and returns an
    AgentThesisResult, yielding each AgentEvent as it fires and then a
    single terminal ("result", ...) or ("error", ...) item.

    `run_pipeline` is a thunk so the caller can bind the analysis result,
    llm client, and providers however it likes (see main.py)."""
    events: queue.Queue[StreamItem | object] = queue.Queue()

    def worker() -> None:
        try:
            result = run_pipeline(lambda event: events.put(("event", event)))
            events.put(("result", result))
        except Exception as exc:  # noqa: BLE001 — surfaced to the client as a terminal error item
            events.put(("error", exc))
        finally:
            events.put(_DONE)

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    try:
        while True:
            item = events.get()
            if item is _DONE:
                break
            yield item  # type: ignore[misc]
    finally:
        thread.join()
