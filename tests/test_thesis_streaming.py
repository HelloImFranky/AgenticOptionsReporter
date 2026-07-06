from datetime import datetime, timezone

from agentic_options_reporter.analysis.composite_score import compute_composite_score
from agentic_options_reporter.models.schemas import (
    AgentEvent,
    AgentThesisResult,
    DomainScore,
    InvestmentThesis,
    QuantInterpretation,
)
from agentic_options_reporter.thesis.streaming import run_thesis_streaming


def _event(agent: str, phase: str) -> AgentEvent:
    return AgentEvent(
        agent=agent, phase=phase, at=datetime.now(timezone.utc).replace(tzinfo=None)
    )


def _result() -> AgentThesisResult:
    return AgentThesisResult(
        run_id=1,
        generated_at=datetime.now(timezone.utc),
        quant_interpretation=QuantInterpretation(
            narrative="x",
            key_factors=[],
            quant_trade_quality=compute_composite_score({}, source="quant", contract_symbol=None),
            technical_domain_score=DomainScore(
                domain="technical", score=0.0, confidence=0.0, evidence=[], factors=[],
                source="agent", generated_at=datetime.now(timezone.utc).replace(tzinfo=None),
            ),
        ),
        risk_assessment=None,
        strategy_suggestion=None,
        investment_thesis=InvestmentThesis(thesis="t", consensus="neutral"),
    )


def test_run_thesis_streaming_yields_events_then_result():
    result = _result()

    def run_pipeline(on_event):
        on_event(_event("quant_interpreter", "started"))
        on_event(_event("quant_interpreter", "completed"))
        return result

    items = list(run_thesis_streaming(run_pipeline))

    kinds = [kind for kind, _ in items]
    assert kinds == ["event", "event", "result"]
    assert items[0][1].agent == "quant_interpreter"
    assert items[0][1].phase == "started"
    assert items[-1][1] is result


def test_run_thesis_streaming_surfaces_exception_as_error_item():
    class Boom(RuntimeError):
        pass

    def run_pipeline(on_event):
        on_event(_event("quant_interpreter", "started"))
        raise Boom("fatal agent")

    items = list(run_thesis_streaming(run_pipeline))

    assert items[0][0] == "event"
    kind, payload = items[-1]
    assert kind == "error"
    assert isinstance(payload, Boom)
    assert "fatal agent" in str(payload)


def test_run_thesis_streaming_preserves_event_order():
    def run_pipeline(on_event):
        for i in range(10):
            on_event(_event(f"agent_{i}", "completed"))
        return _result()

    items = list(run_thesis_streaming(run_pipeline))
    event_agents = [payload.agent for kind, payload in items if kind == "event"]
    assert event_agents == [f"agent_{i}" for i in range(10)]


def test_run_thesis_streaming_result_is_terminal():
    """Nothing is yielded after the terminal result item."""

    def run_pipeline(on_event):
        on_event(_event("quant_interpreter", "started"))
        return _result()

    items = list(run_thesis_streaming(run_pipeline))
    assert items[-1][0] == "result"
    assert [k for k, _ in items].count("result") == 1
