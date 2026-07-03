"""Orchestrates the investment-thesis agent pipeline (specs/agents.yaml).

Coordination only — no LLM calls of its own. Runs quant_interpreter,
then (unless short-circuited) risk_challenger and options_strategy, then
investment_thesis, and assembles the result.
"""

from __future__ import annotations

from datetime import datetime, timezone

from agentic_options_reporter.models.schemas import (
    AgentThesisResult,
    AnalysisResult,
    QuantInterpretation,
)
from agentic_options_reporter.thesis import investment_thesis, options_strategy, quant_interpreter, risk_challenger
from agentic_options_reporter.thesis.llm_client import LlmClient


def run_thesis_pipeline(analysis_result: AnalysisResult, llm_client: LlmClient) -> AgentThesisResult:
    recommendation = analysis_result.recommendation
    top_candidate = next(
        (c for c in analysis_result.candidates if c.contract_symbol == recommendation.contract_symbol),
        None,
    )

    if top_candidate is None:
        # No liquid candidate to assess or size — skip risk/strategy agents
        # entirely rather than asking an LLM to reason about data that
        # doesn't exist (see specs/agents.yaml: no_candidate_short_circuit).
        quant = QuantInterpretation(
            narrative=recommendation.rationale,
            key_factors=[],
            score_breakdown={},
            overall_score=0.0,
        )
        risk = None
        strategy = None
    else:
        quant = quant_interpreter.run(
            llm_client,
            analysis_result.indicators,
            analysis_result.trend,
            analysis_result.volume,
            top_candidate,
        )
        risk = risk_challenger.run(
            llm_client, top_candidate, analysis_result.trend, analysis_result.support_resistance
        )
        strategy = options_strategy.run(llm_client, analysis_result.trend, top_candidate, risk)

    thesis = investment_thesis.run(
        llm_client,
        quant,
        risk,
        strategy,
        recommendation,
        analysis_result.trend,
        analysis_result.volume,
    )

    return AgentThesisResult(
        run_id=analysis_result.run_id,
        generated_at=datetime.now(timezone.utc).replace(tzinfo=None),
        quant_interpretation=quant,
        risk_assessment=risk,
        strategy_suggestion=strategy,
        investment_thesis=thesis,
    )
