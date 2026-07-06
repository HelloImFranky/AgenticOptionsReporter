"""Shared LLM-authored domain-score sub-schema (specs/agents.yaml phase_3).

Each research agent that contributes to the agent-side Trade Quality Score
(thesis/orchestrator.py's composite step) embeds this exact JSON shape
under a "domain_score" key in its response contract, alongside its
existing qualitative fields. `assemble_domain_score` wraps the LLM-authored
score/confidence/evidence with the fixed (non-LLM-authored) domain id,
source, and timestamp to build the full DomainScore — the same pattern
`analyst_consensus` already uses for a provider pass-through, applied here
to a value that instead comes from the model's own independent judgment.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from agentic_options_reporter.models.schemas import DomainId, DomainScore, Score0to100

# Included verbatim in each agent's system prompt so every JSON contract
# asks for the sub-schema identically.
DOMAIN_SCORE_PROMPT_FIELD = (
    '"domain_score": {"score": <0-100 number>, "confidence": <0-100 number>, '
    '"evidence": ["<short phrase>", "..."]}'
)
DOMAIN_SCORE_PROMPT_RULE = (
    "domain_score.score/confidence are YOUR OWN independent 0-100 judgment for "
    "this domain — not a copy of any other number you were given. evidence "
    "should have 1-5 short phrases grounding the score."
)


class LlmDomainScoreFields(BaseModel):
    score: Score0to100
    confidence: Score0to100
    evidence: list[str] = Field(default_factory=list)


def assemble_domain_score(domain: DomainId, fields: LlmDomainScoreFields) -> DomainScore:
    return DomainScore(
        domain=domain,
        score=fields.score,
        confidence=fields.confidence,
        evidence=fields.evidence,
        factors=[],
        source="agent",
        generated_at=datetime.now(timezone.utc).replace(tzinfo=None),
    )
