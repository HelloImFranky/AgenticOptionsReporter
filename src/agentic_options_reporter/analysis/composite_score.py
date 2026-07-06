"""Provider-agnostic composite Trade Quality Score engine.

Weights, the renormalization/confidence-discount rules, and the action
thresholds mirror specs/scoring.yaml exactly; update both together if the
model changes.

The single `compute_composite_score` function here is used identically by
two different callers that never call each other:
- analysis/domain_scoring.py + analysis/scoring.py, feeding it
  deterministically-computed DomainScores (source="quant") during /analyze.
- thesis/orchestrator.py, feeding it LLM-agent-authored DomainScores
  (source="agent") on the Agents tab.

Neither caller is imported here — this module has no notion of "quant" or
"agent" beyond the source tag it's told to stamp on the result.
"""

from __future__ import annotations

from datetime import datetime, timezone

from agentic_options_reporter.models.schemas import (
    DomainScore,
    RecommendationAction,
    ScoreSource,
    TradeQualityScore,
    WeightingProfileId,
)

DOMAIN_IDS: tuple[str, ...] = (
    "technical",
    "risk",
    "liquidity",
    "fundamental",
    "macro",
    "sentiment",
    "relative_strength",
    "statistical_edge",
)

# Each profile's weights sum to 1.0. day_trade favors fast-moving,
# liquidity-sensitive signals; long_term favors fundamentals/macro/relative
# strength; swing sits in between. One global action-threshold table
# applies regardless of profile (see ACTION_THRESHOLDS below) — the profile
# changes how the score is built, not how a given score is read.
WEIGHTING_PROFILES: dict[str, dict[str, float]] = {
    "day_trade": {
        "technical": 0.30,
        "liquidity": 0.20,
        "risk": 0.15,
        "relative_strength": 0.10,
        "statistical_edge": 0.10,
        "sentiment": 0.10,
        "fundamental": 0.03,
        "macro": 0.02,
    },
    "swing": {
        "technical": 0.20,
        "risk": 0.15,
        "liquidity": 0.10,
        "fundamental": 0.15,
        "macro": 0.10,
        "sentiment": 0.10,
        "relative_strength": 0.10,
        "statistical_edge": 0.10,
    },
    "long_term": {
        "technical": 0.10,
        "risk": 0.10,
        "liquidity": 0.05,
        "fundamental": 0.25,
        "macro": 0.15,
        "sentiment": 0.05,
        "relative_strength": 0.20,
        "statistical_edge": 0.10,
    },
}

DEFAULT_WEIGHTING_PROFILE: WeightingProfileId = "swing"

ACTION_THRESHOLDS: list[tuple[float, RecommendationAction]] = [
    (75, "STRONG_BUY"),
    (60, "BUY"),
    (40, "HOLD"),
    (0, "AVOID"),
]


def _action_for_score(score: float) -> RecommendationAction:
    for floor, action in ACTION_THRESHOLDS:
        if score >= floor:
            return action
    return "AVOID"


def _explainability(
    present: dict[str, DomainScore], weights: dict[str, float]
) -> list[str]:
    """Deterministic 'led by X, held back by Y' bullets ranking the present
    domains by score — part of the persisted, API-visible record, not just
    UI text (both the quant and agent composite carry this)."""
    if not present:
        return ["No domain scores were available for this candidate."]
    ranked = sorted(present.items(), key=lambda kv: kv[1].score, reverse=True)
    lines: list[str] = []
    for index, (domain, ds) in enumerate(ranked):
        weight_pct = round(weights[domain] * 100)
        tag = ""
        if len(ranked) > 1 and index == 0:
            tag = " — strongest contributor"
        elif len(ranked) > 1 and index == len(ranked) - 1:
            tag = " — weakest contributor"
        label = domain.replace("_", " ").title()
        lines.append(f"{label} (weight {weight_pct}%): {ds.score:.0f}/100{tag}")
    return lines


def compute_composite_score(
    domain_scores: dict[str, DomainScore],
    *,
    source: ScoreSource,
    weighting_profile: WeightingProfileId = DEFAULT_WEIGHTING_PROFILE,
    contract_symbol: str | None = None,
) -> TradeQualityScore:
    """Blend whichever DomainScores are present using the named weighting
    profile.

    A domain absent from `domain_scores` (no data source for it, or the
    calling agent didn't run) is excluded from both the numerator and the
    weight sum — renormalization over what's actually known, not a
    fabricated neutral value. The resulting `confidence` is further
    discounted by the fraction of the profile's weight that was actually
    covered (`raw_confidence * total_weight`), so a thin-data composite
    can't read as falsely authoritative. Callers/UI must always present
    `composite_score` alongside `confidence`, never the score alone.
    """
    weights = WEIGHTING_PROFILES[weighting_profile]
    present = {d: s for d, s in domain_scores.items() if d in weights and weights[d] > 0}
    total_weight = sum(weights[d] for d in present)

    if total_weight <= 0:
        composite_score = 0.0
        confidence = 0.0
    else:
        composite_score = sum(weights[d] * s.score for d, s in present.items()) / total_weight
        raw_confidence = sum(weights[d] * s.confidence for d, s in present.items()) / total_weight
        confidence = raw_confidence * total_weight

    return TradeQualityScore(
        contract_symbol=contract_symbol,
        domain_scores=present,
        composite_score=round(composite_score, 2),
        confidence=round(confidence, 2),
        recommendation_action=_action_for_score(composite_score),
        weighting_profile=weighting_profile,
        source=source,
        generated_at=datetime.now(timezone.utc).replace(tzinfo=None),
        explainability=_explainability(present, weights),
    )
