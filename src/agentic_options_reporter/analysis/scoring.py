"""Trade Quality Score assembly and recommendation generation.

Per-candidate domain scores come from analysis/domain_scoring.py and
analysis/statistical_edge.py; the composite blend comes from
analysis/composite_score.py. This module's job is just to assemble the 8
domains for each candidate (some candidate-level, some run-level and
shared across candidates of the same run) and pick the top-scoring one.
See specs/scoring.yaml.
"""

from __future__ import annotations

from agentic_options_reporter.analysis.composite_score import compute_composite_score
from agentic_options_reporter.analysis.domain_scoring import (
    fundamental_domain_score,
    liquidity_domain_score,
    macro_domain_score,
    relative_strength_domain_score,
    risk_domain_score,
    sentiment_domain_score,
    technical_domain_score,
)
from agentic_options_reporter.analysis.statistical_edge import statistical_edge_domain_score
from agentic_options_reporter.models.schemas import (
    DomainScore,
    EvaluatedContract,
    FundamentalsSnapshot,
    IndicatorSnapshot,
    MacroObservation,
    NewsArticle,
    PastRunOutcome,
    PriceHistory,
    Recommendation,
    RiskProfile,
    ScoredCandidate,
    SupportResistanceLevel,
    TradeQualityScore,
    TrendAssessment,
    VolumeAssessment,
    WeightingProfileId,
)


def _assemble_domain_scores(
    candidate: EvaluatedContract,
    risk: RiskProfile,
    history: PriceHistory,
    indicators: IndicatorSnapshot,
    trend: TrendAssessment,
    volume: VolumeAssessment,
    levels: list[SupportResistanceLevel],
    fundamentals: FundamentalsSnapshot | None,
    macro_observations: list[MacroObservation],
    news_articles: list[NewsArticle],
    benchmark_history: PriceHistory | None,
    sector_history: PriceHistory | None,
    past_runs: list[PastRunOutcome],
) -> dict[str, DomainScore]:
    option_type = candidate.contract.option_type
    domain_scores: dict[str, DomainScore] = {
        "technical": technical_domain_score(
            option_type, candidate.underlying_price, history, indicators, trend, volume, levels
        ),
        "risk": risk_domain_score(candidate, risk, indicators, levels),
        "liquidity": liquidity_domain_score(candidate, indicators),
    }

    fundamental = fundamental_domain_score(fundamentals)
    if fundamental is not None:
        domain_scores["fundamental"] = fundamental

    macro = macro_domain_score(macro_observations, option_type)
    if macro is not None:
        domain_scores["macro"] = macro

    sentiment = sentiment_domain_score(news_articles, fundamentals, option_type)
    if sentiment is not None:
        domain_scores["sentiment"] = sentiment

    relative_strength = relative_strength_domain_score(
        history, benchmark_history, sector_history, option_type
    )
    if relative_strength is not None:
        domain_scores["relative_strength"] = relative_strength

    statistical_edge = statistical_edge_domain_score(
        option_type, history, candidate.days_to_expiration, risk.breakeven, candidate.underlying_price, past_runs
    )
    if statistical_edge is not None:
        domain_scores["statistical_edge"] = statistical_edge

    return domain_scores


def score_candidates(
    evaluated_contracts: list[EvaluatedContract],
    risk_profiles: list[RiskProfile],
    trend: TrendAssessment,
    volume: VolumeAssessment,
    levels: list[SupportResistanceLevel],
    history: PriceHistory,
    indicators: IndicatorSnapshot,
    *,
    fundamentals: FundamentalsSnapshot | None = None,
    macro_observations: list[MacroObservation] | None = None,
    news_articles: list[NewsArticle] | None = None,
    benchmark_history: PriceHistory | None = None,
    sector_history: PriceHistory | None = None,
    past_runs: list[PastRunOutcome] | None = None,
    weighting_profile: WeightingProfileId = "swing",
) -> list[ScoredCandidate]:
    risk_by_symbol = {rp.contract_symbol: rp for rp in risk_profiles}

    scored: list[ScoredCandidate] = []
    for candidate in evaluated_contracts:
        if not candidate.liquidity_ok:
            continue
        risk = risk_by_symbol.get(candidate.contract.contract_symbol)
        if risk is None:
            continue

        domain_scores = _assemble_domain_scores(
            candidate,
            risk,
            history,
            indicators,
            trend,
            volume,
            levels,
            fundamentals,
            macro_observations or [],
            news_articles or [],
            benchmark_history,
            sector_history,
            past_runs or [],
        )
        trade_quality = compute_composite_score(
            domain_scores,
            source="quant",
            weighting_profile=weighting_profile,
            contract_symbol=candidate.contract.contract_symbol,
        )

        scored.append(
            ScoredCandidate(
                contract_symbol=candidate.contract.contract_symbol,
                option_type=candidate.contract.option_type,
                strike=candidate.contract.strike,
                expiration=candidate.contract.expiration,
                delta=candidate.greeks.delta,
                gamma=candidate.greeks.gamma,
                theta=candidate.greeks.theta,
                vega=candidate.greeks.vega,
                rho=candidate.greeks.rho,
                open_interest=candidate.contract.open_interest,
                spread_pct=candidate.spread_pct,
                volume=candidate.contract.volume,
                max_loss=risk.max_loss,
                max_gain=risk.max_gain,
                breakeven=risk.breakeven,
                reward_risk_ratio=risk.reward_risk_ratio,
                probability_of_profit=risk.probability_of_profit,
                score=trade_quality.composite_score,
                domain_scores=domain_scores,
            )
        )

    scored.sort(key=lambda c: c.score, reverse=True)
    return scored


def build_recommendation(
    candidates: list[ScoredCandidate],
    weighting_profile: WeightingProfileId = "swing",
) -> tuple[Recommendation, TradeQualityScore | None]:
    """Returns (Recommendation, TradeQualityScore | None) — the latter is
    the quant-side Trade Quality Score for the top candidate, None on
    AVOID / no liquid candidates, mirroring Recommendation.contract_symbol."""
    if not candidates:
        return (
            Recommendation(
                action="AVOID",
                contract_symbol=None,
                confidence=0.0,
                rationale="No liquid, scoreable candidates were found in the option chain.",
            ),
            None,
        )

    top = max(candidates, key=lambda c: c.score)
    trade_quality = compute_composite_score(
        top.domain_scores,
        source="quant",
        weighting_profile=weighting_profile,
        contract_symbol=top.contract_symbol,
    )
    explanation = "; ".join(trade_quality.explainability[:3])
    rationale = f"{top.contract_symbol} scored {trade_quality.composite_score:.1f}/100 ({explanation})."

    recommendation = Recommendation(
        action=trade_quality.recommendation_action,
        contract_symbol=top.contract_symbol,
        confidence=round(trade_quality.confidence / 100, 4),
        rationale=rationale,
    )
    return recommendation, trade_quality
