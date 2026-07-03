"""Investment Thesis agent.

Synthesizes every prior agent's output (and, when the recommendation had
no candidate, the deterministic recommendation alone) into one narrative
paragraph plus a consensus label. This is the text the user actually
reads (see specs/agents.yaml). financial_research/news_research/
macro_research are each optional — absent when their provider wasn't
configured (see specs/providers.yaml) — and the synthesis proceeds over
whatever is present rather than erroring.
"""

from __future__ import annotations

from agentic_options_reporter.models.schemas import (
    FinancialResearchFinding,
    InvestmentThesis,
    MacroResearchFinding,
    NewsResearchFinding,
    QuantInterpretation,
    Recommendation,
    RiskAssessment,
    StrategySuggestion,
    TrendAssessment,
    VolumeAssessment,
)
from agentic_options_reporter.thesis.llm_client import LlmClient
from agentic_options_reporter.thesis.parsing import parse_response

_SYSTEM_PROMPT = """\
You are a portfolio manager writing a final investment thesis for a
client. You are given a quant interpretation, optional financial/news/
macro research findings, an optional risk assessment, an optional
strategy suggestion, and the underlying recommendation and market
context. Synthesize them into one coherent paragraph — do not just
concatenate them. If any findings seem to disagree (e.g. strong
fundamentals but bearish news, or a bullish quant read against a
risk-off macro regime), address the tension directly rather than
ignoring it. Some findings may be absent (not applicable) — synthesize
over whatever is present.

Respond with a single JSON object with exactly these keys:
{"thesis": "<one coherent paragraph, 3-6 sentences>",
 "consensus": "bullish" | "bearish" | "neutral" | "mixed"}

Output ONLY the JSON object, no markdown fences, no extra text.
"""


def _build_prompt(
    quant_interpretation: QuantInterpretation,
    financial_research: FinancialResearchFinding | None,
    news_research: NewsResearchFinding | None,
    macro_research: MacroResearchFinding | None,
    risk_assessment: RiskAssessment | None,
    strategy_suggestion: StrategySuggestion | None,
    recommendation: Recommendation,
    trend: TrendAssessment,
    volume: VolumeAssessment,
) -> str:
    parts = [
        f"Deterministic recommendation: {recommendation.action} "
        f"(confidence {recommendation.confidence:.0%}) — {recommendation.rationale}",
        f"Trend: {trend.direction} ({trend.strength})",
        f"Volume: {volume.relative_volume:.2f}x average, flags: {', '.join(volume.flags) or 'none'}",
        f"Quant interpretation: {quant_interpretation.narrative} "
        f"Key factors: {', '.join(quant_interpretation.key_factors)}",
    ]

    if financial_research is not None:
        parts.append(
            f"Financial research: health={financial_research.company_health}, "
            f"growth={financial_research.growth}, profitability={financial_research.profitability}, "
            f"cash_flow={financial_research.cash_flow}, "
            f"analyst_consensus={financial_research.analyst_consensus}. "
            f"{financial_research.narrative}"
        )
    else:
        parts.append("Financial research: not applicable (no financial data provider configured).")

    if news_research is not None:
        parts.append(
            f"News research: sentiment={news_research.sentiment}. {news_research.summary} "
            f"Catalysts: {', '.join(news_research.catalysts) or 'none'}. "
            f"Risks: {', '.join(news_research.risks) or 'none'}."
        )
    else:
        parts.append("News research: not applicable (no news provider configured).")

    if macro_research is not None:
        parts.append(
            f"Macro research: regime={macro_research.regime}. {macro_research.summary} "
            f"Outlook: {macro_research.outlook}"
        )
    else:
        parts.append("Macro research: not applicable (no macro provider configured).")

    if risk_assessment is not None:
        parts.append(
            f"Risk assessment: level={risk_assessment.risk_level}, "
            f"concerns={'; '.join(risk_assessment.concerns)}, "
            f"sizing note={risk_assessment.position_sizing_note}"
        )
    else:
        parts.append("Risk assessment: not applicable (no candidate contract to assess).")
    if strategy_suggestion is not None:
        parts.append(
            f"Suggested strategy: {strategy_suggestion.strategy} — {strategy_suggestion.rationale}"
        )
    else:
        parts.append("Suggested strategy: not applicable (no candidate contract).")
    return "\n".join(parts)


def run(
    llm_client: LlmClient,
    quant_interpretation: QuantInterpretation,
    financial_research: FinancialResearchFinding | None,
    news_research: NewsResearchFinding | None,
    macro_research: MacroResearchFinding | None,
    risk_assessment: RiskAssessment | None,
    strategy_suggestion: StrategySuggestion | None,
    recommendation: Recommendation,
    trend: TrendAssessment,
    volume: VolumeAssessment,
) -> InvestmentThesis:
    user_prompt = _build_prompt(
        quant_interpretation,
        financial_research,
        news_research,
        macro_research,
        risk_assessment,
        strategy_suggestion,
        recommendation,
        trend,
        volume,
    )
    raw = llm_client.complete(_SYSTEM_PROMPT, user_prompt)
    return parse_response(InvestmentThesis, raw, "investment_thesis")
