"""Pure functions turning ApiClient JSON payloads into display-ready data.

Kept separate from app.py so this logic is unit-testable without a Flet
runtime. Input shapes mirror the models in specs/api.yaml.
"""

from __future__ import annotations

from typing import Any

CANDIDATE_COLUMNS = [
    "Contract",
    "Type",
    "Strike",
    "Expiration",
    "Score",
    "Delta",
    "Breakeven",
    "Max Loss",
    "Max Gain",
    "PoP",
]

RUN_COLUMNS = ["Run ID", "Symbol", "Generated At", "Action", "Confidence"]

# Semantic tones, mapped to actual colors in app.py. Kept as plain strings
# here so this module has no dependency on Flet and stays unit-testable.
TONE_SUCCESS = "success"
TONE_WARNING = "warning"
TONE_DANGER = "danger"
TONE_NEUTRAL = "neutral"

_RECOMMENDATION_TONES = {
    "STRONG_BUY": TONE_SUCCESS,
    "BUY": TONE_SUCCESS,
    "HOLD": TONE_WARNING,
    "AVOID": TONE_DANGER,
}

_TREND_TONES = {
    "bullish": TONE_SUCCESS,
    "bearish": TONE_DANGER,
    "neutral": TONE_NEUTRAL,
}

_CONSENSUS_TONES = {
    "bullish": TONE_SUCCESS,
    "bearish": TONE_DANGER,
    "neutral": TONE_NEUTRAL,
    "mixed": TONE_WARNING,
}

_RISK_LEVEL_TONES = {
    "low": TONE_SUCCESS,
    "medium": TONE_WARNING,
    "high": TONE_DANGER,
}

_MACRO_REGIME_TONES = {
    "risk_on": TONE_SUCCESS,
    "risk_off": TONE_DANGER,
    "neutral": TONE_NEUTRAL,
}

# Financial-research finding tones. Mirrors the enums in
# specs/api.yaml (CompanyHealth/GrowthTrend/ProfitabilityLevel/CashFlowState):
# the healthiest value reads success, the weakest danger, the middle neutral.
_COMPANY_HEALTH_TONES = {
    "strong": TONE_SUCCESS,
    "stable": TONE_NEUTRAL,
    "weak": TONE_DANGER,
}

_GROWTH_TONES = {
    "accelerating": TONE_SUCCESS,
    "steady": TONE_NEUTRAL,
    "decelerating": TONE_DANGER,
}

_PROFITABILITY_TONES = {
    "high": TONE_SUCCESS,
    "moderate": TONE_NEUTRAL,
    "low": TONE_DANGER,
}

_CASH_FLOW_TONES = {
    "positive": TONE_SUCCESS,
    "neutral": TONE_NEUTRAL,
    "negative": TONE_DANGER,
}


def recommendation_tone(action: str) -> str:
    return _RECOMMENDATION_TONES.get(action, TONE_NEUTRAL)


def trend_tone(direction: str) -> str:
    return _TREND_TONES.get(direction, TONE_NEUTRAL)


def consensus_tone(consensus: str) -> str:
    return _CONSENSUS_TONES.get(consensus, TONE_NEUTRAL)


def risk_level_tone(risk_level: str) -> str:
    return _RISK_LEVEL_TONES.get(risk_level, TONE_NEUTRAL)


def macro_regime_tone(regime: str) -> str:
    return _MACRO_REGIME_TONES.get(regime, TONE_NEUTRAL)


def company_health_tone(value: str) -> str:
    return _COMPANY_HEALTH_TONES.get(value, TONE_NEUTRAL)


def growth_tone(value: str) -> str:
    return _GROWTH_TONES.get(value, TONE_NEUTRAL)


def profitability_tone(value: str) -> str:
    return _PROFITABILITY_TONES.get(value, TONE_NEUTRAL)


def cash_flow_tone(value: str) -> str:
    return _CASH_FLOW_TONES.get(value, TONE_NEUTRAL)


def quant_score_tone(score: float) -> str:
    """Tone for the quant overall_score (0-100), aligned with the
    recommendation thresholds in specs/scoring.yaml: BUY/STRONG_BUY (>=60)
    reads success, HOLD (>=40) warning, AVOID danger."""
    if score >= 60:
        return TONE_SUCCESS
    if score >= 40:
        return TONE_WARNING
    return TONE_DANGER


def format_recommendation(recommendation: dict[str, Any]) -> str:
    contract = recommendation.get("contract_symbol") or "—"
    rationale = recommendation.get("rationale", "")
    return f"{contract}\n{rationale}" if rationale else contract


def format_timestamp(value: str) -> str:
    """Trim an ISO datetime string down to minute precision for display."""
    return str(value).replace("T", " ")[:16]


def recommended_candidate(
    recommendation: dict[str, Any], candidates: list[dict[str, Any]] | None
) -> dict[str, Any] | None:
    """The scored candidate the recommendation points at (matched by
    contract_symbol), or None for an AVOID / no-contract recommendation."""
    symbol = recommendation.get("contract_symbol")
    if not symbol:
        return None
    for candidate in candidates or []:
        if candidate.get("contract_symbol") == symbol:
            return candidate
    return None


def score_breakdown_items(score_breakdown: dict[str, float] | None) -> list[tuple[str, float]]:
    """Convert raw score factor names into display labels and numeric values."""
    if not score_breakdown:
        return []

    labels = {
        "trend_alignment": "Trend Alignment",
        "volume_confirmation": "Volume Confirmation",
        "support_resistance_proximity": "Support Resistance Proximity",
        "liquidity": "Liquidity",
        "risk_reward": "Risk Reward",
    }
    return [(labels.get(key, key.replace("_", " ").title()), float(value)) for key, value in score_breakdown.items()]


def score_breakdown_summary(score_breakdown: dict[str, float] | None) -> str:
    """A one-line, plain-language read of the score breakdown — what carried
    the score and what dragged on it — to caption the breakdown chart
    instead of restating every factor as raw text. Deterministic (no LLM);
    derived straight from the factor values."""
    items = score_breakdown_items(score_breakdown)
    if not items:
        return ""
    ordered = sorted(items, key=lambda kv: kv[1], reverse=True)
    if len(ordered) == 1:
        label, value = ordered[0]
        return f"Score reflects {label.lower()} ({value:.2f})."
    (top_label, top_value) = ordered[0]
    (bottom_label, bottom_value) = ordered[-1]
    # When every factor sits in a tight band, "strongest vs weakest" is
    # misleading — call it balanced instead.
    if top_value - bottom_value < 0.15:
        return f"Score is balanced across {len(ordered)} factors, all near {top_value:.2f}."
    return (
        f"Score is led by {top_label.lower()} ({top_value:.2f}) and held back by "
        f"{bottom_label.lower()} ({bottom_value:.2f})."
    )


def recommendation_facts(
    recommendation: dict[str, Any], candidates: list[dict[str, Any]] | None = None
) -> list[tuple[str, str]]:
    """Key facts for the recommendation as (label, value) pairs, ready to
    lay out as boxes or a table in either the UI or the PDF. Contract-level
    metrics are pulled from the matching scored candidate; a field absent
    from the candidate payload is simply omitted rather than shown as 0."""
    facts: list[tuple[str, str]] = [("Contract", recommendation.get("contract_symbol") or "—")]
    candidate = recommended_candidate(recommendation, candidates)
    if candidate is None:
        return facts

    def add(label: str, key: str, fmt) -> None:
        value = candidate.get(key)
        if key in candidate and value is not None:
            facts.append((label, fmt(value)))

    add("Type", "option_type", lambda v: str(v).upper())
    add("Strike", "strike", lambda v: f"{v:.2f}")
    add("Expiration", "expiration", lambda v: str(v))
    add("Score", "score", lambda v: f"{v:.1f}")
    add("Delta", "delta", lambda v: f"{v:.3f}")
    add("Breakeven", "breakeven", lambda v: f"{v:.2f}")
    add("Max loss", "max_loss", lambda v: f"{v:.2f}")
    if "max_gain" in candidate:
        max_gain = candidate.get("max_gain")
        facts.append(("Max gain", "unlimited" if max_gain is None else f"{max_gain:.2f}"))
    add("PoP", "probability_of_profit", lambda v: f"{v:.0%}")
    return facts


def technical_snapshot_facts(
    trend: dict[str, Any] | None,
    volume: dict[str, Any] | None,
    indicators: dict[str, Any] | None,
) -> list[tuple[str, str]]:
    """Technical snapshot as (label, value) pairs, for a boxed/table layout
    instead of run-on sentences."""
    facts: list[tuple[str, str]] = []
    if trend:
        direction = str(trend.get("direction", "?")).capitalize()
        facts.append(("Trend", f"{direction} · {trend.get('strength', '?')}"))
        facts.append(("ADX", f"{(trend.get('adx') or 0.0):.1f}"))
    if volume:
        facts.append(("Rel. volume", f"{(volume.get('relative_volume') or 0.0):.2f}x avg"))
        facts.append(("Volume flags", ", ".join(volume.get("flags") or []) or "none"))
    if indicators:
        facts.append(("SMA 20", f"{(indicators.get('sma_20') or 0.0):.2f}"))
        facts.append(("SMA 50", f"{(indicators.get('sma_50') or 0.0):.2f}"))
        facts.append(("RSI 14", f"{(indicators.get('rsi_14') or 0.0):.1f}"))
        facts.append(("ATR 14", f"{(indicators.get('atr_14') or 0.0):.2f}"))
    return facts


def format_money(value: Any) -> str:
    """Compact currency: $3.00T / $500.00M / $1,234. '—' for non-numbers."""
    if not isinstance(value, (int, float)):
        return "—"
    v = float(value)
    for threshold, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if abs(v) >= threshold:
            return f"${v / threshold:.2f}{suffix}"
    return f"${v:,.0f}"


def format_pct(value: Any) -> str:
    """A fraction (0.123) as a percentage (12.3%). '—' for non-numbers."""
    if not isinstance(value, (int, float)):
        return "—"
    return f"{float(value) * 100:.1f}%"


def format_num(value: Any, digits: int = 2) -> str:
    if not isinstance(value, (int, float)):
        return "—"
    return f"{float(value):.{digits}f}"


def fundamentals_metric_facts(metrics: dict[str, Any] | None) -> list[tuple[str, str]]:
    """Company key-stats as (label, value) pairs, filtered to the ones a
    provider actually returned — shared by the Analyze tab's Fundamentals
    card and the PDF report so they can't drift."""
    if not metrics:
        return []
    pairs = [
        ("Market cap", format_money(metrics.get("market_cap"))),
        ("P/E", format_num(metrics.get("pe_ratio"))),
        ("Forward P/E", format_num(metrics.get("forward_pe"))),
        ("PEG", format_num(metrics.get("peg_ratio"))),
        ("Price/Book", format_num(metrics.get("price_to_book"))),
        ("Beta", format_num(metrics.get("beta"))),
        ("Div. yield", format_pct(metrics.get("dividend_yield"))),
        ("Op. margin", format_pct(metrics.get("operating_margin"))),
        ("Profit margin", format_pct(metrics.get("profit_margin"))),
        ("Rev. growth", format_pct(metrics.get("revenue_growth"))),
        ("1w high", format_num(metrics.get("week1_high"))),
        ("1w low", format_num(metrics.get("week1_low"))),
        ("1m high", format_num(metrics.get("month1_high"))),
        ("1m low", format_num(metrics.get("month1_low"))),
        ("52w high", format_num(metrics.get("week52_high"))),
        ("52w low", format_num(metrics.get("week52_low"))),
    ]
    return [(label, value) for label, value in pairs if value != "—"]


def format_next_earnings(calendar: dict[str, Any] | None) -> str | None:
    """One-line 'next earnings' summary, or None when no date is known."""
    if not calendar or not calendar.get("next_date"):
        return None
    line = f"Next earnings: {calendar['next_date']}"
    eps = calendar.get("eps_estimate")
    if isinstance(eps, (int, float)):
        line += f"  ·  EPS est. {eps:.2f}"
    return line


def earnings_surprise_facts(
    earnings: dict[str, Any] | None, limit: int = 4
) -> list[tuple[str, str]]:
    """Recent earnings surprises as (period, 'actual vs estimate · +7.1%')
    pairs, most recent first, capped at `limit`."""
    surprises = (earnings or {}).get("surprises") or []
    facts: list[tuple[str, str]] = []
    for s in surprises[:limit]:
        value = f"{format_num(s.get('actual_eps'))} vs {format_num(s.get('estimate_eps'))} est"
        pct = s.get("surprise_percent")
        if isinstance(pct, (int, float)):
            value += f"  ·  {format_pct(pct)}"
        facts.append((str(s.get("period", "")), value))
    return facts


def insider_activity_header(insider: dict[str, Any] | None) -> str:
    """The 'Insider activity — net buying/selling (N shares)' heading, or ''
    when there's nothing to show."""
    if not insider:
        return ""
    transactions = insider.get("transactions") or []
    net = insider.get("net_shares")
    if not transactions and not isinstance(net, (int, float)):
        return ""
    header = "Insider activity"
    if isinstance(net, (int, float)) and net != 0:
        direction = "net buying" if net > 0 else "net selling"
        header += f" — {direction} ({abs(net):,.0f} shares)"
    return header


def insider_transaction_bars(
    insider: dict[str, Any] | None, limit: int = 8
) -> list[dict[str, Any]]:
    """Recent insider transactions prepared for a bar chart: one dict per
    transaction with `label`, `detail` ('sell 1,000'), `is_buy`, and `ratio`
    (bar length 0..1, sized by |shares| relative to the largest shown). Sells
    are is_buy=False (render red), buys True (green). Empty when no
    transaction has a share count to size a bar."""
    transactions = (insider or {}).get("transactions") or []
    sized = []
    for t in transactions[:limit]:
        shares = t.get("shares")
        if not isinstance(shares, (int, float)) or shares == 0:
            continue
        ttype = t.get("transaction_type") or "txn"
        sized.append(
            {
                "label": t.get("name") or "Insider",
                "detail": f"{ttype} {abs(shares):,.0f}",
                "is_buy": ttype != "sell",
                "magnitude": abs(float(shares)),
            }
        )
    if not sized:
        return []
    largest = max(item["magnitude"] for item in sized)
    for item in sized:
        item["ratio"] = item["magnitude"] / largest if largest else 0.0
    return sized


def format_trend_summary(trend: dict[str, Any]) -> str:
    direction = str(trend.get("direction", "?")).capitalize()
    strength = trend.get("strength", "?")
    adx = trend.get("adx") or 0.0
    return f"Trend: {direction} · {strength} (ADX {adx:.1f})"


def format_volume_summary(volume: dict[str, Any]) -> str:
    relative_volume = volume.get("relative_volume") or 0.0
    flags = ", ".join(volume.get("flags") or []) or "none"
    return f"Volume: {relative_volume:.2f}x average · flags: {flags}"


def format_indicator_summary(indicators: dict[str, Any]) -> str:
    sma_20 = indicators.get("sma_20") or 0.0
    sma_50 = indicators.get("sma_50") or 0.0
    rsi_14 = indicators.get("rsi_14") or 0.0
    atr_14 = indicators.get("atr_14") or 0.0
    return f"SMA20 {sma_20:.2f} · SMA50 {sma_50:.2f} · RSI14 {rsi_14:.1f} · ATR14 {atr_14:.2f}"


def candidates_to_rows(candidates: list[dict[str, Any]]) -> list[list[str]]:
    rows = []
    for candidate in candidates:
        max_gain = candidate.get("max_gain")
        rows.append(
            [
                str(candidate.get("contract_symbol", "")),
                str(candidate.get("option_type", "")).upper(),
                f"{candidate.get('strike', 0):.2f}",
                str(candidate.get("expiration", "")),
                f"{candidate.get('score', 0):.1f}",
                f"{candidate.get('delta', 0):.3f}",
                f"{candidate.get('breakeven', 0):.2f}",
                f"{candidate.get('max_loss', 0):.2f}",
                "unlimited" if max_gain is None else f"{max_gain:.2f}",
                f"{(candidate.get('probability_of_profit') or 0):.0%}",
            ]
        )
    return rows


def runs_to_rows(runs: list[dict[str, Any]]) -> list[list[str]]:
    rows = []
    for run in runs:
        rows.append(
            [
                str(run.get("run_id", "")),
                str(run.get("symbol", "")),
                format_timestamp(run.get("generated_at", "")),
                str(run.get("recommendation_action", "")),
                f"{(run.get('recommendation_confidence') or 0):.0%}",
            ]
        )
    return rows
