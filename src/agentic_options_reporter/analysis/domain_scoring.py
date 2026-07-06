"""Deterministic domain scorers for the Trade Quality Score (specs/scoring.yaml).

Each function below computes one of the 8 domains as a weighted average
over named, 0-1 sub-factors (see `_weighted_avg`), then wraps the result in
a `DomainScore(source="quant")`. `analysis/scoring.py` calls these per
candidate/run and feeds the results to `analysis/composite_score.py`'s
shared engine.

A sub-factor is `None` (rather than a fabricated neutral 0.5) whenever its
underlying data isn't available; `_weighted_avg` excludes it from both the
numerator and the weight sum and reduces the domain's own `confidence`
accordingly — the same completeness convention the composite engine uses
one level up. A whole domain returns `None` (omitted from the run's
`domain_scores` entirely) when its data source is completely absent (e.g.
no FundamentalsSnapshot, no MacroObservations).
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone

from agentic_options_reporter.models.schemas import (
    DomainFactor,
    DomainScore,
    EvaluatedContract,
    FundamentalsSnapshot,
    IndicatorSnapshot,
    MacroObservation,
    NewsArticle,
    PriceHistory,
    RiskProfile,
    SupportResistanceLevel,
    TrendAssessment,
    VolumeAssessment,
)

# ---------------------------------------------------------------------------
# Shared sub-factor building blocks, several carried over verbatim from the
# legacy flat opportunity-score model (analysis/scoring.py, pre-overhaul) —
# they're still valid 0-1 reads on trend/volume/liquidity/risk-reward, now
# reused as sub-factors within the richer Technical/Risk/Liquidity domains.
# ---------------------------------------------------------------------------

_SR_MAX_DISTANCE_PCT = 0.05
_LIQUIDITY_OI_TARGET = 500
_LIQUIDITY_MAX_SPREAD_PCT = 0.10
_RISK_REWARD_FLOOR = 0.5
_RISK_REWARD_CEIL = 2.0
_RS_LOOKBACK_BARS = 21


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _bias(option_type: str) -> str:
    return "bullish" if option_type == "call" else "bearish"


def _trend_alignment(option_type: str, trend: TrendAssessment) -> float:
    bias = _bias(option_type)
    if trend.direction == "neutral":
        return 0.5
    if trend.direction == bias:
        return 1.0 if trend.strength != "weak" else 0.5
    return 0.0


def _volume_confirmation(option_type: str, volume: VolumeAssessment) -> float:
    bias = _bias(option_type)
    if "high_volume" in volume.flags:
        return 1.0
    if bias == "bullish" and "bearish_divergence" in volume.flags:
        return 0.0
    if bias == "bearish" and "bullish_divergence" in volume.flags:
        return 0.0
    if "low_volume" in volume.flags:
        return 0.0
    return 0.5


def _support_resistance_proximity(
    option_type: str, underlying_price: float, levels: list[SupportResistanceLevel]
) -> float:
    relevant_type = "support" if option_type == "call" else "resistance"
    relevant = [lvl for lvl in levels if lvl.level_type == relevant_type]
    if not relevant or underlying_price <= 0:
        return 0.0
    nearest = min(relevant, key=lambda lvl: abs(lvl.price - underlying_price))
    distance_pct = abs(nearest.price - underlying_price) / underlying_price
    return max(0.0, 1.0 - distance_pct / _SR_MAX_DISTANCE_PCT)


def _liquidity(candidate: EvaluatedContract) -> float:
    oi_score = min(candidate.contract.open_interest / _LIQUIDITY_OI_TARGET, 1.0)
    spread_score = max(0.0, 1 - candidate.spread_pct / _LIQUIDITY_MAX_SPREAD_PCT)
    return 0.5 * oi_score + 0.5 * spread_score


def _risk_reward(reward_risk_ratio: float | None) -> float:
    if reward_risk_ratio is None:
        return 1.0  # unlimited upside (e.g. long call) with defined risk
    if reward_risk_ratio >= _RISK_REWARD_CEIL:
        return 1.0
    if reward_risk_ratio <= _RISK_REWARD_FLOOR:
        return 0.0
    return (reward_risk_ratio - _RISK_REWARD_FLOOR) / (_RISK_REWARD_CEIL - _RISK_REWARD_FLOOR)


def _weighted_avg(
    subfactors: list[tuple[str, float | None, float, str]],
) -> tuple[float, float, list[DomainFactor]]:
    """subfactors: (name, value 0-1 or None, weight 0-1, detail). A None
    value is excluded from both the numerator and the weight sum;
    `confidence` is the fraction of the domain's total sub-factor weight
    that was actually present. Returns (score 0-100, confidence 0-100,
    factors)."""
    present = [(name, value, weight, detail) for name, value, weight, detail in subfactors if value is not None]
    factors = [
        DomainFactor(name=name, value=round(value, 4), weight=weight, detail=detail)
        for name, value, weight, detail in present
    ]
    present_weight = sum(weight for _, _, weight, _ in present)
    full_weight = sum(weight for _, _, weight, _ in subfactors)
    if present_weight <= 0 or full_weight <= 0:
        return 0.0, 0.0, factors
    score = 100 * sum(weight * value for _, value, weight, _ in present) / present_weight
    confidence = 100 * (present_weight / full_weight)
    return score, confidence, factors


def _evidence(factors: list[DomainFactor], limit: int = 5) -> list[str]:
    ranked = sorted(factors, key=lambda f: f.value, reverse=True)
    lines = []
    for f in ranked[:limit]:
        label = f.name.replace("_", " ").title()
        line = f"{label}: {f.value:.2f}"
        if f.detail:
            line += f" — {f.detail}"
        lines.append(line)
    return lines


# ---------------------------------------------------------------------------
# Technical (candidate-level)
# ---------------------------------------------------------------------------


def _regression_slope_pct(closes: list[float]) -> float | None:
    """Normalized least-squares slope of `closes` (fractional change per
    bar), used as a lightweight 'market structure' proxy."""
    n = len(closes)
    if n < 5:
        return None
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(closes) / n
    if mean_y == 0:
        return None
    denom = sum((x - mean_x) ** 2 for x in xs)
    if denom == 0:
        return None
    numer = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, closes))
    return (numer / denom) / mean_y


def _moving_average_alignment(
    bias: str, close: float, sma_20: float, sma_50: float, sma_200: float | None
) -> float:
    stack = [close, sma_20, sma_50] + ([sma_200] if sma_200 is not None else [])
    if bias == "bullish":
        fully_aligned = all(stack[i] >= stack[i + 1] for i in range(len(stack) - 1))
        partially_aligned = close >= sma_20
    else:
        fully_aligned = all(stack[i] <= stack[i + 1] for i in range(len(stack) - 1))
        partially_aligned = close <= sma_20
    if fully_aligned:
        return 1.0
    if partially_aligned:
        return 0.5
    return 0.0


def _adx_component(bias: str, trend: TrendAssessment) -> float:
    normalized = min(trend.adx / 40.0, 1.0)
    if trend.direction == bias:
        return normalized
    if trend.direction == "neutral":
        return 0.5
    return max(0.0, 1.0 - normalized)  # a strong opposing trend hurts


def _momentum_component(bias: str, indicators: IndicatorSnapshot) -> float:
    rsi = indicators.rsi_14
    if bias == "bullish":
        rsi_component = 1.0 if 50 <= rsi <= 70 else 0.5 if 40 <= rsi < 80 else 0.0
        macd_component = 1.0 if indicators.macd_histogram > 0 else 0.0
    else:
        rsi_component = 1.0 if 30 <= rsi <= 50 else 0.5 if 20 < rsi <= 60 else 0.0
        macd_component = 1.0 if indicators.macd_histogram < 0 else 0.0
    return 0.5 * rsi_component + 0.5 * macd_component


def _breakout_quality(
    option_type: str,
    underlying_price: float,
    levels: list[SupportResistanceLevel],
    volume: VolumeAssessment,
) -> float:
    relevant_type = "resistance" if option_type == "call" else "support"
    relevant = [lvl for lvl in levels if lvl.level_type == relevant_type]
    if not relevant or underlying_price <= 0:
        return 0.0
    nearest = min(relevant, key=lambda lvl: abs(lvl.price - underlying_price))
    broke_through = (
        underlying_price > nearest.price if option_type == "call" else underlying_price < nearest.price
    )
    if not broke_through:
        return 0.0
    return 1.0 if "high_volume" in volume.flags else 0.6


def _candlestick_component(bias: str, last_bar) -> float | None:
    price_range = last_bar.high - last_bar.low
    if price_range <= 0:
        return None
    position = (last_bar.close - last_bar.low) / price_range
    return position if bias == "bullish" else 1 - position


def technical_domain_score(
    option_type: str,
    underlying_price: float,
    history: PriceHistory,
    indicators: IndicatorSnapshot,
    trend: TrendAssessment,
    volume: VolumeAssessment,
    levels: list[SupportResistanceLevel],
) -> DomainScore:
    bias = _bias(option_type)
    bars = history.bars
    closes = [b.close for b in bars]

    slope_pct = _regression_slope_pct(closes[-20:]) if len(closes) >= 5 else None
    market_structure = None
    if slope_pct is not None:
        aligned = slope_pct if bias == "bullish" else -slope_pct
        market_structure = _clamp(0.5 + aligned / 0.01)

    last_close = closes[-1] if closes else underlying_price
    moving_averages = _moving_average_alignment(
        bias, last_close, indicators.sma_20, indicators.sma_50, indicators.sma_200
    )

    subfactors: list[tuple[str, float | None, float, str]] = [
        ("trend_alignment", _trend_alignment(option_type, trend), 0.15, "Directional bias vs. detected trend"),
        ("market_structure", market_structure, 0.10, "Price regression slope, direction-aligned"),
        ("moving_averages", moving_averages, 0.15, "SMA20/50/200 stack alignment"),
        ("adx", _adx_component(bias, trend), 0.10, "ADX trend strength, direction-aware"),
        ("momentum", _momentum_component(bias, indicators), 0.15, "RSI band + MACD histogram sign"),
        ("volume_confirmation", _volume_confirmation(option_type, volume), 0.10, "Relative volume vs. trend"),
        ("rvol", _clamp(volume.relative_volume / 2.0), 0.05, "Relative volume magnitude"),
        (
            "support_resistance_proximity",
            _support_resistance_proximity(option_type, underlying_price, levels),
            0.10,
            "Distance to nearest supporting level",
        ),
        (
            "breakout_quality",
            _breakout_quality(option_type, underlying_price, levels, volume),
            0.05,
            "Price beyond the adverse S/R level, volume-confirmed",
        ),
        (
            "candlestick_confirmation",
            _candlestick_component(bias, bars[-1]) if bars else None,
            0.05,
            "Last bar's close position within its range",
        ),
    ]
    score, confidence, factors = _weighted_avg(subfactors)
    return DomainScore(
        domain="technical",
        score=score,
        confidence=confidence,
        evidence=_evidence(factors),
        factors=factors,
        source="quant",
        generated_at=_now(),
    )


# ---------------------------------------------------------------------------
# Risk (candidate-level)
# ---------------------------------------------------------------------------


def _stop_quality(
    option_type: str, underlying_price: float, breakeven: float, levels: list[SupportResistanceLevel]
) -> float | None:
    adverse_type = "support" if option_type == "call" else "resistance"
    relevant = [lvl for lvl in levels if lvl.level_type == adverse_type]
    if not relevant or underlying_price <= 0:
        return None
    nearest = min(relevant, key=lambda lvl: abs(lvl.price - underlying_price))
    distance_to_stop = abs(underlying_price - nearest.price)
    distance_to_breakeven = abs(underlying_price - breakeven)
    if distance_to_breakeven <= 0:
        return None
    return _clamp(1 - distance_to_stop / distance_to_breakeven)


def _expected_drawdown(theta: float, mid_price: float) -> float | None:
    if mid_price <= 0:
        return None
    daily_decay_pct = abs(theta) / mid_price
    return _clamp(1 - daily_decay_pct / 0.02)


def _volatility_risk(implied_volatility: float, atr_14: float, underlying_price: float) -> float | None:
    if underlying_price <= 0 or atr_14 <= 0:
        return None
    hv_proxy = (atr_14 / underlying_price) * math.sqrt(252)
    if hv_proxy <= 0:
        return None
    premium = (implied_volatility - hv_proxy) / hv_proxy
    return _clamp(1 - max(premium, 0.0))


def _position_sizing(probability_of_profit: float, reward_risk_ratio: float | None) -> float:
    rr = reward_risk_ratio if reward_risk_ratio is not None else 5.0
    kelly = probability_of_profit - (1 - probability_of_profit) / max(rr, 0.01)
    return _clamp(kelly)


def risk_domain_score(
    candidate: EvaluatedContract,
    risk: RiskProfile,
    indicators: IndicatorSnapshot,
    levels: list[SupportResistanceLevel],
) -> DomainScore:
    option_type = candidate.contract.option_type
    subfactors: list[tuple[str, float | None, float, str]] = [
        ("risk_reward", _risk_reward(risk.reward_risk_ratio), 0.30, "Reward:risk ratio"),
        (
            "stop_quality",
            _stop_quality(option_type, candidate.underlying_price, risk.breakeven, levels),
            0.20,
            "Room to a logical stop before breakeven is breached",
        ),
        (
            "expected_drawdown",
            _expected_drawdown(candidate.greeks.theta, candidate.mid_price),
            0.20,
            "Daily theta decay as % of premium",
        ),
        (
            "volatility_risk",
            _volatility_risk(candidate.implied_volatility, indicators.atr_14, candidate.underlying_price),
            0.15,
            "Implied vol vs. ATR-derived historical-vol proxy",
        ),
        (
            "position_sizing",
            _position_sizing(risk.probability_of_profit, risk.reward_risk_ratio),
            0.15,
            "Simplified Kelly fraction from PoP + reward:risk",
        ),
    ]
    score, confidence, factors = _weighted_avg(subfactors)
    return DomainScore(
        domain="risk",
        score=score,
        confidence=confidence,
        evidence=_evidence(factors),
        factors=factors,
        source="quant",
        generated_at=_now(),
    )


# ---------------------------------------------------------------------------
# Liquidity (candidate-level)
# ---------------------------------------------------------------------------


def _bid_ask_spread_component(spread_pct: float) -> float:
    return _clamp(1 - spread_pct / 0.10)


def _adv_component(volume_sma_20: float) -> float:
    return _clamp(volume_sma_20 / 500_000)


def _slippage_component(spread_pct: float) -> float:
    """A tighter-ceiling variant of the spread proxy — no Level-2/quote-depth
    data is available from any configured provider, so slippage is
    approximated from the spread alone rather than left unmodeled."""
    return _clamp(1 - spread_pct / 0.05)


def _market_impact_component(open_interest: int) -> float:
    return _clamp(open_interest / 2000)


def liquidity_domain_score(candidate: EvaluatedContract, indicators: IndicatorSnapshot) -> DomainScore:
    contract = candidate.contract
    subfactors: list[tuple[str, float | None, float, str]] = [
        ("options_liquidity", _liquidity(candidate), 0.30, "Open interest + spread tightness"),
        ("bid_ask_spread", _bid_ask_spread_component(candidate.spread_pct), 0.25, "Bid/ask spread as % of mid"),
        ("adv", _adv_component(indicators.volume_sma_20), 0.20, "Underlying average-daily-volume proxy"),
        ("slippage", _slippage_component(candidate.spread_pct), 0.15, "Spread-based slippage proxy"),
        ("market_impact", _market_impact_component(contract.open_interest), 0.10, "Open-interest depth proxy"),
    ]
    score, confidence, factors = _weighted_avg(subfactors)
    return DomainScore(
        domain="liquidity",
        score=score,
        confidence=confidence,
        evidence=_evidence(factors),
        factors=factors,
        source="quant",
        generated_at=_now(),
    )


# ---------------------------------------------------------------------------
# Fundamental (run-level; omitted entirely if no FundamentalsSnapshot)
# ---------------------------------------------------------------------------


def fundamental_domain_score(fundamentals: FundamentalsSnapshot | None) -> DomainScore | None:
    if fundamentals is None:
        return None
    metrics = fundamentals.metrics
    ratios = fundamentals.ratios
    statements = fundamentals.statements
    estimates = fundamentals.estimates
    earnings_history = fundamentals.earnings_history

    revenue_growth = (
        _clamp(0.5 + metrics.revenue_growth * 2)
        if metrics and metrics.revenue_growth is not None
        else None
    )
    earnings_growth = (
        _clamp(0.5 + metrics.earnings_growth * 2)
        if metrics and metrics.earnings_growth is not None
        else None
    )

    eps_surprise = None
    if earnings_history and earnings_history.surprises:
        latest = earnings_history.surprises[-1]
        if latest.surprise_percent is not None:
            eps_surprise = _clamp(0.5 + latest.surprise_percent * 2)

    margins = None
    if metrics and (metrics.gross_margin is not None or metrics.profit_margin is not None):
        parts = []
        if metrics.gross_margin is not None:
            parts.append(_clamp(metrics.gross_margin / 0.5))
        if metrics.profit_margin is not None:
            parts.append(_clamp(metrics.profit_margin / 0.2))
        margins = sum(parts) / len(parts)

    fcf = None
    if statements and statements.free_cash_flow is not None:
        if statements.revenue:
            fcf = _clamp(0.5 + (statements.free_cash_flow / statements.revenue) * 2)
        else:
            fcf = 0.7 if statements.free_cash_flow > 0 else 0.2

    debt = _clamp(1 - ratios.debt_to_equity / 2.0) if ratios and ratios.debt_to_equity is not None else None
    roe = _clamp(ratios.return_on_equity / 0.20) if ratios and ratios.return_on_equity is not None else None

    valuation = None
    if metrics and metrics.peg_ratio is not None and metrics.peg_ratio > 0:
        valuation = _clamp(1 - max(metrics.peg_ratio - 1, 0) / 2)

    analyst_revisions = _consensus_rating_score(estimates.consensus_rating if estimates else None, bullish=True)

    subfactors: list[tuple[str, float | None, float, str]] = [
        ("revenue_growth", revenue_growth, 0.12, "YoY revenue growth"),
        ("earnings_growth", earnings_growth, 0.12, "YoY earnings growth"),
        ("eps_surprise", eps_surprise, 0.10, "Latest EPS surprise vs. estimate"),
        ("margins", margins, 0.15, "Gross/net margin quality"),
        ("fcf", fcf, 0.12, "Free cash flow"),
        ("debt", debt, 0.10, "Debt-to-equity"),
        ("roe", roe, 0.12, "Return on equity"),
        ("valuation", valuation, 0.12, "PEG-based valuation"),
        ("analyst_revisions", analyst_revisions, 0.05, "Analyst consensus rating"),
    ]
    score, confidence, factors = _weighted_avg(subfactors)
    if not factors:
        return None
    return DomainScore(
        domain="fundamental",
        score=score,
        confidence=confidence,
        evidence=_evidence(factors),
        factors=factors,
        source="quant",
        generated_at=_now(),
    )


def _consensus_rating_score(consensus_rating: str | None, *, bullish: bool) -> float | None:
    if not consensus_rating or consensus_rating.strip().upper() == "N/A":
        return None
    rating = consensus_rating.strip().lower()
    if any(term in rating for term in ("buy", "outperform", "overweight")):
        value = 1.0
    elif any(term in rating for term in ("sell", "underperform", "underweight")):
        value = 0.0
    elif any(term in rating for term in ("hold", "neutral")):
        value = 0.5
    else:
        return None
    return value if bullish else 1 - value


# ---------------------------------------------------------------------------
# Macro (run-level, bias-aligned; omitted entirely if no observations)
# ---------------------------------------------------------------------------


def macro_domain_score(observations: list[MacroObservation], option_type: str) -> DomainScore | None:
    if not observations:
        return None
    bullish = option_type == "call"

    def align(value: float) -> float:
        return value if bullish else 1 - value

    by_id = {o.metric_id: o for o in observations}

    interest_rates = None
    policy = by_id.get("policy_rate")
    if policy and policy.yoy_change_pct is not None:
        interest_rates = align(_clamp(0.5 - policy.yoy_change_pct / 20.0))

    inflation = None
    cpi = by_id.get("cpi")
    if cpi and cpi.yoy_change_pct is not None:
        inflation = align(_clamp(0.5 - (cpi.yoy_change_pct - 2.0) / 10.0))

    bond_yield_curve = None
    t10, t2 = by_id.get("treasury_10y"), by_id.get("treasury_2y")
    if t10 is not None and t2 is not None:
        spread = t10.value - t2.value
        bond_yield_curve = align(_clamp(0.5 + spread / 2.0))

    gdp = None
    gdp_obs = by_id.get("gdp")
    if gdp_obs and gdp_obs.yoy_change_pct is not None:
        gdp = align(_clamp(gdp_obs.yoy_change_pct / 4.0))

    subfactors: list[tuple[str, float | None, float, str]] = [
        ("interest_rates", interest_rates, 0.30, "Policy rate YoY trend"),
        ("inflation", inflation, 0.25, "CPI YoY trend"),
        ("bond_yield_curve", bond_yield_curve, 0.25, "10y-2y Treasury spread"),
        ("gdp", gdp, 0.20, "GDP YoY growth"),
    ]
    score, confidence, factors = _weighted_avg(subfactors)
    if not factors:
        return None
    # Macro -> single-stock is inherently a blunt instrument, regardless of
    # how many of the 4 sub-factors were servable.
    confidence = min(confidence, 80.0)
    return DomainScore(
        domain="macro",
        score=score,
        confidence=confidence,
        evidence=_evidence(factors),
        factors=factors,
        source="quant",
        generated_at=_now(),
    )


# ---------------------------------------------------------------------------
# Sentiment (run-level, bias-aligned except news_volume; omitted entirely if
# there's neither news nor fundamentals-derived sentiment data)
# ---------------------------------------------------------------------------


def sentiment_domain_score(
    articles: list[NewsArticle], fundamentals: FundamentalsSnapshot | None, option_type: str
) -> DomainScore | None:
    if not articles and fundamentals is None:
        return None
    bullish = option_type == "call"

    def align(value: float) -> float:
        return value if bullish else 1 - value

    # Attention/momentum proxy, deliberately direction-agnostic: the quant
    # engine stays non-LLM, so judging *bullish vs. bearish* sentiment from
    # article text is left to the news_research agent (specs/agents.yaml).
    news_volume = _clamp(len(articles) / 10.0) if articles else None

    earnings_proximity = None
    calendar = fundamentals.earnings_calendar if fundamentals else None
    if calendar and calendar.next_date:
        days_out = (calendar.next_date - date.today()).days
        earnings_proximity = 1.0 if days_out < 0 else _clamp((days_out - 3) / 11.0)

    estimates = fundamentals.estimates if fundamentals else None
    analyst_ratings = _consensus_rating_score(
        estimates.consensus_rating if estimates else None, bullish=bullish
    )

    insider_activity = None
    insider = fundamentals.insider_activity if fundamentals else None
    if insider and insider.net_shares:
        magnitude = _clamp(abs(insider.net_shares) / 100_000.0) * 0.5
        base = 0.5 + (magnitude if insider.net_shares > 0 else -magnitude)
        insider_activity = align(_clamp(base))

    subfactors: list[tuple[str, float | None, float, str]] = [
        ("news_volume", news_volume, 0.25, "Recent news attention (direction-agnostic)"),
        ("earnings_proximity", earnings_proximity, 0.25, "Days to next earnings (event risk)"),
        ("analyst_ratings", analyst_ratings, 0.30, "Analyst consensus rating"),
        ("insider_activity", insider_activity, 0.20, "Net insider buying/selling"),
    ]
    score, confidence, factors = _weighted_avg(subfactors)
    if not factors:
        return None
    return DomainScore(
        domain="sentiment",
        score=score,
        confidence=confidence,
        evidence=_evidence(factors),
        factors=factors,
        source="quant",
        generated_at=_now(),
    )


# ---------------------------------------------------------------------------
# Relative Strength (run-level, bias-aligned; no new provider interface —
# reuses MarketDataProvider for SPY/sector-ETF price history)
# ---------------------------------------------------------------------------


def _trailing_return(history: PriceHistory | None, bars: int = _RS_LOOKBACK_BARS) -> float | None:
    if history is None:
        return None
    closes = [b.close for b in history.bars]
    if len(closes) <= bars:
        return None
    start, end = closes[-bars - 1], closes[-1]
    if start <= 0:
        return None
    return (end - start) / start


def relative_strength_domain_score(
    symbol_history: PriceHistory,
    benchmark_history: PriceHistory | None,
    sector_history: PriceHistory | None,
    option_type: str,
) -> DomainScore | None:
    bullish = option_type == "call"

    def align(value: float) -> float:
        return value if bullish else 1 - value

    symbol_return = _trailing_return(symbol_history)
    if symbol_return is None:
        return None

    vs_market = None
    bench_return = _trailing_return(benchmark_history)
    if bench_return is not None:
        vs_market = align(_clamp(0.5 + (symbol_return - bench_return) * 2))

    vs_sector = None
    sector_return = _trailing_return(sector_history)
    if sector_return is not None:
        vs_sector = align(_clamp(0.5 + (symbol_return - sector_return) * 2))

    subfactors: list[tuple[str, float | None, float, str]] = [
        ("vs_market", vs_market, 0.55, "21-trading-day return vs. SPY"),
        ("vs_sector", vs_sector, 0.45, "21-trading-day return vs. sector ETF"),
    ]
    score, confidence, factors = _weighted_avg(subfactors)
    if not factors:
        return None
    return DomainScore(
        domain="relative_strength",
        score=score,
        confidence=confidence,
        evidence=_evidence(factors),
        factors=factors,
        source="quant",
        generated_at=_now(),
    )


# Sector -> ETF proxy map for the Relative Strength domain. A sector absent
# here simply means vs_sector is omitted and vs_market carries the full
# domain weight (see relative_strength_domain_score / _weighted_avg).
SECTOR_ETF_MAP: dict[str, str] = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Financials": "XLF",
    "Energy": "XLE",
    "Healthcare": "XLV",
    "Health Care": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Industrials": "XLI",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Basic Materials": "XLB",
    "Materials": "XLB",
    "Communication Services": "XLC",
}
