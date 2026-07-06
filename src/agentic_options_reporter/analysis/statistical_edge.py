"""Statistical Edge domain scorer (specs/scoring.yaml).

No outcome-tracking/backtest infrastructure exists elsewhere in this
codebase, so this domain combines two techniques implementable now from
data already in the DB or already fetched for /analyze:

1. Historical hit-rate (historical_win_rate / expectancy / pattern_success):
   query this symbol's past AnalysisRuns (persistence.fetch_recent_runs_for_
   symbol) and check whether the underlying moved in the recommended
   direction ~5 trading days later, using the *current* price history
   (which, at the default 365-day lookback, already spans most recent past
   runs). Below a 5-run minimum sample these three sub-factors are omitted
   (not fabricated as neutral) — the domain gets more informative the
   longer the tool is used against a given symbol.
2. Monte Carlo confidence: a historical bootstrap — resample day-over-day
   % returns from the trailing bars (fixed seed) to estimate P(the forward
   path finishes past breakeven in the required direction) as a
   model-free cross-check against analysis/risk.py's closed-form
   Black-Scholes probability_of_profit.

Weight is capped at 0.10 and confidence at 70 in every weighting profile
(analysis/composite_score.py WEIGHTING_PROFILES) so this proxy can never
dominate the composite score.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import numpy as np

from agentic_options_reporter.analysis.domain_scoring import _clamp, _evidence, _now, _weighted_avg
from agentic_options_reporter.models.schemas import DomainScore, PastRunOutcome, PriceHistory

_MIN_SAMPLE_RUNS = 5
_FORWARD_BARS = 5  # ~1 trading week, mirrors workflow._WEEK_BARS
_BOOTSTRAP_LOOKBACK_BARS = 252
_BOOTSTRAP_RESAMPLES = 750
_BOOTSTRAP_SEED = 20260706
_CONFIDENCE_CAP = 70.0


def _bias(option_type: str) -> str:
    return "bullish" if option_type == "call" else "bearish"


@dataclass
class _HitRateResult:
    win_rate: float | None
    expectancy: float | None
    pattern_success: float | None
    detail: str


def _bars_by_date(history: PriceHistory) -> list[tuple[date, float]]:
    return [(b.dt, b.close) for b in history.bars]


def _forward_return(bars: list[tuple[date, float]], run_date: date) -> float | None:
    """Return from the first bar on/after `run_date` to `_FORWARD_BARS`
    trading days later, or None if the history doesn't reach that far."""
    entry_index = next((i for i, (d, _) in enumerate(bars) if d >= run_date), None)
    if entry_index is None:
        return None
    target_index = entry_index + _FORWARD_BARS
    if target_index >= len(bars):
        return None
    entry_close = bars[entry_index][1]
    if entry_close <= 0:
        return None
    return (bars[target_index][1] - entry_close) / entry_close


def _historical_hit_rate(
    option_type: str, history: PriceHistory, past_runs: list[PastRunOutcome]
) -> _HitRateResult:
    bars = _bars_by_date(history)
    bias = _bias(option_type)

    outcomes: list[tuple[float, str]] = []  # (forward_return, run_bias)
    for run in past_runs:
        if run.action == "AVOID" or run.option_type is None:
            continue
        forward_return = _forward_return(bars, run.generated_at.date())
        if forward_return is None:
            continue
        outcomes.append((forward_return, _bias(run.option_type)))

    n = len(outcomes)
    if n < _MIN_SAMPLE_RUNS:
        return _HitRateResult(
            None,
            None,
            None,
            f"Insufficient run history (n={n}<{_MIN_SAMPLE_RUNS}); based on Monte Carlo simulation only.",
        )

    hits = [
        (fr > 0) if run_bias == "bullish" else (fr < 0)
        for fr, run_bias in outcomes
    ]
    win_rate = _clamp(sum(hits) / n)

    avg_return = sum(fr for fr, _ in outcomes) / n
    aligned_avg_return = avg_return if bias == "bullish" else -avg_return
    expectancy = _clamp(0.5 + aligned_avg_return * 5)

    same_bias_hits = [hit for (fr, run_bias), hit in zip(outcomes, hits) if run_bias == bias]
    if len(same_bias_hits) >= _MIN_SAMPLE_RUNS:
        pattern_success = _clamp(sum(same_bias_hits) / len(same_bias_hits))
        detail = f"Based on {n} past recommendations ({len(same_bias_hits)} matching this bias)."
    else:
        pattern_success = win_rate
        detail = f"Based on {n} past recommendations (insufficient same-bias sample; using overall rate)."

    return _HitRateResult(win_rate, expectancy, pattern_success, detail)


def _monte_carlo_confidence(
    option_type: str,
    closes: list[float],
    days_to_expiration: int,
    breakeven: float,
    underlying_price: float,
) -> float | None:
    if len(closes) < 30 or underlying_price <= 0 or breakeven <= 0:
        return None
    window = np.asarray(closes[-_BOOTSTRAP_LOOKBACK_BARS:], dtype=float)
    if len(window) < 20:
        return None
    daily_returns = np.diff(window) / window[:-1]
    daily_returns = daily_returns[np.isfinite(daily_returns)]
    if len(daily_returns) < 20:
        return None

    horizon_bars = max(round(days_to_expiration * 252 / 365), 1)
    rng = np.random.default_rng(_BOOTSTRAP_SEED)
    sampled = rng.choice(daily_returns, size=(_BOOTSTRAP_RESAMPLES, horizon_bars), replace=True)
    terminal_prices = underlying_price * np.prod(1 + sampled, axis=1)
    favorable = terminal_prices > breakeven if option_type == "call" else terminal_prices < breakeven
    return float(np.mean(favorable))


def statistical_edge_domain_score(
    option_type: str,
    history: PriceHistory,
    days_to_expiration: int,
    breakeven: float,
    underlying_price: float,
    past_runs: list[PastRunOutcome],
) -> DomainScore | None:
    hit_rate = _historical_hit_rate(option_type, history, past_runs)
    closes = [b.close for b in history.bars]
    monte_carlo = _monte_carlo_confidence(option_type, closes, days_to_expiration, breakeven, underlying_price)

    subfactors: list[tuple[str, float | None, float, str]] = [
        ("historical_win_rate", hit_rate.win_rate, 0.35, hit_rate.detail),
        (
            "expectancy",
            hit_rate.expectancy,
            0.25,
            "Average forward return of past recommendations, bias-aligned",
        ),
        (
            "pattern_success",
            hit_rate.pattern_success,
            0.20,
            "Hit rate conditioned on matching directional bias",
        ),
        (
            "monte_carlo_confidence",
            monte_carlo,
            0.20,
            "Historical-bootstrap P(price finishes past breakeven)",
        ),
    ]
    score, confidence, factors = _weighted_avg(subfactors)
    if not factors:
        return None
    confidence = min(confidence, _CONFIDENCE_CAP)
    return DomainScore(
        domain="statistical_edge",
        score=score,
        confidence=confidence,
        evidence=_evidence(factors),
        factors=factors,
        source="quant",
        generated_at=_now(),
    )
