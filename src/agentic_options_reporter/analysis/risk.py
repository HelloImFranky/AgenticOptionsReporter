"""Risk metrics for long single-leg option candidates.

See docs/option_analysis.md#risk-analysisrisk-py.
"""

from __future__ import annotations

import math

from scipy.stats import norm

from agentic_options_reporter.config import get_settings
from agentic_options_reporter.models.schemas import EvaluatedContract, RiskProfile

CONTRACT_MULTIPLIER = 100


def _probability_of_profit(
    option_type: str,
    underlying_price: float,
    breakeven: float,
    sigma: float,
    days_to_expiration: int,
    risk_free_rate: float,
) -> float:
    t = max(days_to_expiration / 365.0, 1 / 365)
    sigma = max(sigma, 1e-4)
    if underlying_price <= 0 or breakeven <= 0:
        return 0.0

    drift = (risk_free_rate - sigma**2 / 2) * t
    d = (math.log(underlying_price / breakeven) + drift) / (sigma * math.sqrt(t))

    if option_type == "call":
        return float(norm.cdf(d))
    return float(norm.cdf(-d))


def compute_risk(
    candidates: list[EvaluatedContract], risk_free_rate: float | None = None
) -> list[RiskProfile]:
    settings = get_settings()
    rate = risk_free_rate if risk_free_rate is not None else settings.risk_free_rate

    profiles: list[RiskProfile] = []
    for candidate in candidates:
        contract = candidate.contract
        premium = candidate.mid_price

        max_loss = premium * CONTRACT_MULTIPLIER

        if contract.option_type == "call":
            breakeven = contract.strike + premium
            max_gain = None  # unlimited upside for a long call
        else:
            breakeven = contract.strike - premium
            max_gain = max(contract.strike - premium, 0.0) * CONTRACT_MULTIPLIER

        reward_risk_ratio = (
            max_gain / max_loss if max_gain is not None and max_loss > 0 else None
        )

        probability_of_profit = _probability_of_profit(
            contract.option_type,
            candidate.underlying_price,
            breakeven,
            candidate.implied_volatility,
            candidate.days_to_expiration,
            rate,
        )

        profiles.append(
            RiskProfile(
                contract_symbol=contract.contract_symbol,
                max_loss=max_loss,
                max_gain=max_gain,
                breakeven=breakeven,
                reward_risk_ratio=reward_risk_ratio,
                probability_of_profit=probability_of_profit,
            )
        )
    return profiles
