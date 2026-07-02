"""Option chain evaluation and Black-Scholes Greeks.

See docs/option_analysis.md#chain-evaluation.
"""

from __future__ import annotations

import math
from datetime import date

from scipy.stats import norm

from agentic_options_reporter.config import get_settings
from agentic_options_reporter.models.schemas import (
    EvaluatedContract,
    Greeks,
    OptionChain,
    OptionContract,
    PriceHistory,
)

_MIN_YEARS_TO_EXPIRATION = 1 / 365
_DEFAULT_IV_GUESS = 0.30
_DEFAULT_IV_FALLBACK = 0.30


def _years_to_expiration(expiration: date, as_of: date) -> float:
    days = max((expiration - as_of).days, 1)
    return max(days / 365.0, _MIN_YEARS_TO_EXPIRATION)


def _bs_price(
    option_type: str, spot: float, strike: float, t: float, r: float, sigma: float
) -> float:
    sigma = max(sigma, 1e-6)
    d1 = (math.log(spot / strike) + (r + sigma**2 / 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if option_type == "call":
        return spot * norm.cdf(d1) - strike * math.exp(-r * t) * norm.cdf(d2)
    return strike * math.exp(-r * t) * norm.cdf(-d2) - spot * norm.cdf(-d1)


def implied_volatility(
    option_type: str,
    market_price: float,
    spot: float,
    strike: float,
    t: float,
    r: float,
    initial_guess: float = _DEFAULT_IV_GUESS,
    max_iterations: int = 50,
    tolerance: float = 1e-5,
) -> float:
    """Newton-Raphson solve for implied volatility; falls back to a default."""
    if market_price <= 0 or spot <= 0 or strike <= 0 or t <= 0:
        return _DEFAULT_IV_FALLBACK

    sigma = initial_guess
    for _ in range(max_iterations):
        price = _bs_price(option_type, spot, strike, t, r, sigma)
        vega = _vega(spot, strike, t, r, sigma)
        diff = price - market_price
        if abs(diff) < tolerance:
            return max(sigma, 1e-4)
        if vega < 1e-8:
            break
        sigma -= diff / vega
        if sigma <= 0 or sigma > 5:
            break
    return sigma if 1e-4 < sigma <= 5 else _DEFAULT_IV_FALLBACK


def _vega(spot: float, strike: float, t: float, r: float, sigma: float) -> float:
    sigma = max(sigma, 1e-6)
    d1 = (math.log(spot / strike) + (r + sigma**2 / 2) * t) / (sigma * math.sqrt(t))
    return spot * norm.pdf(d1) * math.sqrt(t)


def compute_greeks(
    option_type: str,
    spot: float,
    strike: float,
    t: float,
    r: float,
    sigma: float,
) -> Greeks:
    sigma = max(sigma, 1e-6)
    t = max(t, _MIN_YEARS_TO_EXPIRATION)
    sqrt_t = math.sqrt(t)
    d1 = (math.log(spot / strike) + (r + sigma**2 / 2) * t) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t

    gamma = norm.pdf(d1) / (spot * sigma * sqrt_t)
    vega = spot * norm.pdf(d1) * sqrt_t / 100  # per 1% change in IV

    if option_type == "call":
        delta = norm.cdf(d1)
        theta = (
            -(spot * norm.pdf(d1) * sigma) / (2 * sqrt_t)
            - r * strike * math.exp(-r * t) * norm.cdf(d2)
        ) / 365
        rho = (strike * t * math.exp(-r * t) * norm.cdf(d2)) / 100
    else:
        delta = norm.cdf(d1) - 1
        theta = (
            -(spot * norm.pdf(d1) * sigma) / (2 * sqrt_t)
            + r * strike * math.exp(-r * t) * norm.cdf(-d2)
        ) / 365
        rho = (-strike * t * math.exp(-r * t) * norm.cdf(-d2)) / 100

    return Greeks(delta=delta, gamma=gamma, theta=theta, vega=vega, rho=rho)


def evaluate_contract(
    contract: OptionContract,
    underlying_price: float,
    as_of: date,
    risk_free_rate: float,
    min_open_interest: int,
    max_spread_pct: float,
) -> EvaluatedContract:
    t = _years_to_expiration(contract.expiration, as_of)
    days_to_expiration = max((contract.expiration - as_of).days, 0)

    sigma = contract.implied_volatility
    if not sigma or sigma <= 0:
        sigma = implied_volatility(
            contract.option_type,
            contract.mid_price,
            underlying_price,
            contract.strike,
            t,
            risk_free_rate,
        )

    greeks = compute_greeks(
        contract.option_type, underlying_price, contract.strike, t, risk_free_rate, sigma
    )

    spread_pct = contract.spread_pct
    liquidity_ok = (
        contract.open_interest >= min_open_interest
        and contract.bid > 0
        and spread_pct <= max_spread_pct
    )

    return EvaluatedContract(
        contract=contract,
        greeks=greeks,
        liquidity_ok=liquidity_ok,
        mid_price=contract.mid_price,
        spread_pct=spread_pct,
        days_to_expiration=days_to_expiration,
        underlying_price=underlying_price,
        implied_volatility=sigma,
    )


def evaluate_chain(
    chain: OptionChain,
    history: PriceHistory,
    risk_free_rate: float | None = None,
    min_open_interest: int | None = None,
    max_spread_pct: float | None = None,
) -> list[EvaluatedContract]:
    settings = get_settings()
    risk_free_rate = risk_free_rate if risk_free_rate is not None else settings.risk_free_rate
    min_open_interest = (
        min_open_interest if min_open_interest is not None else settings.min_open_interest
    )
    max_spread_pct = (
        max_spread_pct if max_spread_pct is not None else settings.max_spread_pct
    )

    as_of = chain.as_of.date()
    return [
        evaluate_contract(
            contract,
            chain.underlying_price,
            as_of,
            risk_free_rate,
            min_open_interest,
            max_spread_pct,
        )
        for contract in chain.contracts
    ]
