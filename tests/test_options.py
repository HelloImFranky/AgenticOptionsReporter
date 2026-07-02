from datetime import date, timedelta

from agentic_options_reporter.analysis.options import (
    compute_greeks,
    evaluate_chain,
    implied_volatility,
)


def test_call_delta_between_zero_and_one():
    greeks = compute_greeks("call", spot=100, strike=100, t=0.25, r=0.045, sigma=0.3)
    assert 0 < greeks.delta < 1
    assert greeks.gamma > 0
    assert greeks.vega > 0


def test_put_delta_between_minus_one_and_zero():
    greeks = compute_greeks("put", spot=100, strike=100, t=0.25, r=0.045, sigma=0.3)
    assert -1 < greeks.delta < 0


def test_deep_itm_call_delta_near_one():
    greeks = compute_greeks("call", spot=200, strike=100, t=0.25, r=0.045, sigma=0.3)
    assert greeks.delta > 0.9


def test_implied_volatility_round_trip():
    from agentic_options_reporter.analysis.options import _bs_price

    true_sigma = 0.4
    price = _bs_price("call", spot=100, strike=100, t=0.5, r=0.045, sigma=true_sigma)
    solved = implied_volatility("call", price, spot=100, strike=100, t=0.5, r=0.045)
    assert abs(solved - true_sigma) < 0.01


def test_evaluate_chain_marks_liquidity(sample_option_chain, uptrend_history):
    evaluated = evaluate_chain(sample_option_chain, uptrend_history)
    assert len(evaluated) == len(sample_option_chain.contracts)
    assert all(e.liquidity_ok for e in evaluated)  # fixture contracts are all liquid
    assert all(e.greeks.gamma >= 0 for e in evaluated)


def test_evaluate_chain_rejects_illiquid_contracts(sample_option_chain, uptrend_history):
    sample_option_chain.contracts[0].open_interest = 0
    sample_option_chain.contracts[0].bid = 0.0
    evaluated = evaluate_chain(sample_option_chain, uptrend_history)
    assert evaluated[0].liquidity_ok is False
