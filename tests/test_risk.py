from agentic_options_reporter.analysis.options import evaluate_chain
from agentic_options_reporter.analysis.risk import compute_risk


def test_call_breakeven_and_max_gain(sample_option_chain, uptrend_history):
    evaluated = evaluate_chain(sample_option_chain, uptrend_history)
    calls = [e for e in evaluated if e.contract.option_type == "call"]
    risk_profiles = compute_risk(calls)

    for profile, candidate in zip(risk_profiles, calls):
        assert profile.max_gain is None  # unlimited upside
        assert profile.max_loss == candidate.mid_price * 100
        assert profile.breakeven == candidate.contract.strike + candidate.mid_price
        assert profile.reward_risk_ratio is None
        assert 0 <= profile.probability_of_profit <= 1


def test_put_max_gain_is_capped(sample_option_chain, uptrend_history):
    evaluated = evaluate_chain(sample_option_chain, uptrend_history)
    puts = [e for e in evaluated if e.contract.option_type == "put"]
    risk_profiles = compute_risk(puts)

    for profile, candidate in zip(risk_profiles, puts):
        assert profile.max_gain is not None
        assert profile.max_gain >= 0
        assert profile.breakeven == candidate.contract.strike - candidate.mid_price
        if profile.max_loss > 0:
            assert profile.reward_risk_ratio == profile.max_gain / profile.max_loss
