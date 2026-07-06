from datetime import date, timedelta

from agentic_options_reporter.analysis.domain_scoring import (
    fundamental_domain_score,
    liquidity_domain_score,
    macro_domain_score,
    relative_strength_domain_score,
    risk_domain_score,
    sentiment_domain_score,
    technical_domain_score,
)
from agentic_options_reporter.analysis.indicators import compute_indicators
from agentic_options_reporter.analysis.options import evaluate_chain
from agentic_options_reporter.analysis.risk import compute_risk
from agentic_options_reporter.analysis.support_resistance import detect_levels
from agentic_options_reporter.analysis.trend import detect_trend
from agentic_options_reporter.analysis.volume import analyze_volume
from agentic_options_reporter.models.schemas import (
    AnalystEstimates,
    CompanyMetrics,
    EarningsCalendar,
    FinancialRatios,
    FundamentalsSnapshot,
    InsiderActivity,
    InsiderTransaction,
    MacroObservation,
    NewsArticle,
)


def _call_and_put(sample_option_chain, history):
    indicators = compute_indicators(history)
    trend = detect_trend(history, indicators)
    volume = analyze_volume(history, indicators)
    levels = detect_levels(history)
    evaluated = evaluate_chain(sample_option_chain, history)
    risk_profiles = {rp.contract_symbol: rp for rp in compute_risk(evaluated)}

    call = next(c for c in evaluated if c.contract.option_type == "call" and c.liquidity_ok)
    put = next(c for c in evaluated if c.contract.option_type == "put" and c.liquidity_ok)
    return (call, risk_profiles[call.contract.contract_symbol]), (put, risk_profiles[put.contract.contract_symbol]), indicators, trend, volume, levels


def test_technical_domain_score_bounds_and_bias_sensitivity(sample_option_chain, uptrend_history):
    (call, _), (put, _), indicators, trend, volume, levels = _call_and_put(sample_option_chain, uptrend_history)

    call_score = technical_domain_score("call", call.underlying_price, uptrend_history, indicators, trend, volume, levels)
    put_score = technical_domain_score("put", put.underlying_price, uptrend_history, indicators, trend, volume, levels)

    assert 0 <= call_score.score <= 100
    assert 0 <= put_score.score <= 100
    assert call_score.domain == "technical"
    assert call_score.source == "quant"
    # In an uptrend, the bullish (call) read should score higher than bearish.
    assert call_score.score > put_score.score


def test_risk_domain_score_bounds(sample_option_chain, uptrend_history):
    (call, risk), _, indicators, _, _, levels = _call_and_put(sample_option_chain, uptrend_history)
    result = risk_domain_score(call, risk, indicators, levels)
    assert result.domain == "risk"
    assert 0 <= result.score <= 100
    assert 0 <= result.confidence <= 100


def test_liquidity_domain_score_bounds(sample_option_chain, uptrend_history):
    (call, _), _, indicators, _, _, _ = _call_and_put(sample_option_chain, uptrend_history)
    result = liquidity_domain_score(call, indicators)
    assert result.domain == "liquidity"
    assert 0 <= result.score <= 100


def test_fundamental_domain_score_none_without_snapshot():
    assert fundamental_domain_score(None) is None


def test_fundamental_domain_score_present_with_partial_data():
    snapshot = FundamentalsSnapshot(
        ticker="TEST",
        metrics=CompanyMetrics(ticker="TEST", revenue_growth=0.15, earnings_growth=0.10, peg_ratio=1.2),
        ratios=FinancialRatios(ticker="TEST", debt_to_equity=0.4, return_on_equity=0.25),
    )
    result = fundamental_domain_score(snapshot)
    assert result is not None
    assert result.domain == "fundamental"
    assert 0 <= result.score <= 100
    # Only a subset of sub-factors had data; confidence should reflect that.
    assert result.confidence < 100


def test_macro_domain_score_none_without_observations():
    assert macro_domain_score([], "call") is None


def test_macro_domain_score_bias_alignment():
    observations = [
        MacroObservation(
            metric_id="policy_rate", label="Fed funds", value=4.0, unit="percent",
            as_of=date(2026, 1, 1), source="fake", yoy_change_pct=-2.0,  # falling rates
        ),
    ]
    call_score = macro_domain_score(observations, "call")
    put_score = macro_domain_score(observations, "put")
    assert call_score is not None and put_score is not None
    assert call_score.confidence <= 80  # macro is capped
    # Falling rates read bullish for a call, bearish for a put.
    assert call_score.score > put_score.score


def test_sentiment_domain_score_none_without_data():
    assert sentiment_domain_score([], None, "call") is None


def test_sentiment_domain_score_present_with_articles_only():
    articles = [
        NewsArticle(
            headline="x", source="fake", url="https://example.com",
            published_at="2026-01-01T00:00:00Z", summary="",
        )
    ]
    result = sentiment_domain_score(articles, None, "call")
    assert result is not None
    assert result.domain == "sentiment"


def test_sentiment_domain_score_insider_activity_bias_alignment():
    snapshot = FundamentalsSnapshot(
        ticker="TEST",
        insider_activity=InsiderActivity(
            ticker="TEST", net_shares=50_000.0,
            transactions=[InsiderTransaction(name="x", transaction_type="buy", shares=50_000.0)],
        ),
        estimates=AnalystEstimates(ticker="TEST", consensus_rating="N/A"),
        earnings_calendar=EarningsCalendar(ticker="TEST", next_date=date.today() + timedelta(days=60)),
    )
    call_score = sentiment_domain_score([], snapshot, "call")
    put_score = sentiment_domain_score([], snapshot, "put")
    assert call_score is not None and put_score is not None
    # Net insider buying reads bullish for a call, bearish for a put.
    assert call_score.score > put_score.score


def test_relative_strength_domain_score_none_without_symbol_history_length(sample_option_chain, uptrend_history):
    from agentic_options_reporter.models.schemas import Bar, PriceHistory

    short_history = PriceHistory(symbol="TEST", bars=uptrend_history.bars[:5])
    assert relative_strength_domain_score(short_history, None, None, "call") is None


def test_relative_strength_domain_score_bias_alignment(uptrend_history, downtrend_history):
    # Symbol outperforms a flat/declining benchmark -> bullish for a call.
    call_score = relative_strength_domain_score(uptrend_history, downtrend_history, None, "call")
    put_score = relative_strength_domain_score(uptrend_history, downtrend_history, None, "put")
    assert call_score is not None and put_score is not None
    assert call_score.score > put_score.score
    assert call_score.score + put_score.score == 100  # exact bias inversion, single sub-factor present
