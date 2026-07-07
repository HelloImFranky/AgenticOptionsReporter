from agentic_options_reporter.frontend.formatting import (
    CANDIDATE_COLUMNS,
    RUN_COLUMNS,
    TONE_DANGER,
    TONE_NEUTRAL,
    TONE_SUCCESS,
    TONE_WARNING,
    candidates_to_rows,
    consensus_tone,
    earnings_surprise_facts,
    format_indicator_summary,
    format_money,
    format_next_earnings,
    format_num,
    format_pct,
    format_recommendation,
    format_timestamp,
    format_trend_summary,
    format_volume_summary,
    fundamentals_metric_facts,
    insider_activity_header,
    insider_activity_series,
    macro_regime_tone,
    domain_badges,
    domain_id_for_label,
    domain_score_items,
    missing_domain_labels,
    recommendation_facts,
    recommendation_tone,
    recommended_candidate,
    relative_strength_leadership,
    relative_strength_performance_label,
    relative_strength_performance_tone,
    risk_level_tone,
    runs_to_rows,
    score_severity_label,
    score_severity_tone,
    statistical_edge_confidence_label,
    statistical_edge_confidence_tone,
    technical_snapshot_facts,
    trade_quality_agreement_summary,
    trade_quality_summary,
    trade_quality_tone,
    trend_tone,
)


def test_format_recommendation_with_contract():
    text = format_recommendation(
        {
            "action": "BUY",
            "contract_symbol": "AAPL260116C00150000",
            "confidence": 0.732,
            "rationale": "strong trend alignment",
        }
    )
    assert "AAPL260116C00150000" in text
    assert "strong trend alignment" in text


def test_format_recommendation_without_contract():
    text = format_recommendation(
        {"action": "AVOID", "contract_symbol": None, "confidence": 0.0, "rationale": "no candidates"}
    )
    assert "—" in text
    assert "no candidates" in text


def test_format_recommendation_without_rationale():
    text = format_recommendation({"action": "AVOID", "contract_symbol": None, "rationale": ""})
    assert text == "—"


_CANDIDATE = {
    "contract_symbol": "AAPL260116C00150000",
    "option_type": "call",
    "strike": 150.0,
    "expiration": "2026-01-16",
    "score": 82.4,
    "delta": 0.612,
    "breakeven": 152.3,
    "max_loss": 210.0,
    "max_gain": None,
    "probability_of_profit": 0.58,
}


def test_recommended_candidate_matches_by_contract_symbol():
    rec = {"contract_symbol": "AAPL260116C00150000"}
    assert recommended_candidate(rec, [_CANDIDATE]) is _CANDIDATE


def test_recommended_candidate_none_for_avoid():
    assert recommended_candidate({"contract_symbol": None}, [_CANDIDATE]) is None
    assert recommended_candidate({"contract_symbol": "MISSING"}, [_CANDIDATE]) is None


def test_recommendation_facts_pulls_candidate_metrics():
    rec = {"action": "BUY", "contract_symbol": "AAPL260116C00150000", "confidence": 0.73}
    facts = dict(recommendation_facts(rec, [_CANDIDATE]))
    assert facts["Contract"] == "AAPL260116C00150000"
    assert facts["Type"] == "CALL"
    assert facts["Strike"] == "150.00"
    assert facts["Expiration"] == "2026-01-16"
    assert facts["Score"] == "82.4"
    assert facts["Breakeven"] == "152.30"
    assert facts["Max gain"] == "unlimited"   # None -> unlimited
    assert facts["PoP"] == "58%"


def test_recommendation_facts_avoid_shows_only_contract_dash():
    facts = recommendation_facts({"action": "AVOID", "contract_symbol": None}, [])
    assert facts == [("Contract", "—")]


def _domain_score(score: float, confidence: float = 90.0, evidence=None) -> dict:
    return {"score": score, "confidence": confidence, "evidence": evidence or []}


def test_domain_score_items_formats_labels_in_canonical_order():
    items = domain_score_items(
        {
            "liquidity": _domain_score(74.0, evidence=["Spread tight"]),
            "technical": _domain_score(91.0, evidence=["Strong uptrend"]),
        }
    )
    assert items == [
        ("Technical", 91.0, 90.0, ["Strong uptrend"]),
        ("Liquidity", 74.0, 90.0, ["Spread tight"]),
    ]


def test_domain_score_items_empty_for_no_domains():
    assert domain_score_items({}) == []
    assert domain_score_items(None) == []


def test_missing_domain_labels_lists_absent_domains():
    missing = missing_domain_labels({"technical": _domain_score(80.0)})
    assert "Technical" not in missing
    assert "Macro" in missing
    assert "Statistical Edge" in missing


def test_trade_quality_tone_thresholds():
    assert trade_quality_tone(75) == TONE_SUCCESS
    assert trade_quality_tone(50) == TONE_WARNING
    assert trade_quality_tone(10) == TONE_DANGER


def test_score_severity_label_and_tone_mapping():
    assert score_severity_label(82) == "Excellent"
    assert score_severity_label(64) == "Strong"
    assert score_severity_label(46) == "Balanced"
    assert score_severity_label(24) == "Cautious"
    assert score_severity_label(8) == "Weak"
    assert score_severity_tone(82) == TONE_SUCCESS
    assert score_severity_tone(46) == TONE_WARNING
    assert score_severity_tone(8) == TONE_DANGER


def test_trade_quality_summary_uses_explainability():
    summary = trade_quality_summary(
        {"explainability": ["Technical (weight 30%): 91/100 — strongest contributor", "Macro: 40/100"]}
    )
    assert "Technical" in summary


def test_trade_quality_summary_empty_is_blank():
    assert trade_quality_summary({}) == ""
    assert trade_quality_summary(None) == ""


def test_trade_quality_agreement_summary_names_diverging_domain():
    quant = {
        "composite_score": 72.0,
        "domain_scores": {"technical": _domain_score(90.0), "risk": _domain_score(60.0)},
    }
    agent = {
        "composite_score": 48.0,
        "domain_scores": {"technical": _domain_score(88.0), "risk": _domain_score(20.0)},
    }
    summary = trade_quality_agreement_summary(quant, agent)
    assert "Risk" in summary
    assert "diverge" in summary


def test_trade_quality_agreement_summary_broadly_aligned():
    quant = {"composite_score": 70.0, "domain_scores": {"technical": _domain_score(70.0)}}
    agent = {"composite_score": 72.0, "domain_scores": {"technical": _domain_score(72.0)}}
    assert "broadly aligned" in trade_quality_agreement_summary(quant, agent)


def test_trade_quality_agreement_summary_empty_without_both_sources():
    assert trade_quality_agreement_summary(None, {"composite_score": 1}) == ""
    assert trade_quality_agreement_summary({"composite_score": 1}, None) == ""


def test_recommendation_facts_omits_absent_candidate_fields():
    partial = {"contract_symbol": "X", "option_type": "put", "strike": 10.0}
    facts = dict(recommendation_facts({"contract_symbol": "X"}, [partial]))
    assert facts["Type"] == "PUT"
    assert facts["Strike"] == "10.00"
    assert "Breakeven" not in facts   # not in payload -> omitted, not shown as 0
    assert "PoP" not in facts


def test_technical_snapshot_facts_labels_and_values():
    facts = dict(
        technical_snapshot_facts(
            {"direction": "bullish", "strength": "strong", "adx": 31.2},
            {"relative_volume": 1.8, "flags": ["above_average"]},
            {"sma_20": 195.1, "sma_50": 188.4, "rsi_14": 61.2, "atr_14": 3.4},
        )
    )
    assert facts["Trend"] == "Bullish · strong"
    assert facts["ADX"] == "31.2"
    assert facts["Rel. volume"] == "1.80x avg"
    assert facts["Volume flags"] == "above_average"
    assert facts["SMA 20"] == "195.10"
    assert facts["RSI 14"] == "61.2"


def test_technical_snapshot_facts_empty_when_no_inputs():
    assert technical_snapshot_facts(None, None, None) == []


def test_recommendation_tone_mapping():
    assert recommendation_tone("STRONG_BUY") == TONE_SUCCESS
    assert recommendation_tone("BUY") == TONE_SUCCESS
    assert recommendation_tone("HOLD") == TONE_WARNING
    assert recommendation_tone("AVOID") == TONE_DANGER
    assert recommendation_tone("SOMETHING_ELSE") == TONE_NEUTRAL


def test_trend_tone_mapping():
    assert trend_tone("bullish") == TONE_SUCCESS
    assert trend_tone("bearish") == TONE_DANGER
    assert trend_tone("neutral") == TONE_NEUTRAL
    assert trend_tone("unexpected") == TONE_NEUTRAL


def test_format_timestamp_trims_seconds_and_replaces_t():
    assert format_timestamp("2026-07-03T12:34:56.789") == "2026-07-03 12:34"


def test_consensus_tone_mapping():
    assert consensus_tone("bullish") == TONE_SUCCESS
    assert consensus_tone("bearish") == TONE_DANGER
    assert consensus_tone("neutral") == TONE_NEUTRAL
    assert consensus_tone("mixed") == TONE_WARNING
    assert consensus_tone("unexpected") == TONE_NEUTRAL


def test_risk_level_tone_mapping():
    assert risk_level_tone("low") == TONE_SUCCESS
    assert risk_level_tone("medium") == TONE_WARNING
    assert risk_level_tone("high") == TONE_DANGER
    assert risk_level_tone("unexpected") == TONE_NEUTRAL


def test_macro_regime_tone_mapping():
    assert macro_regime_tone("risk_on") == TONE_SUCCESS
    assert macro_regime_tone("risk_off") == TONE_DANGER
    assert macro_regime_tone("neutral") == TONE_NEUTRAL
    assert macro_regime_tone("unexpected") == TONE_NEUTRAL


def test_format_trend_summary():
    text = format_trend_summary({"direction": "bullish", "strength": "strong", "adx": 42.567})
    assert "Bullish" in text
    assert "strong" in text
    assert "42.6" in text


def test_format_volume_summary_with_flags():
    text = format_volume_summary({"relative_volume": 1.8, "flags": ["high_volume"]})
    assert "1.80x" in text
    assert "high_volume" in text


def test_format_volume_summary_no_flags():
    text = format_volume_summary({"relative_volume": 1.0, "flags": []})
    assert "none" in text


def test_format_indicator_summary():
    text = format_indicator_summary(
        {"sma_20": 101.234, "sma_50": 98.5, "rsi_14": 55.1, "atr_14": 2.345}
    )
    assert "101.23" in text
    assert "98.50" in text
    assert "55.1" in text
    assert "2.35" in text


def test_candidates_to_rows_shapes_match_columns():
    rows = candidates_to_rows(
        [
            {
                "contract_symbol": "AAPL260116C00150000",
                "option_type": "call",
                "strike": 150.0,
                "expiration": "2026-01-16",
                "score": 78.456,
                "delta": 0.567,
                "breakeven": 152.5,
                "max_loss": 250.0,
                "max_gain": None,
                "probability_of_profit": 0.612,
            }
        ]
    )
    assert len(rows) == 1
    assert len(rows[0]) == len(CANDIDATE_COLUMNS)
    assert rows[0][0] == "AAPL260116C00150000"
    assert rows[0][1] == "CALL"
    assert rows[0][8] == "unlimited"
    assert rows[0][9] == "61%"


def test_candidates_to_rows_with_capped_max_gain():
    rows = candidates_to_rows(
        [
            {
                "contract_symbol": "AAPL260116P00150000",
                "option_type": "put",
                "strike": 150.0,
                "expiration": "2026-01-16",
                "score": 40.0,
                "delta": -0.3,
                "breakeven": 147.5,
                "max_loss": 250.0,
                "max_gain": 14750.0,
                "probability_of_profit": 0.4,
            }
        ]
    )
    assert rows[0][8] == "14750.00"


def test_runs_to_rows_shapes_match_columns():
    rows = runs_to_rows(
        [
            {
                "run_id": 1,
                "symbol": "AAPL",
                "generated_at": "2026-07-03T12:00:00",
                "recommendation_action": "BUY",
                "recommendation_confidence": 0.65,
            }
        ]
    )
    assert len(rows) == 1
    assert len(rows[0]) == len(RUN_COLUMNS)
    assert rows[0] == ["1", "AAPL", "2026-07-03 12:00", "BUY", "65%"]


def test_empty_lists_produce_no_rows():
    assert candidates_to_rows([]) == []
    assert runs_to_rows([]) == []


# -- fundamentals formatting (shared by the Analyze tab card + PDF report) --


def test_format_money_scales_by_magnitude():
    assert format_money(3.0e12) == "$3.00T"
    assert format_money(5.0e8) == "$500.00M"
    assert format_money(1234) == "$1.23K"
    assert format_money(500) == "$500"
    assert format_money(None) == "—"


def test_format_pct_and_num_handle_missing():
    assert format_pct(0.123) == "12.3%"
    assert format_pct(None) == "—"
    assert format_num(30.5) == "30.50"
    assert format_num("nope") == "—"


def test_fundamentals_metric_facts_filters_absent_fields():
    facts = fundamentals_metric_facts({"pe_ratio": 30.5, "beta": None, "market_cap": 3.0e12})
    labels = {label for label, _ in facts}
    assert ("P/E", "30.50") in facts
    assert ("Market cap", "$3.00T") in facts
    assert "Beta" not in labels          # None -> omitted
    assert fundamentals_metric_facts(None) == []


def test_fundamentals_metric_facts_includes_derived_price_ranges():
    facts = dict(fundamentals_metric_facts({
        "week1_high": 160.0, "week1_low": 150.0,
        "month1_high": 165.0, "month1_low": 148.0, "week52_high": 200.0,
    }))
    assert facts["1w high"] == "160.00"
    assert facts["1w low"] == "150.00"
    assert facts["1m high"] == "165.00"
    assert facts["1m low"] == "148.00"
    assert facts["52w high"] == "200.00"


def test_format_next_earnings():
    assert format_next_earnings({"next_date": "2026-08-01", "eps_estimate": 1.6}) == (
        "Next earnings: 2026-08-01  ·  EPS est. 1.60"
    )
    assert format_next_earnings({"next_date": None}) is None
    assert format_next_earnings(None) is None


def test_earnings_surprise_facts_caps_and_formats():
    earnings = {"surprises": [
        {"period": "2026-03-31", "actual_eps": 1.5, "estimate_eps": 1.4, "surprise_percent": 0.071},
        {"period": "2025-12-31", "actual_eps": 2.1, "estimate_eps": 2.2, "surprise_percent": -0.045},
    ]}
    facts = earnings_surprise_facts(earnings, limit=1)
    assert len(facts) == 1
    period, value = facts[0]
    assert period == "2026-03-31"
    assert "1.50 vs 1.40 est" in value and "7.1%" in value


def test_insider_activity_header_describes_net_flow():
    insider = {
        "net_shares": -500.0,
        "transactions": [
            {"name": "Jane Doe", "transaction_type": "sell", "shares": 1000},
            {"name": "John Roe", "transaction_type": "buy", "shares": 500},
        ],
    }
    header = insider_activity_header(insider)
    assert "net selling" in header and "500 shares" in header
    assert insider_activity_header(None) == ""
    assert insider_activity_header({"transactions": []}) == ""


def test_insider_activity_series_aggregates_net_by_date():
    insider = {"transactions": [
        {"name": "A", "transaction_type": "sell", "shares": 1000, "filed_at": "2026-06-01"},
        {"name": "B", "transaction_type": "buy", "shares": 400, "filed_at": "2026-06-01"},   # same date -> netted
        {"name": "C", "transaction_type": "buy", "shares": 500, "filed_at": "2026-05-01"},
        {"name": "D", "transaction_type": "buy", "shares": None, "filed_at": "2026-04-01"},   # no shares -> skip
        {"name": "E", "transaction_type": "sell", "shares": 700},                              # no date -> skip
    ]}
    series = insider_activity_series(insider)
    # Chronological, one point per active date; same-date trades are netted.
    assert [p["date"] for p in series] == ["2026-05-01", "2026-06-01"]
    assert series[0]["net"] == 500 and series[0]["is_buy"] is True
    assert series[1]["net"] == -600 and series[1]["is_buy"] is False   # 400 buy - 1000 sell
    assert insider_activity_series(None) == []
    assert insider_activity_series({"transactions": []}) == []


def test_insider_activity_series_keeps_most_recent_within_limit():
    txns = [
        {"transaction_type": "buy", "shares": 100, "filed_at": f"2026-01-{d:02d}"}
        for d in range(1, 8)
    ]
    series = insider_activity_series({"transactions": txns}, limit=3)
    # Most recent 3 dates, still oldest-first.
    assert [p["date"] for p in series] == ["2026-01-05", "2026-01-06", "2026-01-07"]


def test_relative_strength_performance_label_tiers():
    assert relative_strength_performance_label(95) == "Exceptional"
    assert relative_strength_performance_label(75) == "Very Strong"
    assert relative_strength_performance_label(65) == "Strong"
    assert relative_strength_performance_label(50) == "Neutral"
    assert relative_strength_performance_label(25) == "Weak"
    assert relative_strength_performance_label(5) == "Very Weak"


def test_relative_strength_performance_tone_matches_trade_quality_bands():
    # Tone nests inside trade_quality_tone's 60/40 bands so the same score
    # always reads the same color everywhere in the app.
    assert relative_strength_performance_tone(95) == TONE_SUCCESS
    assert relative_strength_performance_tone(65) == TONE_SUCCESS
    assert relative_strength_performance_tone(50) == TONE_WARNING
    assert relative_strength_performance_tone(25) == TONE_DANGER


def test_relative_strength_leadership_none_without_factors():
    assert relative_strength_leadership([]) is None
    assert relative_strength_leadership(None) is None


def test_relative_strength_leadership_picks_larger_deviation():
    # vs_market strongly positive, vs_sector only mildly so -> Market Leader.
    factors = [
        {"name": "vs_market", "value": 0.9},
        {"name": "vs_sector", "value": 0.55},
    ]
    assert relative_strength_leadership(factors) == ("Market Leader", TONE_SUCCESS)


def test_relative_strength_leadership_laggard_and_sector_only():
    assert relative_strength_leadership([{"name": "vs_market", "value": 0.1}]) == (
        "Market Laggard", TONE_DANGER,
    )
    assert relative_strength_leadership([{"name": "vs_sector", "value": 0.85}]) == (
        "Sector Leader", TONE_SUCCESS,
    )
    assert relative_strength_leadership([{"name": "vs_sector", "value": 0.15}]) == (
        "Sector Laggard", TONE_DANGER,
    )


def test_relative_strength_leadership_neutral_is_peer_average():
    assert relative_strength_leadership([{"name": "vs_market", "value": 0.52}]) == (
        "Peer Average", TONE_NEUTRAL,
    )


def test_statistical_edge_confidence_label_tiers():
    assert statistical_edge_confidence_label(70) == "Very High Confidence"
    assert statistical_edge_confidence_label(55) == "High Confidence"
    assert statistical_edge_confidence_label(40) == "Moderate Confidence"
    assert statistical_edge_confidence_label(25) == "Low Confidence"
    assert statistical_edge_confidence_label(5) == "Insufficient Data"


def test_statistical_edge_confidence_tone_tiers():
    assert statistical_edge_confidence_tone(70) == TONE_SUCCESS
    assert statistical_edge_confidence_tone(40) == TONE_WARNING
    assert statistical_edge_confidence_tone(5) == TONE_DANGER


def test_domain_badges_relative_strength_includes_performance_and_leadership():
    factors = [{"name": "vs_market", "value": 0.9}, {"name": "vs_sector", "value": 0.6}]
    badges = domain_badges("relative_strength", 90.0, 80.0, factors)
    assert badges[0] == ("Exceptional", TONE_SUCCESS)
    assert badges[1] == ("Market Leader", TONE_SUCCESS)


def test_domain_badges_relative_strength_without_factors_is_performance_only():
    badges = domain_badges("relative_strength", 90.0, 80.0, None)
    assert badges == [("Exceptional", TONE_SUCCESS)]


def test_domain_badges_statistical_edge_is_confidence_only():
    badges = domain_badges("statistical_edge", 55.0, 10.0, None)
    assert badges == [("Insufficient Data", TONE_DANGER)]


def test_domain_badges_other_domains_fall_back_to_generic_severity():
    badges = domain_badges("technical", 65.0, 90.0, None)
    assert badges == [(score_severity_label(65.0), score_severity_tone(65.0))]


def test_domain_id_for_label_round_trips_domain_label():
    assert domain_id_for_label("Relative Strength") == "relative_strength"
    assert domain_id_for_label("Statistical Edge") == "statistical_edge"
    assert domain_id_for_label("not a real label") is None
