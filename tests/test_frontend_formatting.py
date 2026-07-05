from agentic_options_reporter.frontend.formatting import (
    CANDIDATE_COLUMNS,
    RUN_COLUMNS,
    TONE_DANGER,
    TONE_NEUTRAL,
    TONE_SUCCESS,
    TONE_WARNING,
    candidates_to_rows,
    consensus_tone,
    format_indicator_summary,
    format_recommendation,
    format_timestamp,
    format_trend_summary,
    format_volume_summary,
    macro_regime_tone,
    recommendation_facts,
    recommendation_tone,
    recommended_candidate,
    risk_level_tone,
    score_breakdown_items,
    runs_to_rows,
    technical_snapshot_facts,
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


def test_score_breakdown_items_formats_factor_names_and_values():
    items = score_breakdown_items(
        {
            "trend_alignment": 1.0,
            "support_resistance_proximity": 0.2,
        }
    )
    assert items == [("Trend Alignment", 1.0), ("Support Resistance Proximity", 0.2)]


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
