import asyncio
from datetime import date, datetime, timezone

import httpx
import pytest

from agentic_options_reporter.data.async_http import AsyncHttpProviderBase
from agentic_options_reporter.data.macro import (
    DEFAULT_MACRO_METRICS,
    MACRO_METRICS,
    BeaMacroProvider,
    BlsMacroProvider,
    FredMacroProvider,
    ImfMacroProvider,
    MacroProvider,
    MacroProviderError,
    MacroProviderRateLimited,
    MacroProviderRouter,
    MacroProviderUnsupported,
    WorldBankMacroProvider,
    build_macro_provider,
)
from agentic_options_reporter.models.schemas import MacroObservation


@pytest.fixture(autouse=True)
def _reset_provider_cache():
    AsyncHttpProviderBase.clear_shared_cache()
    yield
    AsyncHttpProviderBase.clear_shared_cache()


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch):
    for var in ("FRED_API_KEY", "BLS_API_KEY", "BEA_API_KEY", "AOR_MACRO_PROVIDER_FALLBACK_ORDER"):
        monkeypatch.delenv(var, raising=False)
    for metric_id in MACRO_METRICS:
        monkeypatch.delenv(f"AOR_MACRO_PRIORITY_{metric_id.upper()}", raising=False)


class RecordingTransport:
    def __init__(self, *responses):
        self._responses = list(responses)
        self.requests: list[httpx.Request] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if not self._responses:
            raise AssertionError("No more fake HTTP responses queued")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        status_code, payload = item
        return httpx.Response(status_code, json=payload)


def _client(transport: RecordingTransport) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(transport))


# -- capability declarations --


def test_default_metrics_are_all_registered():
    assert set(DEFAULT_MACRO_METRICS) <= set(MACRO_METRICS)


@pytest.mark.parametrize(
    "provider_cls,expected",
    [
        (FredMacroProvider, {"policy_rate", "treasury_10y", "treasury_2y", "cpi", "gdp"}),
        (BlsMacroProvider, {"cpi"}),
        (BeaMacroProvider, {"gdp"}),
        (ImfMacroProvider, {"cpi", "gdp"}),
        (WorldBankMacroProvider, {"cpi", "gdp"}),
    ],
)
def test_provider_declares_supported_metrics(provider_cls, expected):
    provider = provider_cls(api_key="test-key") if provider_cls.API_KEY_ENV_VAR else provider_cls()
    assert provider.supported_metrics == frozenset(expected)
    assert provider.supports("cpi") == ("cpi" in expected)
    assert not provider.supports("policy_rate") or provider_cls is FredMacroProvider


def test_fetch_unadvertised_metric_raises_unsupported():
    provider = WorldBankMacroProvider()
    with pytest.raises(MacroProviderUnsupported):
        asyncio.run(provider.fetch("policy_rate"))


@pytest.mark.parametrize("provider_cls", [FredMacroProvider, BlsMacroProvider, BeaMacroProvider])
def test_keyed_provider_requires_api_key(provider_cls):
    with pytest.raises(MacroProviderError):
        provider_cls()


@pytest.mark.parametrize("provider_cls", [ImfMacroProvider, WorldBankMacroProvider])
def test_keyless_provider_needs_no_api_key(provider_cls):
    assert isinstance(provider_cls(), MacroProvider)


# -- FRED --


def _fred_observations(*values_and_dates):
    return {"observations": [{"value": v, "date": d} for v, d in values_and_dates]}


def test_fred_fetch_policy_rate():
    transport = RecordingTransport((200, _fred_observations(("5.25", "2026-06-01"))))
    provider = FredMacroProvider(api_key="test-key", client=_client(transport))

    obs = asyncio.run(provider.fetch("policy_rate"))

    assert isinstance(obs, MacroObservation)
    assert obs.metric_id == "policy_rate"
    assert obs.value == 5.25
    assert obs.unit == "percent"
    assert obs.source == "FRED"
    assert obs.yoy_change_pct is None  # rates carry no YoY
    assert transport.requests[0].url.params["series_id"] == "FEDFUNDS"


def test_fred_fetch_cpi_computes_yoy():
    observations = [("310.0", "2026-06-01")] + [("300.0", "2025-06-01")] * 12
    transport = RecordingTransport((200, _fred_observations(*observations)))
    provider = FredMacroProvider(api_key="test-key", client=_client(transport))

    obs = asyncio.run(provider.fetch("cpi"))

    assert obs.value == 310.0
    assert obs.yoy_change_pct == pytest.approx((310.0 - 300.0) / 300.0 * 100)
    assert transport.requests[0].url.params["series_id"] == "CPIAUCSL"


def test_fred_drops_missing_value_marker():
    transport = RecordingTransport(
        (200, _fred_observations((".", "2026-06-01"), ("4.30", "2026-05-01")))
    )
    provider = FredMacroProvider(api_key="test-key", client=_client(transport))

    obs = asyncio.run(provider.fetch("treasury_10y"))

    assert obs.value == 4.30


def test_fred_raises_when_no_observations():
    transport = RecordingTransport((200, _fred_observations()))
    provider = FredMacroProvider(api_key="test-key", client=_client(transport))
    with pytest.raises(MacroProviderError):
        asyncio.run(provider.fetch("gdp"))


# -- BLS --


def _bls_series(*values_and_periods):
    return {
        "status": "REQUEST_SUCCEEDED",
        "Results": {
            "series": [
                {
                    "seriesID": "CUUR0000SA0",
                    "data": [
                        {"year": year, "period": period, "value": value}
                        for value, year, period in values_and_periods
                    ],
                }
            ]
        },
    }


def test_bls_fetch_cpi_computes_yoy():
    entries = [("310.0", "2026", "M06")] + [("300.0", "2025", f"M{i:02d}") for i in range(12, 0, -1)]
    transport = RecordingTransport((200, _bls_series(*entries)))
    provider = BlsMacroProvider(api_key="test-key", client=_client(transport))

    obs = asyncio.run(provider.fetch("cpi"))

    assert obs.value == 310.0
    assert obs.source == "BLS"
    assert obs.yoy_change_pct == pytest.approx((310.0 - 300.0) / 300.0 * 100)
    assert obs.as_of == date(2026, 6, 1)


def test_bls_bad_status_raises():
    transport = RecordingTransport((200, {"status": "REQUEST_NOT_PROCESSED", "message": ["bad key"]}))
    provider = BlsMacroProvider(api_key="test-key", client=_client(transport))
    with pytest.raises(MacroProviderError):
        asyncio.run(provider.fetch("cpi"))


# -- BEA --


def _bea_data(*rows):
    return {
        "BEAAPI": {
            "Results": {
                "Data": [
                    {"TimePeriod": period, "LineNumber": "1", "DataValue": value}
                    for value, period in rows
                ]
            }
        }
    }


def test_bea_fetch_gdp_computes_yoy():
    transport = RecordingTransport(
        (200, _bea_data(
            ("23,000.0", "2026Q2"), ("22,500.0", "2026Q1"), ("22,300.0", "2025Q4"),
            ("22,100.0", "2025Q3"), ("22,000.0", "2025Q2"),
        ))
    )
    provider = BeaMacroProvider(api_key="test-key", client=_client(transport))

    obs = asyncio.run(provider.fetch("gdp"))

    assert obs.value == 23000.0
    assert obs.yoy_change_pct == pytest.approx((23000.0 - 22000.0) / 22000.0 * 100)
    assert obs.as_of == date(2026, 6, 1)


# -- IMF --


def _imf_series(*values_and_periods):
    return {
        "CompactData": {
            "DataSet": {
                "Series": {
                    "Obs": [
                        {"@TIME_PERIOD": period, "@OBS_VALUE": value}
                        for value, period in values_and_periods
                    ]
                }
            }
        }
    }


def test_imf_fetch_cpi_computes_yoy():
    entries = [("310.0", "2026-06")] + [("300.0", f"2025-{m:02d}") for m in range(12, 0, -1)]
    transport = RecordingTransport((200, _imf_series(*entries)))
    provider = ImfMacroProvider(client=_client(transport))

    obs = asyncio.run(provider.fetch("cpi"))

    assert obs.value == 310.0
    assert obs.yoy_change_pct == pytest.approx((310.0 - 300.0) / 300.0 * 100)
    assert obs.as_of == date(2026, 6, 1)


def test_imf_fetch_gdp_parses_quarterly():
    entries = [("23000.0", "2026-Q2"), ("22500.0", "2026-Q1"), ("22300.0", "2025-Q4"),
               ("22100.0", "2025-Q3"), ("22000.0", "2025-Q2")]
    transport = RecordingTransport((200, _imf_series(*entries)))
    provider = ImfMacroProvider(client=_client(transport))

    obs = asyncio.run(provider.fetch("gdp"))

    assert obs.value == 23000.0
    assert obs.as_of == date(2026, 6, 1)


def test_imf_handles_single_observation_bare_dict():
    payload = {"CompactData": {"DataSet": {"Series": {"Obs": {"@TIME_PERIOD": "2026-06", "@OBS_VALUE": "310.0"}}}}}
    transport = RecordingTransport((200, payload))
    provider = ImfMacroProvider(client=_client(transport))

    obs = asyncio.run(provider.fetch("cpi"))

    assert obs.value == 310.0
    assert obs.yoy_change_pct is None


# -- World Bank --


def _worldbank_rows(*values_and_years):
    return [{"page": 1}, [{"date": year, "value": value} for value, year in values_and_years]]


def test_worldbank_fetch_gdp_computes_yoy():
    transport = RecordingTransport(
        (200, _worldbank_rows((28e12, "2025"), (27e12, "2024")))
    )
    provider = WorldBankMacroProvider(client=_client(transport))

    obs = asyncio.run(provider.fetch("gdp"))

    assert obs.value == 28e12
    assert obs.yoy_change_pct == pytest.approx((28e12 - 27e12) / 27e12 * 100)
    assert obs.as_of == date(2025, 12, 31)


def test_worldbank_skips_unpublished_null_years():
    transport = RecordingTransport(
        (200, _worldbank_rows((None, "2026"), (310.0, "2025"), (300.0, "2024")))
    )
    provider = WorldBankMacroProvider(client=_client(transport))

    obs = asyncio.run(provider.fetch("cpi"))

    assert obs.value == 310.0
    assert obs.as_of == date(2025, 12, 31)


def test_worldbank_raises_when_no_observations():
    transport = RecordingTransport((200, _worldbank_rows()))
    provider = WorldBankMacroProvider(client=_client(transport))
    with pytest.raises(MacroProviderError):
        asyncio.run(provider.fetch("gdp"))


# -- shared behavior --


def test_http_429_raises_rate_limited():
    transport = RecordingTransport((429, {}))
    provider = WorldBankMacroProvider(client=_client(transport))
    with pytest.raises(MacroProviderRateLimited):
        asyncio.run(provider.fetch("gdp"))


def test_identical_requests_served_from_cache_across_instances():
    transport = RecordingTransport((200, _worldbank_rows((28e12, "2025"))))

    first = WorldBankMacroProvider(client=_client(transport))
    asyncio.run(first.fetch("gdp"))
    second = WorldBankMacroProvider(client=_client(transport))
    asyncio.run(second.fetch("gdp"))

    assert len(transport.requests) == 1


def test_health_probes_a_supported_metric_and_reports_unhealthy():
    transport = RecordingTransport((503, {}))
    provider = WorldBankMacroProvider(client=_client(transport))

    health = asyncio.run(provider.health())

    assert health.healthy is False
    assert health.provider == "World Bank"


# -- capability-filtering router --


class _StubMacroProvider(MacroProvider):
    def __init__(self, metrics, observations=None, error=None, name="stub"):
        self._metrics = frozenset(metrics)
        self._observations = observations or {}
        self._error = error
        self._name = name
        self.fetched: list[str] = []

    @property
    def supported_metrics(self):
        return self._metrics

    async def fetch(self, metric_id):
        self.fetched.append(metric_id)
        if self._error is not None:
            raise self._error
        return self._observations[metric_id]

    async def health(self):
        from agentic_options_reporter.data.macro import ProviderHealth

        return ProviderHealth(
            provider=self._name, healthy=self._error is None,
            detail="" if self._error is None else str(self._error),
            checked_at=datetime.now(timezone.utc),
        )


def _obs(metric_id, source):
    return MacroObservation(
        metric_id=metric_id, label=metric_id, value=1.0, unit="percent",
        as_of=date(2026, 6, 1), source=source,
    )


def test_router_rejects_empty_client_list():
    with pytest.raises(MacroProviderError):
        MacroProviderRouter([])


def test_router_only_queries_providers_that_advertise_the_metric():
    """The core fix: World Bank is never asked for policy_rate."""
    fred = _StubMacroProvider({"policy_rate", "cpi"}, {"policy_rate": _obs("policy_rate", "FRED")}, name="fred")
    worldbank = _StubMacroProvider({"cpi", "gdp"}, name="worldbank")
    router = MacroProviderRouter([("worldbank", worldbank), ("fred", fred)])

    obs = asyncio.run(router.fetch("policy_rate"))

    assert obs.source == "FRED"
    assert worldbank.fetched == []  # never asked — it doesn't advertise policy_rate


def test_router_raises_unsupported_when_no_provider_serves_metric():
    worldbank = _StubMacroProvider({"cpi", "gdp"})
    router = MacroProviderRouter([("worldbank", worldbank)])

    with pytest.raises(MacroProviderUnsupported):
        asyncio.run(router.fetch("policy_rate"))


def test_router_falls_over_among_supporting_providers():
    first = _StubMacroProvider({"cpi"}, error=MacroProviderRateLimited("429"), name="first")
    second = _StubMacroProvider({"cpi"}, {"cpi": _obs("cpi", "second")}, name="second")
    router = MacroProviderRouter([("first", first), ("second", second)])

    obs = asyncio.run(router.fetch("cpi"))

    assert obs.source == "second"


def test_router_supported_metrics_is_union():
    fred = _StubMacroProvider({"policy_rate", "cpi", "gdp"})
    bls = _StubMacroProvider({"cpi"})
    router = MacroProviderRouter([("fred", fred), ("bls", bls)])
    assert router.supported_metrics == frozenset({"policy_rate", "cpi", "gdp"})


def test_router_applies_per_metric_priority_override(monkeypatch):
    monkeypatch.setenv("AOR_MACRO_PRIORITY_GDP", "worldbank,fred")
    fred = _StubMacroProvider({"gdp"}, {"gdp": _obs("gdp", "FRED")}, name="fred")
    worldbank = _StubMacroProvider({"gdp"}, {"gdp": _obs("gdp", "World Bank")}, name="worldbank")
    # Global order puts fred first, but the override prefers worldbank for GDP.
    router = MacroProviderRouter([("fred", fred), ("worldbank", worldbank)])

    obs = asyncio.run(router.fetch("gdp"))

    assert obs.source == "World Bank"


def test_router_health_aggregates():
    up = _StubMacroProvider({"cpi"}, name="up")
    down = _StubMacroProvider({"gdp"}, error=MacroProviderError("down"), name="down")
    router = MacroProviderRouter([("up", up), ("down", down)])

    health = asyncio.run(router.health())

    assert health.healthy is True
    assert "up: ok" in health.detail


# -- build_macro_provider --


def test_build_macro_provider_always_includes_keyless_sources():
    provider = build_macro_provider()
    assert provider.provider_names == ["imf", "worldbank"]
    # keyless deployment serves cpi/gdp but not rates — which are simply
    # never requested rather than erroring.
    assert provider.supported_metrics == frozenset({"cpi", "gdp"})
    assert provider.supports("policy_rate") is False


def test_build_macro_provider_orders_configured_providers(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    monkeypatch.setenv("BLS_API_KEY", "test-key")
    provider = build_macro_provider()
    assert provider.provider_names == ["fred", "bls", "imf", "worldbank"]


def test_build_macro_provider_respects_fallback_order_env_var(monkeypatch):
    monkeypatch.setenv("FRED_API_KEY", "test-key")
    monkeypatch.setenv("AOR_MACRO_PROVIDER_FALLBACK_ORDER", "worldbank,fred")
    provider = build_macro_provider()
    assert provider.provider_names == ["worldbank", "fred"]
