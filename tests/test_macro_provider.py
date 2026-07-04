import asyncio
from datetime import date, datetime, timezone

import httpx
import pytest

from agentic_options_reporter.data.async_http import AsyncHttpProviderBase
from agentic_options_reporter.data.macro import (
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
from agentic_options_reporter.models.schemas import CpiSnapshot


@pytest.fixture(autouse=True)
def _reset_provider_cache():
    AsyncHttpProviderBase.clear_shared_cache()
    yield
    AsyncHttpProviderBase.clear_shared_cache()


@pytest.fixture(autouse=True)
def _clear_key_env_vars(monkeypatch):
    for var in ("FRED_API_KEY", "BLS_API_KEY", "BEA_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.delenv("AOR_MACRO_PROVIDER_FALLBACK_ORDER", raising=False)


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


_KEYED_PROVIDERS = [FredMacroProvider, BlsMacroProvider, BeaMacroProvider]


@pytest.mark.parametrize("provider_cls", _KEYED_PROVIDERS)
def test_keyed_provider_requires_api_key(provider_cls):
    with pytest.raises(MacroProviderError):
        provider_cls()


@pytest.mark.parametrize("provider_cls", [ImfMacroProvider, WorldBankMacroProvider])
def test_keyless_provider_needs_no_api_key(provider_cls):
    assert isinstance(provider_cls(), MacroProvider)


# -- FRED --


def _fred_observations(*values_and_dates):
    return {"observations": [{"value": v, "date": d} for v, d in values_and_dates]}


def test_fred_get_interest_rates():
    transport = RecordingTransport(
        (200, _fred_observations(("5.25", "2026-06-01"))),  # FEDFUNDS
        (200, _fred_observations(("4.30", "2026-06-01"))),  # DGS10
        (200, _fred_observations(("4.10", "2026-06-01"))),  # DGS2
    )
    provider = FredMacroProvider(api_key="test-key", client=_client(transport))

    rates = asyncio.run(provider.get_interest_rates())

    assert rates.fed_funds_rate == 5.25
    assert rates.ten_year_yield == 4.30
    assert rates.two_year_yield == 4.10
    # All three series fetched concurrently in one bridge.
    assert len(transport.requests) == 3


def test_fred_get_cpi_computes_yoy_change():
    observations = [("310.0", "2026-06-01")] + [("300.0", "2025-06-01")] * 12
    transport = RecordingTransport((200, _fred_observations(*observations)))
    provider = FredMacroProvider(api_key="test-key", client=_client(transport))

    cpi = asyncio.run(provider.get_cpi())

    assert cpi.value == 310.0
    assert cpi.yoy_change_pct == pytest.approx((310.0 - 300.0) / 300.0 * 100)


def test_fred_drops_missing_value_marker():
    transport = RecordingTransport(
        (200, _fred_observations((".", "2026-06-01"), ("310.0", "2026-05-01")))
    )
    provider = FredMacroProvider(api_key="test-key", client=_client(transport))

    cpi = asyncio.run(provider.get_cpi())

    assert cpi.value == 310.0


def test_fred_get_cpi_raises_when_no_observations():
    transport = RecordingTransport((200, _fred_observations()))
    provider = FredMacroProvider(api_key="test-key", client=_client(transport))
    with pytest.raises(MacroProviderError):
        asyncio.run(provider.get_cpi())


def test_fred_get_macro_calendar_returns_empty_list():
    provider = FredMacroProvider(api_key="test-key")
    assert asyncio.run(provider.get_macro_calendar()) == []


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


def test_bls_get_cpi_computes_yoy_change():
    entries = [("310.0", "2026", "M06")] + [
        ("300.0", "2025", f"M{i:02d}") for i in range(12, 0, -1)
    ]
    transport = RecordingTransport((200, _bls_series(*entries)))
    provider = BlsMacroProvider(api_key="test-key", client=_client(transport))

    cpi = asyncio.run(provider.get_cpi())

    assert cpi.value == 310.0
    assert cpi.yoy_change_pct == pytest.approx((310.0 - 300.0) / 300.0 * 100)
    assert cpi.as_of == date(2026, 6, 1)


def test_bls_bad_status_raises():
    transport = RecordingTransport(
        (200, {"status": "REQUEST_NOT_PROCESSED", "message": ["invalid key"]})
    )
    provider = BlsMacroProvider(api_key="test-key", client=_client(transport))
    with pytest.raises(MacroProviderError):
        asyncio.run(provider.get_cpi())


def test_bls_rates_and_gdp_are_unsupported():
    provider = BlsMacroProvider(api_key="test-key")
    with pytest.raises(MacroProviderUnsupported):
        asyncio.run(provider.get_interest_rates())
    with pytest.raises(MacroProviderUnsupported):
        asyncio.run(provider.get_gdp())


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


def test_bea_get_gdp_computes_yoy_growth():
    transport = RecordingTransport(
        (200, _bea_data(
            ("23,000.0", "2026Q2"),
            ("22,500.0", "2026Q1"),
            ("22,300.0", "2025Q4"),
            ("22,100.0", "2025Q3"),
            ("22,000.0", "2025Q2"),
        ))
    )
    provider = BeaMacroProvider(api_key="test-key", client=_client(transport))

    gdp = asyncio.run(provider.get_gdp())

    assert gdp.value == 23000.0
    assert gdp.yoy_growth_pct == pytest.approx((23000.0 - 22000.0) / 22000.0 * 100)
    assert gdp.as_of == date(2026, 6, 1)


def test_bea_cpi_and_rates_are_unsupported():
    provider = BeaMacroProvider(api_key="test-key")
    with pytest.raises(MacroProviderUnsupported):
        asyncio.run(provider.get_cpi())
    with pytest.raises(MacroProviderUnsupported):
        asyncio.run(provider.get_interest_rates())


# -- IMF --


def _imf_series(*values_and_periods):
    observations = [
        {"@TIME_PERIOD": period, "@OBS_VALUE": value} for value, period in values_and_periods
    ]
    return {"CompactData": {"DataSet": {"Series": {"Obs": observations}}}}


def test_imf_get_cpi_computes_yoy_change():
    entries = [("310.0", "2026-06")] + [(f"300.0", f"2025-{m:02d}") for m in range(12, 0, -1)]
    transport = RecordingTransport((200, _imf_series(*entries)))
    provider = ImfMacroProvider(client=_client(transport))

    cpi = asyncio.run(provider.get_cpi())

    assert cpi.value == 310.0
    assert cpi.yoy_change_pct == pytest.approx((310.0 - 300.0) / 300.0 * 100)
    assert cpi.as_of == date(2026, 6, 1)


def test_imf_get_gdp_parses_quarterly_periods():
    entries = [
        ("23000.0", "2026-Q2"),
        ("22500.0", "2026-Q1"),
        ("22300.0", "2025-Q4"),
        ("22100.0", "2025-Q3"),
        ("22000.0", "2025-Q2"),
    ]
    transport = RecordingTransport((200, _imf_series(*entries)))
    provider = ImfMacroProvider(client=_client(transport))

    gdp = asyncio.run(provider.get_gdp())

    assert gdp.value == 23000.0
    assert gdp.yoy_growth_pct == pytest.approx((23000.0 - 22000.0) / 22000.0 * 100)
    assert gdp.as_of == date(2026, 6, 1)


def test_imf_handles_single_observation_returned_as_bare_dict():
    payload = {"CompactData": {"DataSet": {"Series": {"Obs": {"@TIME_PERIOD": "2026-06", "@OBS_VALUE": "310.0"}}}}}
    transport = RecordingTransport((200, payload))
    provider = ImfMacroProvider(client=_client(transport))

    cpi = asyncio.run(provider.get_cpi())

    assert cpi.value == 310.0
    assert cpi.yoy_change_pct is None


def test_imf_rates_are_unsupported():
    provider = ImfMacroProvider()
    with pytest.raises(MacroProviderUnsupported):
        asyncio.run(provider.get_interest_rates())


# -- World Bank --


def _worldbank_rows(*values_and_years):
    return [
        {"page": 1},
        [{"date": year, "value": value} for value, year in values_and_years],
    ]


def test_worldbank_get_gdp_computes_yoy_growth():
    transport = RecordingTransport(
        (200, _worldbank_rows((28_000_000_000_000, "2025"), (27_000_000_000_000, "2024")))
    )
    provider = WorldBankMacroProvider(client=_client(transport))

    gdp = asyncio.run(provider.get_gdp())

    assert gdp.value == 28_000_000_000_000
    assert gdp.yoy_growth_pct == pytest.approx(
        (28_000_000_000_000 - 27_000_000_000_000) / 27_000_000_000_000 * 100
    )
    assert gdp.as_of == date(2025, 12, 31)


def test_worldbank_skips_unpublished_null_years():
    transport = RecordingTransport(
        (200, _worldbank_rows((None, "2026"), (310.0, "2025"), (300.0, "2024")))
    )
    provider = WorldBankMacroProvider(client=_client(transport))

    cpi = asyncio.run(provider.get_cpi())

    assert cpi.value == 310.0
    assert cpi.as_of == date(2025, 12, 31)


def test_worldbank_raises_when_no_observations():
    transport = RecordingTransport((200, _worldbank_rows()))
    provider = WorldBankMacroProvider(client=_client(transport))
    with pytest.raises(MacroProviderError):
        asyncio.run(provider.get_gdp())


def test_worldbank_rates_are_unsupported():
    provider = WorldBankMacroProvider()
    with pytest.raises(MacroProviderUnsupported):
        asyncio.run(provider.get_interest_rates())


# -- Shared adapter behavior --


def test_http_429_raises_rate_limited():
    transport = RecordingTransport((429, {}))
    provider = WorldBankMacroProvider(client=_client(transport))
    with pytest.raises(MacroProviderRateLimited):
        asyncio.run(provider.get_gdp())


def test_identical_requests_are_served_from_cache_across_instances():
    transport = RecordingTransport(
        (200, _worldbank_rows((28_000_000_000_000, "2025")))
    )

    first = WorldBankMacroProvider(client=_client(transport))
    asyncio.run(first.get_gdp())
    second = WorldBankMacroProvider(client=_client(transport))
    asyncio.run(second.get_gdp())

    assert len(transport.requests) == 1


def test_health_reports_unhealthy_instead_of_raising():
    transport = RecordingTransport((503, {}))
    provider = WorldBankMacroProvider(client=_client(transport))

    health = asyncio.run(provider.health())

    assert health.healthy is False
    assert health.provider == "World Bank"


# -- MacroProviderRouter --


class _StubMacroProvider(MacroProvider):
    def __init__(self, cpi=None, error=None, name="stub"):
        self._cpi = cpi
        self._error = error
        self._name = name

    async def get_interest_rates(self):
        raise NotImplementedError

    async def get_cpi(self):
        if self._error is not None:
            raise self._error
        return self._cpi

    async def get_gdp(self):
        raise NotImplementedError

    async def get_macro_calendar(self):
        return []

    async def health(self):
        from agentic_options_reporter.data.macro import ProviderHealth

        return ProviderHealth(
            provider=self._name,
            healthy=self._error is None,
            detail="" if self._error is None else str(self._error),
            checked_at=datetime.now(timezone.utc),
        )


def test_router_rejects_empty_client_list():
    with pytest.raises(MacroProviderError):
        MacroProviderRouter([])


def test_router_falls_through_when_specialist_lacks_series():
    unsupported = _StubMacroProvider(error=MacroProviderUnsupported("no CPI"))
    supported = _StubMacroProvider(
        cpi=CpiSnapshot(value=310.0, yoy_change_pct=3.3, as_of=date(2026, 6, 1))
    )
    router = MacroProviderRouter([("first", unsupported), ("second", supported)])

    cpi = asyncio.run(router.get_cpi())

    assert cpi.value == 310.0


def test_router_health_aggregates():
    healthy = _StubMacroProvider(name="up")
    unhealthy = _StubMacroProvider(error=MacroProviderUnsupported("down"), name="down")
    router = MacroProviderRouter([("up", healthy), ("down", unhealthy)])

    health = asyncio.run(router.health())

    assert health.healthy is True
    assert "up: ok" in health.detail


# -- build_macro_provider --


def test_build_macro_provider_always_includes_keyless_sources():
    provider = build_macro_provider()
    assert provider.provider_names == ["imf", "worldbank"]


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