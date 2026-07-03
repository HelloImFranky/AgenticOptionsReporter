import pytest

from agentic_options_reporter.data.provider_errors import (
    ProviderRateLimited,
    ProviderTimeout,
    ProviderUnavailable,
    ProviderUnsupported,
)
from agentic_options_reporter.data.provider_router import call_with_fallback, classify_requests_error


class _AllFailedError(RuntimeError):
    pass


class _FakeClient:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error
        self.calls = 0

    def fetch(self, *args, **kwargs):
        self.calls += 1
        if self.error is not None:
            raise self.error
        return self.response


def test_call_with_fallback_returns_first_success():
    first = _FakeClient(response="from first")
    second = _FakeClient(response="from second")

    result = call_with_fallback([("first", first), ("second", second)], "fetch", _AllFailedError)

    assert result == "from first"
    assert second.calls == 0


def test_call_with_fallback_advances_on_retryable_error():
    first = _FakeClient(error=ProviderRateLimited("rate limited"))
    second = _FakeClient(response="from second")

    result = call_with_fallback([("first", first), ("second", second)], "fetch", _AllFailedError)

    assert result == "from second"
    assert second.calls == 1


def test_call_with_fallback_does_not_catch_non_retryable_error():
    first = _FakeClient(error=ValueError("not retryable"))
    second = _FakeClient(response="from second")

    with pytest.raises(ValueError):
        call_with_fallback([("first", first), ("second", second)], "fetch", _AllFailedError)
    assert second.calls == 0


def test_call_with_fallback_raises_all_failed_error_with_details_when_every_client_fails():
    first = _FakeClient(error=ProviderTimeout("timed out"))
    second = _FakeClient(error=ProviderUnavailable("down"))

    with pytest.raises(_AllFailedError, match="first:.*timed out.*second:.*down"):
        call_with_fallback([("first", first), ("second", second)], "fetch", _AllFailedError)


def test_provider_unsupported_is_retryable():
    first = _FakeClient(error=ProviderUnsupported("not offered"))
    second = _FakeClient(response="from second")

    result = call_with_fallback([("first", first), ("second", second)], "fetch", _AllFailedError)

    assert result == "from second"


def test_call_with_fallback_passes_args_and_kwargs_through():
    calls = []

    class _RecordingClient:
        def fetch(self, ticker, limit=20):
            calls.append((ticker, limit))
            return "ok"

    result = call_with_fallback([("a", _RecordingClient())], "fetch", _AllFailedError, "AAPL", limit=5)

    assert result == "ok"
    assert calls == [("AAPL", 5)]


# -- classify_requests_error --


class _RateLimited(Exception):
    pass


class _Timeout(Exception):
    pass


class _Unavailable(Exception):
    pass


def _classify(exc):
    return classify_requests_error(
        exc,
        "TestProvider",
        base_error_cls=RuntimeError,
        rate_limited_cls=_RateLimited,
        timeout_cls=_Timeout,
        unavailable_cls=_Unavailable,
    )


def test_classify_requests_error_timeout(fake_requests_module):
    result = _classify(fake_requests_module.exceptions.Timeout("timed out"))
    assert isinstance(result, _Timeout)


def test_classify_requests_error_connection_error_is_unavailable(fake_requests_module):
    result = _classify(fake_requests_module.exceptions.ConnectionError("connection refused"))
    assert isinstance(result, _Unavailable)


def test_classify_requests_error_429_status_is_rate_limited(fake_requests_module):
    exc = fake_requests_module.exceptions.RequestException("too many requests")
    exc.response = type("Resp", (), {"status_code": 429})()
    result = _classify(exc)
    assert isinstance(result, _RateLimited)


def test_classify_requests_error_5xx_status_is_unavailable(fake_requests_module):
    exc = fake_requests_module.exceptions.RequestException("server error")
    exc.response = type("Resp", (), {"status_code": 503})()
    result = _classify(exc)
    assert isinstance(result, _Unavailable)


def test_classify_requests_error_generic_falls_back_to_base_error(fake_requests_module):
    result = _classify(fake_requests_module.exceptions.RequestException("boom"))
    assert isinstance(result, RuntimeError)
    assert not isinstance(result, (_RateLimited, _Timeout, _Unavailable))
