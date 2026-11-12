import pytest
import requests

from pyolymarket import _client
from pyolymarket.custom_exceptions import (
    PolymarketAPIError,
    PolymarketNotFoundError,
    PolymarketRateLimitError,
)


class FakeResponse:
    def __init__(self, status = 200, payload = None, text = None, headers = None):
        self.status_code = status
        self._payload = payload
        self.text = text if text is not None else str(payload)
        self.headers = headers or {}
        self.url = "https://example.test/thing"

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


class FakeSession:
    def __init__(self, response = None, error = None):
        self.response = response
        self.error = error
        self.requests = []

    def request(self, method, url, **kwargs):
        self.requests.append((method, url, kwargs))
        if self.error:
            raise self.error
        return self.response


def test_get_json_decodes_and_builds_url():
    sess = FakeSession(FakeResponse(payload = {"id": "1"}))

    assert _client.get_json("/events/1", params = {"a": 1}, sess = sess) == {"id": "1"}

    method, url, kwargs = sess.requests[0]
    assert method == "GET"
    assert url == _client.GAMMA + "events/1"
    assert kwargs["params"] == {"a": 1}
    assert kwargs["timeout"] == _client.DEFAULT_TIMEOUT


def test_post_json_defaults_to_clob_and_sends_body():
    sess = FakeSession(FakeResponse(payload = []))

    _client.post_json("books", json_body = [{"token_id": "1"}], sess = sess)

    method, url, kwargs = sess.requests[0]
    assert (method, url) == ("POST", _client.CLOB + "books")
    assert kwargs["json"] == [{"token_id": "1"}]


@pytest.mark.parametrize("status, expected", [
    (404, PolymarketNotFoundError),
    (429, PolymarketRateLimitError),
    (500, PolymarketAPIError),
    (400, PolymarketAPIError),
])
def test_http_errors_are_translated(status, expected):
    sess = FakeSession(FakeResponse(status = status, text = "nope",
                                    headers = {"Retry-After": "7"}))

    with pytest.raises(expected) as raised:
        _client.get_json("events/1", sess = sess)

    assert isinstance(raised.value, PolymarketAPIError)
    if status == 429:
        assert raised.value.retry_after == "7"


@pytest.mark.parametrize("error", [
    requests.exceptions.Timeout("slow"),
    requests.exceptions.ConnectionError("down"),
])
def test_transport_errors_are_translated(error):
    with pytest.raises(PolymarketAPIError):
        _client.get_json("events/1", sess = FakeSession(error = error))


def test_non_json_body_raises():
    with pytest.raises(PolymarketAPIError, match = "non-JSON"):
        _client.get_json("events/1", sess = FakeSession(FakeResponse(text = "<html>")))


def test_error_payload_with_200_raises():
    payload = {"type": "not found error", "error": "slug not found"}

    with pytest.raises(PolymarketAPIError, match = "not found error"):
        _client.get_json("events/1", sess = FakeSession(FakeResponse(payload = payload)))


def test_session_retry_policy():
    retry = _client.new_session().get_adapter("https://").max_retries

    assert retry.total == _client.DEFAULT_RETRIES
    assert set(_client.RETRY_STATUSES) <= set(retry.status_forcelist)
    assert _client.new_session(retries = 0).get_adapter("https://").max_retries.total == 0


def test_session_sends_a_user_agent():
    assert "pyolymarket" in _client.new_session().headers["User-Agent"]
