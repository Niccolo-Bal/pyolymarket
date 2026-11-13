"""Shared fixtures. Nothing in this suite touches the network or needs a key."""

import copy

import pytest
import requests

import pyolymarket as pyoly
from pyolymarket import _client
from pyolymarket.custom_exceptions import PolymarketNotFoundError

ENV_VARS = ("PYOLY_OPENAI_API_KEY", "PYOLY_CLOB_API_KEY", "PYOLY_CLOB_SECRET",
            "PYOLY_CLOB_PASSPHRASE", "PYOLY_CLOB_ADDRESS", "PYOLY_CLOB_PRIVATE_KEY")


@pytest.fixture(autouse = True)
def clean_config(tmp_path, monkeypatch):
    """Each test starts with caching off, its own cache directory and no credentials."""
    config = pyoly.config
    saved = (config._caching, config._cache_dir, config.log,
             config.embedding_base_url, config.embedding_model)

    config._caching = "off"
    config.CACHE_DIR = tmp_path / "cache"
    config.log = False
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising = False)

    yield config

    (config._caching, config._cache_dir, config.log,
     config.embedding_base_url, config.embedding_model) = saved


@pytest.fixture(autouse = True)
def no_network(monkeypatch):
    """A test that reaches the real network is a broken test."""
    def blocked(self, request, *args, **kwargs):
        raise AssertionError(f"test tried to reach the network: {request.url}")

    monkeypatch.setattr(requests.adapters.HTTPAdapter, "send", blocked)


class FakeAPI:
    """Stands in for _client.get_json / post_json. Unknown paths answer 404."""

    def __init__(self):
        self.routes = {}
        self.calls = []

    def add(self, path, payload):
        """`payload` is returned as-is, or called with the params/body if callable."""
        self.routes[path] = payload

    def _answer(self, method, path, argument, kwargs):
        # Copied, because callers are free to reuse and mutate their params.
        self.calls.append({"method": method, "path": path,
                           "arg": copy.deepcopy(argument), **kwargs})
        if path not in self.routes:
            raise PolymarketNotFoundError(f"{path} returned 404")
        payload = self.routes[path]
        return payload(argument) if callable(payload) else copy.deepcopy(payload)

    def get_json(self, path, params = None, **kwargs):
        return self._answer("GET", path, params, kwargs)

    def post_json(self, path, json_body = None, **kwargs):
        return self._answer("POST", path, json_body, kwargs)

    def paths(self):
        return [call["path"] for call in self.calls]


@pytest.fixture
def api(monkeypatch):
    fake = FakeAPI()
    monkeypatch.setattr(_client, "get_json", fake.get_json)
    monkeypatch.setattr(_client, "post_json", fake.post_json)
    return fake


def market_payload(market_id = "500", slug = "will-it-rain", **overrides):
    payload = {
        "id": market_id,
        "slug": slug,
        "question": "Will it rain?",
        "conditionId": "0xabc",
        "outcomes": '["Yes", "No"]',
        "outcomePrices": '["0.25", "0.75"]',
        "clobTokenIds": '["111", "222"]',
        "endDate": "2030-01-01T00:00:00Z",
        "active": True,
        "closed": False,
    }
    payload.update(overrides)
    return payload


def event_payload(event_id = "900", slug = "rain-event", **overrides):
    payload = {
        "id": event_id,
        "slug": slug,
        "title": "Rain event",
        "description": "Resolves Yes if it rains.\nFine print.",
        "endDate": "2030-01-01T00:00:00Z",
        "markets": [market_payload()],
    }
    payload.update(overrides)
    return payload


@pytest.fixture
def make_market():
    return market_payload


@pytest.fixture
def make_event():
    return event_payload
