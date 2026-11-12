"""Exercises the cacher against a small, stubbed Gamma pull.

The real refresh walks Polymarket's entire open catalogue and pays for
embeddings, so here the pull and the embedder are both replaced.
"""

import os
import time

import pytest

pd = pytest.importorskip("pandas")
np = pytest.importorskip("numpy")

import pyolymarket as pyoly
from pyolymarket import cacher
from pyolymarket import embedding_logic as emb


def _market(market_id, event_id, description, title = None, has_event = True):
    market = {
        "id": str(market_id),
        "question": f"Question {market_id}?",
        "slug": f"market-{market_id}",
        "conditionId": f"0x{market_id}",
        "clobTokenIds": f'["{market_id}001", "{market_id}002"]',
    }
    if has_event:
        market["events"] = [{
            "id": event_id,
            "slug": f"event-{event_id}",
            "title": title if title is not None else f"Event {event_id}",
            "description": description,
            "endDate": "2030-12-31T00:00:00Z",
            "seriesSlug": None,
        }]
    return market


class Gamma:
    """Serves `pages` from markets/keyset and records what was asked for."""

    def __init__(self, api):
        self.api = api
        self.pages = [[]]
        api.add("markets/keyset", self._page)

    def _page(self, params):
        index = int(params.get("after_cursor", 0))
        page = {"markets": self.pages[index]}
        if index + 1 < len(self.pages):
            page["next_cursor"] = str(index + 1)
        return page

    def serve(self, *markets):
        self.pages = [list(markets)]


@pytest.fixture
def gamma(api):
    return Gamma(api)


@pytest.fixture
def embedded(monkeypatch):
    """Replaces the embedder and records each batch of texts it was sent."""
    batches = []

    def embed_list(texts, model = None):
        assert all(isinstance(t, str) and t for t in texts), texts
        batches.append(list(texts))
        return [[float(len(t)), 1.0, 2.0] for t in texts], len(texts)

    monkeypatch.setattr(emb, "embed_list", embed_list)
    return batches


@pytest.fixture
def csv_level(clean_config):
    clean_config.caching = "csv"
    return clean_config


def test_refresh_builds_events_markets_and_vectors(gamma, embedded, csv_level):
    # A full page is 100 markets. Page one shares one event; page two repeats a
    # market id and adds a second event.
    gamma.pages = [
        [_market(i, "10", "Bitcoin up or down\nfine print") for i in range(1, 101)],
        [_market(100, "10", "Bitcoin up or down\nfine print"),
         _market(101, "20", "Something else")],
    ]
    store = cacher.Cache()

    store.refresh()

    assert [call["arg"].get("after_cursor") for call in gamma.api.calls] == [None, "1"]
    assert gamma.api.calls[0]["arg"]["closed"] == "false"
    assert list(store.events["id"]) == ["10", "20"]
    assert list(store.events["shortenedDesc"]) == ["Bitcoin up or down", "Something else"]
    assert store.events_vecs.shape == (2, 3)
    assert len(store.markets) == 101 and store.markets["market_id"].is_unique
    assert list(store.markets.columns) == ["market_id", "question", "slug", "conditionId",
                                           "clobTokenIds", "event_id"]
    assert store.markets["event_id"].notna().all()
    assert embedded == [["Bitcoin up or down", "Something else"]]


def test_same_text_is_embedded_once(gamma, embedded, csv_level):
    gamma.serve(_market(1, "10", "Up or down?\nJan 1"), _market(2, "20", "Up or down?\nJan 2"))
    store = cacher.Cache()

    store.refresh()

    assert embedded == [["Up or down?"]]
    assert np.array_equal(store.events_vecs[0], store.events_vecs[1])
    assert list(store.events["uniqueDescription"]) == [True, False]


def test_embed_text_falls_back_and_never_sends_blanks(gamma, embedded, csv_level):
    gamma.serve(
        _market(1, "10", "First line\nrest"),
        _market(2, "20", "\nOnly the body"),
        _market(3, "30", "", title = "Title thirty"),
        _market(4, "40", None, title = ""),
        _market(5, "50", "orphan", has_event = False),
    )
    store = cacher.Cache()

    store.refresh()

    assert list(store.events["shortenedDesc"]) == ["First line", "Only the body",
                                                   "Title thirty", "event-40"]
    # The market with no parent event is kept, but creates no event.
    assert len(store.markets) == 5
    assert store.markets["event_id"].isna().sum() == 1


def test_empty_pull_leaves_the_cache_alone(gamma, embedded, csv_level):
    gamma.serve(_market(1, "10", "Kept"))
    store = cacher.Cache()
    store.refresh()

    gamma.serve()
    store.refresh()

    assert list(store.events["id"]) == ["10"]
    assert embedded == [["Kept"]]


def test_second_pass_reuses_vectors_and_refreshes_metadata(gamma, embedded, csv_level):
    gamma.serve(_market(1, "10", "Stays"), _market(2, "20", "Will change"),
                _market(3, "30", "Goes away"))
    store = cacher.Cache()
    store.refresh()
    embedded.clear()

    gamma.serve(_market(1, "10", "Stays", title = "Renamed"),
                _market(2, "20", "Has changed"), _market(4, "40", "Brand new"))
    store.refresh()

    assert embedded == [["Has changed", "Brand new"]]
    assert list(store.events["id"]) == ["10", "20", "40"]
    assert store.events.loc[store.events["id"] == "10", "title"].item() == "Renamed"


def test_csv_level_round_trips_through_disk(gamma, embedded, csv_level):
    gamma.serve(_market(1, "10", "One"), _market(2, "20", "Two words"))
    cacher.Cache().refresh()
    embedded.clear()
    calls_before = len(gamma.api.calls)

    files = sorted(os.listdir(csv_level.CACHE_DIR))
    reloaded = cacher.Cache()

    assert files == ["events.csv", "events_vecs.npy", "markets.csv"]
    assert reloaded.load() is True
    # Read off disk: no new pull, no new embeddings, ids still strings.
    assert len(gamma.api.calls) == calls_before and embedded == []
    assert list(reloaded.events["id"]) == ["10", "20"]
    assert list(reloaded.markets["event_id"]) == ["10", "20"]
    assert reloaded.events_vecs.shape == (2, 3)
    assert np.array_equal(np.stack(reloaded.events["embeddedVector"].to_numpy()),
                          reloaded.events_vecs)
    # Vectors live only in the .npy file, not in the table.
    with open(os.path.join(csv_level.CACHE_DIR, "events.csv")) as handle:
        assert "embeddedVector" not in handle.readline()


@pytest.mark.parametrize("level, suffixes", [
    ("on", {".csv", ".parquet", ".npy"}),
    ("npy", {".parquet", ".npy"}),
])
def test_parquet_levels_round_trip(gamma, embedded, clean_config, level, suffixes):
    pytest.importorskip("pyarrow")
    clean_config.caching = level
    gamma.serve(_market(1, "10", "One"), _market(2, "20", "Two words"))

    cacher.Cache().refresh()
    reloaded = cacher.Cache()

    assert {os.path.splitext(name)[1] for name in os.listdir(clean_config.CACHE_DIR)} == suffixes
    assert reloaded.load() is True
    assert list(reloaded.events["id"]) == ["10", "20"]
    assert reloaded.events_vecs.shape == (2, 3)


def test_off_level_keeps_everything_in_memory(gamma, embedded, clean_config):
    gamma.serve(_market(1, "10", "One"))
    store = cacher.Cache()

    store.refresh()

    assert store.loaded and len(store.events) == 1
    assert not os.path.exists(clean_config.CACHE_DIR)


def test_load_builds_a_missing_cache_only_when_allowed(gamma, embedded, csv_level):
    gamma.serve(_market(1, "10", "One"))
    store = cacher.Cache()

    assert store.load(build = False) is False
    assert gamma.api.calls == [] and not store.loaded

    assert store.ensure_loaded() is True
    assert len(gamma.api.calls) == 1 and store.loaded

    # Already loaded: ensure_loaded does nothing further.
    assert store.ensure_loaded() is True
    assert len(gamma.api.calls) == 1


def _age_cache(directory, seconds):
    old = time.time() - seconds
    os.utime(os.path.join(directory, "events_vecs.npy"), (old, old))


def test_stale_cache_is_refreshed_on_load(gamma, embedded, csv_level):
    gamma.serve(_market(1, "10", "One"))
    cacher.Cache().refresh()
    _age_cache(csv_level.CACHE_DIR, cacher.REFRESH_SECONDS + 60)
    gamma.serve(_market(1, "10", "One"), _market(2, "20", "Two"))

    fresh, offline = cacher.Cache(), cacher.Cache()
    offline.load(build = False)
    fresh.load()

    assert list(offline.events["id"]) == ["10"]
    assert list(fresh.events["id"]) == ["10", "20"]
    assert fresh.age() < 60


def test_stale_cache_survives_a_failed_refresh(gamma, embedded, csv_level):
    gamma.serve(_market(1, "10", "One"))
    cacher.Cache().refresh()
    _age_cache(csv_level.CACHE_DIR, cacher.REFRESH_SECONDS + 60)

    def down(params):
        raise pyoly.PolymarketAPIError("gamma is down")

    gamma.api.add("markets/keyset", down)
    store = cacher.Cache()

    assert store.load() is True
    assert list(store.events["id"]) == ["10"]


def test_mismatched_files_trigger_a_rebuild(gamma, embedded, csv_level):
    gamma.serve(_market(1, "10", "One"), _market(2, "20", "Two"))
    cacher.Cache().refresh()
    np.save(os.path.join(csv_level.CACHE_DIR, "events_vecs.npy"), np.zeros((1, 3)))

    store = cacher.Cache()

    assert store.load(build = False) is False
    assert store.load() is True
    assert store.events_vecs.shape == (2, 3)


def test_explicit_directory(gamma, embedded, csv_level, tmp_path):
    gamma.serve(_market(1, "10", "One"))

    cacher.Cache(tmp_path / "mine").refresh()

    assert sorted(os.listdir(tmp_path / "mine")) == ["events.csv", "events_vecs.npy",
                                                     "markets.csv"]
    assert not os.path.exists(csv_level.CACHE_DIR)


def test_module_functions_drive_the_default_cache(gamma, embedded, csv_level, monkeypatch):
    monkeypatch.setattr(cacher, "default", cacher.Cache())
    gamma.serve(_market(1, "10", "One"))

    assert cacher.events is None and cacher.markets is None

    cacher.cache()

    assert list(cacher.events["id"]) == ["10"]
    assert cacher.events_vecs.shape == (1, 3)
    assert len(cacher.markets) == 1
    assert cacher.load() is True


def test_event_vectorize_reuses_the_cached_vector(gamma, embedded, csv_level, monkeypatch):
    monkeypatch.setattr(cacher, "default", cacher.Cache())
    gamma.serve(_market(1, "10", "One"))
    cacher.cache()
    monkeypatch.setattr(emb, "embed", lambda text, model = None: pytest.fail("embedded again"))

    event = pyoly.Event.from_data({"id": "10", "slug": "event-10", "description": "One"})

    assert list(event.vectorize()) == [3.0, 1.0, 2.0]


def test_event_vectorize_does_not_build_the_cache(gamma, embedded, csv_level, monkeypatch):
    monkeypatch.setattr(cacher, "default", cacher.Cache())
    monkeypatch.setattr(emb, "embed", lambda text, model = None: [9.0])

    event = pyoly.Event.from_data({"id": "10", "slug": "event-10", "description": "One"})

    assert event.vectorize() == [9.0]
    assert gamma.api.calls == []
