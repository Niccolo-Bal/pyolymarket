import sys
import types

import pytest

import pyolymarket as pyoly
from pyolymarket import Event, utils


###### native search


def test_polymarket_search_builds_the_query(api):
    api.add("public-search", {"events": [], "pagination": {}})

    utils.polymarket_search("super bowl", limit_per_type = 3, search_tags = True,
                            ascending = False, page = None)

    assert api.calls[0]["arg"] == {"q": "super bowl", "limit_per_type": 3,
                                   "search_tags": "true", "ascending": "false"}


def test_polymarket_search_rejects_bad_input(api):
    with pytest.raises(ValueError, match = "non-empty"):
        utils.polymarket_search("")
    with pytest.raises(ValueError, match = "limit"):
        utils.polymarket_search("x", limit = 3)

    assert api.calls == []


def test_search_event_always_returns_a_list(api, make_event):
    api.add("public-search", lambda params: {
        "events": [make_event("1", "one"), make_event("2", "two")][:params["limit_per_type"]]})

    many = utils.polymarket_search_event("rain", results = 2)
    one = utils.polymarket_search_event("rain", results = 1)

    assert [e.slug for e in many] == ["one", "two"]
    assert [e.slug for e in one] == ["one"]
    assert all(isinstance(e, Event) for e in many)


def test_search_event_with_no_hits_is_an_empty_list(api):
    api.add("public-search", {"events": None})

    assert utils.polymarket_search_event("nothing") == []

    with pytest.raises(ValueError):
        utils.polymarket_search_event("x", results = 0)


###### DuckDuckGo search


@pytest.mark.parametrize("url, slug", [
    ("https://polymarket.com/event/fed-decision", "fed-decision"),
    ("https://polymarket.com/event/fed-decision?tid=123", "fed-decision"),
    ("https://polymarket.com/event/fed-decision/", "fed-decision"),
    ("https://polymarket.com/event/fed-decision/will-the-fed-cut", "fed-decision"),
    ("https://www.polymarket.com/es/event/fed-decision#comments", "fed-decision"),
    ("https://polymarket.com/event", None),
    ("https://polymarket.com/markets/politics", None),
    ("https://www.eventbrite.com/e/some-event-123", None),
    ("https://notpolymarket.com/event/fed-decision", None),
    ("", None),
    (None, None),
])
def test_event_slug(url, slug):
    assert utils._event_slug(url) == slug


@pytest.fixture
def ddg(monkeypatch):
    """A stand-in ddgs module whose results the test sets."""
    hits = []

    class DDGS:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def text(self, query, max_results = 10):
            return [{"href": url} for url in hits[:max_results]]

    monkeypatch.setitem(sys.modules, "ddgs", types.SimpleNamespace(DDGS = DDGS))
    return hits


def test_ddg_search_skips_bad_links_and_duplicates(api, make_event, ddg):
    api.add("events/slug/one", make_event("1", "one"))
    api.add("events/slug/two", make_event("2", "two"))
    ddg.extend([
        "https://www.eventbrite.com/e/abc",
        "https://polymarket.com/event/one?tid=1",
        "https://polymarket.com/event/one/some-market",
        "https://polymarket.com/event/gone",
        "https://polymarket.com/event/two",
    ])

    found = utils.ddg_search_event("anything", max_results = 5)

    assert [e.slug for e in found] == ["one", "two"]
    assert api.paths() == ["events/slug/one", "events/slug/gone", "events/slug/two"]


def test_ddg_search_stops_at_max_results(api, make_event, ddg):
    api.add("events/slug/one", make_event("1", "one"))
    api.add("events/slug/two", make_event("2", "two"))
    ddg.extend(["https://polymarket.com/event/one", "https://polymarket.com/event/two"])

    found = utils.ddg_search_event("anything")

    assert [e.slug for e in found] == ["one"]
    assert api.paths() == ["events/slug/one"]


def test_ddg_search_with_no_hits_is_an_empty_list(api, ddg):
    assert utils.ddg_search_event("anything") == []

    with pytest.raises(ValueError):
        utils.ddg_search_event("x", max_results = 0)
    with pytest.raises(ValueError):
        utils.ddg_search_event("x", depth = 0)


###### embedded search


@pytest.fixture
def corpus(api, make_event, monkeypatch):
    pd = pytest.importorskip("pandas")
    from pyolymarket import embedding_logic

    for event_id in ("1", "2", "3"):
        api.add("events/" + event_id, make_event(event_id, "event-" + event_id))
    # The query points along the first axis, so event 2 is the closest match.
    monkeypatch.setattr(embedding_logic, "embed", lambda text, model = None: [1.0, 0.0])
    return pd.DataFrame({"id": ["1", "2", "3"],
                         "embeddedVector": [[0.0, 1.0], [5.0, 0.1], [1.0, 1.0]]})


def test_embedded_search_ranks_by_cosine_similarity(corpus):
    assert [e.id for e in utils.embedded_search_event("q", results = 3, data = corpus)] \
        == ["2", "3", "1"]
    assert [e.id for e in utils.embedded_search_event("q", data = corpus)] == ["2"]
    # Asking for more than there is returns everything.
    assert len(utils.embedded_search_event("q", results = 10, data = corpus)) == 3


def test_embedded_search_validates_the_table(corpus):
    pd = pytest.importorskip("pandas")

    with pytest.raises(pyoly.EmbeddingError, match = "embeddedVector"):
        utils.embedded_search_event("q", data = corpus[["id"]])
    with pytest.raises(pyoly.EmbeddingError, match = "empty"):
        utils.embedded_search_event("q", data = pd.DataFrame({"id": [], "embeddedVector": []}))
    with pytest.raises(ValueError):
        utils.embedded_search_event("q", results = 0, data = corpus)


def test_embedded_search_needs_caching_or_data(corpus):
    with pytest.raises(pyoly.EmbeddingError, match = "caching is disabled"):
        utils.embedded_search_event("q")


def test_embedded_search_uses_the_default_cache(corpus, clean_config, monkeypatch):
    np = pytest.importorskip("numpy")
    from pyolymarket import cacher

    store = cacher.Cache()
    store.events = corpus
    store.events_vecs = np.stack(corpus["embeddedVector"].to_numpy())
    monkeypatch.setattr(cacher, "default", store)
    clean_config.caching = True

    assert [e.id for e in utils.embedded_search_event("q", results = 2)] == ["2", "3"]
