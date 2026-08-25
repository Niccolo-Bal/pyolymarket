"""Exercises cacher.cache() against a bounded, stubbed Gamma pull.

The real cache() walks Polymarket's entire open catalogue, which is far too slow
for a test. The transformation and persistence logic is what actually broke
before, so that is what gets covered here.
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pyolymarket as pyoly
from pyolymarket import _client, cacher
from pyolymarket import embedding_logic as emb


def _market(market_id, event_id, description, question):
    return {
        "id": market_id,
        "question": question,
        "slug": f"market-{market_id}",
        "conditionId": f"0x{market_id:0>4}",
        "clobTokenIds": f'["{market_id}001", "{market_id}002"]',
        "events": [{
            "id": event_id,
            "slug": f"event-{event_id}",
            "title": f"Event {event_id}",
            "description": description,
            "endDate": "2026-12-31T00:00:00Z",
            "seriesSlug": None,
        }],
    }


# A full page is 100 markets; anything shorter ends the walk. Page one shares
# one description across every market so the embedding dedupe has something to
# collapse, page two adds a new event and repeats a market id.
PAGE_ONE = [_market(str(i), "10", "Bitcoin up or down\nfine print", f"BTC {i}?")
            for i in range(1, 101)]

PAGES = [
    {"markets": PAGE_ONE, "next_cursor": "page2"},
    {"markets": [
        # Duplicate market id, to prove the dedupe actually runs.
        _market("100", "10", "Bitcoin up or down\nfine print", "BTC 100?"),
        _market("101", "20", "", "Empty description market"),
    ], "next_cursor": None},
]


def main():
    calls = []

    def fake_get_json(path, params = None, **kwargs):
        cursor = params.get("after_cursor")
        calls.append(cursor)
        return PAGES[0] if cursor is None else PAGES[1]

    _client.get_json = fake_get_json
    emb.embed_list = lambda texts, model = None: (
        [[float(len(t) % 7), 1.0, 2.0] for t in texts], len(texts))

    pyoly.config.CACHE_DIR = tempfile.mkdtemp()
    pyoly.config.caching = True

    cacher.cache()

    events, markets, vecs = cacher.events, cacher.markets, cacher.events_vecs
    print("pagination followed:", calls)
    print("events   :", list(events.columns))
    print("markets  :", list(markets.columns))
    print("event rows", len(events), "| vec shape", vecs.shape)
    print("market rows", len(markets), "| unique ids", markets["market_id"].nunique())
    print("events seen:", sorted(events["id"]))
    print("event_id joined for every market:", markets["event_id"].notna().all())
    print("no raw events column:", "events" not in markets.columns)
    print("empty description fell back to full text:",
          events.loc[events["id"] == "20", "shortenedDesc"].iloc[0] == "")

    files = sorted(p.name for p in Path(pyoly.config.CACHE_DIR).iterdir())
    print("files:", files)

    # Reload from disk into fresh globals.
    cacher.events = cacher.markets = cacher.events_vecs = None
    cacher.load()
    print("reloaded:", len(cacher.events), cacher.events_vecs.shape, len(cacher.markets))

    # Second pass: cached events are kept and nothing new needs embedding.
    calls.clear()
    embedded = []
    emb.embed_list = lambda texts, model = None: (
        embedded.append(texts) or ([[0.0, 1.0, 2.0]] * len(texts), 0))
    cacher.cache()
    print("second run rows:", len(cacher.events), "| new embeddings:", embedded)

    # Cache levels.
    for level, expected in (("csv", {".csv"}), ("npy", {".parquet", ".npy"})):
        pyoly.config.CACHE_DIR = tempfile.mkdtemp()
        pyoly.config.caching = level
        cacher.events = None
        cacher.cache()
        suffixes = {p.suffix for p in Path(pyoly.config.CACHE_DIR).iterdir()}
        print(f"level {level!r} wrote {sorted(suffixes)} -> "
              f"{'ok' if suffixes == expected else 'MISMATCH ' + str(expected)}")


if __name__ == "__main__":
    main()
