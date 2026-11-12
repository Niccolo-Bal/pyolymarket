from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlsplit

from . import _client
from .custom_exceptions import EmbeddingError, PolymarketNotFoundError
from .polyclasses import Event
from .config import config

if TYPE_CHECKING:
    import pandas as pd

SEARCH_PARAMS = frozenset({
    "q", "cache", "events_status", "limit_per_type", "page", "events_tag",
    "keep_closed_markets", "sort", "ascending", "search_tags",
    "search_profiles", "recurrence", "exclude_tag_id", "optimized",
})


def polymarket_search(query : str, **params) -> dict:

    if not query:
        raise ValueError("polymarket_search requires a non-empty query")

    unknown = set(params) - SEARCH_PARAMS
    if unknown:
        raise ValueError(
            f"Unrecognized search parameter(s): {', '.join(sorted(unknown))}. "
            f"Accepted parameters are: {', '.join(sorted(SEARCH_PARAMS))}")

    query_params = {"q": query}
    for key, value in params.items():
        if value is None:
            continue
        query_params[key] = str(value).lower() if isinstance(value, bool) else value

    return _client.get_json("public-search", params = query_params)


# Native Polymarket search
def polymarket_search_event(query : str, results : int = 5, page : int | None = None, events_tag : list[str] | None = None,
                            events_status : str | None = None, keep_closed_markets : int | None = None, sort : str | None = None,
                            ascending : bool | None = None, exclude_tag_id : list[int] | None = None, **params) -> Event | list[Event]:
    
    if results < 1:
        raise ValueError("results must be a positive value")

    payload = polymarket_search(query,
                                limit_per_type = results,
                                page = page,
                                events_tag = events_tag,
                                events_status = events_status,
                                keep_closed_markets = keep_closed_markets,
                                sort = sort,
                                ascending = ascending,
                                exclude_tag_id = exclude_tag_id,
                                **params)

    events = [Event.from_data(data) for data in payload.get("events") or []]

    if not events:
        return None

    if results == 1:
        return events[0]

    return events


def _event_slug(url : str) -> str | None:
    """Event slug from a polymarket.com event URL, or None for anything else.
    Handles /event/<slug>, /event/<slug>/<market-slug>, query strings,
    trailing slashes and locale prefixes."""
    try:
        parts = urlsplit(url or "")
    except ValueError:
        return None

    host = (parts.hostname or "").lower()
    if host != "polymarket.com" and not host.endswith(".polymarket.com"):
        return None

    segments = [s for s in parts.path.split("/") if s]
    if "event" not in segments:
        return None
    position = segments.index("event")
    if len(segments) <= position + 1:
        return None
    return segments[position + 1]


# Works better than embedded search but inconsistent
def ddg_search_event(query : str, depth : int = 10, max_results : int = 1) -> Event | list[Event]: 
    
    if max_results < 1: 
        raise ValueError("max_results must be a positive value") 

    if depth < 1: 
        raise ValueError("depth must be a positive value")

    try:
        from ddgs import DDGS
    except ImportError as e:
        raise ImportError(
            "ddg_search_event needs the ddgs package. "
            "Install it with: pip install pyolymarket[search]") from e

    with DDGS() as ddgs:

        search_events = []
        seen_slugs = set()

        for result in ddgs.text(query + " polymarket event", max_results = depth):
            slug = _event_slug(result.get("href"))
            if slug is None or slug in seen_slugs:
                continue
            seen_slugs.add(slug)

            try:
                event = Event(slug = slug)
            except PolymarketNotFoundError:
                # Stale or mangled link; one bad hit shouldn't sink the search.
                continue
            search_events.append(event)

            if len(search_events) == max_results:
                if max_results == 1:
                    return event
                return search_events

    if len(search_events) != 0:
        return search_events

    return None

# Embedded searches through cached embedded list of Events.
def embedded_search_event(query : str, results : int = 1, data : pd.DataFrame | None = None) -> Event | list[Event]: # Fallback allways return result

    if results < 1: 
        raise ValueError("results must be a positive value")

    try:
        import numpy as np

        from . import embedding_logic as emb
    except ImportError as e:
        raise ImportError(
            "embedded_search_event needs numpy and the openai client. "
            "Install them with: pip install pyolymarket[embeddings]") from e

    if data is None:
        if not config.caching:
            raise EmbeddingError("Cannot do embedded search while caching is disabled. "
                                 "Set pyolymarket.config.caching = True, or pass a "
                                 "DataFrame as data=")
        try:
            from .cacher import events, events_vecs
        except ImportError as e:
            raise ImportError(
                "Searching the cached catalogue needs the cache dependencies. "
                "Install them with: pip install pyolymarket[cache]") from e
        if events is None or events_vecs is None:
            raise EmbeddingError(
                "No embedding cache available. Build one with "
                "pyolymarket.cacher.cache().")
    else:
        missing = {"id", "embeddedVector"} - set(data.columns)
        if missing:
            raise EmbeddingError(
                f"data is missing required column(s): {', '.join(sorted(missing))}. "
                "A custom embedding table needs an 'id' column and an "
                "'embeddedVector' column of equal-length vectors.")
        events = data
        if len(data) != 0:
            events_vecs = np.stack(data["embeddedVector"].to_numpy())

    if len(events) == 0:
        raise EmbeddingError("There are no embedded events to search: the "
                             "embedding table is empty.")

    if results > len(events):
        results = len(events)

    vecorized_query = emb.embed(query)

    similarity_vec = events_vecs @ vecorized_query / (
        np.linalg.norm(events_vecs, axis=1) * np.linalg.norm(vecorized_query)
    )
    top_n_idx = np.argpartition(similarity_vec, -results)[-results:]
    ranked_idx = sorted(top_n_idx, key = lambda i : -similarity_vec[i])
    
    if results == 1: 
        return Event(id = events.iloc[ranked_idx[0]]["id"])

    return [Event(id = event_id) for event_id in events.iloc[ranked_idx]["id"]]
