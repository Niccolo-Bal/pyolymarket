from __future__ import annotations

import re
from typing import TYPE_CHECKING

from . import _client
from .custom_exceptions import EmbeddingError
from .polyclasses import Event
from .config import config

if TYPE_CHECKING:
    import pandas as pd

# numpy, pandas and the openai client are only needed by embedded_search_event,
# so they are imported inside it rather than here. That keeps the native and
# DuckDuckGo searches usable on a bare install.

# Cannonical Polymarket Searches

# Gamma's search endpoint silently ignores query parameters it doesn't
# recognize, answering 200 with an unfiltered result set. A typo would look
# like a bad ranking rather than a mistake, so the accepted names are pinned
# here and anything else raises.
SEARCH_PARAMS = frozenset({
    "q", "cache", "events_status", "limit_per_type", "page", "events_tag",
    "keep_closed_markets", "sort", "ascending", "search_tags",
    "search_profiles", "recurrence", "exclude_tag_id", "optimized",
})


def polymarket_search(query : str, **params) -> dict:
    """Raw response from Polymarket's native search.

    Returns {"events": [...], "pagination": {"hasMore", "totalResults"}}, plus
    "tags" and "profiles" when search_tags / search_profiles are set. Any of
    the parameters in SEARCH_PARAMS may be passed as a keyword.
    """
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
        # Gamma wants JSON-style booleans, not Python's capitalized repr.
        query_params[key] = str(value).lower() if isinstance(value, bool) else value

    return _client.get_json("public-search", params = query_params)


def polymarket_search_event(query : str,
                            results : int = 5,
                            page : int | None = None,
                            events_tag : list[str] | None = None,
                            events_status : str | None = None,
                            keep_closed_markets : int | None = None,
                            sort : str | None = None,
                            ascending : bool | None = None,
                            exclude_tag_id : list[int] | None = None,
                            **params) -> Event | list[Event]:
    """Search Polymarket's own index and return Events.

    One request total: search hits arrive with their markets nested, so the
    Events are built from that payload rather than re-fetched. Returns a bare
    Event when results == 1, mirroring the other search helpers here.
    """
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

        for url in [result["href"] for result in ddgs.text(query + " polymarket event", max_results = depth)]:
            if re.search(r"event", url):

                if url == "https://polymarket.com/event":
                    continue

                event = Event(slug = url.split("/")[-1])
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
        from .cacher import events, events_vecs
        if events is None or events_vecs is None:
            raise EmbeddingError(
                "No embedding cache available. Build one with "
                "pyolymarket.cacher.cache(); note that cache level \"csv\" does "
                "not write the vector file embedded search needs.")
    else:
        missing = {"id", "embeddedVector"} - set(data.columns)
        if missing:
            raise EmbeddingError(
                f"data is missing required column(s): {', '.join(sorted(missing))}. "
                "A custom embedding table needs an 'id' column and an "
                "'embeddedVector' column of equal-length vectors.")
        events = data
        events_vecs = np.stack(data["embeddedVector"].to_numpy())

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
