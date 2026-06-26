import re
import numpy as np
from ddgs import DDGS

import embedding_logic as emb
from polyclasses import Event, Market
from config import config
if config.caching:
    from cacher import events, events_vecs


def ddgSearchEvent(query : str, depth : int = 10, max_results : int = 1) -> Event | list[Event]: # Works well, inconsistent
    
    if max_results < 1: 
        raise ValueError("max_results must be a positive value") 

    if depth < 1: 
        raise ValueError("depth must be a positive value")

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


def embeddedSearchEvent(query : str, results : int = 1) -> Event | list[Event]: # Fallback allways return result

    if results < 1: 
        raise ValueError("results must be a positive value")

    vecorized_query = emb.embed(query)

    similarity_vec = events_vecs @ vecorized_query / (
        np.linalg.norm(events_vecs, axis=1) * np.linalg.norm(vecorized_query)
    )
    top_n_idx = np.argpartition(similarity_vec, -results)[-results:]
    
    if results == 1: 
        return Event(id = events.iloc[top_n_idx[0]]["id"])

    return [Event(id) for id in events.iloc[sorted(top_n_idx, key = lambda i : -similarity_vec[i])]["id"]]