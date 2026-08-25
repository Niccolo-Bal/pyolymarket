import asyncio, os, logging
import pandas as pd
import numpy as np
from datetime import datetime

from . import _client
from . import embedding_logic as emb
from .config import config

# On-disk catalogue of open events and markets, plus the event embeddings that
# utils.embedded_search_event ranks against.
#
# Layout under config.CACHE_DIR:
#   events.parquet / events.csv      one row per event, with its embeddedVector
#   markets.parquet / markets.csv    one row per market, joined to its event
#   events_vecs.npy                  the embedding matrix, row-aligned to events

MARKET_COLUMNS = ["id", "question", "slug", "conditionId", "clobTokenIds", "events"]
EVENT_COLUMNS = ["id", "slug", "title", "description", "endDate", "seriesSlug"]

REFRESH_SECONDS = 9000

events: pd.DataFrame | None = None
events_vecs: np.ndarray | None = None
markets: pd.DataFrame | None = None


def _cache_path(name: str) -> str:
    return os.path.join(config.CACHE_DIR, name)


def _write_frame(frame: pd.DataFrame, name: str) -> None:
    """Honour the configured cache level. CSV is for eyeballing, parquet is what
    gets read back, so "csv" alone is inspection-only."""
    level = config.cache_level
    if level == "off":
        return
    if level in ("on", "csv"):
        frame.to_csv(_cache_path(name + ".csv"), index = False)
    if level in ("on", "npy"):
        frame.to_parquet(_cache_path(name + ".parquet"), index = False)


def _read_frame(name: str) -> pd.DataFrame:
    parquet = _cache_path(name + ".parquet")
    if os.path.exists(parquet):
        return pd.read_parquet(parquet)
    return pd.read_csv(_cache_path(name + ".csv"))


def cache():

    global events, events_vecs, markets

    if config.log:
        logger.setLevel(logging.INFO)

    os.makedirs(config.CACHE_DIR, exist_ok = True)

    logger.info(" Starting cache sequence")

    pulled_markets = []
    params = {"limit" : 100, "end_date_min" : datetime.today().strftime("%Y-%m-%d")}
    next_cursor = None

    with alive_bar(title = "Pulling Market Data", disable = not config.log) as bar: 
        with _client.new_session() as session:
            logger.info(" Established connection to gamma API.")
            while True:

                if next_cursor:
                    params["after_cursor"] = next_cursor

                batch = _client.get_json("markets/keyset",
                                         params = params, sess = session)

                pulled_markets.extend(batch["markets"])

                next_cursor = batch.get("next_cursor")
                
                if not next_cursor or len(batch["markets"]) < 100:
                    bar(len(batch["markets"]))
                    break

                bar(100)   


    markets_df = pd.DataFrame(pulled_markets).reindex(columns = MARKET_COLUMNS)
    pulled_events_df = pd.json_normalize(markets_df["events"].str[0]).reindex(
        columns = EVENT_COLUMNS)
    markets_df = markets_df.join(pulled_events_df["id"], rsuffix = ".event")
    markets_df = markets_df.rename(columns = {"id" : "market_id",
                                              "id.event" : "event_id"})
    pulled_events_df.drop_duplicates(subset = ["id"], inplace = True)

    ## Events Logic
    # Compare to already cached events, remove old and embed new
    cached_events = events
    if cached_events is None:
        cached_events = pd.DataFrame(columns = ["id"])
    
    cached_events = pd.merge(cached_events, pulled_events_df["id"], on = "id", how = "inner")
    new_events = pulled_events_df[~pulled_events_df["id"].isin(cached_events["id"])].copy()

    # Embed new events by first line of description; use same vector for same first-line description (ex, crypto up-down markets)
    new_events["shortenedDesc"] = new_events["description"].str.split("\n").str[0]
    new_events["shortenedDesc"] = np.where(new_events["shortenedDesc"] == "", new_events["description"], new_events["shortenedDesc"])
    new_events["uniqueDescription"] = ~new_events.duplicated(subset = ["shortenedDesc"])

    embedding_df = new_events[["shortenedDesc"]].drop_duplicates()
    embedding_list = embedding_df["shortenedDesc"].to_list()

    if embedding_list:
        vectorized_list, tokens = emb.embed_list(embedding_list)
        logger.info(f" Succesfully embeded new events. Tokens used: {tokens}")
        embedding_df["embeddedVector"] = vectorized_list
        new_events = pd.merge(new_events, embedding_df, on = "shortenedDesc")
    else:
        # Nothing new since the last run, so skip the embedding request entirely.
        logger.info(" No new events to embed.")

    final_events_df = pd.concat([cached_events, new_events], ignore_index = True)

    ## Markets Logic
    # Mirrors the events flow: keep only markets still in the pull, deduped on
    # id. Unlike events there is no embedding to preserve across runs, so the
    # fresh pull is the whole answer rather than a merge against the cache.
    markets_df = markets_df.drop(columns = ["events"]).drop_duplicates(
        subset = ["market_id"])

    _write_frame(markets_df, "markets")
    markets = markets_df

    events_vecs = np.stack(final_events_df["embeddedVector"].to_numpy())
    events = final_events_df
    _write_frame(events, "events")
    if config.cache_level in ("on", "npy"):
        np.save(_cache_path("events_vecs.npy"), events_vecs)

    logger.info(" Succesfully refreshed markets and events.")


async def cacheLoop():
    while True:
        cache()
        await asyncio.sleep(REFRESH_SECONDS)


def load():
    """Read the cache off disk into the module-level frames. Builds it first if
    it isn't there yet."""
    global events, events_vecs, markets

    try:
        events = _read_frame("events")
        events_vecs = np.load(_cache_path("events_vecs.npy"))
    except (FileNotFoundError, OSError):
        logger.info(" Cache not found at %s. Refreshing markets.", config.CACHE_DIR)
        cache()
        return

    try:
        markets = _read_frame("markets")
    except (FileNotFoundError, OSError):
        markets = None


def alive_bar(*args, **kwargs):
    """Progress bar if alive-progress is installed, a no-op context otherwise.
    Deferred so the cacher stays importable without the optional extra."""
    try:
        from alive_progress import alive_bar as _alive_bar
    except ImportError:
        from contextlib import contextmanager

        @contextmanager
        def _null_bar(*_a, **_kw):
            yield lambda *_args, **_kwargs: None

        _alive_bar = _null_bar
    return _alive_bar(*args, **kwargs)


###### File initializations

# A library must not touch the root logger; that is the application's call.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

if config.caching:
    load()

######

if __name__ == "__main__":
    config.caching = True
    config.log = True
    logging.basicConfig(level = logging.INFO)
    cache()
