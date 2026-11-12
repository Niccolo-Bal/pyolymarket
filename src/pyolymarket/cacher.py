import asyncio, os, logging, time
import pandas as pd
import numpy as np
from datetime import datetime

from . import _client
from . import embedding_logic as emb
from .config import config
from .custom_exceptions import PolymarketAPIError

MARKET_COLUMNS = ["id", "question", "slug", "conditionId", "clobTokenIds", "events"]
EVENT_COLUMNS = ["id", "slug", "title", "description", "endDate", "seriesSlug"]

REFRESH_SECONDS = 9000

events: pd.DataFrame | None = None
events_vecs: np.ndarray | None = None
markets: pd.DataFrame | None = None


def _cache_path(name: str) -> str:
    return os.path.join(config.CACHE_DIR, name)


def _write_frame(frame: pd.DataFrame, name: str) -> None:
    """Honour the configured cache level: CSV for eyeballing, parquet for
    compactness. Either one can be read back by _read_frame."""
    level = config.cache_level
    if level == "off":
        return
    if level in ("on", "csv"):
        frame.to_csv(_cache_path(name + ".csv"), index = False)
    if level in ("on", "npy"):
        frame.to_parquet(_cache_path(name + ".parquet"), index = False)


def _read_frame(name: str) -> pd.DataFrame:
    """Read whichever copy was written most recently, so a parquet file left
    over from an earlier cache level can't shadow a newer CSV."""
    parquet = _cache_path(name + ".parquet")
    csv = _cache_path(name + ".csv")
    found = [p for p in (parquet, csv) if os.path.exists(p)]
    if not found:
        raise FileNotFoundError(f"No cached {name} table in {config.CACHE_DIR}")

    newest = max(found, key = os.path.getmtime)
    if newest == parquet:
        return pd.read_parquet(parquet)
    # Ids are strings everywhere else; don't let CSV parsing turn them into ints.
    return pd.read_csv(csv, dtype = {"id": str, "market_id": str, "event_id": str})


def _configure_logging() -> None:
    """config.log is an explicit opt-in, so make it visible even when the
    application hasn't set up logging itself."""
    if not config.log:
        return
    logger.setLevel(logging.INFO)
    if not logging.getLogger().handlers and not any(
            type(h) is logging.StreamHandler for h in logger.handlers):
        logger.addHandler(logging.StreamHandler())


def _embed_text(frame: pd.DataFrame) -> pd.Series:
    """The text embedded for each event: first line of its description, then
    the whole description, then the title, then the slug. Never null; empty
    only when an event has none of them."""
    def clean(column: str) -> pd.Series:
        if column not in frame:
            return pd.Series("", index = frame.index, dtype = object)
        return frame[column].fillna("").astype(str).str.strip()

    description = clean("description")
    text = description.str.split("\n").str[0].str.strip()
    for fallback in (description, clean("title"), clean("slug")):
        text = text.where(text != "", fallback)
    return text


def _known_vectors() -> dict:
    """Embeddings already paid for, keyed by the text that produced them."""
    if events is None or not {"shortenedDesc", "embeddedVector"} <= set(events.columns):
        return {}
    return {text: vec for text, vec in zip(events["shortenedDesc"],
                                           events["embeddedVector"])
            if isinstance(text, str) and text and vec is not None}


def cache():

    global events, events_vecs, markets

    _configure_logging()

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

    if not pulled_markets:
        logger.warning(" Gamma returned no markets. Leaving the existing cache untouched.")
        return

    markets_df = pd.DataFrame(pulled_markets).reindex(columns = MARKET_COLUMNS)

    # A market can arrive without a parent event. Give it an empty one so the
    # event rows stay aligned with the market rows for the join below.
    first_events = [evs[0] if isinstance(evs, list) and evs else {}
                    for evs in markets_df["events"]]
    pulled_events_df = pd.json_normalize(first_events).reindex(columns = EVENT_COLUMNS)

    markets_df = markets_df.join(pulled_events_df["id"], rsuffix = ".event")
    markets_df = markets_df.rename(columns = {"id" : "market_id",
                                              "id.event" : "event_id"})
    pulled_events_df = (pulled_events_df.dropna(subset = ["id"])
                        .drop_duplicates(subset = ["id"])
                        .reset_index(drop = True))

    ## Events Logic
    # The catalogue is rebuilt from the fresh pull every time, so titles, end
    # dates and descriptions never go stale. Only the embeddings are carried
    # over: a vector is reused whenever its text has been embedded before,
    # whichever event it belonged to (ex, recurring crypto up-down markets).
    pulled_events_df["shortenedDesc"] = _embed_text(pulled_events_df)

    unembeddable = pulled_events_df["shortenedDesc"] == ""
    if unembeddable.any():
        logger.warning(" Skipping %s event(s) with no description, title or slug to embed.",
                       int(unembeddable.sum()))
        pulled_events_df = pulled_events_df[~unembeddable].reset_index(drop = True)

    pulled_events_df["uniqueDescription"] = ~pulled_events_df.duplicated(
        subset = ["shortenedDesc"])

    vectors = _known_vectors()
    embedding_list = [text for text in pulled_events_df["shortenedDesc"].unique()
                      if text not in vectors]

    if embedding_list:
        vectorized_list, tokens = emb.embed_list(embedding_list)
        logger.info(f" Succesfully embeded new events. Tokens used: {tokens}")
        vectors.update(zip(embedding_list, vectorized_list))
    else:
        # Nothing new since the last run, so skip the embedding request entirely.
        logger.info(" No new events to embed.")

    pulled_events_df["embeddedVector"] = [vectors[text] for text
                                          in pulled_events_df["shortenedDesc"]]
    final_events_df = pulled_events_df

    ## Markets Logic
    # Mirrors the events flow: keep only markets still in the pull, deduped on
    # id. Unlike events there is no embedding to preserve across runs, so the
    # fresh pull is the whole answer rather than a merge against the cache.
    markets_df = markets_df.drop(columns = ["events"]).drop_duplicates(
        subset = ["market_id"])

    _write_frame(markets_df, "markets")
    markets = markets_df

    if len(final_events_df):
        events_vecs = np.stack(final_events_df["embeddedVector"].to_numpy())
    else:
        events_vecs = np.empty((0, 0))
    events = final_events_df

    # On disk the vectors live only in events_vecs.npy, row-aligned to the
    # events table. load() reattaches them as the embeddedVector column.
    _write_frame(events.drop(columns = ["embeddedVector"]), "events")
    if config.cache_level != "off":
        np.save(_cache_path("events_vecs.npy"), events_vecs)

    logger.info(" Succesfully refreshed markets and events.")


async def cacheLoop():
    while True:
        # cache() blocks for as long as the pull takes, so keep it off the event loop.
        await asyncio.to_thread(cache)
        await asyncio.sleep(REFRESH_SECONDS)


def load():
    """Read the cache off disk into the module-level frames. Builds it first if
    it isn't there yet, and refreshes it if it is older than REFRESH_SECONDS."""
    global events, events_vecs, markets

    vecs_path = _cache_path("events_vecs.npy")

    try:
        frame = _read_frame("events")
        vecs = np.load(vecs_path)
    except (FileNotFoundError, OSError):
        logger.info(" Cache not found at %s. Refreshing markets.", config.CACHE_DIR)
        cache()
        return

    if len(frame) != len(vecs):
        logger.warning(" Cached events and vectors are out of step. Rebuilding the cache.")
        cache()
        return

    frame["embeddedVector"] = list(vecs)
    events, events_vecs = frame, vecs

    try:
        markets = _read_frame("markets")
    except (FileNotFoundError, OSError):
        markets = None

    if time.time() - os.path.getmtime(vecs_path) > REFRESH_SECONDS:
        logger.info(" Cache is older than %s seconds. Refreshing markets.", REFRESH_SECONDS)
        try:
            cache()
        except PolymarketAPIError as e:
            logger.warning(" Could not refresh the cache (%s). Using the stale copy.", e)


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
