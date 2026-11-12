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

# A library must not touch the root logger; that is the application's call.
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


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


def _pull_markets() -> list[dict]:
    """Every open market on Gamma, walked page by page."""
    pulled_markets = []
    # The keyset endpoint only returns open markets by default; say so anyway.
    params = {"limit" : 100, "closed" : "false",
              "end_date_min" : datetime.today().strftime("%Y-%m-%d")}
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

    return pulled_markets


class Cache:
    """A catalogue of open events and markets, plus the event embeddings that
    embedded search ranks against.

    `events` has one row per event (with its `embeddedVector`), `markets` one
    row per market joined to its event, and `events_vecs` is the embedding
    matrix, row-aligned to `events`. All three are None until refresh() or
    load() has run. Nothing happens at construction or import.

    `directory` defaults to config.CACHE_DIR, read at the time it is used.
    """

    def __init__(self, directory: str | None = None):
        self._directory = None if directory is None else str(directory)
        self.events: pd.DataFrame | None = None
        self.events_vecs: np.ndarray | None = None
        self.markets: pd.DataFrame | None = None

    @property
    def directory(self) -> str:
        return self._directory or config.CACHE_DIR

    @property
    def loaded(self) -> bool:
        return self.events is not None and self.events_vecs is not None

    def _path(self, name: str) -> str:
        return os.path.join(self.directory, name)

    def age(self) -> float | None:
        """Seconds since the cache on disk was last written, None if there is none."""
        try:
            return time.time() - os.path.getmtime(self._path("events_vecs.npy"))
        except OSError:
            return None

    def _write_frame(self, frame: pd.DataFrame, name: str) -> None:
        """Honour the configured cache level: CSV for eyeballing, parquet for
        compactness. Either one can be read back by _read_frame."""
        level = config.cache_level
        if level == "off":
            return
        if level in ("on", "csv"):
            frame.to_csv(self._path(name + ".csv"), index = False)
        if level in ("on", "npy"):
            frame.to_parquet(self._path(name + ".parquet"), index = False)

    def _read_frame(self, name: str) -> pd.DataFrame:
        """Read whichever copy was written most recently, so a parquet file left
        over from an earlier cache level can't shadow a newer CSV."""
        parquet = self._path(name + ".parquet")
        csv = self._path(name + ".csv")
        found = [p for p in (parquet, csv) if os.path.exists(p)]
        if not found:
            raise FileNotFoundError(f"No cached {name} table in {self.directory}")

        newest = max(found, key = os.path.getmtime)
        if newest == parquet:
            return pd.read_parquet(parquet)
        # Ids are strings everywhere else; don't let CSV parsing turn them into ints.
        return pd.read_csv(csv, dtype = {"id": str, "market_id": str, "event_id": str})

    def _known_vectors(self) -> dict:
        """Embeddings already paid for, keyed by the text that produced them."""
        events = self.events
        if events is None or not {"shortenedDesc", "embeddedVector"} <= set(events.columns):
            return {}
        return {text: vec for text, vec in zip(events["shortenedDesc"],
                                               events["embeddedVector"])
                if isinstance(text, str) and text and vec is not None}

    def refresh(self) -> None:
        """Pull the open catalogue from Gamma, embed any event text not seen
        before, and write the result to disk at the configured cache level."""

        _configure_logging()

        if config.cache_level != "off":
            os.makedirs(self.directory, exist_ok = True)

        logger.info(" Starting cache sequence")

        pulled_markets = _pull_markets()

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

        vectors = self._known_vectors()
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

        if len(final_events_df):
            events_vecs = np.stack(final_events_df["embeddedVector"].to_numpy())
        else:
            events_vecs = np.empty((0, 0))

        # On disk the vectors live only in events_vecs.npy, row-aligned to the
        # events table. load() reattaches them as the embeddedVector column.
        self._write_frame(markets_df, "markets")
        self._write_frame(final_events_df.drop(columns = ["embeddedVector"]), "events")
        if config.cache_level != "off":
            np.save(self._path("events_vecs.npy"), events_vecs)

        self.markets = markets_df
        self.events = final_events_df
        self.events_vecs = events_vecs

        logger.info(" Succesfully refreshed markets and events.")

    def load(self, build: bool = True) -> bool:
        """Read the cache off disk. With `build` set (the default) it is built
        first if it isn't there, and refreshed if it is older than
        REFRESH_SECONDS; without it this never touches the network. Returns
        whether a cache ended up loaded."""

        try:
            frame = self._read_frame("events")
            vecs = np.load(self._path("events_vecs.npy"))
        except (FileNotFoundError, OSError):
            if not build:
                return False
            logger.info(" Cache not found at %s. Refreshing markets.", self.directory)
            self.refresh()
            return self.loaded

        if len(frame) != len(vecs):
            if not build:
                return False
            logger.warning(" Cached events and vectors are out of step. Rebuilding the cache.")
            self.refresh()
            return self.loaded

        frame["embeddedVector"] = list(vecs)
        self.events, self.events_vecs = frame, vecs

        try:
            self.markets = self._read_frame("markets")
        except (FileNotFoundError, OSError):
            self.markets = None

        age = self.age()
        if build and age is not None and age > REFRESH_SECONDS:
            logger.info(" Cache is older than %s seconds. Refreshing markets.", REFRESH_SECONDS)
            try:
                self.refresh()
            except PolymarketAPIError as e:
                logger.warning(" Could not refresh the cache (%s). Using the stale copy.", e)

        return True

    def ensure_loaded(self, build: bool = True) -> bool:
        """Load the cache if this instance hasn't already. See load()."""
        if self.loaded:
            return True
        return self.load(build = build)


# The cache the rest of the library uses. cacher.events, cacher.events_vecs and
# cacher.markets read through to it.
default = Cache()


def cache() -> None:
    """Build or refresh the default cache. See Cache.refresh()."""
    default.refresh()


def load(build: bool = True) -> bool:
    """Read the default cache off disk. See Cache.load()."""
    return default.load(build = build)


async def cacheLoop():
    while True:
        # cache() blocks for as long as the pull takes, so keep it off the event loop.
        await asyncio.to_thread(cache)
        await asyncio.sleep(REFRESH_SECONDS)


def __getattr__(name: str):
    if name in ("events", "events_vecs", "markets"):
        return getattr(default, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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


if __name__ == "__main__":
    config.caching = True
    config.log = True
    logging.basicConfig(level = logging.INFO)
    cache()
