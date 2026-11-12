from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from . import _client
from .custom_exceptions import PolymarketAPIError
from .config import config

if TYPE_CHECKING:
    import numpy as np

# Contains logic for Event and Market classes, to allow for more seemless analysis of
# Polymarket 

# Polymarket API docs: https://docs.polymarket.com/api-reference/


def _parse_time(value) -> datetime | None:
    if not value:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo = timezone.utc)
    return parsed


def _json_field(data: dict, key: str) -> list:
    raw = data.get(key)
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return []


def _to_float(value) -> float | None:
    """Gamma sends some numbers as strings and omits others entirely."""
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_bool(value) -> bool | None:
    return None if value is None else bool(value)


class Event:
    """A Polymarket event: one question page holding one or more markets.

    Load one with Event(id=...) or Event(slug=...). The full Gamma payload is
    in `.data`; the properties below cover the fields used most.
    """

    def __init__(self, id: str = None, slug : str = None, vec : list[np.array] = None):
        
        if id:
            self.id = id
            self.slug = None
            self.data = _client.get_json("events/" + str(id))
            try:
                self.slug = self.data["slug"]
            except KeyError as e:
                raise PolymarketAPIError(
                    "Polymarket returned no slug in event initialization", obj = self) from e

        elif slug:
            self.slug = slug
            self.id = None
            self.data = _client.get_json("events/slug/" + str(slug))
            try:
                self.id = self.data["id"]
            except KeyError as e:
                raise PolymarketAPIError(
                    "Polymarket returned no id in event initialization", obj = self) from e

        else:
            raise ValueError("Event instance must be initialized with either an id or slug")
        self.vec = vec
        self.data_timestamp = datetime.now()
        self._markets = None

    @classmethod
    def from_data(cls, data : dict, vec : list[np.array] = None) -> "Event":
        event = cls.__new__(cls)
        event.data = data
        event.id = data.get("id")
        event.slug = data.get("slug")
        event.vec = vec
        event.data_timestamp = datetime.now()
        event._markets = None
        return event

    @property
    def markets(self) -> list["Market"]:
        if self._markets is None:
            self._markets = [Market.from_data(m)
                             for m in (self.data.get("markets") or [])]
            # Nested market payloads don't name their parent event.
            for market in self._markets:
                if market.event_id is None:
                    market.event_id = self.id
        return self._markets

    @property
    def title(self) -> str | None:
        return self.data.get("title")

    @property
    def description(self) -> str | None:
        return self.data.get("description")

    @property
    def start_date(self) -> datetime | None:
        return _parse_time(self.data.get("startDate"))

    @property
    def end_date(self) -> datetime | None:
        return _parse_time(self.data.get("endDate"))

    @property
    def active(self) -> bool | None:
        return _to_bool(self.data.get("active"))

    @property
    def closed(self) -> bool | None:
        return _to_bool(self.data.get("closed"))

    @property
    def volume(self) -> float | None:
        """Total traded volume across the event's markets, in USDC."""
        return _to_float(self.data.get("volume"))

    @property
    def liquidity(self) -> float | None:
        return _to_float(self.data.get("liquidity"))

    @property
    def tags(self) -> list[str]:
        """Tag labels, e.g. ["Politics", "Elections"]."""
        return [tag.get("label") or tag.get("slug") for tag
                in (self.data.get("tags") or []) if isinstance(tag, dict)]

    @property
    def url(self) -> str | None:
        return f"https://polymarket.com/event/{self.slug}" if self.slug else None

    def vectorize(self, in_place : bool = True):
        """Embed this event's description (see cacher for the exact text used).
        Stores the vector on `.vec` unless in_place is False."""
        from . import embedding_logic as emb

        vec = self._cached_vec() if config.caching else None

        if vec is None:
            # Same fallback chain the cacher uses: first line of the
            # description, the whole description, the title, then the slug.
            description = (self.data.get("description") or "").strip()
            text = (description.split("\n")[0].strip() or description
                    or (self.title or "").strip() or str(self.slug or ""))
            vec = emb.embed(text)
        if in_place:
            self.vec = vec
        return vec

    def _cached_vec(self):
        """Reuse the embedding the cacher already paid for, if this event is in it."""
        if self.id is None:
            return None

        try:
            from . import cacher
        except ImportError:
            return None

        # Only what is already on disk or in memory: embedding one event is not
        # a reason to pull the whole catalogue.
        if not cacher.default.ensure_loaded(build = False):
            return None

        events = cacher.default.events
        if "embeddedVector" not in events:
            return None

        hit = events.loc[events["id"].astype(str) == str(self.id), "embeddedVector"]
        if hit.empty:
            return None
        return hit.iloc[0]

    def refresh(self):
        """Re-fetch the event from Gamma and restamp `data_timestamp`."""
        self.data = _client.get_json("events/" + str(self.id))
        self.data_timestamp = datetime.now()
        self._markets = None

    def to_frame(self):
        """Markets in this event as a DataFrame, one row per market."""
        try:
            import pandas as pd
        except ImportError as e:
            raise ImportError("to_frame needs pandas. Install it with: "
                              "pip install pyolymarket[cache]") from e

        return pd.DataFrame(self.data.get("markets") or [])

    def __str__(self):
        return f"Event {self.slug}. Last Refreshed : {self.data_timestamp}"

    def __repr__(self):
        return f"Event(slug={self.slug!r}, id={self.id!r})"


class Market:
    """One tradable question inside an event.

    Load one with Market(id=...) or Market(slug=...), or take them from
    Event.markets. The full Gamma payload is in `.data`. The quote properties
    (best_bid, best_ask, last_trade_price) are a snapshot from when the data
    was fetched; the methods at the bottom ask the CLOB for live values.
    """

    def __init__(self, id : str = None, slug : str = None):
        if id:
            self.id = id
            self.slug = None
            self.data = _client.get_json("markets/" + str(id))
            try:
                self.slug = self.data["slug"]
            except KeyError as e:
                raise PolymarketAPIError(
                    "Polymarket returned no slug in market initialization", obj = self) from e
        elif slug:
            self.slug = slug
            self.id = None
            self.data = _client.get_json("markets/slug/" + str(slug))
            try:
                self.id = self.data["id"]
            except KeyError as e:
                raise PolymarketAPIError(
                    "Polymarket returned no id in market initialization", obj = self) from e
        else:
            raise ValueError("Market instance must be initialized with either an id or slug")
        self.data_timestamp = datetime.now()
        events = self.data.get("events") or []
        self.event_id = events[0]["id"] if events else None
        self.__resolution = None

    @classmethod
    def from_data(cls, data : dict) -> Market:
        market = cls.__new__(cls)
        market.data = data
        market.id = data.get("id")
        market.slug = data.get("slug")
        market.data_timestamp = datetime.now()
        events = data.get("events") or []
        market.event_id = events[0].get("id") if events else None
        market._Market__resolution = None
        return market

    def refresh(self):
        """Re-fetch the market from Gamma and restamp `data_timestamp`."""
        self.data = _client.get_json("markets/" + str(self.id))
        self.data_timestamp = datetime.now()

    @property
    def question(self) -> str | None:
        return self.data.get("question")

    @property
    def description(self) -> str | None:
        return self.data.get("description")

    @property
    def start_date(self) -> datetime | None:
        return _parse_time(self.data.get("startDate"))

    @property
    def end_date(self) -> datetime | None:
        return _parse_time(self.data.get("endDate"))

    @property
    def active(self) -> bool | None:
        return _to_bool(self.data.get("active"))

    @property
    def closed(self) -> bool | None:
        return _to_bool(self.data.get("closed"))

    @property
    def accepting_orders(self) -> bool | None:
        return _to_bool(self.data.get("acceptingOrders"))

    @property
    def volume(self) -> float | None:
        """Total traded volume, in USDC."""
        value = self.data.get("volumeNum")
        return _to_float(self.data.get("volume") if value is None else value)

    @property
    def liquidity(self) -> float | None:
        value = self.data.get("liquidityNum")
        return _to_float(self.data.get("liquidity") if value is None else value)

    @property
    def best_bid(self) -> float | None:
        """Best bid for the first outcome, as of `data_timestamp`."""
        return _to_float(self.data.get("bestBid"))

    @property
    def best_ask(self) -> float | None:
        """Best ask for the first outcome, as of `data_timestamp`."""
        return _to_float(self.data.get("bestAsk"))

    @property
    def last_trade_price(self) -> float | None:
        return _to_float(self.data.get("lastTradePrice"))

    @property 
    def resolved(self) -> bool:
        """Whether the market has closed. Gamma's `closed` flag is the source
        of truth: a market can resolve before its end date, or keep trading
        past it. The end date is only a fallback for payloads without the flag."""
        closed = self.data.get("closed")
        if closed is not None:
            return bool(closed)
        end = _parse_time(self.data.get("endDate"))
        return end is not None and datetime.now(timezone.utc) > end
    
    @property
    def resolution(self):
        """"pending" while the market is open, otherwise the UMA oracle's
        status, or "unresolved" if it has none."""
        if not self.resolved:
            return "pending"
        if self.__resolution:
            return self.__resolution

        self.refresh()
        status = self.data.get("umaResolutionStatus")
        if not status:
            return "unresolved"

        self.__resolution = status
        return self.__resolution

    ###### CLOB bridge
    #
    # Gamma identifies a market by id/slug, the CLOB by conditionId and by one
    # token id per outcome. These properties are the join between the two.

    @property
    def condition_id(self) -> str | None:
        return self.data.get("conditionId")

    @property
    def outcomes(self) -> list[str]:
        return _json_field(self.data, "outcomes")

    @property
    def outcome_prices(self) -> list[float]:
        return [float(p) for p in _json_field(self.data, "outcomePrices")]

    @property
    def token_ids(self) -> list[str]:
        return _json_field(self.data, "clobTokenIds")

    def token_id(self, outcome : str | int = 0) -> str:
        """Token id for one outcome, given either its label or its index.

        Labels are matched against `outcomes` rather than assuming index 0 is
        "Yes"; multi-outcome and negative-risk markets don't follow that
        convention.
        """
        tokens = self.token_ids
        if not tokens:
            raise PolymarketAPIError(
                "Market has no clobTokenIds; it may not be order-book traded",
                obj = self)

        if isinstance(outcome, int):
            return tokens[outcome]

        labels = self.outcomes
        for label, token in zip(labels, tokens):
            if label.casefold() == str(outcome).casefold():
                return token
        raise ValueError(f"Unrecognized outcome {outcome!r}. "
                         f"Market outcomes are: {', '.join(labels)}")

    def book(self, outcome : str | int = 0) -> dict:
        """Live order book for one outcome. See clob.book()."""
        from . import clob

        return clob.book(self.token_id(outcome))

    def midpoint(self, outcome : str | int = 0) -> float:
        """Live midpoint between best bid and best ask for one outcome."""
        from . import clob

        return clob.midpoint(self.token_id(outcome))

    def spread(self, outcome : str | int = 0) -> float:
        """Live bid-ask spread for one outcome."""
        from . import clob

        return clob.spread(self.token_id(outcome))

    def price(self, side : str, outcome : str | int = 0) -> float:
        """Live best bid ("BUY") or best ask ("SELL"). See clob.price()."""
        from . import clob

        return clob.price(self.token_id(outcome), side)

    def price_history(self,
                      outcome : str | int = 0,
                      interval : str | None = "1d",
                      start_ts : int | None = None,
                      end_ts : int | None = None,
                      fidelity : int | None = None,
                      as_frame : bool = False):
        """Price timeseries for one outcome. See clob.price_history()."""
        from . import clob

        if start_ts is not None or end_ts is not None:
            interval = None
        return clob.price_history(self.token_id(outcome), interval = interval,
                                  start_ts = start_ts, end_ts = end_ts,
                                  fidelity = fidelity, as_frame = as_frame)

    def __str__(self):
        return f"Market {self.slug}. Last refreshed {self.data_timestamp}"

    def __repr__(self):
        return f"Market(slug={self.slug!r}, id={self.id!r})"
