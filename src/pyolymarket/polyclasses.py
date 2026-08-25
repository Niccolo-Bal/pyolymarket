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
    """Gamma timestamps are ISO 8601 strings, sometimes Z-suffixed and sometimes
    naive. Normalize to an aware UTC datetime so comparisons are safe."""
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
    """Gamma returns clobTokenIds, outcomes and outcomePrices as JSON-encoded
    strings rather than arrays, so they need a second decode pass."""
    raw = data.get(key)
    if raw is None:
        return []
    if isinstance(raw, list):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return []


class Event:

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
        """Build an Event from a payload already in hand, without another
        request. Search results and cached rows arrive fully populated, so
        re-fetching them would only cost latency."""
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
        """Markets nested in this event's payload, as Market objects. No extra
        requests: the event payload already carries them in full."""
        if self._markets is None:
            self._markets = [Market.from_data(m)
                             for m in (self.data.get("markets") or [])]
        return self._markets

    @property
    def title(self) -> str | None:
        return self.data.get("title")

    @property
    def end_date(self) -> datetime | None:
        return _parse_time(self.data.get("endDate"))

    def vectorize(self, in_place : bool = True):
        from . import embedding_logic as emb

        vec = self._cached_vec() if config.caching else None

        if vec is None:
            vec = emb.embed((self.data.get("description") or "").split("\n")[0])
            if not vec:
                vec = emb.embed(self.data.get("description") or "")
        if in_place:
            self.vec = vec
        return vec

    def _cached_vec(self):
        """Reuse the embedding the cacher already paid for, if this event is in it."""
        try:
            from .cacher import events
        except (ImportError, AttributeError):
            return None

        if events is None or self.id is None or "embeddedVector" not in events:
            return None

        hit = events.loc[events["id"].astype(str) == str(self.id), "embeddedVector"]
        if hit.empty:
            return None
        return hit.iloc[0]

    def refresh(self):
        self.data = _client.get_json("events/" + str(self.id))
        self.data_timestamp = datetime.now()
        self._markets = None

    def to_frame(self):
        """Markets in this event as a DataFrame, one row per market."""
        import pandas as pd

        return pd.DataFrame(self.data.get("markets") or [])

    def __str__(self):
        return f"Event {self.slug}. Last Refreshed : {self.data_timestamp}"

    def __repr__(self):
        return f"Event(slug={self.slug!r}, id={self.id!r})"


class Market:

    def __init__(self, id : str = None, slug : str = None):
        if id:
            self.id = id
            self.slug = None
            self.data = _client.get_json("markets/" + str(id))
            self.slug = self.data["slug"]
        elif slug:
            self.slug = slug
            self.id = None
            self.data = _client.get_json("markets/slug/" + str(slug))
            self.id = self.data["id"]
        else:
            raise ValueError("Market instance must be initialized with either an id or slug")
        self.data_timestamp = datetime.now()
        events = self.data.get("events") or []
        self.event_id = events[0]["id"] if events else None
        self.__resolution = None

    @classmethod
    def from_data(cls, data : dict) -> "Market":
        """Build a Market from a payload already in hand, without another
        request. Markets nested inside an event or a search hit are complete."""
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
        self.data = _client.get_json("markets/" + str(self.id))
        self.data_timestamp = datetime.now()

    @property 
    def resolved(self) -> bool:
        end = _parse_time(self.data.get("endDate"))
        if end is None:
            return bool(self.data.get("closed"))
        return datetime.now(timezone.utc) > end
    
    @property
    def resolution(self):
        if not self.resolved:
            return "pending"
        if self.__resolution:
            return self.__resolution

        self.refresh()
        status = self.data.get("umaResolutionStatus")
        if not status:
            # Past its end date but the oracle has not reported yet. Don't
            # memoize this, it is the one answer that is expected to change.
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
        from . import clob

        return clob.book(self.token_id(outcome))

    def midpoint(self, outcome : str | int = 0) -> float:
        from . import clob

        return clob.midpoint(self.token_id(outcome))

    def spread(self, outcome : str | int = 0) -> float:
        from . import clob

        return clob.spread(self.token_id(outcome))

    def price(self, side : str, outcome : str | int = 0) -> float:
        from . import clob

        return clob.price(self.token_id(outcome), side)

    def price_history(self,
                      outcome : str | int = 0,
                      interval : str | None = "1d",
                      start_ts : int | None = None,
                      end_ts : int | None = None,
                      fidelity : int | None = None,
                      as_frame : bool = False):
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
