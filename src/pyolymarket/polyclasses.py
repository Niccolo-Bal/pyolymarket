import json, re
import numpy as np
from datetime import datetime

from . import _client
from ._client import GAMMA as gamma
from .custom_exceptions import PolymarketAPIError
from .config import config

# Contains logic for Event and Market classes, to allow for more seemless analysis of
# Polymarket 

# Polymarket API docs: https://docs.polymarket.com/api-reference/


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

    def vectorize(self, in_place : bool = True):
        from . import embedding_logic as emb
        if config.caching and False: # TODO: pull from cache
            pass
        else: 
            vec = emb.embed(self.data["description"].split("\n")[0])
            if not vec:
                vec = emb.embed(self.data["description"])
        if in_place:
            self.vec = vec
        return vec

    def refresh(self):
        self.data = _client.get_json("events/" + str(self.id))
        self.data_timestamp = datetime.now()

    def __str__(self):
        return f"Event {self.slug}. Last Refreshed : {self.data_timestamp}"


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

    def refresh(self):
        self.data = _client.get_json("markets/" + str(self.id))
        self.data_timestamp = datetime.now()

    @property 
    def resolved(self) -> bool:
        return datetime.now() > self.data["endDate"]
    
    @property
    def resolution(self):
        if not self.resolved():
            return "pending"
        if self.__resolution:
            return self.__resolution
        self.refresh()
        self.__resolution = self.data["umaResolutionStatus"]
        if not self.__resolution:
            # TODO: Error handling
            pass

    def __str__(self):
        return f"Market {self.slug}. Last refreshed {self.data_timestamp}"

if __name__ == "__main__":
    pass