import requests, json, re
import numpy as np
from datetime import datetime
from custom_exceptions import PollymarketAPIError
from config import config

# Contains logic for Event and Market classes, to allow for more seemless analysis of
# Polymarket 

# Polymarket API docs: https://docs.polymarket.com/api-reference/

gamma = "https://gamma-api.polymarket.com/"


class Event:

    def __init__(self, id: str = None, slug : str = None, vec : list[np.array] = None):
        
        if id:
            self.data = requests.get(gamma + "events/" + id).json()
            self.id = id
            try: 
                self.slug = self.data["slug"]
            except KeyError as e:
                if "type" in self.data:
                    raise PollymarketAPIError(
                        f"Polymarket threw error in event initialization: {self.data["type"]}", obj = self)
                else: raise e
        
        elif slug:
            self.data = requests.get(gamma + "events/slug/" + slug).json()
            self.slug = slug
            try:
                self.id = self.data["id"]
            except KeyError:
                if "type" in self.data:
                    raise PollymarketAPIError(
                        f"Polymarket threw error in event initialization: {self.data["type"]}", obj = self)
                else: raise e

        else:
            raise ValueError("Event instance must be initialized with either an id or slug")
        self.vec = vec
        self.data_timestamp = datetime.now()

    def vectorize(self, in_place : bool = True):
        import embedding_logic as emb
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
        self.data = requests.get(gamma + "events/" + self.id).json
        self.data_timestamp = datetime.now()

    def __str__(self):
        return f"Event {self.slug}. Last Refreshed : {self.data_timestamp}"


class Market:

    def __init__(self, id : str = None, slug : str = None):
        if id:
            self.data = requests.get(gamma + "markets/" + id).json()
            self.id = id
            self.slug = self.data["slug"]
        elif slug:
            self.data = requests.get(gamma + "markets/slug/" + slug).json()
            self.id = self.data["id"]
            self.slug = slug
        else:
            raise ValueError("Market instance must be initialized with either an id or slug")
        self.data_timestamp = datetime.now()
        self.event_id = self.data["events"].json()["id"]
        self.__resolution = None

    def refresh(self):
        self.data = requests.get(gamma + "markets/" + self.id).json()
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
            #
            pass

    def __str__(self):
        return f"Market {self.slug}. Last refreshed {self.data_timestamp}"

if __name__ == "__main__":
    # print(getPolymarketId("bitcoin-up-or-down-june-4-2026-7pm-et", type = "event"))
    pass