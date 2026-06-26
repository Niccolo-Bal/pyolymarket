from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from polyclasses import Market, Event, gamma

class PollymarketAPIError(Exception):
    def __init__(self, message : str, obj : Event | Market = None):
        if obj:
            self.obj = obj
            if type(obj) == Event:
                super().__init__(
                    f"Error at {gamma + "events/" + (obj.slug if obj.slug else obj.id)}: {message}")
            elif type(obj) == Market:
                super().__init__(
                    f"Error at {gamma + "markets/" + (obj.slug if obj.slug else obj.id)}: {message}")
        super().__init__(message)