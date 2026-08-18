from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .polyclasses import Event, Market


class PolymarketAPIError(Exception):
    """Base for every error raised while talking to a Polymarket API."""

    def __init__(self, message: str, obj: "Event | Market | None" = None):
        self.obj = obj
        if obj is not None:
            super().__init__(f"{_describe(obj)}: {message}")
        else:
            super().__init__(message)


class PolymarketNotFoundError(PolymarketAPIError):
    """The requested event or market does not exist (HTTP 404)."""


class PolymarketRateLimitError(PolymarketAPIError):
    """Rate limited (HTTP 429) and still failing after the client's retries."""

    def __init__(self, message: str, obj: "Event | Market | None" = None,
                 retry_after: str | None = None):
        super().__init__(message, obj)
        self.retry_after = retry_after

class EmbeddingError(Exception):
    def __init__(self, message: str):
        super().__init__(message)


def _describe(obj: "Event | Market") -> str:
    """Retrieve URL from where exception originated"""
    from ._client import GAMMA

    kind = "events/" if type(obj).__name__ == "Event" else "markets/"
    ident = getattr(obj, "slug", None) or getattr(obj, "id", None) or "<unknown>"
    return f"Error at {GAMMA}{kind}{ident}"
