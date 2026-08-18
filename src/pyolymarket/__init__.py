from . import embedding_logic as embedding
from . import cacher as cacher
from . import utils as utils
from .utils import ddgSearchEvent, embeddedSearchEvent
from .polyclasses import Event, Market
from .custom_exceptions import (
    PolymarketAPIError,
    PolymarketNotFoundError,
    PolymarketRateLimitError,
)

try:
    from importlib.metadata import version as _version
    __version__ = _version("pyolymarket")
except Exception:
    __version__ = "0.0.0+unknown"

__all__ = [
    "embedding", "cacher", "utils", "Event", "Market",
    "ddgSearchEvent", "embeddedSearchEvent",
    "PolymarketAPIError", "PolymarketNotFoundError", "PolymarketRateLimitError",
]
