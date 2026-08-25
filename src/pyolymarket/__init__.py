import importlib
from typing import TYPE_CHECKING

from .config import config
from .polyclasses import Event, Market
from .custom_exceptions import (
    EmbeddingError,
    PolymarketAPIError,
    PolymarketNotFoundError,
    PolymarketRateLimitError,
)

try:
    from importlib.metadata import version as _version
    __version__ = _version("pyolymarket")
except Exception:
    __version__ = "0.0.0+unknown"

# The Gamma and CLOB surfaces need nothing but requests. Embeddings, the disk
# cache and DuckDuckGo search each pull a heavier optional dependency, so they
# are resolved on first access instead of at import. That keeps `import
# pyolymarket` cheap and, in the cacher's case, free of side effects: importing
# it can kick off a full catalogue build.

_LAZY_MODULES = {"clob", "cacher", "utils", "embedding_logic"}

# Which extra to point at when a lazy module's own imports are unsatisfied.
_MODULE_EXTRAS = {"cacher": "cache", "embedding_logic": "embeddings"}

_LAZY_ATTRS = {
    "embed": "embedding_logic",
    "embed_list": "embedding_logic",
    "ddg_search_event": "utils",
    "embedded_search_event": "utils",
    "polymarket_search": "utils",
    "polymarket_search_event": "utils",
}

if TYPE_CHECKING:
    from . import cacher, clob, embedding_logic, utils
    from .embedding_logic import embed, embed_list
    from .utils import (
        ddg_search_event,
        embedded_search_event,
        polymarket_search,
        polymarket_search_event,
    )


def _import(name: str):
    try:
        return importlib.import_module("." + name, __name__)
    except ImportError as e:
        extra = _MODULE_EXTRAS.get(name)
        if extra is None:
            raise
        raise ImportError(
            f"pyolymarket.{name} needs the optional {extra!r} dependencies "
            f"({e}). Install them with: pip install pyolymarket[{extra}]") from e


def __getattr__(name: str):
    if name in _LAZY_MODULES:
        module = _import(name)
        globals()[name] = module
        return module

    if name in _LAZY_ATTRS:
        module = _import(_LAZY_ATTRS[name])
        attr = getattr(module, name)
        globals()[name] = attr
        return attr

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | _LAZY_MODULES | set(_LAZY_ATTRS))


__all__ = [
    "Event", "Market", "config",
    "clob", "cacher", "utils", "embedding_logic",
    "embed", "embed_list",
    "polymarket_search", "polymarket_search_event",
    "ddg_search_event", "embedded_search_event",
    "EmbeddingError", "PolymarketAPIError", "PolymarketNotFoundError",
    "PolymarketRateLimitError",
]
