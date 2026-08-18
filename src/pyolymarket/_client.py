from __future__ import annotations

import logging
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .custom_exceptions import (
    PolymarketAPIError,
    PolymarketNotFoundError,
    PolymarketRateLimitError,
)

# Single place where HTTP happens. Everything else in the library goes through
# get()/get_json() so timeouts, retries and error translation stay consistent.

GAMMA = "https://gamma-api.polymarket.com/"

DEFAULT_TIMEOUT = 10.0
DEFAULT_RETRIES = 3
RETRY_STATUSES = (429, 500, 502, 503, 504)

logger = logging.getLogger(__name__)


def _build_session(retries: int = DEFAULT_RETRIES) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total = retries,
        connect = retries,
        read = retries,
        status = retries,
        backoff_factor = 0.5,
        status_forcelist = RETRY_STATUSES,
        allowed_methods = frozenset(["GET", "HEAD", "OPTIONS"]),
        respect_retry_after_header = True,
        raise_on_status = False,
    )
    adapter = HTTPAdapter(max_retries = retry, pool_maxsize = 20)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    session.headers.update({"Accept": "application/json"})
    return session


_session: requests.Session | None = None


def session() -> requests.Session:
    """Process-wide session, created on first use so importing costs nothing."""
    global _session
    if _session is None:
        _session = _build_session()
    return _session


def new_session(retries: int = DEFAULT_RETRIES) -> requests.Session:
    """A separate session with the same retry policy, for callers that want
    to scope connection reuse to one bulk job (see cacher.cache)."""
    return _build_session(retries)


def _raise_for_status(response: requests.Response, url: str) -> None:
    if response.ok:
        return

    detail = response.text[:200].strip()

    if response.status_code == 404:
        raise PolymarketNotFoundError(f"{url} returned 404: {detail}")
    if response.status_code == 429:
        raise PolymarketRateLimitError(
            f"{url} rate limited after retries (429): {detail}",
            retry_after = response.headers.get("Retry-After"))
    raise PolymarketAPIError(f"{url} returned HTTP {response.status_code}: {detail}")


def get(path: str,
        params: dict[str, Any] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        base: str = GAMMA,
        sess: requests.Session | None = None) -> requests.Response:
    """GET with a timeout and retries. Raises a Polymarket* exception on failure."""
    url = path if path.startswith("http") else base + path.lstrip("/")
    sess = sess or session()

    try:
        response = sess.get(url, params = params, timeout = timeout)
    except requests.exceptions.Timeout as e:
        raise PolymarketAPIError(f"{url} timed out after {timeout}s") from e
    except requests.exceptions.RequestException as e:
        raise PolymarketAPIError(f"{url} request failed: {e}") from e

    _raise_for_status(response, url)
    return response


def get_json(path: str,
             params: dict[str, Any] | None = None,
             timeout: float = DEFAULT_TIMEOUT,
             base: str = GAMMA,
             sess: requests.Session | None = None) -> Any:
    """GET and decode JSON. Gamma signals some errors with HTTP 200 and a
    body containing "type", so check for that too."""
    response = get(path, params = params, timeout = timeout, base = base, sess = sess)

    try:
        payload = response.json()
    except ValueError as e:
        raise PolymarketAPIError(
            f"{response.url} returned non-JSON body: {response.text[:200]!r}") from e

    if isinstance(payload, dict) and "type" in payload and "error" in payload:
        raise PolymarketAPIError(f"{response.url} returned error: {payload['type']}")

    return payload
