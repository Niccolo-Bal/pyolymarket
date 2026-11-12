from __future__ import annotations

from typing import Any, Literal

import requests

from . import _client
from ._client import CLOB as clob_base
from .custom_exceptions import PolymarketAPIError

Side = Literal["BUY", "SELL"]

INTERVALS = ("max", "all", "1m", "1w", "1d", "6h", "1h")

HISTORY_TOKEN_PARAM = "market"


def _get(path: str,
         params: dict[str, Any] | None = None,
         sess: requests.Session | None = None) -> Any:
    return _client.get_json(path, params = params, base = clob_base, sess = sess)


def _post(path: str,
          body: Any,
          sess: requests.Session | None = None) -> Any:
    return _client.post_json(path, json_body = body, base = clob_base, sess = sess)


def _token_bodies(token_ids: list[str]) -> list[dict[str, str]]:
    return [{"token_id": str(t)} for t in token_ids]


###### Order books


def book(token_id: str, sess: requests.Session | None = None) -> dict[str, Any]:
    """Full order book for one outcome token.

    Bids and asks are sorted best-first here; the API's own ordering is not
    consistent, so `result["bids"][0]` is only the best bid because of this.
    """
    payload = _get("book", {"token_id": str(token_id)}, sess = sess)
    return _sort_book(payload)


def books(token_ids: list[str],
          sess: requests.Session | None = None) -> list[dict[str, Any]]:
    """Order books for several tokens in one request."""
    payload = _post("books", _token_bodies(token_ids), sess = sess)
    return [_sort_book(b) for b in payload]


def _sort_book(payload: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return payload
    for side, reverse in (("bids", True), ("asks", False)):
        levels = payload.get(side)
        if levels:
            payload[side] = sorted(levels, key = lambda lv: float(lv["price"]),
                                   reverse = reverse)
    return payload


###### Quotes


def price(token_id: str, side: Side,
          sess: requests.Session | None = None) -> float:
    """Best resting price on one side of an outcome token's book.

    `side` names the side of the book, not the trade you intend to make:
    "BUY" answers with the best bid and "SELL" with the best ask, so the price
    to acquire a token is the "SELL" one.
    """
    side = _check_side(side)
    payload = _get("price", {"token_id": str(token_id), "side": side}, sess = sess)
    return float(payload["price"])


def prices(pairs: list[tuple[str, Side]],
           sess: requests.Session | None = None) -> dict[str, dict[str, str]]:
    """Batch of (token_id, side) quotes, keyed by token id then side."""
    body = [{"token_id": str(t), "side": _check_side(s)} for t, s in pairs]
    return _post("prices", body, sess = sess)


def midpoint(token_id: str, sess: requests.Session | None = None) -> float:
    """Midpoint between the best bid and best ask."""
    payload = _get("midpoint", {"token_id": str(token_id)}, sess = sess)
    return float(payload["mid"])


def midpoints(token_ids: list[str],
              sess: requests.Session | None = None) -> dict[str, str]:
    """Midpoints for several tokens in one request, keyed by token id."""
    return _post("midpoints", _token_bodies(token_ids), sess = sess)


def spread(token_id: str, sess: requests.Session | None = None) -> float:
    """Best ask minus best bid."""
    payload = _get("spread", {"token_id": str(token_id)}, sess = sess)
    return float(payload["spread"])


def spreads(token_ids: list[str],
            sess: requests.Session | None = None) -> dict[str, str]:
    """Spreads for several tokens in one request, keyed by token id."""
    return _post("spreads", _token_bodies(token_ids), sess = sess)


def last_trade_price(token_id: str,
                     sess: requests.Session | None = None) -> dict[str, Any]:
    """Most recent trade. `side` comes back empty when the token never traded."""
    return _get("last-trade-price", {"token_id": str(token_id)}, sess = sess)


def last_trade_prices(token_ids: list[str],
                      sess: requests.Session | None = None) -> list[dict[str, Any]]:
    """Most recent trade for several tokens in one request."""
    return _post("last-trades-prices", _token_bodies(token_ids), sess = sess)


def _check_side(side: str) -> str:
    normalized = str(side).upper()
    if normalized not in ("BUY", "SELL"):
        raise ValueError(f'Unrecognized side: {side!r}. side must be "BUY" or "SELL"')
    return normalized


###### Price history


def price_history(token_id: str, interval: str | None = None, start_ts: int | None = None, end_ts: int | None = None,
                  fidelity: int | None = None, as_frame: bool = False, sess: requests.Session | None = None):
    """Price timeseries for one token. Pass `interval`, or both `start_ts` and
    `end_ts` (unix seconds). `fidelity` is the resolution in minutes. Returns
    {"t", "p"} dicts, or a time-indexed DataFrame with as_frame."""

    if interval is None and (start_ts is None or end_ts is None):
        raise ValueError("price_history needs either interval, or both "
                         "start_ts and end_ts")
    if interval is not None and interval not in INTERVALS:
        raise ValueError(f"Unrecognized interval: {interval!r}. "
                         f"interval must be one of {', '.join(INTERVALS)}")

    params: dict[str, Any] = {HISTORY_TOKEN_PARAM: str(token_id)}
    if interval is not None:
        params["interval"] = interval
    if start_ts is not None:
        params["startTs"] = int(start_ts)
    if end_ts is not None:
        params["endTs"] = int(end_ts)
    if fidelity is not None:
        params["fidelity"] = int(fidelity)

    history = _get("prices-history", params, sess = sess).get("history", [])

    if not as_frame:
        return history
    return _history_frame(history)


def batch_price_history(token_ids: list[str], interval: str | None = None, start_ts: int | None = None, end_ts: int | None = None,
                        fidelity: int | None = None, sess: requests.Session | None = None) -> dict[str, list]:
    """Price history for up to 20 tokens, keyed by token id."""

    if len(token_ids) > 20:
        raise ValueError(f"batch_price_history accepts at most 20 tokens, "
                         f"got {len(token_ids)}")

    body: dict[str, Any] = {"markets": [str(t) for t in token_ids]}
    if interval is not None:
        body["interval"] = interval
    if start_ts is not None:
        body["start_ts"] = int(start_ts)
    if end_ts is not None:
        body["end_ts"] = int(end_ts)
    if fidelity is not None:
        body["fidelity"] = int(fidelity)

    return _post("batch-prices-history", body, sess = sess).get("history", {})


def _history_frame(history: list[dict[str, Any]]):
    try:
        import pandas as pd
    except ImportError as e:
        raise ImportError("as_frame=True needs pandas. Install it with: "
                          "pip install pyolymarket[cache]") from e

    frame = pd.DataFrame(history, columns = ["t", "p"])
    frame["t"] = pd.to_datetime(frame["t"], unit = "s", utc = True)
    return frame.set_index("t").rename(columns = {"p": "price"})


###### Market metadata


def market(condition_id: str, sess: requests.Session | None = None) -> dict[str, Any]:
    """The CLOB's view of a market, including its outcome `tokens`."""
    return _get("markets/" + str(condition_id), sess = sess)


def markets(next_cursor: str | None = None,
            sess: requests.Session | None = None) -> dict[str, Any]:
    """One page of markets. Feed `next_cursor` back in until it is "LTE="."""
    params = {"next_cursor": next_cursor} if next_cursor else None
    return _get("markets", params, sess = sess)


def market_by_token(token_id: str,
                    sess: requests.Session | None = None) -> dict[str, Any]:
    """The condition id and token pair a token belongs to."""
    return _get("markets-by-token/" + str(token_id), sess = sess)


def tick_size(token_id: str, sess: requests.Session | None = None) -> float:
    """Smallest price increment the token's book accepts."""
    payload = _get("tick-size", {"token_id": str(token_id)}, sess = sess)
    return float(payload["minimum_tick_size"])


def neg_risk(token_id: str, sess: requests.Session | None = None) -> bool:
    """Whether the token belongs to a negative-risk (multi-outcome) market."""
    payload = _get("neg-risk", {"token_id": str(token_id)}, sess = sess)
    return bool(payload["neg_risk"])


def server_time(sess: requests.Session | None = None) -> int:
    """CLOB clock as unix seconds. Useful for diagnosing signature timestamps."""
    return int(_get("time", sess = sess))


def ok(sess: requests.Session | None = None) -> bool:
    """Whether the CLOB is up."""
    try:
        return _get("ok", sess = sess) == "OK"
    except PolymarketAPIError:
        return False


###### Authenticated reads (L2 credentials)


def _auth_get(path: str,
              params: dict[str, Any] | None = None,
              sess: requests.Session | None = None) -> Any:
    from . import _auth

    return _client.get_json(path, params = params, base = clob_base, sess = sess,
                            headers = _auth.l2_headers("GET", "/" + path.lstrip("/")))


def open_orders(market: str | None = None,
                asset_id: str | None = None,
                sess: requests.Session | None = None) -> Any:
    """Your own resting orders, optionally filtered by condition or token id."""
    params = {k: v for k, v in (("market", market), ("asset_id", asset_id))
              if v is not None}
    return _auth_get("data/orders", params or None, sess = sess)


def user_trades(market: str | None = None,
                asset_id: str | None = None,
                sess: requests.Session | None = None) -> Any:
    """Your own fills. The CLOB has no public trade feed; for everyone's trades
    use the separate Data API at data-api.polymarket.com."""
    params = {k: v for k, v in (("market", market), ("asset_id", asset_id))
              if v is not None}
    return _auth_get("data/trades", params or None, sess = sess)


def balance_allowance(asset_type: str = "COLLATERAL",
                      token_id: str | None = None,
                      sess: requests.Session | None = None) -> Any:
    """Your balance and allowance for USDC ("COLLATERAL") or one outcome token
    ("CONDITIONAL", with token_id)."""
    params: dict[str, Any] = {"asset_type": asset_type}
    if token_id is not None:
        params["token_id"] = str(token_id)
    return _auth_get("balance-allowance", params, sess = sess)


def order_scoring(order_id: str, sess: requests.Session | None = None) -> Any:
    """Whether one of your orders is currently earning liquidity rewards."""
    return _auth_get("order-scoring", {"order_id": str(order_id)}, sess = sess)


def derive_api_key(nonce: int = 0, sess: requests.Session | None = None) -> Any:
    """Re-derive the L2 credentials for a wallet. Needs L1 (wallet) signing."""
    from . import _auth

    return _client.get_json("auth/derive-api-key", base = clob_base, sess = sess,
                            headers = _auth.l1_headers(nonce = nonce))


def create_api_key(nonce: int = 0, sess: requests.Session | None = None) -> Any:
    """Mint fresh L2 credentials for a wallet. Needs L1 (wallet) signing."""
    from . import _auth

    headers = _auth.l1_headers(nonce = nonce)
    if sess is not None:
        return _client.post_json("auth/api-key", base = clob_base, sess = sess,
                                 headers = headers)

    # Minting a key is not a read, so it must not go through the shared
    # session, whose retry policy replays POSTs.
    with _client.new_session(retries = 0) as one_shot:
        return _client.post_json("auth/api-key", base = clob_base,
                                 sess = one_shot, headers = headers)
