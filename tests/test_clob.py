import pytest

from pyolymarket import _client, clob


def test_book_is_sorted_best_first(api):
    api.add("book", {"bids": [{"price": "0.10", "size": "1"}, {"price": "0.12", "size": "1"}],
                     "asks": [{"price": "0.20", "size": "1"}, {"price": "0.13", "size": "1"}]})

    book = clob.book("111")

    assert [level["price"] for level in book["bids"]] == ["0.12", "0.10"]
    assert [level["price"] for level in book["asks"]] == ["0.13", "0.20"]
    assert api.calls[0]["arg"] == {"token_id": "111"}
    assert api.calls[0]["base"] == _client.CLOB


def test_books_posts_token_bodies(api):
    api.add("books", [{"bids": [], "asks": []}])

    clob.books(["111", 222])

    assert api.calls[0]["method"] == "POST"
    assert api.calls[0]["arg"] == [{"token_id": "111"}, {"token_id": "222"}]


def test_single_quotes_parse_to_floats(api):
    api.add("price", {"price": "0.012"})
    api.add("midpoint", {"mid": "0.0125"})
    api.add("spread", {"spread": "0.001"})
    api.add("tick-size", {"minimum_tick_size": 0.001})
    api.add("neg-risk", {"neg_risk": False})

    assert clob.price("111", "sell") == 0.012
    assert api.calls[0]["arg"] == {"token_id": "111", "side": "SELL"}
    assert clob.midpoint("111") == 0.0125
    assert clob.spread("111") == 0.001
    assert clob.tick_size("111") == 0.001
    assert clob.neg_risk("111") is False


def test_unknown_side_is_rejected_before_any_request(api):
    with pytest.raises(ValueError, match = "BUY"):
        clob.price("111", "buy sideways")
    with pytest.raises(ValueError):
        clob.prices([("111", "HOLD")])

    assert api.calls == []


def test_prices_batch_body(api):
    api.add("prices", {})

    clob.prices([("111", "buy"), ("222", "SELL")])

    assert api.calls[0]["arg"] == [{"token_id": "111", "side": "BUY"},
                                   {"token_id": "222", "side": "SELL"}]


def test_price_history_parameters(api):
    api.add("prices-history", {"history": [{"t": 1, "p": 0.5}]})

    assert clob.price_history("111", interval = "1w", fidelity = 60) == [{"t": 1, "p": 0.5}]
    clob.price_history("111", start_ts = 10, end_ts = 20)

    assert api.calls[0]["arg"] == {"market": "111", "interval": "1w", "fidelity": 60}
    assert api.calls[1]["arg"] == {"market": "111", "startTs": 10, "endTs": 20}


def test_price_history_validation(api):
    with pytest.raises(ValueError, match = "interval"):
        clob.price_history("111")
    with pytest.raises(ValueError, match = "interval"):
        clob.price_history("111", start_ts = 10)
    with pytest.raises(ValueError, match = "Unrecognized interval"):
        clob.price_history("111", interval = "2d")

    assert api.calls == []


def test_price_history_missing_history_key(api):
    api.add("prices-history", {})

    assert clob.price_history("111", interval = "1d") == []


def test_price_history_as_frame(api):
    pd = pytest.importorskip("pandas")
    api.add("prices-history", {"history": [{"t": 1700000000, "p": 0.5},
                                           {"t": 1700000060, "p": 0.6}]})

    frame = clob.price_history("111", interval = "1d", as_frame = True)

    assert list(frame.columns) == ["price"]
    assert list(frame["price"]) == [0.5, 0.6]
    assert frame.index[0] == pd.Timestamp(1700000000, unit = "s", tz = "UTC")


def test_batch_price_history(api):
    api.add("batch-prices-history", {"history": {"111": []}})

    assert clob.batch_price_history(["111"], interval = "1d", start_ts = 5) == {"111": []}
    assert api.calls[0]["arg"] == {"markets": ["111"], "interval": "1d", "start_ts": 5}

    with pytest.raises(ValueError, match = "at most 20"):
        clob.batch_price_history([str(i) for i in range(21)], interval = "1d")


def test_markets_cursor(api):
    api.add("markets", {"data": [], "next_cursor": "LTE="})

    clob.markets()
    clob.markets(next_cursor = "abc")

    assert api.calls[0]["arg"] is None
    assert api.calls[1]["arg"] == {"next_cursor": "abc"}


def test_ok_swallows_api_errors(api):
    assert clob.ok() is False

    api.add("ok", "OK")
    assert clob.ok() is True


def test_server_time(api):
    api.add("time", 1787698109)

    assert clob.server_time() == 1787698109


def test_authenticated_reads_sign_the_path_without_the_query(api, monkeypatch):
    from pyolymarket import _auth

    signed = []
    monkeypatch.setattr(_auth, "l2_headers",
                        lambda method, path, **kw: signed.append((method, path)) or {"H": "1"})
    api.add("data/orders", [])
    api.add("balance-allowance", {})

    clob.open_orders(market = "0xabc")
    clob.balance_allowance(token_id = 111)

    assert signed == [("GET", "/data/orders"), ("GET", "/balance-allowance")]
    assert api.calls[0]["arg"] == {"market": "0xabc"}
    assert api.calls[0]["headers"] == {"H": "1"}
    assert api.calls[1]["arg"] == {"asset_type": "COLLATERAL", "token_id": "111"}


def test_create_api_key_is_never_retried(api, monkeypatch):
    from pyolymarket import _auth

    monkeypatch.setattr(_auth, "l1_headers", lambda nonce = 0: {"POLY_NONCE": str(nonce)})
    api.add("auth/api-key", {"apiKey": "k"})

    assert clob.create_api_key(nonce = 3) == {"apiKey": "k"}

    call = api.calls[0]
    assert call["method"] == "POST"
    assert call["headers"] == {"POLY_NONCE": "3"}
    assert call["sess"].get_adapter("https://").max_retries.total == 0
