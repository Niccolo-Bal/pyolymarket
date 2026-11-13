from datetime import datetime, timezone

import pytest

import pyolymarket as pyoly
from pyolymarket import Event, Market
from pyolymarket.polyclasses import _json_field, _parse_time


###### helpers


def test_parse_time_normalizes_to_aware_utc():
    expected = datetime(2024, 11, 5, 12, tzinfo = timezone.utc)

    assert _parse_time("2024-11-05T12:00:00Z") == expected
    assert _parse_time("2024-11-05T12:00:00") == expected
    assert _parse_time(None) is None
    assert _parse_time("not a date") is None


def test_json_field_decodes_strings_and_passes_lists():
    assert _json_field({"k": '["a", "b"]'}, "k") == ["a", "b"]
    assert _json_field({"k": ["a"]}, "k") == ["a"]
    assert _json_field({}, "k") == []
    assert _json_field({"k": "not json"}, "k") == []


###### Event


def test_event_by_id_and_by_slug(api, make_event):
    api.add("events/900", make_event())
    api.add("events/slug/rain-event", make_event())

    by_id = Event(id = "900")
    by_slug = Event(slug = "rain-event")

    assert (by_id.id, by_id.slug) == ("900", "rain-event")
    assert (by_slug.id, by_slug.slug) == ("900", "rain-event")
    assert api.paths() == ["events/900", "events/slug/rain-event"]


def test_event_needs_an_id_or_slug():
    with pytest.raises(ValueError):
        Event()


def test_missing_event_raises_not_found(api):
    with pytest.raises(pyoly.PolymarketNotFoundError):
        Event(slug = "nope")


def test_event_without_slug_in_payload_raises_api_error(api):
    api.add("events/1", {"id": "1"})

    with pytest.raises(pyoly.PolymarketAPIError, match = "no slug"):
        Event(id = "1")


def test_event_properties(make_event):
    event = Event.from_data(make_event(
        startDate = "2029-01-01T00:00:00Z", active = True, closed = False,
        volume = 1234.5, liquidity = "99.5",
        tags = [{"label": "Weather", "slug": "weather"}, {"slug": "rain"}]))

    assert event.title == "Rain event"
    assert event.description.startswith("Resolves Yes")
    assert event.start_date == datetime(2029, 1, 1, tzinfo = timezone.utc)
    assert event.end_date == datetime(2030, 1, 1, tzinfo = timezone.utc)
    assert event.active is True and event.closed is False
    assert event.volume == 1234.5 and event.liquidity == 99.5
    assert event.tags == ["Weather", "rain"]
    assert event.url == "https://polymarket.com/event/rain-event"


def test_event_properties_tolerate_missing_fields():
    event = Event.from_data({"id": "1", "slug": "bare"})

    assert event.title is None and event.end_date is None
    assert event.active is None and event.volume is None
    assert event.tags == [] and event.markets == []


def test_event_markets_are_built_without_requests(api, make_event):
    event = Event.from_data(make_event())

    markets = event.markets

    assert [m.slug for m in markets] == ["will-it-rain"]
    assert markets[0].event_id == "900"
    assert event.markets is markets
    assert api.calls == []


def test_event_refresh_refetches_and_resets_markets(api, make_event):
    event = Event.from_data(make_event())
    event.markets
    api.add("events/900", make_event(title = "Renamed", markets = []))

    event.refresh()

    assert event.title == "Renamed" and event.markets == []


def test_event_to_frame(make_event):
    pytest.importorskip("pandas")

    frame = Event.from_data(make_event()).to_frame()

    assert list(frame["slug"]) == ["will-it-rain"]


@pytest.mark.parametrize("data, expected", [
    ({"description": "First line\nSecond"}, "First line"),
    ({"description": "\nBody only"}, "Body only"),
    ({"description": "", "title": "The title"}, "The title"),
    ({"description": None, "title": None}, "the-slug"),
])
def test_vectorize_text_fallback(monkeypatch, data, expected):
    from pyolymarket import embedding_logic

    monkeypatch.setattr(embedding_logic, "embed", lambda text, model = None: [text])
    event = Event.from_data({"id": "1", "slug": "the-slug", **data})

    assert event.vectorize() == [expected]
    assert event.vec == [expected]


def test_vectorize_not_in_place(monkeypatch):
    from pyolymarket import embedding_logic

    monkeypatch.setattr(embedding_logic, "embed", lambda text, model = None: [1.0])
    event = Event.from_data({"id": "1", "slug": "s", "description": "d"})

    assert event.vectorize(in_place = False) == [1.0]
    assert event.vec is None


###### Market


def test_market_by_id_and_by_slug(api, make_market):
    api.add("markets/500", make_market(events = [{"id": "900"}]))
    api.add("markets/slug/will-it-rain", make_market())

    by_id = Market(id = "500")
    by_slug = Market(slug = "will-it-rain")

    assert (by_id.slug, by_id.event_id) == ("will-it-rain", "900")
    assert (by_slug.id, by_slug.event_id) == ("500", None)


def test_market_needs_an_id_or_slug():
    with pytest.raises(ValueError):
        Market()


def test_market_without_slug_in_payload_raises_api_error(api):
    api.add("markets/5", {"id": "5"})

    with pytest.raises(pyoly.PolymarketAPIError, match = "no slug"):
        Market(id = "5")


def test_market_properties(make_market):
    market = Market.from_data(make_market(
        description = "Rules.", startDate = "2029-01-01T00:00:00Z",
        acceptingOrders = True, volume = "10.5", volumeNum = 10.5,
        liquidity = "3", bestBid = 0.24, bestAsk = 0.26, lastTradePrice = 0.25))

    assert market.question == "Will it rain?"
    assert market.description == "Rules."
    assert market.condition_id == "0xabc"
    assert market.outcomes == ["Yes", "No"]
    assert market.outcome_prices == [0.25, 0.75]
    assert market.token_ids == ["111", "222"]
    assert market.end_date == datetime(2030, 1, 1, tzinfo = timezone.utc)
    assert market.active is True and market.closed is False
    assert market.accepting_orders is True
    assert market.volume == 10.5 and market.liquidity == 3.0
    assert (market.best_bid, market.best_ask, market.last_trade_price) == (0.24, 0.26, 0.25)


def test_market_properties_tolerate_missing_fields():
    market = Market.from_data({"id": "1", "slug": "bare"})

    assert market.question is None and market.volume is None
    assert market.best_bid is None and market.accepting_orders is None
    assert market.outcomes == [] and market.outcome_prices == []


def test_token_id_by_label_and_index(make_market):
    market = Market.from_data(make_market())

    assert market.token_id("Yes") == "111"
    assert market.token_id("no") == "222"
    assert market.token_id(1) == "222"
    assert market.token_id() == "111"

    with pytest.raises(ValueError, match = "Yes, No"):
        market.token_id("Maybe")


def test_token_id_without_tokens_raises(make_market):
    market = Market.from_data(make_market(clobTokenIds = None))

    with pytest.raises(pyoly.PolymarketAPIError):
        market.token_id("Yes")


@pytest.mark.parametrize("fields, expected", [
    # The closed flag wins over the end date, in both directions.
    ({"closed": True, "endDate": "2999-01-01T00:00:00Z"}, True),
    ({"closed": False, "endDate": "2000-01-01T00:00:00Z"}, False),
    # Without the flag, fall back to the end date.
    ({"closed": None, "endDate": "2000-01-01T00:00:00Z"}, True),
    ({"closed": None, "endDate": "2999-01-01T00:00:00Z"}, False),
    ({"closed": None, "endDate": None}, False),
])
def test_resolved(make_market, fields, expected):
    assert Market.from_data(make_market(**fields)).resolved is expected


def test_resolution_is_pending_while_open(api, make_market):
    assert Market.from_data(make_market()).resolution == "pending"
    assert api.calls == []


def test_resolution_reports_and_remembers_the_oracle_status(api, make_market):
    market = Market.from_data(make_market(closed = True))
    api.add("markets/500", make_market(closed = True, umaResolutionStatus = "resolved"))

    assert market.resolution == "resolved"
    assert market.resolution == "resolved"
    assert api.paths() == ["markets/500"]


def test_resolution_unresolved_without_oracle_status(api, make_market):
    market = Market.from_data(make_market(closed = True))
    api.add("markets/500", make_market(closed = True))

    assert market.resolution == "unresolved"


def test_market_quotes_go_to_the_clob_with_the_right_token(api, make_market):
    market = Market.from_data(make_market())
    api.add("midpoint", {"mid": "0.25"})
    api.add("spread", {"spread": "0.02"})
    api.add("price", {"price": "0.24"})
    api.add("book", {"bids": [], "asks": []})

    assert market.midpoint("No") == 0.25
    assert market.spread() == 0.02
    assert market.price("buy", "Yes") == 0.24
    assert market.book("Yes") == {"bids": [], "asks": []}

    assert [call["arg"]["token_id"] for call in api.calls] == ["222", "111", "111", "111"]
    assert api.calls[2]["arg"]["side"] == "BUY"


def test_market_price_history_drops_interval_for_explicit_range(api, make_market):
    market = Market.from_data(make_market())
    api.add("prices-history", {"history": [{"t": 1, "p": 0.5}]})

    market.price_history()
    market.price_history(start_ts = 10, end_ts = 20)

    assert api.calls[0]["arg"] == {"market": "111", "interval": "1d"}
    assert api.calls[1]["arg"] == {"market": "111", "startTs": 10, "endTs": 20}
