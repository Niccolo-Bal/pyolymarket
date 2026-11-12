import base64
import hashlib
import hmac

from pyolymarket import _auth

CREDS = {"api_key": "key", "secret": base64.urlsafe_b64encode(b"secret-bytes").decode(),
         "passphrase": "pass", "address": "0xabc"}


def _expected(message):
    digest = hmac.new(b"secret-bytes", message.encode(), hashlib.sha256).digest()
    return base64.urlsafe_b64encode(digest).decode()


def test_l2_headers(monkeypatch):
    monkeypatch.setattr(_auth.time, "time", lambda: 1700000000.9)

    headers = _auth.l2_headers("get", "/data/orders", creds = CREDS)

    assert headers == {
        "POLY_API_KEY": "key",
        "POLY_ADDRESS": "0xabc",
        "POLY_SIGNATURE": _expected("1700000000GET/data/orders"),
        "POLY_PASSPHRASE": "pass",
        "POLY_TIMESTAMP": "1700000000",
    }


def test_l2_headers_sign_the_body(monkeypatch):
    monkeypatch.setattr(_auth.time, "time", lambda: 1700000000)
    body = _auth.serialize_body({"a": 1, "b": [1, 2]})

    headers = _auth.l2_headers("POST", "/order", body = body, creds = CREDS)

    assert body == '{"a":1,"b":[1,2]}'
    assert headers["POLY_SIGNATURE"] == _expected("1700000000POST/order" + body)
