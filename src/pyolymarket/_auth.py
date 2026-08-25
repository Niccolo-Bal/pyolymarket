from __future__ import annotations

import base64
import hashlib
import hmac
import json
import time
from typing import Any

from .config import config

CHAIN_ID = 137
L1_MESSAGE = "This message attests that I control the given wallet"


def serialize_body(body: Any) -> str:
    return json.dumps(body, separators = (",", ":"))


def l2_headers(method: str,
               path: str,
               body: str | None = None,
               creds: dict[str, str] | None = None) -> dict[str, str]:
    """Sign a request with the L2 (API key) scheme.

    `path` is the path only; the query string is deliberately excluded from the
    signed message. `body` must already be serialized, ideally by
    serialize_body(), so the signature matches what is actually transmitted.
    """
    creds = creds or config.clob_creds
    timestamp = str(int(time.time()))

    message = timestamp + method.upper() + path
    if body is not None:
        message += body

    secret = base64.urlsafe_b64decode(creds["secret"])
    digest = hmac.new(secret, message.encode("utf-8"), hashlib.sha256).digest()

    return {
        "POLY_API_KEY": creds["api_key"],
        "POLY_ADDRESS": creds["address"],
        "POLY_SIGNATURE": base64.urlsafe_b64encode(digest).decode("utf-8"),
        "POLY_PASSPHRASE": creds["passphrase"],
        "POLY_TIMESTAMP": timestamp,
    }


def l1_headers(nonce: int = 0,
               private_key: str | None = None,
               address: str | None = None) -> dict[str, str]:
    """Sign the ClobAuth EIP-712 struct with the wallet key.

    Only needed for POST /auth/api-key and GET /auth/derive-api-key. Requires
    the optional eth-account dependency.
    """
    try:
        from eth_account import Account
        from eth_account.messages import encode_typed_data
    except ImportError as e:
        raise ImportError(
            "L1 (wallet) authentication needs eth-account. "
            "Install it with: pip install pyolymarket[trading]") from e

    private_key = private_key or config.clob_private_key
    address = address or Account.from_key(private_key).address
    timestamp = str(int(time.time()))

    typed_data = {
        "domain": {"name": "ClobAuthDomain", "version": "1", "chainId": CHAIN_ID},
        "types": {
            "ClobAuth": [
                {"name": "address", "type": "address"},
                {"name": "timestamp", "type": "string"},
                {"name": "nonce", "type": "uint256"},
                {"name": "message", "type": "string"},
            ],
        },
        "primaryType": "ClobAuth",
        "message": {
            "address": address,
            "timestamp": timestamp,
            "nonce": nonce,
            "message": L1_MESSAGE,
        },
    }

    signed = Account.sign_message(
        encode_typed_data(full_message = typed_data), private_key)

    # HexBytes.hex() stopped emitting the 0x prefix in newer releases, and the
    # server's hex decoder requires it, so prefix explicitly rather than
    # depending on which version is installed.
    signature = signed.signature.hex()
    if not signature.startswith("0x"):
        signature = "0x" + signature

    return {
        "POLY_ADDRESS": address,
        "POLY_SIGNATURE": signature,
        "POLY_TIMESTAMP": timestamp,
        "POLY_NONCE": str(nonce),
    }
