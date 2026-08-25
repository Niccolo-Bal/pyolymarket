<p align="center"> <img src="media/pyoly-logo.png" width="100" height="82" /> </p> 

<br>

# Pyolymarket 

Pyolymarket is a Python wrapper for Polymarket's public APIs. It loads events and markets as objects you can inspect and refresh, reads live order books and price history from the CLOB, searches with either Polymarket's own index or embedding similarity, and optionally caches the catalogue on disk.

## Setup

For installation run:

```shell
pip install pyolymarket
```

The core install only needs `requests`, and covers the Gamma API, the CLOB and native search. The heavier features live behind extras:

| Extra | Pulls in | Enables |
| --- | --- | --- |
| `cache` | pandas, numpy, pyarrow, alive-progress | `cacher`, DataFrame output |
| `embeddings` | numpy, openai | `embed`, `embedded_search_event` |
| `search` | ddgs | `ddg_search_event` |
| `trading` | eth-account | deriving CLOB API credentials |
| `all` | all of the above | everything |

```shell
pip install pyolymarket[all]
```

Asking for a feature without its extra raises an error naming the one to install, so nothing fails silently.

Embedding search needs an OpenAI-compatible key in `PYOLY_OPENAI_API_KEY`. To keep embeddings local instead, point `config.embedding_base_url` at any OpenAI-compatible server (Ollama, LM Studio, vLLM, llama.cpp) and set `config.embedding_model`.

### If forking
Clone this repo, then build with: 

```shell
python -m build 
```

And install the library directly through the wheel:

```shell
pip install dist/<.whl file>
```

## Example Usage

### Search

Polymarket's own search index is the fastest way in. Results arrive with their markets nested, so one request gets you all the way to tradeable tokens:

```python
import pyolymarket as pyoly

event = pyoly.polymarket_search_event("fed rate cut", results = 1)
for market in event.markets:
    print(market.data["question"], market.outcome_prices)
```

`polymarket_search` returns the raw response instead, including `pagination` with `hasMore` and `totalResults`. Both accept the endpoint's filters (`events_tag`, `events_status`, `sort`, `ascending`, `page`, `keep_closed_markets`, `exclude_tag_id`, `search_tags`, `search_profiles`). Unrecognized names raise rather than being silently ignored by the API.

There are two other searches: `ddg_search_event` goes through DuckDuckGo, and `embedded_search_event` ranks the local cache by embedding similarity.

### Order books and prices

A market's `conditionId` and per-outcome token ids are the join between Gamma and the CLOB. The `Market` methods handle that mapping, so outcomes can be named rather than indexed:

```python
market = event.markets[0]

market.midpoint("Yes")          # 0.8645
market.spread("Yes")
market.price("BUY", "Yes")
market.book("Yes")              # bids and asks, sorted best-first
market.price_history("Yes", interval = "1d", fidelity = 60, as_frame = True)
```

The `clob` module exposes the same endpoints directly, keyed by token id, plus batch forms that quote many tokens in one request:

```python
from pyolymarket import clob

clob.midpoints(market.token_ids)
clob.books(market.token_ids)
clob.batch_price_history(market.token_ids, interval = "1w", fidelity = 60)
clob.tick_size(market.token_id("Yes"))
```

Authenticated reads (`open_orders`, `user_trades`, `balance_allowance`, `order_scoring`) need L2 credentials in `PYOLY_CLOB_API_KEY`, `PYOLY_CLOB_SECRET`, `PYOLY_CLOB_PASSPHRASE` and `PYOLY_CLOB_ADDRESS`. Derive them once from a wallet key in `PYOLY_CLOB_PRIVATE_KEY` with `clob.derive_api_key()`. Order placement is not implemented.

### Caching

The cache stores open events and markets on disk, along with the event embeddings that `embedded_search_event` ranks against:

```python
pyoly.config.caching = True     # or "csv" / "npy" to choose what gets written
pyoly.cacher.cache()
pyoly.embedded_search_event("who wins the election", results = 5)
```

Note that `cache()` walks Polymarket's entire open catalogue, so the first run takes a while.

## Version

Currently in `v0.1.2`.

Most recent changes:
- Added the CLOB API: order books, quotes, price history, market parameters
- Added native API search
- Split heavy dependencies into extras; the core install is now `requests` only
- Local embedding model support via any OpenAI-compatible endpoint
- Bug fixes

## Documentation

### [GitPages documentation](tbd) - In progress

### [PyPi web page](https://pypi.org/project/pyolymarket/)
