<p align="center"> <img src="https://raw.githubusercontent.com/Niccolo-Bal/pyolymarket/HEAD/media/pyoly-logo.png" width="150" height="125" /> </p> 

<br>

# Pyolymarket 

Pyolymarket is a Python wrapper for Polymarket's public Gamma and CLOB APIs. It loads events and markets as objects you can inspect and refresh, optionally caches that catalogue on disk, and searches events by embedding similarity instead of exact keywords. The Gamma and public CLOB features need no secrets. Embedding search and the cache need an OpenAI-compatible embedding model, and the authenticated CLOB reads need CLOB credentials.

Trading through CLOB has not yet been implemented.

## Setup

For installation run:

```shell
pip install pyolymarket
```

<br>

Embedding search needs an OpenAI-compatible key in `PYOLY_OPENAI_API_KEY`.

CLOB usage requires environment variables at: `PYOLY_CLOB_API_KEY`, `PYOLY_CLOB_SECRET`,`PYOLY_CLOB_PASSPHRASE`, `PYOLY_CLOB_ADDRESS`, and `PYOLY_CLOB_PRIVATE_KEY`.

If you already have these set under other names, point the library at them with `pyoly.config`:

```python
import pyolymarket as pyoly

pyoly.config.CLOB_API_KEY_ENV = "MY_SILLY_CLOB_API_KEY"
```

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

Worked examples live in `tests/`, as four notebooks:

- `01_gamma_basics.ipynb` - events, markets and resolution through the Gamma API
- `02_clob_basics.ipynb` - quotes, order books and price history through the CLOB
- `03_embedding_basics.ipynb` - embedding search and the disk cache
- `04_utils_basics.ipynb` - finding events, and configuring the rest of the library

## Tests

The test suite is mocked end to end, so it needs no keys and no network:

```shell
pip install -e ".[all,tests]"
pytest
```

## Version

Currently in `v0.2.0`.

Most recent changes:
- The search helpers always return a list (breaking: `results = 1` used to return a bare `Event`)
- The cache lives in a `cacher.Cache` class and is no longer loaded at import
- More properties on `Event` and `Market` (volume, liquidity, tags, best bid/ask, ...)
- Supports Python 3.10+
- Test suite and bug fixes

### Releasing

The version number is read from the git tag, so a release is a tag:

```shell
git tag v0.2.0
git push origin v0.2.0
```

Pushing a `v*` tag runs `.github/workflows/publish.yml`, which tests, builds and uploads that exact commit to PyPI.

## Documentation

### [GitPages documentation](tbd) - Not yet implemented

### [PyPi web page](https://pypi.org/project/pyolymarket/)
