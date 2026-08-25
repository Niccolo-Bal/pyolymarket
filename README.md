<p align="center"> <img src="media/pyoly-logo.png" width="150" height="125" /> </p> 

<br>

# Pyolymarket 

Pyolymarket is a Python wrapper for Polymarket's public Gamma and CLOB APIs. It loads events and markets as objects you can inspect and refresh, optionally caches that catalogue on disk, and searches events by embedding similarity instead of exact keywords. All current features are available with no secretes needed, but Caching requires an OpenAI-compatable embedding model, and future CLOB trading implementation will require CLOB secretes.

## Setup

For installation run:

```shell
pip install pyolymarket
```

Embedding search needs an OpenAI-compatible key in `PYOLY_OPENAI_API_KEY`.

CLOB usage requires environment variables at: `PYOLY_CLOB_API_KEY`, `PYOLY_CLOB_SECRET`,`PYOLY_CLOB_PASSPHRASE`, `PYOLY_CLOB_ADDRESS`, and `PYOLY_CLOB_PRIVATE_KEY`.

If you already have these variables set, you can change the endpoints using `pyoly.config`:

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

## Version

Currently in `v0.1.2`.

Most recent changes:
- Added native API search
- Added CLOB parsing + integration
- Bug fixes

## Documentation

### [GitPages documentation](tbd) - In progress

### [PyPi web page](https://pypi.org/project/pyolymarket/)
