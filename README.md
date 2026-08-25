<p align="center"> <img src="media/pyoly-logo.png" width="100" height="82" /> </p> 

<br>

# Pyolymarket 

Pyolymarket is a Python wrapper for Polymarket's public Gamma API. It loads events and markets as objects you can inspect and refresh, optionally caches that catalogue on disk, and searches events by embedding similarity instead of exact keywords.

## Setup

For installation run:

```shell
pip install pyolymarket
```

Embedding search needs an OpenAI-compatible key in `PYOLY_OPENAI_API_KEY`.

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

Worked examples live in `tests/` 

## Version

Currently in `v0.1.2`.

Most recent changes:
- Added native API search
- Added CLOB parsing + integration
- Bug fixes

## Documentation

### [GitPages documentation](tbd) - In progress

### [PyPi web page](https://pypi.org/project/pyolymarket/)
