import os
import subprocess
import sys
from pathlib import Path

import pytest

import pyolymarket as pyoly

SRC = str(Path(__file__).resolve().parents[1] / "src")


def test_public_names_resolve():
    pytest.importorskip("pandas")

    for name in pyoly.__all__:
        assert getattr(pyoly, name) is not None
    assert set(pyoly.__all__) <= set(dir(pyoly))


def test_unknown_attribute():
    with pytest.raises(AttributeError):
        pyoly.does_not_exist


def test_import_is_cheap_and_has_no_side_effects(tmp_path):
    """Importing the package, and the cacher with caching on, must not load the
    optional modules, reach the network or write to disk."""
    code = (
        "import sys, os\n"
        "import pyolymarket as pyoly\n"
        "lazy = [m for m in ('pyolymarket.cacher', 'pyolymarket.clob', "
        "'pyolymarket.utils', 'pyolymarket.embedding_logic', 'pandas', 'openai') "
        "if m in sys.modules]\n"
        "assert not lazy, lazy\n"
        "pyoly.config.caching = True\n"
        "try:\n"
        "    import pandas\n"
        "except ImportError:\n"
        "    sys.exit(0)\n"
        "from pyolymarket import cacher\n"
        "assert cacher.events is None and cacher.markets is None\n"
        "assert os.listdir('.') == [], os.listdir('.')\n"
    )
    env = {**os.environ, "PYTHONPATH": SRC, "PYTHONDONTWRITEBYTECODE": "1"}

    result = subprocess.run([sys.executable, "-c", code], cwd = tmp_path, env = env,
                            capture_output = True, text = True)

    assert result.returncode == 0, result.stderr
