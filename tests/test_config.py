import os

import pytest


def test_caching_accepts_bools_and_levels(clean_config):
    config = clean_config

    assert config.caching is False and config.cache_level == "off"

    config.caching = True
    assert config.caching is True and config.cache_level == "on"

    config.caching = "csv"
    assert config.caching is True and config.cache_level == "csv"

    config.caching = False
    assert config.cache_level == "off"


def test_unknown_cache_level_is_rejected(clean_config):
    with pytest.raises(ValueError):
        clean_config.caching = "sometimes"


def test_cache_dir_follows_the_working_directory(clean_config, tmp_path, monkeypatch):
    clean_config._cache_dir = None
    monkeypatch.chdir(tmp_path)

    assert clean_config.CACHE_DIR == os.path.join(os.getcwd(), "___pyolymarket_cache___")

    clean_config.CACHE_DIR = tmp_path / "elsewhere"
    assert clean_config.CACHE_DIR == str(tmp_path / "elsewhere")


def test_embedding_key(clean_config, monkeypatch):
    with pytest.raises(EnvironmentError, match = "PYOLY_OPENAI_API_KEY"):
        clean_config.emb_api_key

    # A local OpenAI-compatible server needs no key.
    clean_config.embedding_base_url = "http://localhost:11434/v1"
    assert clean_config.emb_api_key

    monkeypatch.setenv("PYOLY_OPENAI_API_KEY", "sk-test")
    assert clean_config.emb_api_key == "sk-test"


def test_clob_creds_names_every_missing_variable(clean_config, monkeypatch):
    monkeypatch.setenv("PYOLY_CLOB_API_KEY", "key")

    with pytest.raises(EnvironmentError) as raised:
        clean_config.clob_creds

    message = str(raised.value)
    assert "PYOLY_CLOB_SECRET" in message and "PYOLY_CLOB_ADDRESS" in message
    assert "PYOLY_CLOB_API_KEY" not in message


def test_clob_creds_complete(clean_config, monkeypatch):
    for name in ("API_KEY", "SECRET", "PASSPHRASE", "ADDRESS"):
        monkeypatch.setenv("PYOLY_CLOB_" + name, name.lower())

    assert clean_config.clob_creds == {"api_key": "api_key", "secret": "secret",
                                       "passphrase": "passphrase", "address": "address"}
