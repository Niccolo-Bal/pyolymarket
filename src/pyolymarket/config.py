import os

CACHE_LEVELS = ("off", "on", "csv", "npy")


class Config:
    EMB_API_KEY_ENV =  "PYOLY_OPENAI_API_KEY" 

    embedding_model = "text-embedding-3-small"
    embedding_base_url = None

    CLOB_API_KEY_ENV = "PYOLY_CLOB_API_KEY"
    CLOB_SECRET_ENV = "PYOLY_CLOB_SECRET"
    CLOB_PASSPHRASE_ENV = "PYOLY_CLOB_PASSPHRASE"
    CLOB_ADDRESS_ENV = "PYOLY_CLOB_ADDRESS"
    CLOB_PRIVATE_KEY_ENV = "PYOLY_CLOB_PRIVATE_KEY"

    _caching = "off"
    log = False
    _cache_dir = None

    @property
    def emb_api_key(self):
        key = os.getenv(self.EMB_API_KEY_ENV)
        if not key:
            if self.embedding_base_url:
                # Local endpoints don't need a key
                return "foo"
            raise EnvironmentError(
                f"Missing env var: {self.EMB_API_KEY_ENV}. Make sure a "
                f"{self.embedding_model} API key is properly set up before using "
                "embedding features, or point config.embedding_base_url at a "
                "local OpenAI-compatible server. "
                "Change the variable name with pyolymarket.config.EMB_API_KEY_ENV.")
        return key

    @property
    def CACHE_DIR(self):
        """Read at access time rather than import time, so the cache lands
        relative to the working directory the caller actually has."""
        if self._cache_dir is None:
            return os.path.join(os.getcwd(), "___pyolymarket_cache___")
        return self._cache_dir

    @CACHE_DIR.setter
    def CACHE_DIR(self, path):
        self._cache_dir = str(path)

    @property
    def caching(self) -> bool:
        return self._caching != "off"

    @caching.setter
    def caching(self, value):
        """Accepts a bool for the common case, or one of the level strings."""
        if isinstance(value, bool):
            self._caching = "on" if value else "off"
        else:
            self.set_cache_level(value)

    @property
    def cache_level(self) -> str:
        return self._caching

    def set_cache_level(self, level: str):
        if level not in CACHE_LEVELS:
            raise ValueError(f'Unrecongnized level argument: {level}. '
                             f'level must be {", ".join(CACHE_LEVELS)}')
        self._caching = level

    @property
    def clob_creds(self) -> dict[str, str]:
        """L2 credentials for authenticated CLOB reads."""
        names = {"api_key": self.CLOB_API_KEY_ENV,
                 "secret": self.CLOB_SECRET_ENV,
                 "passphrase": self.CLOB_PASSPHRASE_ENV,
                 "address": self.CLOB_ADDRESS_ENV}

        creds = {field: os.getenv(env) for field, env in names.items()}
        missing = [names[field] for field, value in creds.items() if not value]
        if missing:
            raise EnvironmentError(
                f"Missing env var(s): {', '.join(missing)}. Authenticated CLOB "
                "reads need L2 credentials; derive them with "
                "pyolymarket.clob.derive_api_key().")
        return creds

    @property
    def clob_private_key(self) -> str:
        key = os.getenv(self.CLOB_PRIVATE_KEY_ENV)
        if not key:
            raise EnvironmentError(
                f"Missing env var: {self.CLOB_PRIVATE_KEY_ENV}. Wallet signing "
                "is only needed to create or derive CLOB API credentials.")
        return key


config = Config()
