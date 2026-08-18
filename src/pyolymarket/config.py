import os

class Config:
    API_KEY_ENV =  "OPENAI_API_KEY"
    _caching = "off"
    log = False
    CACHE_DIR = os.getcwd() + "/pyolymarket_cache"

    @property
    def api_key(self):
        key = os.getenv(self.API_KEY_ENV)
        if not key:
            raise EnvironmentError(
                f"Missing env var: {self.API_KEY_ENV}. Make sure text-embedding-3-small "
                "API key is proppery set up before using embedding features. "
                "Change with pyolymarket.config.API_KEY_ENV.")
        return key

    def set_cache_level(self, level: str):
        if level not in ["off", "on", "csv", "npy"]:
            raise ValueError(f'Unrecongnized level argument: {level}.' 
                             'level must be "off", "on", "csv" or "npy"')
        self._caching = level

config = Config() 