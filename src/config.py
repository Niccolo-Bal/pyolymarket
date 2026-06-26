import os

class Config:
    API_KEY_ENV =  "OPENAI_API_KEY"
    caching = False
    

    @property
    def api_key(self):
        key = os.getenv(self.API_KEY_ENV)
        if not key:
            raise EnvironmentError(
                f"Missing env var: {self.API_KEY_ENV}. Make sure text-embedding-3-small "
                "API key is proppery set up before using embedding features. "
                "Change with pyolymarket.config.API_KEY_ENV.")
        return key

config = Config() 