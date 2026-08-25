import logging
import numpy as np
from .config import config
from openai import OpenAI

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_OpenAIClient = None
_client_settings = None


def _openai() -> OpenAI:
    """Cached client, rebuilt if the configured endpoint changes so switching
    to a local server mid-session takes effect."""
    global _OpenAIClient, _client_settings

    settings = (config.embedding_base_url, config.api_key)
    if _OpenAIClient is None or _client_settings != settings:
        base_url, api_key = settings
        _OpenAIClient = OpenAI(api_key = api_key, base_url = base_url)
        _client_settings = settings
    return _OpenAIClient


def embed(text : str, model : str | None = None) -> list[np.ndarray[float]]:

    if text == "": return None

    return _openai().embeddings.create(
        input = text, model = model or config.embedding_model
    ).data[0].embedding

def embed_list(texts : list[str], model : str | None = None) -> tuple[list[np.ndarray[float]], int]:

    model = model or config.embedding_model

    if len(texts) > 2048:
        list_1, tokens_1 = embed_list(texts[:(len(texts) // 2)], model = model)
        list_2, tokens_2 = embed_list(texts[(len(texts) // 2) :], model = model)
        return list_1 + list_2, tokens_1 + tokens_2

    response = _openai().embeddings.create(
        input = texts, model = model)
    
    if response.usage.total_tokens > 10 ** 5:
        logger.warning("Large embedding request (%s tokens). Make sure this was "
                       "not sent in error.", response.usage.total_tokens)

    return [item.embedding for item in response.data], response.usage.total_tokens

if __name__ == "__main__":
    pass
