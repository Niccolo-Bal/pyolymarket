import logging
import numpy as np
from .config import config
from openai import OpenAI

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

_OpenAIClient = None
_client_settings = None


def _openai() -> OpenAI:
    # Initializing like this lets you import the file without
    # throwing for invalid key/url when not caching
    global _OpenAIClient, _client_settings

    settings = (config.embedding_base_url, config.emb_api_key)
    if _OpenAIClient is None or _client_settings != settings:
        base_url, emb_api_key = settings
        _OpenAIClient = OpenAI(emb_api_key = emb_api_key, base_url = base_url)
        _client_settings = settings
    return _OpenAIClient


def embed(text : str, model : str | None = None) -> list[np.ndarray[float]]:

    if text == "": return None

    return _openai().embeddings.create(
        input = text, model = model or config.embedding_model
    ).data[0].embedding

def _embed_list_help(texts, model):
    
    model = model or config.embedding_model

    if len(texts) > 2048:
        vecs_1, tokens_1 = _embed_list_help(texts[:(len(texts) // 2)], model = model)
        vecs_2, tokens_2 = _embed_list_help(texts[(len(texts) // 2) :], model = model)
        return vecs_1 + vecs_2, tokens_1 + tokens_2

    response = _openai().embeddings.create(
        input = texts, model = model)

    return [item.embedding for item in response.data], response.usage.total_tokens

def embed_list(texts: list[str], model: str | None = None) -> tuple[list[np.ndarray[float]], int]:

    vecs, tokens = _embed_list_help(texts, model)

    if tokens > 10 ** 5:
            logger.warning("Large embedding request (%s tokens). Make sure this was "
                           "not sent in error.", tokens)

    return vecs, tokens

if __name__ == "__main__":
    pass
