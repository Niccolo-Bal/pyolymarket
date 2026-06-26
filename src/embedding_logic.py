import pandas as pd
import numpy as np
import os, json
from config import config
from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

embedder = "text-embedding-3-small"
api_key = config.api_key

OpenAIClient = OpenAI(api_key = os.getenv(api_key))

def embed(text : str) -> list[np.ndarray[float]]:

    if text == "": return None

    return OpenAIClient.embeddings.create(
        input = text, model = embedder
    ).data[0].embedding

def embedList(texts : list[str]) -> tuple[list[np.ndarray[float]], int]:

    if len(texts) > 2048:
        list_1, tokens_1 = embedList(texts[:(len(texts) // 2)])
        list_2, tokens_2 = embedList(texts[(len(texts) // 2) :])
        return list_1 + list_2, tokens_1 + tokens_2

    response = OpenAIClient.embeddings.create(
        input = texts, model = embedder)
    
    if response.usage.total_tokens > 10 ** 5:
        print(f"Warning: large embedding request ({response.usage.total_tokens}). Make sure this was not sent in error.")

    return [item.embedding for item in response.data], response.usage.total_tokens

if __name__ == "__main__":
    pass