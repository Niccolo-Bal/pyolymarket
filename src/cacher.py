import embedding_logic as emb
import polyclasses as poly
import asyncio, requests, json, os, logging, re
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from alive_progress import alive_bar
from config import config

# Final child file before Bot_Logic/main.py. Contains functions for caching data and searching for events.

config.caching = True

def cacheMarkets(log: bool = False):

    if log:
        logging.basicConfig(level = logging.INFO, force = True)

    logger.info("Starting cache sequence")

    markets = []
    params = {"limit" : 100, "end_date_min" : datetime.today().strftime("%Y-%m-%d")}
    next_cursor = None

    with alive_bar(title = "Pulling Market Data", disable = not log) as bar: 
        with requests.Session() as session:
            logger.info("Established connection to gamma API.")
            while True:

                if next_cursor:
                    params["after_cursor"] = next_cursor

                response = session.get("https://gamma-api.polymarket.com/markets/keyset", 
                                        params = params)
                batch = response.json()
                
                if "type" in batch:
                    raise ValueError(f"Error in API request: {batch["type"]}")
                    
                markets.extend(batch["markets"])

                next_cursor = batch.get("next_cursor")
                
                if not next_cursor or len(batch["markets"]) < 100:
                    bar(len(batch["markets"]))
                    break

                bar(100)   


    markets_df = pd.DataFrame(markets)[
        ["id", "question", "slug", "events"]]
    pulled_events_df = pd.json_normalize(markets_df["events"].str[0])[
        ["id", "slug", "title", "description", "endDate", "seriesSlug"]]
    markets_df = markets_df.join(pulled_events_df["id"], rsuffix=".event")
    markets_df.rename(columns = {"id" : "market_id", "id.event" : "event_id"})
    pulled_events_df.drop_duplicates(subset = ["id"], inplace = True)

    ## Events Logic
    # Compare to already cached events, remove old and embed new
    try: 
        cached_events = events
    except NameError:
        cached_events = pd.DataFrame(columns = ["id"])
    
    cached_events = pd.merge(cached_events, pulled_events_df["id"], on = "id", how = "inner")
    new_events = pulled_events_df[~pulled_events_df["id"].isin(cached_events["id"])]

    # Embed new events by first line of description; use same vector for same first-line description (ex, crypto up-down markets)
    new_events["shortenedDesc"] = new_events["description"].str.split("\n").str[0]
    new_events["shortenedDesc"] = np.where(new_events["shortenedDesc"] == "", new_events["description"], new_events["shortenedDesc"])
    new_events["uniqueDescription"] = ~new_events.duplicated(subset = ["shortenedDesc"])

    embedding_df = new_events[["shortenedDesc"]].drop_duplicates()
    embedding_list = embedding_df["shortenedDesc"].to_list()
    vectorized_list, tokens = emb.embedList(embedding_list) 
    logger.info(f"Succesfully embeded new events. Tokens used: {tokens}")
    embedding_df["embeddedVector"] = vectorized_list

    new_events = pd.merge(new_events, embedding_df, on = "shortenedDesc")
    final_events_df = pd.concat([cached_events, new_events])

    ## Markets Logic
    # TODO: Markets Logic; mirror events logic

    markets_df.to_csv("cache/markets.csv", index = False)
    markets_df.to_parquet("cache/markets.parquet", index = False)
    
    events_vecs = np.stack(final_events_df["embeddedVector"].to_numpy())
    events = final_events_df
    events.to_csv("cache/events.csv", index = False)
    events.to_parquet("cache/events.parquet", index = False)
    np.save("cache/events_vecs.npy", events_vecs)

    logger.info("Succesfully refreshed markets and events.")

    logging.basicConfig(level = logging.CRITICAL, force = True)


async def cacheLoop():
    cacheMarkets()
    await asyncio.sleep(9000)

###### File initializations

if config.caching:
    os.chdir(Path(__file__).resolve().parent)
    logging.basicConfig(level = logging.CRITICAL)
    logger = logging.getLogger(__name__)

    try:
        events = pd.read_parquet("cache/events.parquet")
        events_vecs = np.load("cache/events_vecs.npy")
    except FileNotFoundError:
        print(".npy or parquet not found. Refreshing markets.")
        cacheMarkets()
        events = pd.read_parquet("cache/events.parquet")
        events_vecs = np.load("cache/events_vecs.npy")
        print("Makets refreshed.")

######

if __name__ == "__main__":
    pass