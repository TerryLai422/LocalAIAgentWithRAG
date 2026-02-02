# vector.py
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd

CSV_PATH = "realistic_restaurant_reviews2.csv"
df = pd.read_csv(CSV_PATH)

embeddings = OllamaEmbeddings(model="mxbai-embed-large", base_url="http://192.168.51.147:11434")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

if add_documents:
    documents = []
    ids = []
    
    for i, row in df.iterrows():
        document = Document(
            page_content=row["Shop"] + ": Rating[" + str(row["Rating"]) + "], " + row["Title"] + " - " + row["Review"],
            metadata={"shop": row["Shop"], "rating": row["Rating"], "date": row["Date"]},
            id=str(i)
        )
        ids.append(str(i))
        documents.append(document)
        
vector_store = Chroma(
    collection_name="restaurant_reviews",
    persist_directory=db_location,
    embedding_function=embeddings
)

if add_documents:
    vector_store.add_documents(documents=documents, ids=ids)

retriever = vector_store.as_retriever(
    search_kwargs={"k": 5}
)

# -------------------------
# Helper functions for stats
# -------------------------
def get_shop_stats_from_df(n_top=10, min_reviews=1):
    """
    Compute per-shop stats from the CSV dataframe.
    Returns a sorted list of tuples: (shop, avg_rating, count)
    """
    grouped = df.groupby("Shop")["Rating"].agg(["mean", "count"]).reset_index()
    grouped = grouped[grouped["count"] >= min_reviews]
    grouped = grouped.sort_values(by=["mean", "count"], ascending=[False, False])
    results = [(row["Shop"], float(row["mean"]), int(row["count"])) for _, row in grouped.head(n_top).iterrows()]
    return results

def get_shop_stats_from_chroma(n_top=10, min_reviews=1):
    """
    Compute per-shop stats from Chroma collection metadata.
    Useful if you don't want to read the CSV in main.py.
    """
    col = vector_store._client.get_collection("restaurant_reviews")
    # fetch all metadatas; adjust if collection API differs
    all_items = col.get(include=["metadatas", "ids"], where={})
    metadatas = all_items.get("metadatas", [])
    # metadatas is a list of dicts or list of lists depending on API; normalize:
    flat = []
    for m in metadatas:
        if isinstance(m, list):
            flat.extend(m)
        else:
            flat.append(m)
    # compute stats
    stats = {}
    for m in flat:
        shop = m.get("shop")
        rating = m.get("rating")
        if shop is None or rating is None:
            continue
        stats.setdefault(shop, []).append(float(rating))
    rows = []
    for shop, ratings in stats.items():
        avg = sum(ratings) / len(ratings)
        cnt = len(ratings)
        if cnt >= min_reviews:
            rows.append((shop, avg, cnt))
    rows.sort(key=lambda x: (x[1], x[2]), reverse=True)
    return rows[:n_top]
