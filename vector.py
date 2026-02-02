# vector.py (append or modify existing file)
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os
import pandas as pd
import numpy as np

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

retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# -------------------------
# Aggregation helpers
# -------------------------
def get_shop_stats_from_df(n_top=10, min_reviews=1):
    grouped = df.groupby("Shop")["Rating"].agg(["mean", "count"]).reset_index()
    grouped = grouped[grouped["count"] >= min_reviews]
    grouped = grouped.sort_values(by=["mean", "count"], ascending=[False, False])
    results = [(row["Shop"], float(row["mean"]), int(row["count"])) for _, row in grouped.head(n_top).iterrows()]
    return results

# -------------------------
# Semantic intent detection
# -------------------------
# Labeled exemplars: add or refine these to match your dataset and phrasing.
_INTENT_EXEMPLARS = [
    ("Which shops have the best rating?", "aggregation"),
    ("Top rated shops", "aggregation"),
    ("Which restaurants are highest rated?", "aggregation"),
    ("Show me shops with most 5-star reviews", "aggregation"),
    ("Which shops have the most positive reviews?", "aggregation"),
    ("What do customers say about the pizza at X?", "other"),
    ("Summarize reviews for shop Y", "other"),
    ("Give me pros and cons of the restaurant", "other"),
    ("Find reviews mentioning delivery time", "other"),
    ("Is the pizza spicy?", "other"),
]

# Precompute exemplar embeddings
_exemplar_texts = [t for t, _ in _INTENT_EXEMPLARS]
_exemplar_labels = [lbl for _, lbl in _INTENT_EXEMPLARS]

try:
    _exemplar_embeddings = embeddings.embed_documents(_exemplar_texts)
except Exception:
    # fallback if embed_documents not available
    _exemplar_embeddings = [embeddings.embed_query(t) for t in _exemplar_texts]

_exemplar_embeddings = np.array(_exemplar_embeddings)

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0.0
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def is_aggregation_semantic(question: str, threshold: float = 0.72) -> (bool, dict):
    """
    Return (is_aggregation, debug_info)
    debug_info contains top_label, top_score, and all scores if you want to log/tune.
    """
    # embed question
    try:
        q_emb = np.array(embeddings.embed_query(question))
    except Exception:
        q_emb = np.array(embeddings.embed_documents([question])[0])

    # compute similarities
    scores = [_cosine_sim(q_emb, ex) for ex in _exemplar_embeddings]
    top_idx = int(np.argmax(scores))
    top_score = scores[top_idx]
    top_label = _exemplar_labels[top_idx]

    debug = {
        "top_label": top_label,
        "top_score": top_score,
        "all_scores": list(scores),
        "threshold": threshold
    }

    is_agg = (top_label == "aggregation") and (top_score >= threshold)
    return is_agg, debug