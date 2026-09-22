"""Gemini embeddings + ChromaDB retrieval for RAG few-shot prompting.

Retrieval is built to prevent leakage during evaluation: `build_index` can
exclude a set of review_ids (a held-out k-fold) from the index entirely, so
a review can never be retrieved as its own (or its fold-mate's) example.
"""

import chromadb
import pandas as pd
from google.genai import errors as genai_errors
from google.genai import types
from sklearn.model_selection import KFold
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from reviewlens.classify import get_client
from reviewlens.config import settings

EMBED_BATCH_SIZE = 100


@retry(
    retry=retry_if_exception_type(genai_errors.ServerError),
    stop=stop_after_attempt(7),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    reraise=True,
)
def _embed_chunk(contents: list[types.Content]) -> list[list[float]]:
    client = get_client()
    resp = client.models.embed_content(model=settings.gemini_embed_model, contents=contents)
    return [e.values for e in resp.embeddings]


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed a list of documents independently (not concatenated) via Gemini."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        chunk = texts[start:start + EMBED_BATCH_SIZE]
        contents = [types.Content(parts=[types.Part(text=t)]) for t in chunk]
        vectors.extend(_embed_chunk(contents))
    return vectors


def get_chroma_client(persist_dir=None) -> chromadb.ClientAPI:
    path = persist_dir or settings.chroma_dir
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def build_index(
    labeled_df: pd.DataFrame,
    collection_name: str = "labeled_reviews",
    exclude_review_ids: set[int] | None = None,
    persist_dir=None,
) -> chromadb.Collection:
    """(Re)build a Chroma collection from labeled reviews (needs review_id, text, theme, sentiment).

    Pass exclude_review_ids to hold out a k-fold from the retrievable pool.
    """
    exclude_review_ids = exclude_review_ids or set()
    pool = labeled_df[~labeled_df["review_id"].isin(exclude_review_ids)]

    client = get_chroma_client(persist_dir)
    try:
        client.delete_collection(collection_name)
    except chromadb.errors.NotFoundError:
        pass
    collection = client.create_collection(collection_name)

    if pool.empty:
        return collection

    embeddings = embed_texts(pool["text"].tolist())
    collection.add(
        ids=[str(rid) for rid in pool["review_id"]],
        embeddings=embeddings,
        documents=pool["text"].tolist(),
        metadatas=[{"theme": t, "sentiment": s} for t, s in zip(pool["theme"], pool["sentiment"])],
    )
    return collection


def retrieve(query_text: str, collection: chromadb.Collection, k: int = 3) -> list[dict]:
    """Return up to k nearest labeled examples for a query review."""
    if collection.count() == 0:
        return []
    n_results = min(k, collection.count())
    query_embedding = embed_texts([query_text])[0]
    result = collection.query(query_embeddings=[query_embedding], n_results=n_results)

    examples = []
    for doc, meta, dist in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        examples.append({"text": doc, "theme": meta["theme"], "sentiment": meta["sentiment"], "distance": dist})
    return examples


def kfold_retrieve_all(
    labeled_df: pd.DataFrame,
    k_examples: int = 3,
    n_folds: int = 5,
    seed: int = 42,
    collection_name: str = "kfold_eval",
    persist_dir=None,
) -> dict[int, list[dict]]:
    """For every labeled review, retrieve k few-shot examples from a k-fold-held-out index -
    a review's own fold is excluded from the index it's queried against, so it (and its
    fold-mates) can never leak into its own retrieved examples.

    Returns {review_id: [examples]}.
    """
    n_folds = min(n_folds, len(labeled_df)) or 1
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    review_ids = labeled_df["review_id"].to_numpy()

    retrieved: dict[int, list[dict]] = {}
    for _, held_out_idx in kf.split(review_ids):
        held_out_ids = set(review_ids[held_out_idx].tolist())
        collection = build_index(
            labeled_df,
            collection_name=collection_name,
            exclude_review_ids=held_out_ids,
            persist_dir=persist_dir,
        )
        for idx in held_out_idx:
            row = labeled_df.iloc[idx]
            retrieved[int(row["review_id"])] = retrieve(row["text"], collection, k=k_examples)

    return retrieved
