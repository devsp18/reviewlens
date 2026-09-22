"""Gemini-based review classification: structured output, batching, caching, retries."""

import hashlib
import importlib
import json
import time
from collections.abc import Callable

import pandas as pd
from google import genai
from google.genai import types

from reviewlens import db
from reviewlens.config import settings
from reviewlens.gemini_retry import gemini_retry
from reviewlens.schema import BatchClassification

BATCH_SIZE = 20

_client: genai.Client | None = None


class MalformedResponse(Exception):
    """Raised when Gemini's structured output doesn't parse or doesn't cover every review."""


def get_client() -> genai.Client:
    global _client
    if _client is None:
        _client = genai.Client(api_key=settings.gemini_api_key)
    return _client


def cache_key(prompt_version: str, text: str) -> str:
    return hashlib.sha256(f"{prompt_version}:{text}".encode()).hexdigest()


def get_cached(prompt_version: str, text: str, db_path=None) -> dict | None:
    key = cache_key(prompt_version, text)
    with db.get_connection(db_path) as conn:
        row = conn.execute("SELECT response FROM llm_cache WHERE cache_key = ?", (key,)).fetchone()
    return json.loads(row[0]) if row else None


def set_cached(prompt_version: str, text: str, response: dict, db_path=None) -> None:
    key = cache_key(prompt_version, text)
    with db.get_connection(db_path) as conn:
        conn.execute(
            "INSERT OR REPLACE INTO llm_cache (cache_key, response) VALUES (?, ?)",
            (key, json.dumps(response)),
        )
        conn.commit()


@gemini_retry(extra_exception_types=(MalformedResponse,))
def _call_gemini(prompt: str) -> BatchClassification:
    client = get_client()
    resp = client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=BatchClassification,
        ),
    )
    if resp.parsed is None:
        raise MalformedResponse(f"Could not parse Gemini response as BatchClassification: {resp.text[:300]!r}")
    return resp.parsed


def classify_reviews(
    df: pd.DataFrame,
    prompt_version: str,
    db_path=None,
    progress_callback: Callable[[int, int], None] | None = None,
    retriever: Callable[[str], list[dict]] | None = None,
) -> pd.DataFrame:
    """Classify reviews (needs columns: id, cleaned_text) with the given prompt version.

    Cached per-review by hash(prompt_version + text), so re-running is free for
    anything already classified. Returns review_id, theme, sentiment, severity.

    v3 requires `retriever(text) -> list[{text, theme, sentiment}]` (see
    reviewlens.rag) to fetch its few-shot examples; the cache key doesn't
    encode which examples were retrieved, so only reuse a v3 cache within a
    single, consistent retrieval context (e.g. one evaluation run).
    """
    if prompt_version == "v3" and retriever is None:
        raise ValueError("prompt v3 requires a retriever callable (see reviewlens.rag)")

    prompts_module = importlib.import_module(f"reviewlens.prompts.{prompt_version}")

    results: list[dict] = []
    unstored: list[dict] = []
    pending_ids: list[int] = []
    pending_texts: list[str] = []

    def flush_store() -> None:
        nonlocal unstored
        if not unstored:
            return
        _store_classifications(pd.DataFrame(unstored), prompt_version, db_path)
        unstored = []

    def flush_batch() -> None:
        nonlocal pending_ids, pending_texts
        if not pending_texts:
            return
        if prompt_version == "v3":
            examples = [retriever(t) for t in pending_texts]
            prompt = prompts_module.build_prompt(pending_texts, examples)
            db.store_retrievals(
                [{"review_id": rid, "examples": ex} for rid, ex in zip(pending_ids, examples)],
                prompt_version, db_path,
            )
        else:
            prompt = prompts_module.build_prompt(pending_texts)
        batch_result = _call_gemini(prompt)
        by_index = {c.review_index: c for c in batch_result.classifications}

        if set(by_index) != set(range(len(pending_texts))):
            raise MalformedResponse(
                f"Expected indices 0..{len(pending_texts) - 1}, got {sorted(by_index)}"
            )

        for i, (rid, text) in enumerate(zip(pending_ids, pending_texts)):
            c = by_index[i]
            record = {"theme": c.theme.value, "sentiment": c.sentiment.value, "severity": c.severity}
            set_cached(prompt_version, text, record, db_path)
            results.append({"review_id": rid, **record})
            unstored.append({"review_id": rid, **record})
        flush_store()
        pending_ids, pending_texts = [], []

    total = len(df)
    for done, (_, row) in enumerate(df.iterrows()):
        cached = get_cached(prompt_version, row["cleaned_text"], db_path)
        if cached:
            record = {"review_id": row["id"], **cached}
            results.append(record)
            unstored.append(record)
            if len(unstored) >= BATCH_SIZE:
                flush_store()
        else:
            pending_ids.append(row["id"])
            pending_texts.append(row["cleaned_text"])
            if len(pending_texts) >= BATCH_SIZE:
                flush_batch()
                time.sleep(0.5)
        if progress_callback:
            progress_callback(done + 1, total)
    flush_batch()
    flush_store()

    return pd.DataFrame(results, columns=["review_id", "theme", "sentiment", "severity"])


def _store_classifications(result_df: pd.DataFrame, prompt_version: str, db_path=None) -> None:
    if result_df.empty:
        return
    with db.get_connection(db_path) as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO classifications
               (review_id, prompt_version, theme, sentiment, severity, raw_response)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [
                (r["review_id"], prompt_version, r["theme"], r["sentiment"], r["severity"],
                 json.dumps({"theme": r["theme"], "sentiment": r["sentiment"], "severity": r["severity"]}))
                for r in result_df.to_dict("records")
            ],
        )
        conn.commit()
