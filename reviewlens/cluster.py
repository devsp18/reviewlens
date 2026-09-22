"""Group negative/neutral reviews into pain points via embedding clustering, then have
Gemini write a title, summary, and suggested next step for each cluster."""

import numpy as np
import pandas as pd
from google.genai import types
from hdbscan import HDBSCAN
from pydantic import BaseModel, Field

from reviewlens import db
from reviewlens.classify import MalformedResponse, get_client
from reviewlens.config import settings
from reviewlens.gemini_retry import gemini_retry
from reviewlens.rag import embed_texts

MIN_CLUSTER_SIZE = 3


class PainPointSummary(BaseModel):
    title: str = Field(description="Short, specific pain-point title, max 8 words")
    summary: str = Field(description="1-2 sentence summary of the underlying problem")
    next_step: str = Field(description="One concrete, actionable next step for the product team")


@gemini_retry(extra_exception_types=(MalformedResponse,))
def _summarize_cluster(texts: list[str]) -> PainPointSummary:
    sample = texts[:10]
    numbered = "\n".join(f"- {t}" for t in sample)
    prompt = f"""These are negative/neutral app reviews describing the same underlying problem.

Reviews:
{numbered}

Write a short, specific pain-point title (max 8 words), a 1-2 sentence summary of the
underlying problem, and one concrete next step the product team should take."""

    client = get_client()
    resp = client.models.generate_content(
        model=settings.gemini_model,
        contents=prompt,
        config=types.GenerateContentConfig(response_mime_type="application/json", response_schema=PainPointSummary),
    )
    if resp.parsed is None:
        raise MalformedResponse(f"Could not parse pain-point summary: {resp.text[:300]!r}")
    return resp.parsed


def _weighted_severity(group: pd.DataFrame) -> float:
    """Mean severity weighted so low-star reviews count more (weight = 6 - rating)."""
    weights = 6 - group["rating"]
    if weights.sum() == 0:
        return float(group["severity"].mean())
    return float((group["severity"] * weights).sum() / weights.sum())


def _count_in_window(group: pd.DataFrame, start_days_ago: int, end_days_ago: int) -> int:
    now = pd.Timestamp.now().normalize()
    dates = pd.to_datetime(group["review_date"])
    lower = now - pd.Timedelta(days=end_days_ago)
    upper = now - pd.Timedelta(days=start_days_ago)
    return int(((dates > lower) & (dates <= upper)).sum())


def build_pain_points(
    app_id: str,
    prompt_version: str = "v1",
    min_cluster_size: int = MIN_CLUSTER_SIZE,
    db_path=None,
) -> pd.DataFrame:
    """Cluster this app's negative/neutral classified reviews into pain points, summarize
    each with Gemini, persist to the pain_points table, and return the stored rows."""
    with db.get_connection(db_path) as conn:
        pool = pd.read_sql(
            """SELECT r.id AS review_id, r.cleaned_text, r.rating, r.thumbs_up, r.review_date,
                      c.theme, c.sentiment, c.severity
               FROM classifications c JOIN reviews r ON r.id = c.review_id
               WHERE r.app_id = ? AND c.prompt_version = ? AND c.sentiment IN ('negative', 'neutral')""",
            conn,
            params=(app_id, prompt_version),
        )

    if pool.empty:
        db.store_pain_points([], app_id, prompt_version, db_path)
        return db.load_pain_points(app_id, prompt_version, db_path)

    records = []
    for theme, group in pool.groupby("theme"):
        group = group.reset_index(drop=True)
        if len(group) < min_cluster_size:
            labels = np.zeros(len(group), dtype=int)
        else:
            embeddings = np.array(embed_texts(group["cleaned_text"].tolist()))
            normalized = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            labels = HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean").fit_predict(normalized)

        group = group.assign(cluster_label=labels)
        for cluster_label, cluster_df in group.groupby("cluster_label"):
            if cluster_label == -1:
                continue  # HDBSCAN noise - not a coherent pain point
            texts = cluster_df["cleaned_text"].tolist()
            summary = _summarize_cluster(texts)
            quotes = cluster_df["cleaned_text"].sample(min(3, len(cluster_df)), random_state=42).tolist()
            records.append(
                {
                    "theme": theme,
                    "cluster_label": int(cluster_label),
                    "title": summary.title,
                    "summary": summary.summary,
                    "next_step": summary.next_step,
                    "review_count": len(cluster_df),
                    "weighted_severity": _weighted_severity(cluster_df),
                    "last_30_count": _count_in_window(cluster_df, 0, 30),
                    "prev_30_count": _count_in_window(cluster_df, 30, 60),
                    "reach_sum": int(cluster_df["thumbs_up"].sum()),
                    "example_quotes": quotes,
                    "example_review_ids": cluster_df["review_id"].tolist(),
                }
            )

    db.store_pain_points(records, app_id, prompt_version, db_path)
    return db.load_pain_points(app_id, prompt_version, db_path)
