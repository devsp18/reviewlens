"""Storage helpers for human labels (Labeling Studio) and PM pain-point rankings."""

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from reviewlens.config import REPO_ROOT

QUEUE_PATH = REPO_ROOT / "data" / "labeled" / "label_queue.csv"
EVAL_SET_PATH = REPO_ROOT / "data" / "labeled" / "eval_set.csv"
PM_RANKINGS_PATH = REPO_ROOT / "data" / "labeled" / "pm_rankings.csv"

LABEL_COLUMNS = ["review_id", "theme", "sentiment", "labeler", "labeled_at"]
RANKING_COLUMNS = ["reviewer", "app_id", "pain_point", "rank", "rating", "ranked_at"]


def load_queue() -> pd.DataFrame:
    if not QUEUE_PATH.exists():
        return pd.DataFrame(columns=["id", "app_id", "app_name", "rating", "cleaned_text"])
    return pd.read_csv(QUEUE_PATH)


def load_labels(path: Path = EVAL_SET_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=LABEL_COLUMNS)
    return pd.read_csv(path)


def save_label(review_id: int, theme: str, sentiment: str, labeler: str, path: Path = EVAL_SET_PATH) -> None:
    """Upsert one human label, keyed by review_id (a re-label overwrites the prior one)."""
    labels = load_labels(path)
    labels = labels[labels["review_id"] != review_id]
    new_row = pd.DataFrame(
        [
            {
                "review_id": review_id,
                "theme": theme,
                "sentiment": sentiment,
                "labeler": labeler,
                "labeled_at": datetime.now(UTC).isoformat(),
            }
        ]
    )
    labels = pd.concat([labels, new_row], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(path, index=False)


def delete_label(review_id: int, path: Path = EVAL_SET_PATH) -> None:
    labels = load_labels(path)
    labels = labels[labels["review_id"] != review_id]
    labels.to_csv(path, index=False)


def load_rankings(path: Path = PM_RANKINGS_PATH) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=RANKING_COLUMNS)
    return pd.read_csv(path)


def save_ranking(
    reviewer: str, app_id: str, pain_point: str, rank: int, rating: int, path: Path = PM_RANKINGS_PATH
) -> None:
    """Upsert one reviewer's rank/rating for one pain point, keyed by (reviewer, app_id, pain_point)."""
    rankings = load_rankings(path)
    mask = (rankings["reviewer"] == reviewer) & (rankings["app_id"] == app_id) & (rankings["pain_point"] == pain_point)
    rankings = rankings[~mask]
    new_row = pd.DataFrame(
        [
            {
                "reviewer": reviewer,
                "app_id": app_id,
                "pain_point": pain_point,
                "rank": rank,
                "rating": rating,
                "ranked_at": datetime.now(UTC).isoformat(),
            }
        ]
    )
    rankings = pd.concat([rankings, new_row], ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    rankings.to_csv(path, index=False)
