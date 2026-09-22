"""SQLite schema and access helpers for reviews, classifications, and the LLM cache."""

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path

import pandas as pd

from reviewlens.config import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS reviews (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    app_id TEXT NOT NULL,
    app_name TEXT NOT NULL,
    review_id TEXT NOT NULL,
    user_name TEXT,
    rating INTEGER,
    thumbs_up INTEGER DEFAULT 0,
    text TEXT NOT NULL,
    cleaned_text TEXT NOT NULL,
    language TEXT,
    review_date TEXT,
    app_version TEXT,
    source TEXT NOT NULL DEFAULT 'scrape',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(app_id, review_id)
);

CREATE TABLE IF NOT EXISTS classifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id INTEGER NOT NULL REFERENCES reviews(id),
    prompt_version TEXT NOT NULL,
    theme TEXT,
    sentiment TEXT,
    severity INTEGER,
    reasoning TEXT,
    raw_response TEXT,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(review_id, prompt_version)
);

CREATE TABLE IF NOT EXISTS retrievals (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    review_id INTEGER NOT NULL REFERENCES reviews(id),
    prompt_version TEXT NOT NULL,
    retrieved_examples TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(review_id, prompt_version)
);

CREATE TABLE IF NOT EXISTS pain_points (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    app_id TEXT NOT NULL,
    prompt_version TEXT NOT NULL,
    theme TEXT NOT NULL,
    cluster_label INTEGER NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    next_step TEXT NOT NULL,
    review_count INTEGER NOT NULL,
    weighted_severity REAL NOT NULL,
    last_30_count INTEGER NOT NULL,
    prev_30_count INTEGER NOT NULL,
    reach_sum INTEGER NOT NULL,
    example_quotes TEXT NOT NULL,
    example_review_ids TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    UNIQUE(app_id, prompt_version, theme, cluster_label)
);

CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key TEXT PRIMARY KEY,
    response TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


@contextmanager
def get_connection(db_path: Path | None = None):
    path = db_path or settings.db_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        yield conn
    finally:
        conn.close()


def init_db(db_path: Path | None = None) -> None:
    with get_connection(db_path) as conn:
        conn.executescript(SCHEMA)
        _migrate(conn)
        conn.commit()


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns to already-existing tables that predate them (SQLite's
    CREATE TABLE IF NOT EXISTS won't retroactively add new columns)."""
    cols = [r[1] for r in conn.execute("PRAGMA table_info(classifications)").fetchall()]
    if "reasoning" not in cols:
        conn.execute("ALTER TABLE classifications ADD COLUMN reasoning TEXT")


def upsert_reviews(df: pd.DataFrame, db_path: Path | None = None) -> int:
    """Insert cleaned reviews, skipping any (app_id, review_id) already stored. Returns rows inserted."""
    columns = [
        "app_id", "app_name", "review_id", "user_name", "rating", "thumbs_up",
        "text", "cleaned_text", "language", "review_date", "app_version", "source",
    ]
    records = df[columns].to_dict("records")

    with get_connection(db_path) as conn:
        cur = conn.executemany(
            f"""INSERT OR IGNORE INTO reviews ({", ".join(columns)})
                VALUES ({", ".join("?" for _ in columns)})""",
            [tuple(r[c] for c in columns) for r in records],
        )
        conn.commit()
        return cur.rowcount


def review_counts_by_app(db_path: Path | None = None) -> pd.DataFrame:
    with get_connection(db_path) as conn:
        return pd.read_sql("SELECT app_id, app_name, COUNT(*) AS review_count FROM reviews GROUP BY app_id, app_name", conn)


def sample_reviews(n: int = 5, db_path: Path | None = None) -> pd.DataFrame:
    with get_connection(db_path) as conn:
        return pd.read_sql(f"SELECT * FROM reviews ORDER BY RANDOM() LIMIT {int(n)}", conn)


def all_reviews(db_path: Path | None = None) -> pd.DataFrame:
    with get_connection(db_path) as conn:
        return pd.read_sql("SELECT * FROM reviews", conn)


def store_pain_points(records: list[dict], app_id: str, prompt_version: str, db_path: Path | None = None) -> None:
    """records need: theme, cluster_label, title, summary, next_step, review_count,
    weighted_severity, last_30_count, prev_30_count, reach_sum, example_quotes (list),
    example_review_ids (list)."""
    with get_connection(db_path) as conn:
        conn.execute("DELETE FROM pain_points WHERE app_id = ? AND prompt_version = ?", (app_id, prompt_version))
        conn.executemany(
            """INSERT INTO pain_points
               (app_id, prompt_version, theme, cluster_label, title, summary, next_step,
                review_count, weighted_severity, last_30_count, prev_30_count, reach_sum,
                example_quotes, example_review_ids)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    app_id, prompt_version, r["theme"], r["cluster_label"], r["title"], r["summary"],
                    r["next_step"], r["review_count"], r["weighted_severity"], r["last_30_count"],
                    r["prev_30_count"], r["reach_sum"], json.dumps(r["example_quotes"]),
                    json.dumps(r["example_review_ids"]),
                )
                for r in records
            ],
        )
        conn.commit()


def load_pain_points(app_id: str, prompt_version: str, db_path: Path | None = None) -> pd.DataFrame:
    with get_connection(db_path) as conn:
        return pd.read_sql(
            "SELECT * FROM pain_points WHERE app_id = ? AND prompt_version = ?",
            conn, params=(app_id, prompt_version),
        )


def store_retrievals(records: list[dict], prompt_version: str, db_path: Path | None = None) -> None:
    """records: [{'review_id': int, 'examples': [...]}] - examples are JSON-serialized as-is."""
    if not records:
        return
    with get_connection(db_path) as conn:
        conn.executemany(
            """INSERT OR REPLACE INTO retrievals (review_id, prompt_version, retrieved_examples)
               VALUES (?, ?, ?)""",
            [(r["review_id"], prompt_version, json.dumps(r["examples"])) for r in records],
        )
        conn.commit()
