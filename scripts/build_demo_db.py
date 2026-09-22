#!/usr/bin/env python
"""Build data/sample/demo.db: a small, self-contained SQLite snapshot (the committed
300-review sample, already classified, plus any pain points built for those apps) so
Demo Mode (DEMO_MODE=true) never needs to call the Gemini API."""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reviewlens import db  # noqa: E402

SAMPLE_CSV = Path(__file__).resolve().parent.parent / "data" / "sample" / "reviews_sample.csv"
DEMO_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "sample" / "demo.db"


def main() -> None:
    if not SAMPLE_CSV.exists():
        print(f"{SAMPLE_CSV} not found - run scripts/make_sample.py first.")
        return

    sample = pd.read_csv(SAMPLE_CSV)
    sample_ids = set(sample["id"].tolist())
    app_ids = set(sample["app_id"].tolist())

    DEMO_DB_PATH.unlink(missing_ok=True)
    db.init_db(DEMO_DB_PATH)

    # Insert with explicit ids (not upsert_reviews' autoincrement path) so
    # classifications/pain_points' review_id references stay valid.
    with db.get_connection() as main_conn:
        full_reviews = pd.read_sql(
            f"SELECT * FROM reviews WHERE id IN ({','.join('?' * len(sample_ids))})",
            main_conn, params=list(sample_ids),
        )
    with db.get_connection(DEMO_DB_PATH) as demo_conn:
        full_reviews.to_sql("reviews", demo_conn, if_exists="append", index=False)

    with db.get_connection() as main_conn:
        classifications = pd.read_sql("SELECT * FROM classifications", main_conn)
        pain_points = pd.read_sql("SELECT * FROM pain_points", main_conn)

    demo_classifications = classifications[classifications["review_id"].isin(sample_ids)]
    demo_pain_points = pain_points[pain_points["app_id"].isin(app_ids)]

    with db.get_connection(DEMO_DB_PATH) as demo_conn:
        demo_classifications.drop(columns=["id"]).to_sql("classifications", demo_conn, if_exists="append", index=False)
        demo_pain_points.drop(columns=["id"]).to_sql("pain_points", demo_conn, if_exists="append", index=False)

    print(f"Built {DEMO_DB_PATH}: {len(sample)} reviews, {len(demo_classifications)} classifications, {len(demo_pain_points)} pain points.")


if __name__ == "__main__":
    main()
