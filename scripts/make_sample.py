#!/usr/bin/env python
"""Build the small, committed data/sample/ demo set from the full SQLite database.

Stratifies by (app, star rating) so the public demo represents every app and
sentiment range without needing to scrape anything.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reviewlens import db

SAMPLE_SIZE = 300
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "sample" / "reviews_sample.csv"


def main() -> None:
    all_reviews = db.all_reviews()
    groups = list(all_reviews.groupby(["app_id", "rating"]))
    per_group = max(1, SAMPLE_SIZE // len(groups))

    parts = [g.sample(n=min(per_group, len(g)), random_state=42) for _, g in groups]
    sample = pd.concat(parts, ignore_index=True)
    sample = sample.sample(n=min(SAMPLE_SIZE, len(sample)), random_state=42).sort_values(["app_id", "review_date"])

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(sample)} rows to {OUT_PATH}")
    print(sample.groupby(["app_id", "rating"]).size())


if __name__ == "__main__":
    main()
