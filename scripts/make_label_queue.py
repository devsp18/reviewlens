#!/usr/bin/env python
"""Build the fixed 500-review queue for the Labeling Studio.

Stratified by (app, star rating), same approach as make_sample.py but a
separate pool and larger. Deterministic (random_state=42) so re-running
never reshuffles which reviews are in the eval set - progress like
"214/500 labeled" stays meaningful across sessions.
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reviewlens import db

QUEUE_SIZE = 500
OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "labeled" / "label_queue.csv"


def main() -> None:
    all_reviews = db.all_reviews()
    groups = list(all_reviews.groupby(["app_id", "rating"]))
    per_group = max(1, QUEUE_SIZE // len(groups))

    parts = [g.sample(n=min(per_group, len(g)), random_state=42) for _, g in groups]
    queue = pd.concat(parts, ignore_index=True)
    queue = queue.sample(n=min(QUEUE_SIZE, len(queue)), random_state=42).reset_index(drop=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    queue[["id", "app_id", "app_name", "rating", "cleaned_text"]].to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(queue)} rows to {OUT_PATH}")


if __name__ == "__main__":
    main()
