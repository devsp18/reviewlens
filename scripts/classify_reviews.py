#!/usr/bin/env python
"""CLI to classify stored reviews with a given prompt version.

Usage:
    python scripts/classify_reviews.py --prompt-version v1 --input data/sample/reviews_sample.csv
    python scripts/classify_reviews.py --prompt-version v1 --app com.squareup.cash --limit 500
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reviewlens import db
from reviewlens.classify import classify_reviews


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt-version", required=True, help="e.g. v1, v2, v3")
    parser.add_argument("--input", help="CSV of reviews to classify (needs id, cleaned_text columns)")
    parser.add_argument("--app", help="Classify all stored reviews for this app_id instead of a CSV")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.input:
        df = pd.read_csv(args.input)
    else:
        df = db.all_reviews()
        if args.app:
            df = df[df["app_id"] == args.app]

    if args.limit:
        df = df.head(args.limit)

    print(f"Classifying {len(df)} reviews with prompt {args.prompt_version}...")
    pbar = tqdm(total=len(df))

    def progress(done: int, total: int) -> None:
        pbar.n = done
        pbar.refresh()

    result = classify_reviews(df, args.prompt_version, progress_callback=progress)
    pbar.close()

    print("\nTheme distribution:")
    print(result["theme"].value_counts().to_string())
    print("\nSentiment distribution:")
    print(result["sentiment"].value_counts().to_string())


if __name__ == "__main__":
    main()
