#!/usr/bin/env python
"""CLI to fetch, clean, and store Play Store reviews for one app.

Usage:
    python scripts/fetch_reviews.py --app com.squareup.cash --name "Cash App" --count 10000
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reviewlens.ingest import fetch_and_store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app", required=True, help="Play Store package id, e.g. com.squareup.cash")
    parser.add_argument("--name", required=True, help="Human-readable app name")
    parser.add_argument("--count", type=int, default=10000, help="Target number of reviews to fetch")
    args = parser.parse_args()

    print(f"Fetching up to {args.count} reviews for {args.name} ({args.app})...")
    inserted = fetch_and_store(args.app, args.name, count=args.count)
    print(f"Inserted {inserted} new reviews for {args.name}.")


if __name__ == "__main__":
    main()
