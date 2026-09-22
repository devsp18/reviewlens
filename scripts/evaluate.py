#!/usr/bin/env python
"""Eval Lab: run prompt v1/v2/v3 against the human-labeled eval set and score them,
plus PM pain-point ranking agreement. Saves real results to results/ - nothing here
is fabricated, and if the labeled set is empty this says so rather than inventing numbers.

Usage:
    python scripts/evaluate.py
    python scripts/evaluate.py --versions v1 v2
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from reviewlens import db
from reviewlens.evaluate import (
    evaluate_pm_agreement,
    evaluate_prompt_version,
    load_eval_reviews,
)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--versions", nargs="+", default=["v1", "v2", "v3"])
    args = parser.parse_args()

    db.init_db()
    eval_df = load_eval_reviews()

    if eval_df.empty:
        print(
            "No human labels found in data/labeled/eval_set.csv - nothing to evaluate.\n"
            "Label reviews in the Labeling Studio first (app/pages/4_🏷_Labeling_Studio.py), "
            "then re-run this script. Refusing to fabricate results."
        )
        return

    print(f"Evaluating {len(eval_df)} human-labeled reviews...")
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    all_results = {}
    for version in args.versions:
        print(f"\n--- {version} ---")
        result = evaluate_prompt_version(version, eval_df)
        if not result:
            continue
        all_results[version] = result
        print(f"theme accuracy={result['theme_accuracy']:.3f} macro-F1={result['theme_macro_f1']:.3f}")
        print(f"sentiment accuracy={result['sentiment_accuracy']:.3f} macro-F1={result['sentiment_macro_f1']:.3f}")

    pm_agreement = evaluate_pm_agreement()
    if pm_agreement:
        print(f"\n--- PM ranking agreement ({pm_agreement['n_reviewers']} reviewers) ---")
        print(json.dumps(pm_agreement["by_app"], indent=2, default=str))
    else:
        print("\nNo PM rankings found in data/labeled/pm_rankings.csv yet - skipping agreement metrics.")

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_labeled_reviews": len(eval_df),
        "prompt_versions": all_results,
        "pm_agreement": pm_agreement,
    }
    out_path = RESULTS_DIR / "eval_results.json"
    out_path.write_text(json.dumps(output, indent=2, default=str))
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
