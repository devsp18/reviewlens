"""Eval Lab metrics: prompt v1/v2/v3 accuracy against human labels, plus PM ranking agreement.

Every number here comes from a real run against data/labeled/eval_set.csv (human
labels from the Labeling Studio) and data/labeled/pm_rankings.csv (human PM
rankings). Nothing is fabricated or hardcoded - if the labeled set is empty,
functions here return empty/None rather than making up a result.
"""

from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import classification_report, confusion_matrix

from reviewlens import db, labeling
from reviewlens.classify import classify_reviews
from reviewlens.rag import build_index, retrieve


def load_eval_reviews(db_path=None) -> pd.DataFrame:
    """Human-labeled reviews with their text, joined from eval_set.csv + the reviews table."""
    labels = labeling.load_labels()
    if labels.empty:
        return pd.DataFrame(columns=["id", "cleaned_text", "theme", "sentiment"])
    reviews = db.all_reviews(db_path)
    merged = labels.merge(reviews[["id", "cleaned_text"]], left_on="review_id", right_on="id")
    return merged[["id", "cleaned_text", "theme", "sentiment"]].rename(
        columns={"theme": "true_theme", "sentiment": "true_sentiment"}
    )


def _v3_retriever_factory(labeled_pool: pd.DataFrame, held_out_ids: set[int], persist_dir=None):
    """Build a retriever closure whose index excludes held_out_ids (this eval's own
    labeled set), so v3 never retrieves a review's own label as its few-shot example."""
    rag_df = labeled_pool.rename(columns={"true_theme": "theme", "true_sentiment": "sentiment"})
    rag_df = rag_df.rename(columns={"cleaned_text": "text", "id": "review_id"})
    collection = build_index(
        rag_df,
        collection_name="eval_v3",
        exclude_review_ids=held_out_ids,
        persist_dir=persist_dir,
    )

    def retriever(text: str) -> list[dict]:
        return retrieve(text, collection, k=3)

    return retriever


def evaluate_prompt_version(prompt_version: str, eval_df: pd.DataFrame, db_path=None) -> dict:
    """Classify every review in eval_df with prompt_version and score against its human labels."""
    if eval_df.empty:
        return {}

    to_classify = eval_df.rename(columns={"id": "id", "cleaned_text": "cleaned_text"})[["id", "cleaned_text"]]

    if prompt_version == "v3":
        # Hold out this whole eval set from its own retrieval index (k-fold-safe: it's
        # never available to retrieve itself, matching the no-leakage rule from Phase 5).
        retriever = _v3_retriever_factory(eval_df, set(eval_df["id"].tolist()))
        predictions = classify_reviews(to_classify, prompt_version, db_path=db_path, retriever=retriever)
    else:
        predictions = classify_reviews(to_classify, prompt_version, db_path=db_path)

    merged = eval_df.merge(predictions, left_on="id", right_on="review_id")

    theme_report = classification_report(merged["true_theme"], merged["theme"], output_dict=True, zero_division=0)
    sentiment_report = classification_report(
        merged["true_sentiment"], merged["sentiment"], output_dict=True, zero_division=0
    )
    theme_labels = sorted(set(merged["true_theme"]) | set(merged["theme"]))
    sentiment_labels = sorted(set(merged["true_sentiment"]) | set(merged["sentiment"]))

    misclassified = merged[
        (merged["theme"] != merged["true_theme"]) | (merged["sentiment"] != merged["true_sentiment"])
    ]

    return {
        "prompt_version": prompt_version,
        "n": len(merged),
        "theme_accuracy": theme_report["accuracy"],
        "theme_macro_f1": theme_report["macro avg"]["f1-score"],
        "theme_report": theme_report,
        "theme_confusion_matrix": confusion_matrix(merged["true_theme"], merged["theme"], labels=theme_labels).tolist(),
        "theme_labels": theme_labels,
        "sentiment_accuracy": sentiment_report["accuracy"],
        "sentiment_macro_f1": sentiment_report["macro avg"]["f1-score"],
        "sentiment_report": sentiment_report,
        "sentiment_confusion_matrix": confusion_matrix(
            merged["true_sentiment"], merged["sentiment"], labels=sentiment_labels
        ).tolist(),
        "sentiment_labels": sentiment_labels,
        "misclassified_examples": misclassified[
            [
                "id",
                "cleaned_text",
                "true_theme",
                "theme",
                "true_sentiment",
                "sentiment",
                "reasoning",
            ]
        ].to_dict("records"),
    }


def evaluate_pm_agreement(rankings_path: Path = labeling.PM_RANKINGS_PATH) -> dict:
    """Spearman rank correlation and top-5 overlap between each pair of PM reviewers'
    rankings, per app. Returns {} if fewer than 2 reviewers have ranked the same app."""
    rankings = labeling.load_rankings(rankings_path)
    if rankings.empty:
        return {}

    results = {}
    for app_id, app_df in rankings.groupby("app_id"):
        reviewers = sorted(app_df["reviewer"].unique())
        if len(reviewers) < 2:
            continue
        pairs = []
        for i in range(len(reviewers)):
            for j in range(i + 1, len(reviewers)):
                r1, r2 = reviewers[i], reviewers[j]
                a = app_df[app_df["reviewer"] == r1].set_index("pain_point")["rank"]
                b = app_df[app_df["reviewer"] == r2].set_index("pain_point")["rank"]
                common = a.index.intersection(b.index)
                if len(common) < 2:
                    continue
                rho, pval = spearmanr(a.loc[common], b.loc[common])
                top5_a = set(a.nsmallest(5).index)
                top5_b = set(b.nsmallest(5).index)
                overlap = len(top5_a & top5_b) / max(1, len(top5_a | top5_b))
                pairs.append(
                    {
                        "reviewer_a": r1,
                        "reviewer_b": r2,
                        "n_common_pain_points": len(common),
                        "spearman_rho": rho,
                        "spearman_pvalue": pval,
                        "top5_jaccard_overlap": overlap,
                    }
                )
        if pairs:
            results[app_id] = pairs

    return {
        "n_reviewers": rankings["reviewer"].nunique(),
        "metric": "Spearman rank correlation (rho) and top-5 Jaccard overlap, pairwise across all PM reviewers per app",
        "by_app": results,
    }
