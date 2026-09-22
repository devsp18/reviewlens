"""Eval Lab metrics computation. All Gemini calls are mocked."""

import pandas as pd
import pytest

from reviewlens import labeling
from reviewlens.evaluate import evaluate_pm_agreement, evaluate_prompt_version


def test_evaluate_prompt_version_computes_accuracy(mocker):
    # 2 of 3 predictions match ground truth theme, 3 of 3 match sentiment
    predictions = pd.DataFrame({
        "review_id": [1, 2, 3],
        "theme": ["UI/UX", "Pricing", "Other"],
        "sentiment": ["positive", "negative", "neutral"],
        "severity": [1, 4, 1],
        "reasoning": ["a", "b", "c"],
    })
    mocker.patch("reviewlens.evaluate.classify_reviews", return_value=predictions)

    eval_df = pd.DataFrame({
        "id": [1, 2, 3],
        "cleaned_text": ["t1", "t2", "t3"],
        "true_theme": ["UI/UX", "Pricing", "Feature Request"],  # id 3 mismatches
        "true_sentiment": ["positive", "negative", "neutral"],
    })

    result = evaluate_prompt_version("v1", eval_df)
    assert result["n"] == 3
    assert result["theme_accuracy"] == pytest.approx(2 / 3)
    assert result["sentiment_accuracy"] == 1.0
    assert len(result["misclassified_examples"]) == 1
    assert result["misclassified_examples"][0]["id"] == 3


def test_evaluate_prompt_version_empty_eval_set_returns_empty():
    assert evaluate_prompt_version("v1", pd.DataFrame()) == {}


def test_evaluate_pm_agreement_no_rankings_returns_empty(tmp_path):
    empty_path = tmp_path / "pm_rankings.csv"
    assert evaluate_pm_agreement(empty_path) == {}


def test_evaluate_pm_agreement_single_reviewer_skipped(tmp_path):
    path = tmp_path / "pm_rankings.csv"
    pd.DataFrame([
        {"reviewer": "alice", "app_id": "app1", "pain_point": "Crashes", "rank": 1, "rating": 5, "ranked_at": ""},
    ]).to_csv(path, index=False)
    result = evaluate_pm_agreement(path)
    assert result["by_app"] == {}


def test_evaluate_pm_agreement_perfect_agreement_gives_rho_one(tmp_path):
    path = tmp_path / "pm_rankings.csv"
    pd.DataFrame([
        {"reviewer": "alice", "app_id": "app1", "pain_point": "Crashes", "rank": 1, "rating": 5, "ranked_at": ""},
        {"reviewer": "alice", "app_id": "app1", "pain_point": "Billing", "rank": 2, "rating": 3, "ranked_at": ""},
        {"reviewer": "bob", "app_id": "app1", "pain_point": "Crashes", "rank": 1, "rating": 5, "ranked_at": ""},
        {"reviewer": "bob", "app_id": "app1", "pain_point": "Billing", "rank": 2, "rating": 3, "ranked_at": ""},
    ]).to_csv(path, index=False)
    result = evaluate_pm_agreement(path)
    pair = result["by_app"]["app1"][0]
    assert pair["spearman_rho"] == pytest.approx(1.0)
    assert pair["top5_jaccard_overlap"] == 1.0


def test_labeling_save_and_load_round_trip(tmp_path):
    path = tmp_path / "eval_set.csv"
    labeling.save_label(1, "UI/UX", "positive", "tester", path)
    labels = labeling.load_labels(path)
    assert len(labels) == 1
    assert labels.iloc[0]["theme"] == "UI/UX"


def test_labeling_save_label_upserts_by_review_id(tmp_path):
    path = tmp_path / "eval_set.csv"
    labeling.save_label(1, "UI/UX", "positive", "tester", path)
    labeling.save_label(1, "Pricing", "negative", "tester", path)  # re-label
    labels = labeling.load_labels(path)
    assert len(labels) == 1
    assert labels.iloc[0]["theme"] == "Pricing"


def test_labeling_delete_label(tmp_path):
    path = tmp_path / "eval_set.csv"
    labeling.save_label(1, "UI/UX", "positive", "tester", path)
    labeling.delete_label(1, path)
    assert labeling.load_labels(path).empty
