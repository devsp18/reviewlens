"""Cache behavior and batching/storage logic. All Gemini calls are mocked -
these tests must run with no network access and no API key."""

import pandas as pd
import pytest

from reviewlens import db
from reviewlens.classify import cache_key, classify_reviews, get_cached, set_cached
from reviewlens.schema import BatchClassification


def test_cache_key_is_deterministic():
    assert cache_key("v1", "hello world") == cache_key("v1", "hello world")


def test_cache_key_differs_by_prompt_version():
    assert cache_key("v1", "hello world") != cache_key("v2", "hello world")


def test_cache_key_differs_by_text():
    assert cache_key("v1", "hello") != cache_key("v1", "world")


def test_cache_miss_returns_none(db_path):
    assert get_cached("v1", "never cached", db_path) is None


def test_cache_round_trip(db_path):
    record = {"theme": "UI/UX", "sentiment": "positive", "severity": 1, "reasoning": "nice"}
    set_cached("v1", "great app", record, db_path)
    assert get_cached("v1", "great app", db_path) == record


def _fake_batch_classification(pending_texts):
    return BatchClassification(classifications=[
        {
            "review_index": i, "theme": "UI/UX", "sentiment": "positive",
            "severity": 1, "reasoning": "mocked",
        }
        for i in range(len(pending_texts))
    ])


def test_classify_reviews_uses_cache_on_rerun(db_path, mocker):
    mock_call = mocker.patch("reviewlens.classify._call_gemini")
    mock_call.side_effect = lambda prompt: _fake_batch_classification(
        [line for line in prompt.split("\n") if line and line[0].isdigit()]
    )

    df = pd.DataFrame({"id": [1, 2], "cleaned_text": ["review one", "review two"]})

    first = classify_reviews(df, "v1", db_path=db_path)
    assert mock_call.call_count == 1
    assert len(first) == 2

    second = classify_reviews(df, "v1", db_path=db_path)
    assert mock_call.call_count == 1  # no new API call - served entirely from cache
    assert len(second) == 2


def test_classify_reviews_persists_to_classifications_table(db_path, mocker):
    mocker.patch(
        "reviewlens.classify._call_gemini",
        side_effect=lambda prompt: _fake_batch_classification(["x"]),
    )
    df = pd.DataFrame({"id": [42], "cleaned_text": ["a single review"]})
    classify_reviews(df, "v1", db_path=db_path)

    with db.get_connection(db_path) as conn:
        row = conn.execute("SELECT theme, sentiment, severity, reasoning FROM classifications WHERE review_id = 42").fetchone()
    assert row == ("UI/UX", "positive", 1, "mocked")


def test_v3_without_retriever_raises(db_path):
    df = pd.DataFrame({"id": [1], "cleaned_text": ["text"]})
    with pytest.raises(ValueError, match="retriever"):
        classify_reviews(df, "v3", db_path=db_path)
