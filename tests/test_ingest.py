"""Ingestion cleaning: emoji handling, whitespace, dedup, language filter, date normalization."""

import pandas as pd

from reviewlens.ingest import clean_and_filter, clean_text, detect_language


def test_clean_text_normalizes_whitespace():
    assert clean_text("hello    world  \n\t") == "hello world"


def test_clean_text_demojizes_single_emoji():
    assert clean_text("crashes 😡") == "crashes :enraged_face:"


def test_clean_text_pads_back_to_back_emoji():
    # Regression: default emoji.demojize delimiters fuse adjacent emoji into one
    # unreadable token (":grinning_face::thumbs_up:" -> looked like one word).
    result = clean_text("😀👍")
    assert result == ":grinning_face: :thumbs_up:"
    assert "::" not in result


def test_detect_language_english():
    assert detect_language("This app is great and works well for me") == "en"


def test_detect_language_too_short_returns_none():
    assert detect_language("hi") is None


def test_detect_language_empty_returns_none():
    assert detect_language("") is None


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame({
        "app_id": ["app1", "app1", "app1", "app1", "app1"],
        "text": [
            "This app works great for my daily tasks",
            "This app works great for my daily tasks",  # exact duplicate
            "",  # empty
            "Cette application est terrible",  # non-English
            "Crashes constantly on my phone every day",
        ],
        "review_date": ["2026-01-01", "2026-01-02", "2026-01-03", "2026-01-04", "not-a-date"],
    })


def test_clean_and_filter_drops_empty_reviews():
    result = clean_and_filter(_sample_df())
    assert "" not in result["cleaned_text"].tolist()


def test_clean_and_filter_dedupes_exact_matches():
    result = clean_and_filter(_sample_df())
    texts = result["cleaned_text"].tolist()
    assert len(texts) == len(set(texts))


def test_clean_and_filter_keeps_only_english():
    result = clean_and_filter(_sample_df())
    assert (result["language"] == "en").all()


def test_clean_and_filter_drops_unparseable_dates():
    result = clean_and_filter(_sample_df())
    assert result["review_date"].notna().all()
    assert "not-a-date" not in result["review_date"].tolist()
