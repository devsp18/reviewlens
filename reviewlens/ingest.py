"""Fetching, cleaning, and storing app reviews (Play Store scrape or CSV upload)."""

import time

import emoji
import pandas as pd
from google_play_scraper import Sort, reviews
from langdetect import LangDetectException, detect

from reviewlens import db

REQUIRED_CSV_COLUMNS = ["text", "rating"]


def clean_text(text: str) -> str:
    """Normalize whitespace and turn emoji into readable text (e.g. crashes 😡 -> crashes :enraged_face:).
    Padded delimiters keep back-to-back emoji (😀👍 -> :grinning_face: :thumbs_up:) from fusing
    into one unreadable token - the whitespace normalization below then collapses the padding."""
    text = emoji.demojize(str(text), delimiters=(" :", ": "))
    return " ".join(text.split()).strip()


def detect_language(text: str) -> str | None:
    if not text or len(text) < 3:
        return None
    try:
        return detect(text)
    except LangDetectException:
        return None


def clean_and_filter(df: pd.DataFrame, keep_languages: tuple[str, ...] = ("en",)) -> pd.DataFrame:
    """Clean text, drop empty/duplicate reviews, filter to the given languages, normalize dates."""
    df = df.copy()
    df["cleaned_text"] = df["text"].map(clean_text)
    df = df[df["cleaned_text"].str.len() > 0]
    df = df.drop_duplicates(subset=["app_id", "cleaned_text"])

    df["language"] = df["cleaned_text"].map(detect_language)
    df = df[df["language"].isin(keep_languages)]

    df["review_date"] = pd.to_datetime(df["review_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["review_date"])

    return df.reset_index(drop=True)


def fetch_app_reviews(
    app_id: str,
    app_name: str,
    count: int = 10000,
    lang: str = "en",
    country: str = "us",
    sort: Sort = Sort.NEWEST,
    page_size: int = 200,
) -> pd.DataFrame:
    """Fetch up to `count` raw reviews for a Play Store app via paginated scraping."""
    collected = []
    token = None
    while len(collected) < count:
        batch_size = min(page_size, count - len(collected))
        result, token = reviews(
            app_id,
            lang=lang,
            country=country,
            sort=sort,
            count=batch_size,
            continuation_token=token,
        )
        if not result:
            break
        collected.extend(result)
        if token is None:
            break
        time.sleep(0.2)  # be polite to the unofficial endpoint

    if not collected:
        return pd.DataFrame(
            columns=[
                "app_id",
                "app_name",
                "review_id",
                "user_name",
                "rating",
                "thumbs_up",
                "text",
                "review_date",
                "app_version",
                "source",
            ]
        )

    df = pd.DataFrame(collected)
    return pd.DataFrame(
        {
            "app_id": app_id,
            "app_name": app_name,
            "review_id": df["reviewId"],
            "user_name": df["userName"],
            "rating": df["score"],
            "thumbs_up": df["thumbsUpCount"].fillna(0).astype(int),
            "text": df["content"].fillna(""),
            "review_date": df["at"],
            "app_version": df.get("reviewCreatedVersion"),
            "source": "scrape",
        }
    )


def fetch_and_store(app_id: str, app_name: str, count: int = 10000) -> int:
    """Fetch, clean, and store reviews for one app. Returns the number of new rows inserted."""
    raw = fetch_app_reviews(app_id, app_name, count=count)
    if raw.empty:
        return 0
    cleaned = clean_and_filter(raw)
    db.init_db()
    return db.upsert_reviews(cleaned)


def ingest_csv(path: str, column_mapping: dict[str, str], app_id: str, app_name: str) -> int:
    """Load a CSV with user-mapped columns (e.g. {'text': 'Review Text', 'rating': 'Stars'}),
    clean it, and store it. `column_mapping` maps our field names -> the CSV's column names."""
    raw = pd.read_csv(path)
    raw = raw.rename(columns={csv_col: field for field, csv_col in column_mapping.items()})

    for col in REQUIRED_CSV_COLUMNS:
        if col not in raw.columns:
            raise ValueError(f"CSV is missing a mapped '{col}' column")

    raw["app_id"] = app_id
    raw["app_name"] = app_name
    if "review_id" not in raw.columns:
        raw["review_id"] = pd.Series(range(len(raw))).astype(str)
    for optional in ["user_name", "thumbs_up", "review_date", "app_version"]:
        if optional not in raw.columns:
            raw[optional] = None
    raw["thumbs_up"] = raw["thumbs_up"].fillna(0).astype(int)
    raw["source"] = "csv"

    cleaned = clean_and_filter(raw)
    db.init_db()
    return db.upsert_reviews(cleaned)
