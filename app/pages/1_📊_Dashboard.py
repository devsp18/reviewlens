"""Themes, sentiment, and classification progress for a selected app."""

import sys
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from reviewlens import db
from reviewlens.classify import classify_reviews
from reviewlens.config import settings

st.set_page_config(page_title="ReviewLens - Dashboard", page_icon="🔎", layout="wide")
st.title("Dashboard")
if settings.demo_mode:
    st.caption("Demo Mode - showing precomputed results, classification is disabled to avoid spending API credits.")

db.init_db()
reviews = db.all_reviews()

if reviews.empty:
    st.warning("No reviews in the database yet. Run scripts/fetch_reviews.py first.")
    st.stop()

apps = sorted(reviews["app_name"].unique())
app_name = st.sidebar.selectbox("App", apps)
prompt_version = st.sidebar.selectbox("Prompt version", ["v1", "v2"])
app_reviews = reviews[reviews["app_name"] == app_name]

with db.get_connection() as conn:
    classified = pd.read_sql("SELECT * FROM classifications WHERE prompt_version = ?", conn, params=(prompt_version,))

merged = app_reviews.merge(classified, left_on="id", right_on="review_id", how="left")
unclassified_count = merged["theme"].isna().sum()

col1, col2 = st.columns([3, 1])
with col1:
    st.caption(
        f"{len(app_reviews)} reviews for {app_name} - {len(app_reviews) - unclassified_count} classified with {prompt_version}"
    )
with col2:
    run = st.button(
        f"Classify {unclassified_count} remaining",
        disabled=unclassified_count == 0 or settings.demo_mode,
    )

if run and not settings.demo_mode:
    progress_bar = st.progress(0.0)
    status = st.empty()
    start = time.time()
    to_classify = app_reviews[app_reviews["id"].isin(merged.loc[merged["theme"].isna(), "id"])]

    def on_progress(done: int, total: int) -> None:
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0
        eta = (total - done) / rate if rate > 0 else 0
        progress_bar.progress(done / total)
        status.text(f"{done}/{total} classified - est. {eta:.0f}s remaining")

    classify_reviews(to_classify, prompt_version, progress_callback=on_progress)
    st.success("Classification complete.")
    st.rerun()

if unclassified_count < len(app_reviews):
    classified_rows = merged.dropna(subset=["theme"])
    tab1, tab2 = st.tabs(["Themes", "Sentiment"])
    with tab1:
        theme_counts = classified_rows["theme"].value_counts().reset_index()
        theme_counts.columns = ["theme", "count"]
        fig = px.bar(theme_counts, x="count", y="theme", orientation="h", color="theme")
        fig.update_layout(showlegend=False, yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, width="stretch")
    with tab2:
        sentiment_counts = classified_rows["sentiment"].value_counts().reset_index()
        sentiment_counts.columns = ["sentiment", "count"]
        fig = px.pie(sentiment_counts, names="sentiment", values="count", hole=0.5)
        st.plotly_chart(fig, width="stretch")
else:
    st.info("Run classification above to see theme and sentiment breakdowns.")
