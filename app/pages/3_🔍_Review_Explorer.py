"""Search and filter every review, and inspect RAG-retrieved examples for v3 predictions."""

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from reviewlens import db

st.set_page_config(page_title="ReviewLens - Review Explorer", page_icon="🔎", layout="wide")
st.title("Review Explorer")

db.init_db()
reviews = db.all_reviews()
if reviews.empty:
    st.warning("No reviews in the database yet. Run scripts/fetch_reviews.py first.")
    st.stop()

with db.get_connection() as conn:
    classifications = pd.read_sql("SELECT * FROM classifications", conn)
    retrievals = pd.read_sql("SELECT * FROM retrievals", conn)

col1, col2, col3, col4 = st.columns(4)
with col1:
    app_filter = st.selectbox("App", ["All"] + sorted(reviews["app_name"].unique()))
with col2:
    rating_filter = st.selectbox("Rating", ["All"] + sorted(reviews["rating"].unique().tolist(), reverse=True))
with col3:
    prompt_version = st.selectbox("Classified with", ["Any"] + sorted(classifications["prompt_version"].unique()) if not classifications.empty else ["Any"])
with col4:
    theme_options = ["All"]
    if not classifications.empty:
        theme_options += sorted(classifications["theme"].unique())
    theme_filter = st.selectbox("Theme", theme_options)

query = st.text_input("Search review text")

filtered = reviews.copy()
if app_filter != "All":
    filtered = filtered[filtered["app_name"] == app_filter]
if rating_filter != "All":
    filtered = filtered[filtered["rating"] == rating_filter]
if query:
    filtered = filtered[filtered["cleaned_text"].str.contains(query, case=False, na=False)]

if prompt_version != "Any" and not classifications.empty:
    cls_subset = classifications[classifications["prompt_version"] == prompt_version]
    filtered = filtered.merge(cls_subset, left_on="id", right_on="review_id", how="inner", suffixes=("", "_cls"))
    if theme_filter != "All":
        filtered = filtered[filtered["theme"] == theme_filter]
elif not classifications.empty:
    filtered = filtered.merge(
        classifications.drop_duplicates("review_id"), left_on="id", right_on="review_id",
        how="left", suffixes=("", "_cls"),
    )
    if theme_filter != "All":
        filtered = filtered[filtered["theme"] == theme_filter]

st.caption(f"{len(filtered)} reviews")

for _, row in filtered.head(50).iterrows():
    with st.container(border=True):
        header = f"**{row['app_name']}** - {'★' * int(row['rating'])} - {row['review_date']}"
        if "theme" in row and pd.notna(row.get("theme")):
            header += f" - {row['theme']} / {row['sentiment']} (severity {row.get('severity', '?')})"
        st.markdown(header)
        st.write(row["cleaned_text"])

        review_retrievals = retrievals[retrievals["review_id"] == row["id"]]
        if not review_retrievals.empty:
            with st.expander(f"RAG examples retrieved ({len(review_retrievals)} prompt version(s))"):
                for _, rret in review_retrievals.iterrows():
                    st.caption(f"prompt {rret['prompt_version']}:")
                    for ex in json.loads(rret["retrieved_examples"]):
                        st.text(f'  "{ex["text"]}" -> {ex["theme"]} / {ex["sentiment"]} (dist={ex["distance"]:.3f})')

if len(filtered) > 50:
    st.caption(f"Showing first 50 of {len(filtered)} matches - narrow your filters to see more precisely.")
