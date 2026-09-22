"""Landing page: hero, app picker, and headline metrics."""

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.components.metrics import animated_metric
from reviewlens import db

st.set_page_config(page_title="ReviewLens", page_icon="🔎", layout="wide")

db.init_db()
reviews = db.all_reviews()

st.title("ReviewLens")
st.caption("Turn 10,000 reviews into your next roadmap in minutes.")
st.write(
    "Point ReviewLens at a Play Store app (or upload a CSV) and it classifies every review "
    "by theme and sentiment with Gemini, groups the negative ones into pain points, and "
    "ranks them into an evidence-backed feature backlog - quotes, affected-user counts, "
    "and suggested next steps included."
)

st.divider()

col1, col2, col3 = st.columns(3)
with col1:
    animated_metric("Reviews analyzed", len(reviews), accent="#4F46E5")
with col2:
    with db.get_connection() as conn:
        pain_point_count = conn.execute("SELECT COUNT(*) FROM pain_points").fetchone()[0]
    animated_metric("Pain points found", pain_point_count, accent="#8B5CF6")
with col3:
    results_path = Path(__file__).resolve().parent.parent / "results" / "eval_results.json"
    if results_path.exists():
        results = json.loads(results_path.read_text())
        versions = results.get("prompt_versions", {})
        best_accuracy = max((v["theme_accuracy"] for v in versions.values()), default=None)
        if best_accuracy is not None:
            animated_metric("Best theme accuracy", round(best_accuracy * 100, 1), suffix="%", accent="#10B981")
        else:
            st.metric("Theme accuracy", "Not yet evaluated")
    else:
        st.metric("Theme accuracy", "Not yet evaluated")

st.divider()

if reviews.empty:
    st.warning(
        'No reviews yet. Run `python scripts/fetch_reviews.py --app <package_id> --name "App Name"` to get started.'
    )
else:
    st.subheader("Pick an app")
    summary = (
        reviews.groupby(["app_id", "app_name"])
        .agg(reviews=("id", "count"), avg_rating=("rating", "mean"))
        .reset_index()
        .sort_values("reviews", ascending=False)
    )
    for _, row in summary.iterrows():
        with st.container(border=True):
            c1, c2, c3 = st.columns([3, 2, 2])
            with c1:
                st.markdown(f"**{row['app_name']}**")
                st.caption(row["app_id"])
            with c2:
                st.write(f"{row['reviews']:,} reviews")
                st.caption(f"avg {row['avg_rating']:.1f} stars")
            with c3:
                st.page_link("pages/1_📊_Dashboard.py", label="Open Dashboard", icon="📊")
