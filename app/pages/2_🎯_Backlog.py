"""Ranked, evidence-backed feature backlog: pain points scored and sorted by Priority Score."""

import json
import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from reviewlens import db
from reviewlens.cluster import build_pain_points
from reviewlens.prioritize import score_pain_points

st.set_page_config(page_title="ReviewLens - Backlog", page_icon="🔎", layout="wide")
st.title("Feature Backlog")

db.init_db()
reviews = db.all_reviews()
if reviews.empty:
    st.warning("No reviews in the database yet. Run scripts/fetch_reviews.py first.")
    st.stop()

apps = sorted(reviews["app_name"].unique())
app_name = st.sidebar.selectbox("App", apps)
prompt_version = st.sidebar.selectbox("Classified with", ["v1", "v2", "v3"])
app_id = reviews.loc[reviews["app_name"] == app_name, "app_id"].iloc[0]

st.sidebar.subheader("Priority weights")
w_freq = st.sidebar.slider("Frequency", 0.0, 1.0, 0.3, 0.05)
w_sev = st.sidebar.slider("Severity", 0.0, 1.0, 0.3, 0.05)
w_trend = st.sidebar.slider("Trend", 0.0, 1.0, 0.2, 0.05)
w_reach = st.sidebar.slider("Reach", 0.0, 1.0, 0.2, 0.05)
weights = {"frequency": w_freq, "severity": w_sev, "trend": w_trend, "reach": w_reach}

pain_points = db.load_pain_points(app_id, prompt_version)

if st.sidebar.button("Rebuild pain points" if not pain_points.empty else "Build pain points", type="primary"):
    with st.spinner("Clustering reviews and writing pain-point summaries with Gemini..."):
        pain_points = build_pain_points(app_id, prompt_version)
    st.rerun()

if pain_points.empty:
    st.info(
        f"No pain points yet for {app_name} ({prompt_version}). Classify this app's reviews on "
        "the Dashboard page first, then click 'Build pain points' in the sidebar."
    )
    st.stop()

scored = score_pain_points(pain_points, weights)

# --- exports ---
csv_bytes = scored.to_csv(index=False).encode("utf-8")


def to_markdown(df) -> str:
    lines = [f"# {app_name} - Feature Backlog", ""]
    for rank, row in enumerate(df.itertuples(), start=1):
        lines += [
            f"## #{rank}. {row.title} (Priority Score: {row.priority_score:.0f}/100)",
            "",
            (
                f"**Theme:** {row.theme}  |  **Mentions:** {row.review_count}  |  "
                f"**Trend (30d):** {row.last_30_count} vs {row.prev_30_count} prior"
            ),
            "",
            row.summary,
            "",
            f"**Suggested next step:** {row.next_step}",
            "",
            "**Representative quotes:**",
        ]
        for q in json.loads(row.example_quotes):
            lines.append(f'> "{q}"')
        lines.append("")
    return "\n".join(lines)


col1, col2, _ = st.columns([1, 1, 4])
with col1:
    st.download_button("Export CSV", csv_bytes, file_name=f"{app_id}_backlog.csv", mime="text/csv")
with col2:
    st.download_button(
        "Export Markdown (PRD)", to_markdown(scored),
        file_name=f"{app_id}_backlog.md", mime="text/markdown",
    )

st.caption(
    "Priority Score = weighted average of frequency, severity, trend, and reach "
    "(each rescaled 0-100). See reviewlens/prioritize.py for the exact formula."
)

for rank, row in enumerate(scored.itertuples(), start=1):
    badge = {1: "🥇", 2: "🥈", 3: "🥉"}.get(rank, f"#{rank}")
    with st.container(border=True):
        header_col, score_col = st.columns([4, 1])
        with header_col:
            st.markdown(f"### {badge} {row.title}")
            st.caption(f"{row.theme} - {row.review_count} mentions")
        with score_col:
            st.metric("Priority", f"{row.priority_score:.0f}")

        trend_arrow = "📈" if row.last_30_count > row.prev_30_count else (
            "📉" if row.last_30_count < row.prev_30_count else "➡️"
        )
        st.write(row.summary)
        st.caption(
            f"Frequency {row.frequency_score:.0f} | Severity {row.severity_score:.0f} | "
            f"Trend {trend_arrow} {row.last_30_count} vs {row.prev_30_count} | Reach {row.reach_score:.0f}"
        )
        st.info(f"**Suggested next step:** {row.next_step}")

        with st.expander("Representative quotes"):
            for q in json.loads(row.example_quotes):
                st.markdown(f'> "{q}"')
