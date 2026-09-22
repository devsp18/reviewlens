"""Human ground-truth labeling: theme/sentiment cards, and PM pain-point ranking."""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st
from streamlit_shortcuts import shortcut_button

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from reviewlens import db, labeling
from reviewlens.schema import SENTIMENT_VALUES, THEME_VALUES

st.set_page_config(page_title="ReviewLens - Labeling Studio", page_icon="🔎", layout="wide")
st.title("Labeling Studio")
st.caption("Human ground truth for the Eval Lab. Labels here are never auto-generated.")

mode = st.sidebar.radio("Mode", ["Label Reviews", "PM Ranking"])

# --------------------------------------------------------------------------
# Mode 1: theme + sentiment labeling
# --------------------------------------------------------------------------
if mode == "Label Reviews":
    labeler = st.sidebar.text_input("Your name (recorded with each label)", value="satyam")

    queue = labeling.load_queue()
    if queue.empty:
        st.warning("No label queue yet. Run scripts/make_label_queue.py first.")
        st.stop()

    labels = labeling.load_labels()
    labeled_ids = set(labels["review_id"])
    skipped_ids = st.session_state.setdefault("skipped_ids", set())
    history = st.session_state.setdefault("history", [])  # list of (review_id, action)

    remaining = queue[~queue["id"].isin(labeled_ids) & ~queue["id"].isin(skipped_ids)]

    progress_col, undo_col, reset_col = st.columns([4, 1, 1])
    with progress_col:
        st.progress(len(labeled_ids) / len(queue))
        st.caption(f"{len(labeled_ids)} / {len(queue)} labeled ({len(skipped_ids)} skipped this session)")
    with undo_col:
        if shortcut_button("Undo", shortcut="ctrl+z", disabled=not history, hint=True):
            rid, action = history.pop()
            if action == "label":
                labeling.delete_label(rid)
            elif action == "skip":
                skipped_ids.discard(rid)
            st.rerun()
    with reset_col:
        if st.button("Reset skips", disabled=not skipped_ids):
            skipped_ids.clear()
            st.rerun()

    if remaining.empty:
        st.success("Nothing left to label right now.")
        st.stop()

    row = remaining.iloc[0]
    st.session_state.setdefault("pending_theme", None)

    with st.container(border=True):
        st.caption(f"{row['app_name']} - {'★' * int(row['rating'])}{'☆' * (5 - int(row['rating']))}")
        st.markdown(f"### {row['cleaned_text']}")

    st.write("**Theme** (press 1-9 for the first nine, 0 for the tenth)")
    theme_keys = [str(i) for i in range(1, 10)] + ["0"]
    cols = st.columns(5)
    for i, theme in enumerate(THEME_VALUES):
        with cols[i % 5]:
            selected = st.session_state["pending_theme"] == theme
            if shortcut_button(
                theme, shortcut=theme_keys[i], key=f"theme_{theme}",
                type="primary" if selected else "secondary",
            ):
                st.session_state["pending_theme"] = theme
                st.rerun()

    st.write("**Sentiment** (P / N / U - saves and advances)")
    sent_cols = st.columns(4)
    sentiment_keys = {"positive": "p", "negative": "n", "neutral": "u"}
    for i, sentiment in enumerate(SENTIMENT_VALUES):
        with sent_cols[i]:
            if shortcut_button(sentiment.capitalize(), shortcut=sentiment_keys[sentiment], key=f"sent_{sentiment}"):
                if st.session_state["pending_theme"] is None:
                    st.warning("Pick a theme first.")
                else:
                    labeling.save_label(int(row["id"]), st.session_state["pending_theme"], sentiment, labeler)
                    history.append((int(row["id"]), "label"))
                    st.session_state["pending_theme"] = None
                    st.rerun()
    with sent_cols[3]:
        if shortcut_button("Skip", shortcut="s", key="skip_btn"):
            skipped_ids.add(int(row["id"]))
            history.append((int(row["id"]), "skip"))
            st.session_state["pending_theme"] = None
            st.rerun()

# --------------------------------------------------------------------------
# Mode 2: PM pain-point ranking
# --------------------------------------------------------------------------
else:
    st.info(
        "Pain points here are grouped by theme as a stand-in until Phase 6's embedding "
        "clustering ships finer-grained pain points. Rate each on importance to derive a rank."
    )
    reviewer = st.sidebar.text_input("Reviewer name", value="")
    reviews = db.all_reviews()
    apps = sorted(reviews["app_name"].unique())
    app_name = st.sidebar.selectbox("App", apps)
    app_id = reviews.loc[reviews["app_name"] == app_name, "app_id"].iloc[0]

    with db.get_connection() as conn:
        classified = pd.read_sql(
            "SELECT c.*, r.app_id, r.cleaned_text FROM classifications c "
            "JOIN reviews r ON r.id = c.review_id WHERE r.app_id = ?",
            conn, params=(app_id,),
        )

    if classified.empty:
        st.warning("No classified reviews for this app yet - classify it on the Dashboard page first.")
        st.stop()

    pain_pool = classified[classified["sentiment"].isin(["negative", "neutral"])]
    pain_points = (
        pain_pool.groupby("theme")
        .agg(mentions=("review_id", "count"), avg_severity=("severity", "mean"))
        .reset_index()
        .sort_values(["mentions", "avg_severity"], ascending=False)
        .head(15)
    )

    if not reviewer:
        st.warning("Enter your name in the sidebar to start ranking.")
        st.stop()

    existing = labeling.load_rankings()
    existing = existing[(existing["reviewer"] == reviewer) & (existing["app_id"] == app_id)]

    st.subheader(f"Top pain points - {app_name}")
    ratings: dict[str, int] = {}
    for _, pp in pain_points.iterrows():
        theme = pp["theme"]
        prior = existing.loc[existing["pain_point"] == theme, "rating"]
        default = int(prior.iloc[0]) if len(prior) else 3
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(f"**{theme}** - {pp['mentions']} mentions, avg severity {pp['avg_severity']:.1f}")
                examples = pain_pool[pain_pool["theme"] == theme]["cleaned_text"].head(2)
                for ex in examples:
                    st.caption(f'"{ex}"')
            with c2:
                ratings[theme] = st.select_slider(
                    "Importance", options=[1, 2, 3, 4, 5], value=default, key=f"rate_{app_id}_{theme}"
                )

    if st.button("Submit rankings", type="primary"):
        ordered = sorted(ratings.items(), key=lambda kv: -kv[1])
        for rank, (theme, rating) in enumerate(ordered, start=1):
            labeling.save_ranking(reviewer, app_id, theme, rank, rating)
        st.success(f"Saved rankings for {reviewer} on {app_name}.")
