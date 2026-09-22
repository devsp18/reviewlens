"""Real accuracy numbers: prompt v1 vs v2 vs v3, confusion matrices, and error analysis.

Reads results/eval_results.json, produced by scripts/evaluate.py from real human
labels. Never fabricates a number - if that file doesn't exist yet (or is stale),
this page says so instead of inventing something plausible-looking.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

RESULTS_PATH = Path(__file__).resolve().parent.parent.parent / "results" / "eval_results.json"

st.set_page_config(page_title="ReviewLens - Eval Lab", page_icon="🔎", layout="wide")
st.title("Eval Lab")

if not RESULTS_PATH.exists():
    st.warning(
        "No eval results yet. This page only shows real, computed numbers - nothing here "
        "is estimated or hardcoded.\n\n"
        "To generate results: label reviews in the Labeling Studio (Label Reviews mode), "
        "then run `python scripts/evaluate.py` from the project root."
    )
    st.stop()

results = json.loads(RESULTS_PATH.read_text())
prompt_results = results.get("prompt_versions", {})

if not prompt_results:
    st.warning("results/eval_results.json exists but has no scored prompt versions. Re-run scripts/evaluate.py.")
    st.stop()

st.caption(f"{results['n_labeled_reviews']} human-labeled reviews - generated {results['generated_at']}")

# --- accuracy comparison ---
rows = []
for version, r in prompt_results.items():
    rows.append({"prompt": version, "metric": "theme accuracy", "value": r["theme_accuracy"]})
    rows.append({"prompt": version, "metric": "theme macro-F1", "value": r["theme_macro_f1"]})
    rows.append({"prompt": version, "metric": "sentiment accuracy", "value": r["sentiment_accuracy"]})
    rows.append({"prompt": version, "metric": "sentiment macro-F1", "value": r["sentiment_macro_f1"]})
comparison = pd.DataFrame(rows)

fig = px.bar(comparison, x="prompt", y="value", color="metric", barmode="group")
fig.update_layout(yaxis_range=[0, 1], yaxis_tickformat=".0%")
st.plotly_chart(fig, width="stretch")

# --- confusion matrix ---
st.subheader("Confusion matrix")
col1, col2 = st.columns(2)
with col1:
    version = st.selectbox("Prompt version", list(prompt_results.keys()))
with col2:
    axis = st.selectbox("Axis", ["theme", "sentiment"])

r = prompt_results[version]
labels = r[f"{axis}_labels"]
matrix = r[f"{axis}_confusion_matrix"]
heatmap = go.Figure(
    data=go.Heatmap(
        z=matrix,
        x=labels,
        y=labels,
        colorscale="Blues",
        text=matrix,
        texttemplate="%{text}",
    )
)
heatmap.update_layout(xaxis_title="Predicted", yaxis_title="True", height=500)
st.plotly_chart(heatmap, width="stretch")

# --- misclassified gallery ---
st.subheader("Misclassified examples")
misclassified = r["misclassified_examples"]
if not misclassified:
    st.success(f"No misclassifications for {version} on this eval set.")
else:
    st.caption(f"{len(misclassified)} of {r['n']} misclassified ({len(misclassified) / r['n']:.1%})")
    for ex in misclassified[:30]:
        with st.container(border=True):
            st.write(ex["cleaned_text"])
            st.markdown(
                f"True: **{ex['true_theme']}** / **{ex['true_sentiment']}**  →  "
                f"Predicted: **{ex['theme']}** / **{ex['sentiment']}**"
            )
            if ex.get("reasoning"):
                st.caption(f"Model reasoning: {ex['reasoning']}")

# --- PM agreement ---
st.subheader("PM ranking agreement")
pm = results.get("pm_agreement")
if not pm:
    st.info("No PM rankings yet - rank pain points in the Labeling Studio's PM Ranking mode with 2+ reviewers.")
else:
    st.caption(f"{pm['n_reviewers']} reviewers - {pm['metric']}")
    for app_id, pairs in pm["by_app"].items():
        st.markdown(f"**{app_id}**")
        st.dataframe(pd.DataFrame(pairs), hide_index=True)
