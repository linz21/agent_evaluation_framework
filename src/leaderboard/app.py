"""
Streamlit leaderboard for the Agent Evaluation Framework — ranks both
agent versions across all 4 metrics with bootstrap CIs and significance
testing, plus a drill-down view into individual question comparisons.

Usage:
    streamlit run src/leaderboard/app.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import streamlit as st

from src.stats.bootstrap import bootstrap_ci_bca
from src.stats.permutation import permutation_test

VERSIONS = ["qwen3-4b", "claude-sonnet-4.5"]
VERSION_LABELS = {"qwen3-4b": "Qwen3-4B (local)", "claude-sonnet-4.5": "Claude Sonnet 4.5 (API)"}
RANDOM_SEED = 42

st.set_page_config(page_title="Agent Evaluation Leaderboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #F7F5F0; }
    h1, h2, h3 { color: #1F3A5F; font-family: 'Georgia', serif; }
    .metric-card {
        background: white; border-radius: 6px; padding: 16px 20px;
        border-left: 4px solid #C9A227; margin-bottom: 10px;
    }
    .sig-badge {
        display: inline-block; padding: 2px 10px; border-radius: 12px;
        font-size: 0.8em; font-weight: 600; margin-left: 8px;
    }
    .sig-yes { background: #E3EDE0; color: #2C5F2D; }
    .sig-no { background: #EFEFEF; color: #6B6B6B; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_results():
    results = {}
    for v in VERSIONS:
        with open(f"data/results/{v}.json") as f:
            results[v] = json.load(f)
    return results


@st.cache_data
def compute_metric_stats(_results_tuple, field: str, as_binary: bool, higher_is_better: bool):
    results = dict(_results_tuple)
    data_by_version = {}
    for v in VERSIONS:
        vals = [r[field] for r in results[v] if r[field] is not None]
        if as_binary:
            vals = [1.0 if x else 0.0 for x in vals]
        data_by_version[v] = np.array(vals, dtype=float)

    stats = {}
    for v in VERSIONS:
        ci = bootstrap_ci_bca(data_by_version[v].tolist(), n_iterations=5000, random_seed=RANDOM_SEED)
        stats[v] = ci

    perm = permutation_test(
        data_by_version[VERSIONS[0]].tolist(), data_by_version[VERSIONS[1]].tolist(),
        n_iterations=5000, random_seed=RANDOM_SEED,
    )
    return stats, perm


results = load_results()
results_tuple = tuple(sorted(results.items()))

st.title("🌽 Agent Evaluation Leaderboard")
st.caption(
    "Comparing two configurations of the Crop Advisory ReAct Agent across 40 human-reviewed "
    "test questions — bootstrap confidence intervals and permutation significance tests, "
    "not single-number comparisons."
)

st.header("Leaderboard")

metrics_config = [
    ("Task Accuracy", "accuracy_correct", True, True),
    ("Hallucination Rate", "hallucinated", True, False),
    ("Tool Selection Correctness", "tool_selection_correct", True, True),
    ("Latency (seconds)", "latency_seconds", False, False),
]

cols = st.columns(len(metrics_config))
for col, (label, field, as_binary, higher_better) in zip(cols, metrics_config):
    stats, perm = compute_metric_stats(results_tuple, field, as_binary, higher_better)
    with col:
        st.markdown(f"**{label}**")
        for v in VERSIONS:
            s = stats[v]
            unit = "" if as_binary else "s"
            st.markdown(
                f"<div class='metric-card'>{VERSION_LABELS[v]}<br>"
                f"<span style='font-size:1.4em; color:#1F3A5F; font-weight:700;'>"
                f"{s['point_estimate']:.3f}{unit}</span><br>"
                f"<span style='color:#888; font-size:0.85em;'>"
                f"95% CI [{s['ci_lower']:.3f}, {s['ci_upper']:.3f}]</span></div>",
                unsafe_allow_html=True,
            )
        sig = perm["p_value"] < 0.05
        badge_class = "sig-yes" if sig else "sig-no"
        badge_text = f"Significant (p={perm['p_value']:.3f})" if sig else f"Not significant (p={perm['p_value']:.3f})"
        st.markdown(f"<span class='sig-badge {badge_class}'>{badge_text}</span>", unsafe_allow_html=True)

st.divider()

# ── DRILL-DOWN ──────────────────────────────────────────────────────────
st.header("Drill Down Into Individual Results")

categories = sorted(set(r["category"] for r in results[VERSIONS[0]]))
selected_category = st.selectbox("Category", categories)

questions_in_category = sorted(set(
    r["question"] for r in results[VERSIONS[0]] if r["category"] == selected_category
))
selected_question = st.selectbox("Question", questions_in_category)

col1, col2 = st.columns(2)
for col, version in zip([col1, col2], VERSIONS):
    entry = next(r for r in results[version] if r["question"] == selected_question)
    with col:
        st.subheader(VERSION_LABELS[version])
        st.markdown(f"**Correct:** {entry['accuracy_correct']}  |  **Hallucinated:** {entry['hallucinated']}  |  **Tools OK:** {entry['tool_selection_correct']}")
        st.markdown(f"**Latency:** {entry['latency_seconds']:.1f}s  |  **Tools called:** {entry['tools_called']}")
        st.markdown("**Answer:**")
        st.info(entry["agent_answer"])
        with st.expander("Accuracy reasoning"):
            st.write(entry["accuracy_reasoning"])
        with st.expander("Hallucination reasoning"):
            st.write(entry["hallucination_reasoning"])
            if entry["unsupported_claims"]:
                st.write("Unsupported claims:", entry["unsupported_claims"])
        with st.expander("Retrieved context (tool observations)"):
            st.text(entry.get("retrieved_context") or "(none)")
        with st.expander("Memory context"):
            st.text(entry.get("memory_context") or "(none)")

st.caption(
    "Built on src/stats/ (validated bootstrap CI + permutation testing) and "
    "src/eval/metrics.py (RAGAS-aligned hallucination checking). See README for full methodology."
)
