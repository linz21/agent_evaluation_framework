"""
Streamlit leaderboard for the Agent Evaluation Framework — ranks the agent
versions across all 4 metrics with bootstrap CIs and significance testing,
plus a drill-down view into individual question comparisons.

Usage:
    streamlit run src/leaderboard/app.py
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import streamlit as st

from src.stats.bootstrap import bootstrap_ci_bca
from src.stats.permutation import permutation_test
from src.stats.paired import mcnemar_exact_test, wilcoxon_signed_rank

ALL_VERSIONS = ["qwen3-4b", "claude-sonnet-4.5", "claude-sonnet-5.5"]
VERSION_LABELS = {
    "qwen3-4b": "Qwen3-4B (local)",
    "claude-sonnet-4.5": "Claude Sonnet 4.5 (API, retired model)",
    "claude-sonnet-5.5": "Claude Sonnet 5.5 (API)",
}
RANDOM_SEED = 42

st.set_page_config(page_title="Agent Evaluation Leaderboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #F7F5F0; }
    h1, h2, h3 { color: #1F3A5F; font-family: 'Georgia', serif; }
    .metric-card {
        background: white; border-radius: 6px; padding: 12px 16px;
        border-left: 4px solid #C9A227; margin-bottom: 8px;
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
    for v in ALL_VERSIONS:
        path = ROOT / "data" / "results" / f"{v}.json"
        if path.exists():
            with open(path) as f:
                results[v] = json.load(f)
    return results


def metric_values(rows, field, as_binary):
    vals = [r[field] for r in rows if r[field] is not None]
    if as_binary:
        vals = [1.0 if x else 0.0 for x in vals]
    return np.array(vals, dtype=float)


@st.cache_data
def version_ci(_rows_tuple, version, field, as_binary):
    rows = dict(_rows_tuple)[version]
    data = metric_values(rows, field, as_binary)
    return bootstrap_ci_bca(data.tolist(), n_iterations=5000, random_seed=RANDOM_SEED)


@st.cache_data
def pair_tests(_rows_tuple, va, vb, field, as_binary):
    """Paired test (same questions in both versions) + the original unpaired
    permutation test for reference."""
    rows = dict(_rows_tuple)
    a_by_q = {r["question"]: r for r in rows[va]}
    b_by_q = {r["question"]: r for r in rows[vb]}
    pairs = [(a_by_q[q][field], b_by_q[q][field]) for q in a_by_q
             if q in b_by_q and a_by_q[q][field] is not None and b_by_q[q][field] is not None]
    a = np.array([float(x) for x, _ in pairs])
    b = np.array([float(y) for _, y in pairs])
    if as_binary:
        paired = mcnemar_exact_test(a, b)
        paired["name"] = "McNemar exact"
    else:
        paired = wilcoxon_signed_rank(a, b)
        paired["name"] = "Wilcoxon signed-rank"
    unpaired = permutation_test(
        metric_values(rows[va], field, as_binary).tolist(),
        metric_values(rows[vb], field, as_binary).tolist(),
        n_iterations=5000, random_seed=RANDOM_SEED,
    )
    return paired, unpaired


results = load_results()
VERSIONS = [v for v in ALL_VERSIONS if v in results]
results_tuple = tuple(sorted(results.items()))

st.title("🌽 Agent Evaluation Leaderboard")
st.caption(
    f"Comparing {len(VERSIONS)} configurations of the Crop Advisory ReAct Agent across 40 "
    "human-reviewed test questions — bootstrap confidence intervals and significance tests, "
    "not single-number comparisons."
)

st.header("Leaderboard")

st.markdown("**Pick two versions to test against each other** (all versions are always shown with their CIs):")
sel_a, sel_b = st.columns(2)
default_b = VERSIONS.index("claude-sonnet-5.5") if "claude-sonnet-5.5" in VERSIONS else len(VERSIONS) - 1
version_a = sel_a.selectbox("Version A", VERSIONS, index=0, format_func=lambda v: VERSION_LABELS[v])
version_b = sel_b.selectbox("Version B", VERSIONS, index=default_b, format_func=lambda v: VERSION_LABELS[v])

metrics_config = [
    ("Task Accuracy", "accuracy_correct", True, True),
    ("Hallucination Rate", "hallucinated", True, False),
    ("Tool Selection Correctness", "tool_selection_correct", True, True),
    ("Latency (seconds)", "latency_seconds", False, False),
]

cols = st.columns(len(metrics_config))
for col, (label, field, as_binary, higher_better) in zip(cols, metrics_config):
    with col:
        st.markdown(f"**{label}**")
        for v in VERSIONS:
            s = version_ci(results_tuple, v, field, as_binary)
            unit = "" if as_binary else "s"
            st.markdown(
                f"<div class='metric-card'>{VERSION_LABELS[v]}<br>"
                f"<span style='font-size:1.4em; color:#1F3A5F; font-weight:700;'>"
                f"{s['point_estimate']:.3f}{unit}</span><br>"
                f"<span style='color:#888; font-size:0.85em;'>"
                f"95% CI [{s['ci_lower']:.3f}, {s['ci_upper']:.3f}]</span></div>",
                unsafe_allow_html=True,
            )
        if version_a == version_b:
            st.caption("Pick two different versions to compare.")
            continue
        paired, unpaired = pair_tests(results_tuple, version_a, version_b, field, as_binary)
        sig = paired["p_value"] < 0.05
        badge_class = "sig-yes" if sig else "sig-no"
        p = paired["p_value"]
        p_txt = "<0.001" if p < 0.001 else f"{p:.3f}"
        badge_text = f"Significant (p={p_txt})" if sig else f"Not significant (p={p_txt})"
        st.markdown(f"<span class='sig-badge {badge_class}'>{badge_text}</span>", unsafe_allow_html=True)
        st.caption(f"{paired['name']} (paired, n={paired['n_pairs']}). "
                   f"Unpaired permutation p={unpaired['p_value']:.3f}.")

st.info(
    "**How to read the tests.** Every version answered the same 40 questions, so the main test "
    "is *paired* (McNemar for yes/no metrics, Wilcoxon for latency), which compares versions "
    "question by question. The unpaired permutation p-value is shown for reference. Results "
    "near p = 0.05 are borderline with only 40 questions and one run each. The paired analysis "
    "was added after the first (unpaired) results were seen. Sonnet 4.5 results are historical "
    "(that model retires 2026-11-30)."
)

st.divider()

# ── DRILL-DOWN ──────────────────────────────────────────────────────────
st.header("Drill Down Into Individual Results")

categories = sorted(set(r["category"] for r in results[VERSIONS[0]]))
selected_category = st.selectbox("Category", categories)

questions_in_category = sorted(set(
    r["question"] for r in results[VERSIONS[0]] if r["category"] == selected_category
))
selected_question = st.selectbox("Question", questions_in_category)

for col, version in zip(st.columns(len(VERSIONS)), VERSIONS):
    entry = next((r for r in results[version] if r["question"] == selected_question), None)
    with col:
        st.subheader(VERSION_LABELS[version])
        if entry is None:
            st.write("(no result for this question)")
            continue
        st.markdown(f"**Correct:** {entry['accuracy_correct']}  |  **Hallucinated:** {entry['hallucinated']}  |  **Tools OK:** {entry['tool_selection_correct']}")
        lat = entry["latency_seconds"]
        st.markdown(f"**Latency:** {f'{lat:.1f}s' if lat is not None else 'n/a'}  |  **Tools called:** {entry['tools_called']}")
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
    "Built on src/stats/ (validated bootstrap CI, permutation and paired testing) and "
    "src/eval/metrics.py (RAGAS-aligned hallucination checking). See README for full methodology."
)
