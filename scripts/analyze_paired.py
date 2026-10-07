"""
Paired comparison of the two agent versions (Qwen3-4B vs. Claude) on the
same golden-dataset questions. Complements scripts/analyze_results.py,
which treats the two versions as independent samples; report both.

Results are matched by question text, and a question is used for a metric
only if BOTH versions have a non-null value for it (judge failures are
dropped for the pair, not counted as 0 or 1).

Usage:
    python scripts/analyze_paired.py
"""

import json
import sys

sys.path.insert(0, ".")

import numpy as np

from src.stats.paired import paired_sign_flip_test, mcnemar_exact_test, wilcoxon_signed_rank

VERSIONS = ["qwen3-4b", "claude-sonnet-4.5"]
RANDOM_SEED = 42

# (field, label, binary, higher_is_better)
METRICS = [
    ("accuracy_correct", "Task Accuracy (higher = better)", True, True),
    ("hallucinated", "Hallucination Rate (LOWER = better)", True, False),
    ("tool_selection_correct", "Tool Selection Correctness (higher = better)", True, True),
    ("latency_seconds", "Latency in seconds (LOWER = better)", False, False),
]


def load(version_id: str) -> dict:
    with open(f"data/results/{version_id}.json") as f:
        return {r["question"]: r for r in json.load(f)}


def main():
    a_res, b_res = load(VERSIONS[0]), load(VERSIONS[1])
    questions = [q for q in a_res if q in b_res]
    print(f"{len(questions)} questions answered by both versions "
          f"({VERSIONS[0]} = A, {VERSIONS[1]} = B)")

    for field, label, binary, higher_better in METRICS:
        pairs = [(a_res[q][field], b_res[q][field]) for q in questions
                 if a_res[q][field] is not None and b_res[q][field] is not None]
        a = np.array([float(x) for x, _ in pairs])
        b = np.array([float(y) for _, y in pairs])

        print(f"\n{'=' * 70}\nMETRIC: {label}\n{'=' * 70}")
        print(f"n pairs: {len(a)}   A mean: {a.mean():.4f}   B mean: {b.mean():.4f}   "
              f"A - B: {a.mean() - b.mean():+.4f}")

        sf = paired_sign_flip_test(a, b, random_seed=RANDOM_SEED)
        print(f"Paired sign-flip permutation test: p = {sf['p_value']:.4f}")

        if binary:
            mc = mcnemar_exact_test(a, b)
            print(f"McNemar exact test: p = {mc['p_value']:.4f}   "
                  f"(A-only = {mc['a_only']}, B-only = {mc['b_only']}, "
                  f"discordant = {mc['n_discordant']})")
            p = mc["p_value"]
        else:
            w = wilcoxon_signed_rank(a, b)
            print(f"Wilcoxon signed-rank test: p = {w['p_value']:.2e}")
            p = w["p_value"]

        if p < 0.05:
            a_better = (a.mean() > b.mean()) == higher_better
            print(f"  -> significant at alpha=0.05: "
                  f"{VERSIONS[0] if a_better else VERSIONS[1]} performs better")
        else:
            print("  -> not significant at alpha=0.05")

    print(f"\n{'=' * 70}")
    print("NOTE: run alongside scripts/analyze_results.py (unpaired). Results")
    print("near p = 0.05 should be reported as borderline, and this paired")
    print("analysis was added after seeing the unpaired results.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
