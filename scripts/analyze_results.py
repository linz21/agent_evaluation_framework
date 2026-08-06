"""
Statistical comparison of the two agent versions (Qwen3-4B vs. Claude
Sonnet 4.5) across all 4 metrics, using the validated statistics module
(src/stats/): BCa bootstrap confidence intervals per version, and a
permutation test for the pairwise difference.

Only 2 versions are being compared (see configs/config.yaml's comment on
why Qwen2.5-1.5B was dropped), so only ONE pairwise comparison is needed
per metric — multiple-comparison correction (src/stats/multiple_
comparison.py) isn't required for this specific run, though it remains
validated and ready if a third version is added later.

Usage:
    python scripts/analyze_results.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, ".")

import numpy as np

from src.stats.bootstrap import bootstrap_ci_bca
from src.stats.permutation import permutation_test

VERSIONS = ["qwen3-4b", "claude-sonnet-4.5"]
RANDOM_SEED = 42  # matches configs/config.yaml's statistics.random_seed


def load_results(version_id: str) -> list[dict]:
    with open(f"data/results/{version_id}.json") as f:
        return json.load(f)


def extract_metric(results: list[dict], field: str, as_binary: bool = True) -> np.ndarray:
    """
    Extracts a metric as a numeric array, dropping any None values (e.g.
    from a judge error) rather than silently treating them as 0 or 1,
    which would bias the result.
    """
    values = [r[field] for r in results if r[field] is not None]
    if as_binary:
        values = [1.0 if v else 0.0 for v in values]
    return np.array(values, dtype=float)


def print_metric_comparison(metric_name: str, data_by_version: dict, higher_is_better: bool = True):
    print(f"\n{'='*70}")
    print(f"METRIC: {metric_name}")
    print(f"{'='*70}")

    for version_id in VERSIONS:
        data = data_by_version[version_id]
        ci = bootstrap_ci_bca(data.tolist(), n_iterations=10000, random_seed=RANDOM_SEED)
        print(f"\n{version_id} (n={len(data)}):")
        print(f"  Point estimate: {ci['point_estimate']:.4f}")
        print(f"  95% BCa CI: [{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}]")

    perm_result = permutation_test(
        data_by_version[VERSIONS[0]].tolist(),
        data_by_version[VERSIONS[1]].tolist(),
        n_iterations=10000,
        random_seed=RANDOM_SEED,
    )
    print(f"\nPermutation test ({VERSIONS[0]} - {VERSIONS[1]}):")
    print(f"  Observed difference: {perm_result['observed_statistic']:.4f}")
    print(f"  p-value: {perm_result['p_value']:.4f}")
    significant = perm_result['p_value'] < 0.05
    print(f"  Significant at alpha=0.05: {significant}")
    if significant:
        better = VERSIONS[0] if (perm_result['observed_statistic'] > 0) == higher_is_better else VERSIONS[1]
        print(f"  -> {better} performs significantly better on this metric")


def main():
    results = {v: load_results(v) for v in VERSIONS}

    accuracy = {v: extract_metric(results[v], "accuracy_correct") for v in VERSIONS}
    print_metric_comparison("Task Accuracy (higher = better)", accuracy, higher_is_better=True)

    hallucination = {v: extract_metric(results[v], "hallucinated") for v in VERSIONS}
    print_metric_comparison("Hallucination Rate (LOWER = better)", hallucination, higher_is_better=False)

    tool_selection = {v: extract_metric(results[v], "tool_selection_correct") for v in VERSIONS}
    print_metric_comparison("Tool Selection Correctness (higher = better)", tool_selection, higher_is_better=True)

    latency = {v: extract_metric(results[v], "latency_seconds", as_binary=False) for v in VERSIONS}
    print_metric_comparison("Latency in seconds (LOWER = better)", latency, higher_is_better=False)

    print(f"\n{'='*70}")
    print("NOTE: statistical significance does not by itself establish")
    print("practical importance — review the actual point estimates and")
    print("CIs above alongside the qualitative findings from manual")
    print("inspection (see inspect_results.py) for full context.")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
