"""
Multiple comparison correction — necessary because comparing 3 agent
versions pairwise (A vs B, A vs C, B vs C) means running 3 separate
significance tests. At an uncorrected alpha=0.05 per test, the probability
of AT LEAST ONE false positive across all 3 tests (the family-wise error
rate) is higher than 5% — roughly 1 - (0.95)^3 ≈ 14% if the tests were
independent. This module corrects for that inflation.

Two methods, both implemented since they represent a real, meaningful
tradeoff:

1. BONFERRONI: simplest, most conservative. Divides alpha by the number
   of comparisons (or equivalently, multiplies each p-value by the number
   of comparisons). Controls the family-wise error rate (probability of
   ANY false positive) strictly, at the cost of reduced statistical power
   — a real difference between two agent versions is more likely to be
   missed (a false negative) as the number of comparisons grows.

2. BENJAMINI-HOCHBERG (BH): controls the FALSE DISCOVERY RATE instead
   (expected proportion of false positives AMONG the tests called
   significant) rather than the probability of any false positive at
   all. Less conservative than Bonferroni, more statistical power —
   generally the more practical choice when several comparisons are being
   made and some tolerance for a controlled, known rate of false
   positives is acceptable (which is reasonable here: correctly
   identifying real performance differences between agent versions
   matters more than an extremely strict zero-tolerance guarantee against
   any single false positive).

This project reports BOTH methods together rather than picking one
silently, since the choice has a real, honest tradeoff that the
leaderboard's audience should be able to see, not have hidden.
"""

import numpy as np


def bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> dict:
    """
    Returns corrected significance decisions and the adjusted alpha
    threshold — NOT just adjusted p-values in isolation, so results can
    be reported with the actual threshold used, not a number requiring
    the reader to already know the correction formula.
    """
    n = len(p_values)
    adjusted_alpha = alpha / n
    significant = [p < adjusted_alpha for p in p_values]
    adjusted_p_values = [min(p * n, 1.0) for p in p_values]

    return {
        "method": "Bonferroni",
        "n_comparisons": n,
        "original_alpha": alpha,
        "adjusted_alpha": adjusted_alpha,
        "p_values": p_values,
        "adjusted_p_values": adjusted_p_values,
        "significant": significant,
    }


def benjamini_hochberg_correction(p_values: list[float], alpha: float = 0.05) -> dict:
    """
    Benjamini-Hochberg procedure: sort p-values ascending, find the
    largest p-value satisfying p_(i) <= (i/n) * alpha, and call that one
    AND all smaller p-values significant.
    """
    n = len(p_values)
    p_array = np.asarray(p_values)
    sorted_indices = np.argsort(p_array)
    sorted_p = p_array[sorted_indices]

    thresholds = np.array([(i + 1) / n * alpha for i in range(n)])
    below_threshold = sorted_p <= thresholds

    if not np.any(below_threshold):
        largest_significant_rank = -1  # nothing significant
    else:
        # Largest rank where p_(i) <= threshold_(i)
        largest_significant_rank = np.max(np.where(below_threshold)[0])

    significant_sorted = np.zeros(n, dtype=bool)
    if largest_significant_rank >= 0:
        significant_sorted[: largest_significant_rank + 1] = True

    # Map back to original order
    significant = np.zeros(n, dtype=bool)
    significant[sorted_indices] = significant_sorted

    # BH-adjusted p-values (step-up, monotonic)
    adjusted_sorted = np.minimum.accumulate((sorted_p * n / (np.arange(n) + 1))[::-1])[::-1]
    adjusted_sorted = np.clip(adjusted_sorted, 0, 1)
    adjusted_p_values = np.empty(n)
    adjusted_p_values[sorted_indices] = adjusted_sorted

    return {
        "method": "Benjamini-Hochberg (FDR)",
        "n_comparisons": n,
        "original_alpha": alpha,
        "p_values": list(p_values),
        "adjusted_p_values": adjusted_p_values.tolist(),
        "significant": significant.tolist(),
    }


def compare_all_versions_pairwise(results_by_version: dict, metric_fn, alpha: float = 0.05,
                                  n_iterations: int = 10000, random_seed: int = None) -> dict:
    """
    Runs a permutation test for EVERY pair of agent versions on a given
    metric, then applies BOTH correction methods to the resulting set of
    p-values. This is the actual function the benchmark runner/leaderboard
    calls — the two correction functions above are the reusable building
    blocks it's built from.

    Args:
        results_by_version: {version_id: [per-question metric values]}
        metric_fn: function(a, b) -> float, passed through to
            permutation_test (see permutation.py)
    """
    from src.stats.permutation import permutation_test

    version_ids = list(results_by_version.keys())
    pairs = [(a, b) for i, a in enumerate(version_ids) for b in version_ids[i + 1:]]

    raw_results = []
    for a_id, b_id in pairs:
        result = permutation_test(
            results_by_version[a_id], results_by_version[b_id],
            statistic_fn=metric_fn, n_iterations=n_iterations, random_seed=random_seed,
        )
        raw_results.append({"pair": (a_id, b_id), **result})

    p_values = [r["p_value"] for r in raw_results]
    bonferroni = bonferroni_correction(p_values, alpha)
    bh = benjamini_hochberg_correction(p_values, alpha)

    for i, r in enumerate(raw_results):
        r["bonferroni_significant"] = bonferroni["significant"][i]
        r["bonferroni_adjusted_p"] = bonferroni["adjusted_p_values"][i]
        r["bh_significant"] = bh["significant"][i]
        r["bh_adjusted_p"] = bh["adjusted_p_values"][i]

    return {
        "pairwise_results": raw_results,
        "n_comparisons": len(pairs),
        "alpha": alpha,
    }
