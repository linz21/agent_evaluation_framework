"""
Paired significance tests for comparing two agent versions that answered
the SAME golden-dataset questions.

WHY PAIRED: src/stats/permutation.py shuffles the pooled results of the
two versions as if they were independent samples. But both versions
answer the same 40 questions, so each question gives a matched pair of
results. A paired test compares the two versions WITHIN each question,
which removes question-to-question difficulty from the comparison and is
usually more sensitive.

Two tests:
  - paired_sign_flip_test: for any metric (0/1 or continuous). Under the
    null (no difference between versions), the sign of each question's
    difference (A - B) is equally likely to be + or -, so we randomly flip
    signs and see how often the mean difference is as extreme as observed.
  - mcnemar_exact_test: for 0/1 metrics. Only the DISCORDANT questions
    (one version right, the other wrong) carry information; under the
    null, each discordant question is a fair coin flip (exact binomial).
"""

import numpy as np
from scipy import stats


def paired_sign_flip_test(a, b, n_iterations: int = 100000, random_seed: int = None) -> dict:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired test requires equal-length, question-aligned arrays")
    d = a - b
    n = len(d)
    observed = d.mean()

    rng = np.random.default_rng(random_seed)
    signs = rng.choice([-1.0, 1.0], size=(n_iterations, n))
    permuted = (signs * d).mean(axis=1)
    p_value = float(np.mean(np.abs(permuted) >= abs(observed) - 1e-12))

    return {"test": "paired sign-flip permutation", "observed_mean_diff": float(observed),
            "p_value": p_value, "n_pairs": n, "n_iterations": n_iterations}


def mcnemar_exact_test(a, b) -> dict:
    """a, b: 0/1 arrays aligned by question. Exact two-sided McNemar test."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if len(a) != len(b):
        raise ValueError("paired test requires equal-length, question-aligned arrays")
    a_only = int(((a == 1) & (b == 0)).sum())
    b_only = int(((a == 0) & (b == 1)).sum())
    n_disc = a_only + b_only
    p_value = 1.0 if n_disc == 0 else float(stats.binomtest(a_only, n_disc, 0.5).pvalue)
    return {"test": "McNemar exact", "a_only": a_only, "b_only": b_only,
            "n_discordant": n_disc, "p_value": p_value, "n_pairs": len(a)}


def wilcoxon_signed_rank(a, b) -> dict:
    """Paired non-parametric test for continuous metrics (e.g. latency)."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return {"test": "Wilcoxon signed-rank", "p_value": float(stats.wilcoxon(a, b).pvalue),
            "n_pairs": len(a)}
