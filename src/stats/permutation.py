"""
Permutation test for comparing two agent versions on a given metric —
e.g., "is Claude Sonnet 4.5's task accuracy significantly higher than
Qwen3-4B's, or could this difference plausibly be due to chance given the
sample size?"

WHY A PERMUTATION TEST (rather than a t-test or z-test): permutation
tests make NO distributional assumption (no normality, no assumption that
variance is equal across groups) — they only assume EXCHANGEABILITY under
the null hypothesis (if there's truly no difference between agent
versions, it shouldn't matter which "version label" was attached to which
observed result). This is a weaker, more defensible assumption than
normality for this project's actual metrics: task-accuracy scores and
hallucination flags are bounded/binary, not continuous and normally
distributed, and sample sizes (up to ~100 questions per category) are
modest — exactly the conditions where a permutation test's lack of
distributional assumptions matters most.

This is a Monte Carlo (approximate) permutation test, not exact
enumeration of every possible permutation — exact enumeration is only
computationally feasible for very small samples (C(n_a+n_b, n_a) grows
extremely fast). 10,000 random permutations gives a p-value accurate to
roughly +/- 0.01 at the 0.05 significance level, which is sufficient
precision for this project's sample sizes.
"""

import numpy as np


def permutation_test(group_a: list[float], group_b: list[float],
                     statistic_fn=None, n_iterations: int = 10000,
                     random_seed: int = None, alternative: str = "two-sided") -> dict:
    """
    Args:
        group_a, group_b: the two samples being compared (e.g., per-
            question accuracy scores for two different agent versions)
        statistic_fn: function(a, b) -> float, the test statistic to
            permute. Defaults to difference in means (mean(a) - mean(b)),
            which works for both continuous metrics (latency) and
            binary/proportion metrics (accuracy, hallucination rate),
            since the mean of a 0/1 variable IS the proportion.
        alternative: "two-sided" (default), "greater" (is A > B), or
            "less" (is A < B)

    Returns a dict with the observed statistic, p-value, and the number
    of permutations run — NOT just a bare p-value, so the result can be
    reported with full context (sample sizes, iteration count) rather
    than a number stripped of the methodology that produced it.
    """
    if statistic_fn is None:
        statistic_fn = lambda a, b: np.mean(a) - np.mean(b)

    rng = np.random.default_rng(random_seed)
    group_a = np.asarray(group_a)
    group_b = np.asarray(group_b)
    n_a, n_b = len(group_a), len(group_b)

    observed_statistic = statistic_fn(group_a, group_b)

    pooled = np.concatenate([group_a, group_b])
    n_total = len(pooled)

    permuted_statistics = np.empty(n_iterations)
    for i in range(n_iterations):
        shuffled = rng.permutation(pooled)
        perm_a = shuffled[:n_a]
        perm_b = shuffled[n_a:]
        permuted_statistics[i] = statistic_fn(perm_a, perm_b)

    if alternative == "two-sided":
        p_value = np.mean(np.abs(permuted_statistics) >= np.abs(observed_statistic))
    elif alternative == "greater":
        p_value = np.mean(permuted_statistics >= observed_statistic)
    elif alternative == "less":
        p_value = np.mean(permuted_statistics <= observed_statistic)
    else:
        raise ValueError(f"alternative must be 'two-sided', 'greater', or 'less', got {alternative!r}")

    return {
        "observed_statistic": float(observed_statistic),
        "p_value": float(p_value),
        "alternative": alternative,
        "n_iterations": n_iterations,
        "n_a": n_a,
        "n_b": n_b,
    }
