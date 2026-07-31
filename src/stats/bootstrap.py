"""
Bootstrap confidence intervals for evaluation metrics.

Two methods implemented:

1. PERCENTILE bootstrap — the simpler, more widely-known method. Resample
   the data with replacement B times, compute the statistic on each
   resample, take the alpha/2 and 1-alpha/2 percentiles of the resulting
   distribution as the CI bounds.

2. BCa (Bias-Corrected and accelerated) bootstrap — Efron's refinement,
   correcting for two things the plain percentile method ignores:
     - BIAS: whether the bootstrap distribution is centered on the
       observed statistic or systematically shifted from it
     - ACCELERATION: whether the statistic's variance itself changes
       across the range of the parameter (skewness of the sampling
       distribution) — relevant here since metrics like accuracy or
       hallucination rate are PROPORTIONS, which are naturally more
       variable near 0.5 than near 0 or 1, violating the plain
       percentile method's implicit symmetry assumption.

BCa is the more statistically appropriate choice for this project's
actual metrics (bounded proportions, often skewed with small samples),
and is offered as the default; percentile is kept available since it's
simpler to explain and is the more commonly recognized method.

References: Efron & Tibshirani (1993), "An Introduction to the Bootstrap",
Chapter 14 (BCa method).
"""

import numpy as np
from scipy import stats


def bootstrap_ci_percentile(data: list[float], statistic_fn=np.mean,
                            n_iterations: int = 10000, confidence_level: float = 0.95,
                            random_seed: int = None) -> dict:
    """
    Standard percentile bootstrap. Simple and widely understood, but can
    be inaccurate for skewed statistics or small samples — see BCa above
    for the more rigorous alternative used by default elsewhere in this
    project.
    """
    rng = np.random.default_rng(random_seed)
    data = np.asarray(data)
    n = len(data)

    point_estimate = statistic_fn(data)
    boot_stats = np.empty(n_iterations)
    for i in range(n_iterations):
        resample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = statistic_fn(resample)

    alpha = 1 - confidence_level
    ci_lower = np.percentile(boot_stats, 100 * (alpha / 2))
    ci_upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))

    return {
        "method": "percentile",
        "point_estimate": float(point_estimate),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "confidence_level": confidence_level,
        "n_iterations": n_iterations,
        "n_observations": n,
    }


def bootstrap_ci_bca(data: list[float], statistic_fn=np.mean,
                     n_iterations: int = 10000, confidence_level: float = 0.95,
                     random_seed: int = None) -> dict:
    """
    BCa bootstrap — corrects the percentile method's CI bounds using:

    z0 (bias correction): how far the median of the bootstrap distribution
    is from the observed statistic, expressed as a normal quantile. If the
    bootstrap distribution is centered exactly on the observed statistic,
    z0 = 0 and no bias correction is needed.

    a (acceleration): estimated via the jackknife (leave-one-out) — how
    much the statistic's standard error changes as each single observation
    is removed. For a perfectly symmetric statistic (e.g., a sample mean
    of normally-distributed data), a = 0. For proportions/rates — this
    project's actual metrics — a is typically nonzero, especially with
    small samples, making this correction genuinely relevant rather than
    a theoretical nicety.

    Falls back to the percentile method's bounds (with a logged notice)
    in the rare degenerate case where the jackknife estimates have zero
    variance (e.g., a constant statistic across all leave-one-out samples).
    """
    rng = np.random.default_rng(random_seed)
    data = np.asarray(data)
    n = len(data)

    point_estimate = statistic_fn(data)

    boot_stats = np.empty(n_iterations)
    for i in range(n_iterations):
        resample = rng.choice(data, size=n, replace=True)
        boot_stats[i] = statistic_fn(resample)

    # Bias correction z0: proportion of bootstrap stats LESS than the
    # observed point estimate, converted to a normal quantile
    prop_less = np.mean(boot_stats < point_estimate)
    # Guard against exactly 0 or 1 (would give +/- infinity)
    prop_less = np.clip(prop_less, 1e-6, 1 - 1e-6)
    z0 = stats.norm.ppf(prop_less)

    # Acceleration via jackknife
    jackknife_stats = np.empty(n)
    for i in range(n):
        jackknife_sample = np.delete(data, i)
        jackknife_stats[i] = statistic_fn(jackknife_sample)
    jackknife_mean = np.mean(jackknife_stats)
    numerator = np.sum((jackknife_mean - jackknife_stats) ** 3)
    denominator = 6 * (np.sum((jackknife_mean - jackknife_stats) ** 2) ** 1.5)

    if denominator == 0:
        # Degenerate case (e.g., constant data) — fall back to percentile
        alpha = 1 - confidence_level
        ci_lower = np.percentile(boot_stats, 100 * (alpha / 2))
        ci_upper = np.percentile(boot_stats, 100 * (1 - alpha / 2))
        return {
            "method": "percentile (BCa degenerate, fell back)",
            "point_estimate": float(point_estimate),
            "ci_lower": float(ci_lower),
            "ci_upper": float(ci_upper),
            "confidence_level": confidence_level,
            "n_iterations": n_iterations,
            "n_observations": n,
        }

    a = numerator / denominator

    alpha = 1 - confidence_level
    z_alpha_lower = stats.norm.ppf(alpha / 2)
    z_alpha_upper = stats.norm.ppf(1 - alpha / 2)

    # BCa-adjusted percentiles
    def _adjusted_percentile(z_alpha):
        numer = z0 + z_alpha
        denom = 1 - a * (z0 + z_alpha)
        adjusted_z = z0 + numer / denom
        return stats.norm.cdf(adjusted_z) * 100

    pct_lower = _adjusted_percentile(z_alpha_lower)
    pct_upper = _adjusted_percentile(z_alpha_upper)
    pct_lower = np.clip(pct_lower, 0.01, 99.99)
    pct_upper = np.clip(pct_upper, 0.01, 99.99)

    ci_lower = np.percentile(boot_stats, pct_lower)
    ci_upper = np.percentile(boot_stats, pct_upper)

    return {
        "method": "BCa",
        "point_estimate": float(point_estimate),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "confidence_level": confidence_level,
        "n_iterations": n_iterations,
        "n_observations": n,
        "bias_correction_z0": float(z0),
        "acceleration_a": float(a),
    }
