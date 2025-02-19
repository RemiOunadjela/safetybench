"""Statistical testing utilities for moderation metric comparisons."""

from __future__ import annotations

from typing import Callable

import numpy as np


def bootstrap_ci(
    data: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_bootstrap: int = 10_000,
    confidence: float = 0.95,
    seed: int | None = None,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for an arbitrary statistic.

    Args:
        data: 1-D array of observations.
        statistic: Function that computes a scalar from a 1-D array.
        n_bootstrap: Number of bootstrap resamples.
        confidence: Confidence level (e.g. 0.95 for 95% CI).
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper).
    """
    data = np.asarray(data)
    rng = np.random.default_rng(seed)

    point_estimate = float(statistic(data))
    n = len(data)

    bootstrap_stats = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        sample = rng.choice(data, size=n, replace=True)
        bootstrap_stats[i] = statistic(sample)

    alpha = 1 - confidence
    ci_lower = float(np.percentile(bootstrap_stats, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_stats, 100 * (1 - alpha / 2)))

    return point_estimate, ci_lower, ci_upper


def mcnemar_test(
    predictions_a: np.ndarray,
    predictions_b: np.ndarray,
    labels: np.ndarray,
) -> dict[str, float]:
    """McNemar's test for comparing two classifiers on the same dataset.

    Tests whether the two models have statistically different error rates.

    Args:
        predictions_a: Binary predictions from model A.
        predictions_b: Binary predictions from model B.
        labels: Ground-truth binary labels.

    Returns:
        Dict with 'statistic', 'p_value', 'n_discordant_a', 'n_discordant_b'.
    """
    from scipy.stats import chi2

    predictions_a = np.asarray(predictions_a, dtype=bool)
    predictions_b = np.asarray(predictions_b, dtype=bool)
    labels = np.asarray(labels, dtype=bool)

    correct_a = predictions_a == labels
    correct_b = predictions_b == labels

    # b01: A wrong, B right
    b01 = int((~correct_a & correct_b).sum())
    # b10: A right, B wrong
    b10 = int((correct_a & ~correct_b).sum())

    n_discordant = b01 + b10
    if n_discordant == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "n_discordant_a": b10,
            "n_discordant_b": b01,
        }

    # Edwards continuity correction
    statistic = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p_value = 1 - chi2.cdf(statistic, df=1)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "n_discordant_a": b10,
        "n_discordant_b": b01,
    }


def permutation_test(
    metric_a: float,
    metric_b: float,
    data_a: np.ndarray,
    data_b: np.ndarray,
    statistic: Callable[[np.ndarray], float] = np.mean,
    n_permutations: int = 10_000,
    seed: int | None = None,
) -> dict[str, float]:
    """Two-sample permutation test for comparing a metric between two groups.

    Args:
        metric_a: Observed metric value for group A.
        metric_b: Observed metric value for group B.
        data_a: Raw data for group A.
        data_b: Raw data for group B.
        statistic: Function to compute the statistic.
        n_permutations: Number of random permutations.
        seed: Random seed.

    Returns:
        Dict with 'observed_diff', 'p_value'.
    """
    rng = np.random.default_rng(seed)
    observed_diff = abs(metric_a - metric_b)

    combined = np.concatenate([data_a, data_b])
    n_a = len(data_a)
    count = 0

    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_a = statistic(combined[:n_a])
        perm_b = statistic(combined[n_a:])
        if abs(perm_a - perm_b) >= observed_diff:
            count += 1

    return {
        "observed_diff": float(observed_diff),
        "p_value": float(count / n_permutations),
    }
