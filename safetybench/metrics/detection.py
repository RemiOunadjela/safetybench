"""Detection and coverage metrics for content moderation systems.

These metrics measure how effectively a moderation system catches violating
content, with emphasis on proactive detection (catching violations before
user reports) and zero-view enforcement.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def proactive_detection_rate(
    flagged: np.ndarray,
    user_reported: np.ndarray,
    true_violations: np.ndarray,
) -> float:
    """Fraction of true violations caught by the system before any user report.

    PDR = |proactive_catches| / |true_violations|

    Args:
        flagged: Boolean array -- content flagged by the automated system.
        user_reported: Boolean array -- content reported by users.
        true_violations: Boolean array -- ground-truth violation labels.

    Returns:
        PDR in [0, 1].
    """
    flagged = np.asarray(flagged, dtype=bool)
    user_reported = np.asarray(user_reported, dtype=bool)
    true_violations = np.asarray(true_violations, dtype=bool)

    total_violations = true_violations.sum()
    if total_violations == 0:
        return 0.0

    proactive_catches = flagged & true_violations & ~user_reported
    return float(proactive_catches.sum() / total_violations)


def action_rate(
    actioned: np.ndarray,
    reviewed: np.ndarray,
) -> float:
    """Fraction of reviewed content that was actioned (removed, restricted, etc.).

    Args:
        actioned: Boolean array -- content that received a moderation action.
        reviewed: Boolean array -- content that was reviewed (by human or model).

    Returns:
        Action rate in [0, 1].
    """
    actioned = np.asarray(actioned, dtype=bool)
    reviewed = np.asarray(reviewed, dtype=bool)

    total_reviewed = reviewed.sum()
    if total_reviewed == 0:
        return 0.0

    return float((actioned & reviewed).sum() / total_reviewed)


def zero_view_violation_rate(
    view_counts: np.ndarray,
    actioned: np.ndarray,
    true_violations: np.ndarray,
) -> float:
    """Fraction of true violations actioned before accumulating any views.

    This is the gold standard for proactive moderation -- violations removed
    before a single user sees them.

    Args:
        view_counts: Integer array -- view count at time of action.
        actioned: Boolean array -- content that was actioned.
        true_violations: Boolean array -- ground-truth violation labels.

    Returns:
        0VV rate in [0, 1].
    """
    view_counts = np.asarray(view_counts, dtype=np.int64)
    actioned = np.asarray(actioned, dtype=bool)
    true_violations = np.asarray(true_violations, dtype=bool)

    total_violations = true_violations.sum()
    if total_violations == 0:
        return 0.0

    zero_view_catches = (view_counts == 0) & actioned & true_violations
    return float(zero_view_catches.sum() / total_violations)


def detection_rate_by_category(
    predictions: np.ndarray,
    labels: np.ndarray,
    categories: pd.Series,
) -> dict[str, float]:
    """Per-category detection (recall) rates.

    Args:
        predictions: Boolean array of model predictions.
        labels: Boolean array of ground-truth labels.
        categories: Series of category strings aligned with predictions/labels.

    Returns:
        Dict mapping category name to recall.
    """
    predictions = np.asarray(predictions, dtype=bool)
    labels = np.asarray(labels, dtype=bool)

    results: dict[str, float] = {}
    for cat in categories.unique():
        mask = categories == cat
        cat_labels = labels[mask]
        cat_preds = predictions[mask]
        positives = cat_labels.sum()
        if positives == 0:
            results[cat] = 0.0
        else:
            results[cat] = float((cat_preds & cat_labels).sum() / positives)

    return results
