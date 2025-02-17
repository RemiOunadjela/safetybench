"""Quality metrics for content moderation: precision, FPR, appeal rates."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import precision_recall_curve


def false_positive_rate_at_threshold(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> float:
    """False positive rate at a given score threshold.

    FPR = FP / (FP + TN)

    Args:
        scores: Continuous model scores in [0, 1].
        labels: Binary ground-truth labels.
        threshold: Decision threshold.

    Returns:
        FPR in [0, 1].
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)

    predictions = scores >= threshold
    negatives = ~labels
    n_negatives = negatives.sum()

    if n_negatives == 0:
        return 0.0

    fp = (predictions & negatives).sum()
    return float(fp / n_negatives)


def false_positive_rate_curve(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """FPR at multiple thresholds.

    Args:
        scores: Continuous model scores.
        labels: Binary ground-truth labels.
        thresholds: Array of thresholds. Defaults to 50 evenly spaced values.

    Returns:
        Tuple of (thresholds, fpr_values).
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 50)

    thresholds = np.asarray(thresholds, dtype=np.float64)
    fpr_values = np.array([
        false_positive_rate_at_threshold(scores, labels, t) for t in thresholds
    ])
    return thresholds, fpr_values


def appeal_overturn_rate(
    appealed: np.ndarray,
    overturned: np.ndarray,
) -> float:
    """Fraction of appeals that resulted in the original decision being overturned.

    A high overturn rate signals precision problems in the moderation pipeline.

    Args:
        appealed: Boolean array -- actions that were appealed.
        overturned: Boolean array -- appeals where the decision was reversed.

    Returns:
        Overturn rate in [0, 1].
    """
    appealed = np.asarray(appealed, dtype=bool)
    overturned = np.asarray(overturned, dtype=bool)

    n_appealed = appealed.sum()
    if n_appealed == 0:
        return 0.0

    return float((overturned & appealed).sum() / n_appealed)


def precision_recall_at_thresholds(
    scores: np.ndarray,
    labels: np.ndarray,
    thresholds: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Precision and recall at specified thresholds.

    If thresholds is None, uses sklearn's automatic threshold selection
    from the precision_recall_curve.

    Returns:
        Dict with keys 'precision', 'recall', 'thresholds'.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)

    if thresholds is None:
        precision, recall, thresholds_out = precision_recall_curve(labels, scores)
        return {
            "precision": precision[:-1],
            "recall": recall[:-1],
            "thresholds": thresholds_out,
        }

    thresholds = np.asarray(thresholds, dtype=np.float64)
    precision_vals = []
    recall_vals = []
    for t in thresholds:
        preds = scores >= t
        tp = (preds & labels).sum()
        fp = (preds & ~labels).sum()
        fn = (~preds & labels).sum()

        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        precision_vals.append(prec)
        recall_vals.append(rec)

    return {
        "precision": np.array(precision_vals),
        "recall": np.array(recall_vals),
        "thresholds": thresholds,
    }
