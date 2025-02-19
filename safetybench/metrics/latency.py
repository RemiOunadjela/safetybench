"""Time-to-action and latency metrics for moderation pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd


def median_time_to_action(
    created_at: pd.Series,
    actioned_at: pd.Series,
    actioned: np.ndarray | None = None,
) -> float:
    """Median elapsed time (in seconds) from content creation to moderation action.

    Args:
        created_at: Timestamps of content creation.
        actioned_at: Timestamps of moderation action. NaT for non-actioned items.
        actioned: Optional boolean mask. If None, uses non-NaT entries in actioned_at.

    Returns:
        Median time-to-action in seconds.
    """
    deltas = _compute_deltas(created_at, actioned_at, actioned)
    if len(deltas) == 0:
        return float("nan")
    return float(np.median(deltas))


def time_to_action_percentiles(
    created_at: pd.Series,
    actioned_at: pd.Series,
    percentiles: list[float] | None = None,
    actioned: np.ndarray | None = None,
) -> dict[str, float]:
    """Time-to-action at specified percentiles.

    Args:
        created_at: Timestamps of content creation.
        actioned_at: Timestamps of moderation action.
        percentiles: List of percentiles (e.g. [50, 90, 95, 99]). Defaults to [50, 90, 95, 99].
        actioned: Optional boolean mask.

    Returns:
        Dict mapping 'p{N}' to time in seconds.
    """
    if percentiles is None:
        percentiles = [50.0, 90.0, 95.0, 99.0]

    deltas = _compute_deltas(created_at, actioned_at, actioned)
    if len(deltas) == 0:
        return {f"p{int(p)}": float("nan") for p in percentiles}

    return {
        f"p{int(p)}": float(np.percentile(deltas, p))
        for p in percentiles
    }


def time_to_action_by_category(
    created_at: pd.Series,
    actioned_at: pd.Series,
    categories: pd.Series,
    actioned: np.ndarray | None = None,
) -> dict[str, float]:
    """Median time-to-action broken down by violation category.

    Args:
        created_at: Timestamps of content creation.
        actioned_at: Timestamps of moderation action.
        categories: Category labels.
        actioned: Optional boolean mask.

    Returns:
        Dict mapping category to median TTA in seconds.
    """
    if actioned is None:
        mask = actioned_at.notna()
    else:
        mask = np.asarray(actioned, dtype=bool)

    results: dict[str, float] = {}
    for cat in categories.unique():
        cat_mask = (categories == cat) & mask
        if cat_mask.sum() == 0:
            results[cat] = float("nan")
            continue
        deltas = (actioned_at[cat_mask] - created_at[cat_mask]).dt.total_seconds().values
        valid = deltas[~np.isnan(deltas)]
        results[cat] = float(np.median(valid)) if len(valid) > 0 else float("nan")

    return results


def _compute_deltas(
    created_at: pd.Series,
    actioned_at: pd.Series,
    actioned: np.ndarray | None,
) -> np.ndarray:
    """Compute time deltas in seconds for actioned items."""
    if actioned is None:
        mask = actioned_at.notna()
    else:
        mask = np.asarray(actioned, dtype=bool)

    if mask.sum() == 0:
        return np.array([])

    deltas = (actioned_at[mask] - created_at[mask]).dt.total_seconds().values
    return deltas[~np.isnan(deltas)]
