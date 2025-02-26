"""Threshold analysis for moderation classifiers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from safetybench.metrics.quality import (
    false_positive_rate_at_threshold,
    precision_recall_at_thresholds,
)


@dataclass
class ThresholdPoint:
    threshold: float
    precision: float
    recall: float
    fpr: float
    f1: float


class ThresholdAnalyzer:
    """Analyze classifier performance across thresholds.

    Produces precision-recall-FPR curves and recommends operating points
    based on different optimization targets.
    """

    def __init__(
        self,
        thresholds: np.ndarray | None = None,
        n_points: int = 100,
    ):
        if thresholds is not None:
            self.thresholds = np.asarray(thresholds)
        else:
            self.thresholds = np.linspace(0.01, 0.99, n_points)

    def analyze(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
    ) -> pd.DataFrame:
        """Compute metrics at each threshold.

        Returns:
            DataFrame with columns: threshold, precision, recall, fpr, f1.
        """
        scores = np.asarray(scores, dtype=np.float64)
        labels = np.asarray(labels, dtype=bool)

        pr = precision_recall_at_thresholds(scores, labels, self.thresholds)
        precision = pr["precision"]
        recall = pr["recall"]

        fpr_vals = np.array([
            false_positive_rate_at_threshold(scores, labels, t)
            for t in self.thresholds
        ])

        f1 = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )

        return pd.DataFrame({
            "threshold": self.thresholds,
            "precision": precision,
            "recall": recall,
            "fpr": fpr_vals,
            "f1": f1,
        })

    def optimal_threshold(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        strategy: str = "f1",
        max_fpr: float | None = None,
    ) -> ThresholdPoint:
        """Find the optimal threshold under a given strategy.

        Strategies:
            'f1': Maximize F1 score.
            'youden': Maximize Youden's J statistic (TPR - FPR).
            'recall_at_fpr': Maximize recall subject to FPR <= max_fpr.

        Args:
            scores: Model scores.
            labels: Ground-truth labels.
            strategy: Optimization strategy.
            max_fpr: Maximum acceptable FPR (required for 'recall_at_fpr').

        Returns:
            ThresholdPoint at the optimal operating point.
        """
        df = self.analyze(scores, labels)

        if strategy == "f1":
            best_idx = df["f1"].idxmax()
        elif strategy == "youden":
            j = df["recall"] - df["fpr"]
            best_idx = j.idxmax()
        elif strategy == "recall_at_fpr":
            if max_fpr is None:
                raise ValueError("max_fpr required for recall_at_fpr strategy")
            valid = df[df["fpr"] <= max_fpr]
            if len(valid) == 0:
                raise ValueError(f"No threshold achieves FPR <= {max_fpr}")
            best_idx = valid["recall"].idxmax()
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        row = df.iloc[best_idx]
        return ThresholdPoint(
            threshold=float(row["threshold"]),
            precision=float(row["precision"]),
            recall=float(row["recall"]),
            fpr=float(row["fpr"]),
            f1=float(row["f1"]),
        )

    def auc(self, scores: np.ndarray, labels: np.ndarray) -> float:
        """Compute ROC AUC."""
        return float(roc_auc_score(labels, scores))

    def category_thresholds(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        categories: pd.Series,
        strategy: str = "f1",
    ) -> dict[str, ThresholdPoint]:
        """Find optimal threshold per violation category.

        Returns:
            Dict mapping category to ThresholdPoint.
        """
        results: dict[str, ThresholdPoint] = {}
        for cat in categories.unique():
            if cat == "none":
                continue
            mask = (categories == cat) | (~labels.astype(bool))
            sub_scores = scores[mask]
            sub_labels = labels[mask]
            if sub_labels.sum() < 5:
                continue
            results[cat] = self.optimal_threshold(sub_scores, sub_labels, strategy=strategy)
        return results
