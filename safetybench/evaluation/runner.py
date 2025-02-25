"""Main evaluation engine for content moderation models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from safetybench.metrics.detection import (
    action_rate,
    detection_rate_by_category,
    proactive_detection_rate,
    zero_view_violation_rate,
)
from safetybench.metrics.latency import median_time_to_action, time_to_action_percentiles
from safetybench.metrics.quality import (
    appeal_overturn_rate,
    false_positive_rate_at_threshold,
    precision_recall_at_thresholds,
)
from safetybench.metrics.statistical import bootstrap_ci


@dataclass
class EvaluationConfig:
    """Configuration for an evaluation run."""

    threshold: float = 0.5
    bootstrap_samples: int = 1_000
    confidence_level: float = 0.95
    compute_ci: bool = True
    categories: list[str] | None = None


@dataclass
class EvaluationResult:
    """Container for evaluation results."""

    metrics: dict[str, Any] = field(default_factory=dict)
    per_category: dict[str, dict[str, float]] = field(default_factory=dict)
    per_market: dict[str, dict[str, float]] = field(default_factory=dict)
    confidence_intervals: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "per_category": self.per_category,
            "per_market": self.per_market,
            "confidence_intervals": {
                k: {"estimate": v[0], "ci_lower": v[1], "ci_upper": v[2]}
                for k, v in self.confidence_intervals.items()
            },
            "metadata": self.metadata,
        }

    def summary(self) -> pd.DataFrame:
        rows = []
        for name, value in self.metrics.items():
            row = {"metric": name, "value": value}
            if name in self.confidence_intervals:
                _, lo, hi = self.confidence_intervals[name]
                row["ci_lower"] = lo
                row["ci_upper"] = hi
            rows.append(row)
        return pd.DataFrame(rows)


class EvaluationRunner:
    """Runs a full evaluation suite on moderation data.

    Expects a DataFrame with the standard schema produced by SyntheticDataGenerator
    or any DataFrame with compatible columns.

    Required columns: is_violation, model_score, flagged, actioned
    Optional columns: user_reported, view_count, created_at, actioned_at,
                      appealed, overturned, category, market
    """

    def __init__(self, config: EvaluationConfig | None = None):
        self.config = config or EvaluationConfig()

    def evaluate(self, df: pd.DataFrame) -> EvaluationResult:
        result = EvaluationResult()
        result.metadata = {
            "n_samples": len(df),
            "n_violations": int(df["is_violation"].sum()),
            "threshold": self.config.threshold,
        }

        self._compute_core_metrics(df, result)
        self._compute_latency_metrics(df, result)
        self._compute_quality_metrics(df, result)

        if "category" in df.columns:
            self._compute_per_category(df, result)

        if "market" in df.columns:
            self._compute_per_market(df, result)

        if self.config.compute_ci:
            self._compute_confidence_intervals(df, result)

        return result

    def _compute_core_metrics(self, df: pd.DataFrame, result: EvaluationResult) -> None:
        labels = df["is_violation"].values
        flagged = df["flagged"].values
        actioned = df["actioned"].values

        result.metrics["action_rate"] = action_rate(actioned, np.ones(len(df), dtype=bool))

        if "user_reported" in df.columns:
            result.metrics["proactive_detection_rate"] = proactive_detection_rate(
                flagged, df["user_reported"].values, labels
            )

        if "view_count" in df.columns:
            result.metrics["zero_view_violation_rate"] = zero_view_violation_rate(
                df["view_count"].values, actioned, labels
            )

    def _compute_latency_metrics(self, df: pd.DataFrame, result: EvaluationResult) -> None:
        if "created_at" not in df.columns or "actioned_at" not in df.columns:
            return

        result.metrics["median_time_to_action"] = median_time_to_action(
            df["created_at"], df["actioned_at"]
        )

        tta_pcts = time_to_action_percentiles(df["created_at"], df["actioned_at"])
        for key, val in tta_pcts.items():
            result.metrics[f"tta_{key}"] = val

    def _compute_quality_metrics(self, df: pd.DataFrame, result: EvaluationResult) -> None:
        scores = df["model_score"].values
        labels = df["is_violation"].values

        result.metrics["fpr"] = false_positive_rate_at_threshold(
            scores, labels, self.config.threshold
        )

        pr = precision_recall_at_thresholds(
            scores, labels, np.array([self.config.threshold])
        )
        result.metrics["precision"] = float(pr["precision"][0])
        result.metrics["recall"] = float(pr["recall"][0])

        if "appealed" in df.columns and "overturned" in df.columns:
            result.metrics["appeal_overturn_rate"] = appeal_overturn_rate(
                df["appealed"].values, df["overturned"].values
            )

    def _compute_per_category(self, df: pd.DataFrame, result: EvaluationResult) -> None:
        violations = df[df["is_violation"]]
        if len(violations) == 0:
            return

        cats = violations["category"]
        det_rates = detection_rate_by_category(
            violations["flagged"].values, violations["is_violation"].values, cats
        )
        for cat, rate in det_rates.items():
            result.per_category.setdefault(cat, {})["detection_rate"] = rate

        # Per-category FPR
        for cat in df["category"].unique():
            if cat == "none":
                continue
            mask = (df["category"] == cat) | (~df["is_violation"])
            subset = df[mask]
            if len(subset) == 0:
                continue
            fpr = false_positive_rate_at_threshold(
                subset["model_score"].values,
                subset["is_violation"].values,
                self.config.threshold,
            )
            result.per_category.setdefault(cat, {})["fpr"] = fpr

    def _compute_per_market(self, df: pd.DataFrame, result: EvaluationResult) -> None:
        for market in df["market"].unique():
            mdf = df[df["market"] == market]
            sub_result = EvaluationRunner(self.config).evaluate(
                mdf.drop(columns=["market"], errors="ignore")
            )
            result.per_market[market] = sub_result.metrics

    def _compute_confidence_intervals(
        self, df: pd.DataFrame, result: EvaluationResult
    ) -> None:
        labels = df["is_violation"].values.astype(bool)
        flagged = df["flagged"].values.astype(bool)
        actioned = df["actioned"].values.astype(bool)

        # Bootstrap CI for action rate
        actioned_binary = actioned.astype(float)
        result.confidence_intervals["action_rate"] = bootstrap_ci(
            actioned_binary,
            n_bootstrap=self.config.bootstrap_samples,
            confidence=self.config.confidence_level,
            seed=42,
        )

        # Bootstrap CI for precision among flagged
        if flagged.sum() > 0:
            flagged_correct = (flagged & labels).astype(float)[flagged]
            result.confidence_intervals["precision"] = bootstrap_ci(
                flagged_correct,
                n_bootstrap=self.config.bootstrap_samples,
                confidence=self.config.confidence_level,
                seed=42,
            )
