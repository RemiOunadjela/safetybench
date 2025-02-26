"""Model comparison utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from safetybench.evaluation.runner import EvaluationConfig, EvaluationResult, EvaluationRunner
from safetybench.metrics.statistical import mcnemar_test


@dataclass
class ComparisonResult:
    """Results from comparing two or more models."""

    results: dict[str, EvaluationResult]
    pairwise_tests: dict[str, dict[str, Any]]
    summary_table: pd.DataFrame

    def winner(self, metric: str = "proactive_detection_rate") -> str | None:
        best_name = None
        best_val = -1.0
        for name, res in self.results.items():
            val = res.metrics.get(metric, -1.0)
            if val > best_val:
                best_val = val
                best_name = name
        return best_name


class ModelComparator:
    """Compare multiple models on the same dataset.

    Each model is represented by a column of scores in the input DataFrame.
    """

    def __init__(self, config: EvaluationConfig | None = None):
        self.config = config or EvaluationConfig()

    def compare(
        self,
        df: pd.DataFrame,
        score_columns: dict[str, str],
    ) -> ComparisonResult:
        """Run evaluation for each model and produce pairwise comparisons.

        Args:
            df: DataFrame with standard moderation columns.
            score_columns: Dict mapping model name to score column name.

        Returns:
            ComparisonResult with per-model results and significance tests.
        """
        runner = EvaluationRunner(self.config)
        results: dict[str, EvaluationResult] = {}

        for model_name, score_col in score_columns.items():
            model_df = df.copy()
            model_df["model_score"] = model_df[score_col]
            model_df["flagged"] = model_df["model_score"] >= self.config.threshold
            model_df["actioned"] = model_df["flagged"]
            if "user_reported" in model_df.columns:
                model_df["actioned"] = model_df["flagged"] | model_df["user_reported"]

            results[model_name] = runner.evaluate(model_df)

        pairwise = self._pairwise_tests(df, score_columns)
        summary = self._build_summary(results)

        return ComparisonResult(
            results=results,
            pairwise_tests=pairwise,
            summary_table=summary,
        )

    def _pairwise_tests(
        self,
        df: pd.DataFrame,
        score_columns: dict[str, str],
    ) -> dict[str, dict[str, Any]]:
        labels = df["is_violation"].values.astype(bool)
        model_names = list(score_columns.keys())
        tests: dict[str, dict[str, Any]] = {}

        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                name_a = model_names[i]
                name_b = model_names[j]
                preds_a = (df[score_columns[name_a]].values >= self.config.threshold)
                preds_b = (df[score_columns[name_b]].values >= self.config.threshold)

                key = f"{name_a}_vs_{name_b}"
                tests[key] = mcnemar_test(preds_a, preds_b, labels)

        return tests

    def _build_summary(self, results: dict[str, EvaluationResult]) -> pd.DataFrame:
        rows = []
        for model_name, res in results.items():
            row = {"model": model_name}
            row.update(res.metrics)
            rows.append(row)
        return pd.DataFrame(rows).set_index("model")
