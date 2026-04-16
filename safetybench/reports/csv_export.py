"""CSV export for evaluation results."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from safetybench.evaluation.runner import EvaluationResult


class CsvExporter:
    """Exports evaluation results to a flat CSV for downstream analysis and CI tracking.

    Each row represents one metric at a given scope (overall, per-category,
    per-market).  Confidence interval bounds are included where available.

    Columns: scope, metric, value, ci_lower, ci_upper
    """

    def generate(self, result: EvaluationResult) -> str:
        """Return the CSV content as a string."""
        rows = self._build_rows(result)
        buf = io.StringIO()
        writer = csv.DictWriter(
            buf,
            fieldnames=["scope", "metric", "value", "ci_lower", "ci_upper"],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
        return buf.getvalue()

    def write(self, result: EvaluationResult, path: str | Path) -> None:
        """Write CSV to *path*."""
        Path(path).write_text(self.generate(result))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_rows(self, result: EvaluationResult) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []

        # Overall metrics (with CI where present)
        for name, value in result.metrics.items():
            row: dict[str, Any] = {
                "scope": "overall",
                "metric": name,
                "value": value,
                "ci_lower": "",
                "ci_upper": "",
            }
            if name in result.confidence_intervals:
                _, lo, hi = result.confidence_intervals[name]
                row["ci_lower"] = lo
                row["ci_upper"] = hi
            rows.append(row)

        # Per-category metrics
        for category, cat_metrics in sorted(result.per_category.items()):
            for name, value in sorted(cat_metrics.items()):
                rows.append({
                    "scope": f"category:{category}",
                    "metric": name,
                    "value": value,
                    "ci_lower": "",
                    "ci_upper": "",
                })

        # Per-market metrics
        for market, mkt_metrics in sorted(result.per_market.items()):
            for name, value in sorted(mkt_metrics.items()):
                rows.append({
                    "scope": f"market:{market}",
                    "metric": name,
                    "value": value,
                    "ci_lower": "",
                    "ci_upper": "",
                })

        return rows
