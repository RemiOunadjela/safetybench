"""Markdown report generation for evaluation results."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from safetybench.evaluation.runner import EvaluationResult


class MarkdownReportGenerator:
    """Generates Markdown evaluation reports."""

    def generate(
        self,
        result: EvaluationResult,
        title: str = "Moderation Model Evaluation",
    ) -> str:
        sections = [
            self._header(title, result),
            self._metrics_table(result),
        ]

        if result.per_category:
            sections.append(self._category_table(result))

        if result.per_market:
            sections.append(self._market_table(result))

        if result.confidence_intervals:
            sections.append(self._ci_table(result))

        return "\n\n".join(sections) + "\n"

    def write(
        self,
        result: EvaluationResult,
        path: str | Path,
        title: str = "Moderation Model Evaluation",
    ) -> None:
        content = self.generate(result, title)
        Path(path).write_text(content)

    def _header(self, title: str, result: EvaluationResult) -> str:
        meta = result.metadata
        lines = [
            f"# {title}",
            "",
            f"- **Samples:** {meta.get('n_samples', 'N/A'):,}",
            f"- **Violations:** {meta.get('n_violations', 'N/A'):,}",
            f"- **Threshold:** {meta.get('threshold', 'N/A')}",
            f"- **Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        ]
        return "\n".join(lines)

    def _metrics_table(self, result: EvaluationResult) -> str:
        lines = [
            "## Core Metrics",
            "",
            "| Metric | Value |",
            "|--------|-------|",
        ]
        for name, value in result.metrics.items():
            formatted = self._format_value(name, value)
            lines.append(f"| {self._display_name(name)} | {formatted} |")
        return "\n".join(lines)

    def _category_table(self, result: EvaluationResult) -> str:
        all_metrics = set()
        for cat_metrics in result.per_category.values():
            all_metrics.update(cat_metrics.keys())
        metric_names = sorted(all_metrics)

        header = "| Category | " + " | ".join(self._display_name(m) for m in metric_names) + " |"
        sep = "|" + "|".join(["--------"] * (len(metric_names) + 1)) + "|"

        lines = ["## Per-Category Breakdown", "", header, sep]
        for cat, cat_metrics in sorted(result.per_category.items()):
            vals = " | ".join(
                self._format_value(m, cat_metrics.get(m, float("nan")))
                for m in metric_names
            )
            lines.append(f"| {cat} | {vals} |")
        return "\n".join(lines)

    def _market_table(self, result: EvaluationResult) -> str:
        all_metrics = set()
        for mkt_metrics in result.per_market.values():
            all_metrics.update(mkt_metrics.keys())
        metric_names = sorted(all_metrics)

        header = "| Market | " + " | ".join(self._display_name(m) for m in metric_names) + " |"
        sep = "|" + "|".join(["--------"] * (len(metric_names) + 1)) + "|"

        lines = ["## Cross-Market Comparison", "", header, sep]
        for market, mkt_metrics in sorted(result.per_market.items()):
            vals = " | ".join(
                self._format_value(m, mkt_metrics.get(m, float("nan")))
                for m in metric_names
            )
            lines.append(f"| {market.upper()} | {vals} |")
        return "\n".join(lines)

    def _ci_table(self, result: EvaluationResult) -> str:
        lines = [
            "## Confidence Intervals",
            "",
            "| Metric | Estimate | 95% CI Lower | 95% CI Upper |",
            "|--------|----------|-------------|-------------|",
        ]
        for name, (est, lo, hi) in result.confidence_intervals.items():
            lines.append(
                f"| {self._display_name(name)} | {est:.4f} | {lo:.4f} | {hi:.4f} |"
            )
        return "\n".join(lines)

    @staticmethod
    def _display_name(metric: str) -> str:
        return metric.replace("_", " ").title()

    @staticmethod
    def _format_value(metric: str, value: float) -> str:
        if "time" in metric or "tta" in metric:
            return f"{value:.1f}s"
        return f"{value:.4f}"
