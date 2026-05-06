"""Markdown report generation for evaluation results."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np

from safetybench.evaluation.runner import EvaluationResult

_LATENCY_METRICS = {"median_time_to_action", "tta_p50", "tta_p90", "tta_p95", "tta_p99"}
_LATENCY_LABELS = {
    "median_time_to_action": "Median",
    "tta_p50": "P50",
    "tta_p90": "P90",
    "tta_p95": "P95",
    "tta_p99": "P99",
}


class MarkdownReportGenerator:
    """Generates Markdown evaluation reports."""

    def generate(
        self,
        result: EvaluationResult,
        title: str = "Moderation Model Evaluation",
        verbose: bool = False,
    ) -> str:
        sections = [
            self._header(title, result),
            self._metrics_table(result, verbose=verbose),
        ]

        if verbose and any(k in result.metrics for k in _LATENCY_METRICS):
            sections.append(self._latency_table(result))

        if result.per_category:
            sections.append(self._category_table(result))
            sections.append(self._category_summary_table(result))

        if result.per_market:
            sections.append(self._market_table(result))

        if result.confidence_intervals and not verbose:
            sections.append(self._ci_table(result))

        return "\n\n".join(sections) + "\n"

    def write(
        self,
        result: EvaluationResult,
        path: str | Path,
        title: str = "Moderation Model Evaluation",
        verbose: bool = False,
    ) -> None:
        content = self.generate(result, title, verbose=verbose)
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

    def _metrics_table(self, result: EvaluationResult, verbose: bool = False) -> str:
        cis = result.confidence_intervals if verbose else {}
        has_ci = bool(cis)

        if has_ci:
            lines = [
                "## Core Metrics",
                "",
                "| Metric | Value | 95% CI |",
                "|--------|-------|--------|",
            ]
        else:
            lines = [
                "## Core Metrics",
                "",
                "| Metric | Value |",
                "|--------|-------|",
            ]

        for name, value in result.metrics.items():
            if verbose and name in _LATENCY_METRICS:
                continue
            formatted = self._format_value(name, value)
            if has_ci and name in cis:
                _, lo, hi = cis[name]
                ci_str = f"[{lo:.4f}, {hi:.4f}]"
                lines.append(f"| {self._display_name(name)} | {formatted} | {ci_str} |")
            elif has_ci:
                lines.append(f"| {self._display_name(name)} | {formatted} | — |")
            else:
                lines.append(f"| {self._display_name(name)} | {formatted} |")
        return "\n".join(lines)

    def _latency_table(self, result: EvaluationResult) -> str:
        lines = [
            "## Latency Percentiles",
            "",
            "| Percentile | Time to Action |",
            "|------------|----------------|",
        ]
        for key in ("median_time_to_action", "tta_p50", "tta_p90", "tta_p95", "tta_p99"):
            if key in result.metrics:
                label = _LATENCY_LABELS[key]
                lines.append(f"| {label} | {result.metrics[key]:.1f}s |")
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

    def _category_summary_table(self, result: EvaluationResult) -> str:
        all_metrics: set[str] = set()
        for cat_metrics in result.per_category.values():
            all_metrics.update(cat_metrics.keys())
        metric_names = sorted(all_metrics)

        lines = [
            "## Category Summary Statistics",
            "",
            "| Metric | Mean | Std | Min | Max |",
            "|--------|------|-----|-----|-----|",
        ]
        for metric in metric_names:
            vals = np.array([
                m[metric] for m in result.per_category.values() if metric in m
            ], dtype=float)
            if len(vals) == 0:
                continue
            mean = self._format_value(metric, float(np.mean(vals)))
            std = self._format_value(metric, float(np.std(vals)))
            mn = self._format_value(metric, float(np.min(vals)))
            mx = self._format_value(metric, float(np.max(vals)))
            lines.append(f"| {self._display_name(metric)} | {mean} | {std} | {mn} | {mx} |")
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
