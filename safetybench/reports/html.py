"""HTML report generation using Jinja2 templates."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape

from safetybench.evaluation.runner import EvaluationResult


class HTMLReportGenerator:
    """Generates HTML evaluation reports."""

    def __init__(self) -> None:
        self.env = Environment(
            loader=PackageLoader("safetybench", "templates"),
            autoescape=select_autoescape(["html"]),
        )

    def generate(
        self,
        result: EvaluationResult,
        title: str = "Moderation Model Evaluation",
    ) -> str:
        template = self.env.get_template("report.html")
        return template.render(
            title=title,
            generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
            metadata=result.metadata,
            metrics=result.metrics,
            per_category=result.per_category,
            per_market=result.per_market,
            confidence_intervals=result.confidence_intervals,
            format_value=self._format_value,
            display_name=self._display_name,
        )

    def write(
        self,
        result: EvaluationResult,
        path: str | Path,
        title: str = "Moderation Model Evaluation",
    ) -> None:
        content = self.generate(result, title)
        Path(path).write_text(content)

    @staticmethod
    def _display_name(metric: str) -> str:
        return metric.replace("_", " ").title()

    @staticmethod
    def _format_value(metric: str, value: float) -> str:
        if "time" in metric or "tta" in metric:
            return f"{value:.1f}s"
        return f"{value:.4f}"
