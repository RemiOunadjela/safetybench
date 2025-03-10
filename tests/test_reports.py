"""Tests for report generation."""

import tempfile
from pathlib import Path

import pytest

from safetybench.evaluation.runner import EvaluationConfig, EvaluationRunner
from safetybench.generators.synthetic import GeneratorConfig, SyntheticDataGenerator
from safetybench.reports.html import HTMLReportGenerator
from safetybench.reports.markdown import MarkdownReportGenerator


@pytest.fixture
def eval_result():
    config = GeneratorConfig(n_samples=1000, seed=42)
    df = SyntheticDataGenerator(config).generate()
    runner = EvaluationRunner(EvaluationConfig(compute_ci=True, bootstrap_samples=200))
    return runner.evaluate(df)


class TestMarkdownReport:
    def test_generate(self, eval_result):
        gen = MarkdownReportGenerator()
        md = gen.generate(eval_result)
        assert "# Moderation Model Evaluation" in md
        assert "Core Metrics" in md

    def test_contains_metrics(self, eval_result):
        gen = MarkdownReportGenerator()
        md = gen.generate(eval_result)
        assert "Action Rate" in md
        assert "Precision" in md

    def test_contains_categories(self, eval_result):
        gen = MarkdownReportGenerator()
        md = gen.generate(eval_result)
        assert "Per-Category" in md

    def test_contains_ci(self, eval_result):
        gen = MarkdownReportGenerator()
        md = gen.generate(eval_result)
        assert "Confidence Intervals" in md

    def test_write_to_file(self, eval_result):
        gen = MarkdownReportGenerator()
        with tempfile.NamedTemporaryFile(suffix=".md", delete=False) as f:
            gen.write(eval_result, f.name)
            content = Path(f.name).read_text()
            assert len(content) > 100


class TestHTMLReport:
    def test_generate(self, eval_result):
        gen = HTMLReportGenerator()
        html = gen.generate(eval_result)
        assert "<html" in html
        assert "Core Metrics" in html

    def test_write_to_file(self, eval_result):
        gen = HTMLReportGenerator()
        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            gen.write(eval_result, f.name)
            content = Path(f.name).read_text()
            assert "<table>" in content
