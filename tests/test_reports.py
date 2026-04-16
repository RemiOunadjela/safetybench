"""Tests for report generation."""

import csv
import io
import json
import tempfile
from pathlib import Path

import pytest

from safetybench.evaluation.runner import EvaluationConfig, EvaluationRunner
from safetybench.generators.synthetic import GeneratorConfig, SyntheticDataGenerator
from safetybench.reports.csv_export import CsvExporter
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


class TestCsvExporter:
    def test_generate_returns_csv_string(self, eval_result):
        exporter = CsvExporter()
        csv_str = exporter.generate(eval_result)
        assert "scope,metric,value,ci_lower,ci_upper" in csv_str

    def test_overall_rows_present(self, eval_result):
        exporter = CsvExporter()
        csv_str = exporter.generate(eval_result)
        rows = list(csv.DictReader(io.StringIO(csv_str)))
        overall = [r for r in rows if r["scope"] == "overall"]
        assert len(overall) > 0
        metrics_in_csv = {r["metric"] for r in overall}
        assert "precision" in metrics_in_csv
        assert "recall" in metrics_in_csv

    def test_per_category_rows_present(self, eval_result):
        exporter = CsvExporter()
        csv_str = exporter.generate(eval_result)
        rows = list(csv.DictReader(io.StringIO(csv_str)))
        cat_rows = [r for r in rows if r["scope"].startswith("category:")]
        assert len(cat_rows) > 0

    def test_per_market_rows_present(self, eval_result):
        exporter = CsvExporter()
        csv_str = exporter.generate(eval_result)
        rows = list(csv.DictReader(io.StringIO(csv_str)))
        mkt_rows = [r for r in rows if r["scope"].startswith("market:")]
        assert len(mkt_rows) > 0

    def test_ci_bounds_filled_for_bootstrapped_metrics(self, eval_result):
        exporter = CsvExporter()
        csv_str = exporter.generate(eval_result)
        rows = list(csv.DictReader(io.StringIO(csv_str)))
        ci_rows = [r for r in rows if r["ci_lower"] != ""]
        assert len(ci_rows) > 0
        for r in ci_rows:
            assert float(r["ci_lower"]) <= float(r["value"])
            assert float(r["ci_upper"]) >= float(r["value"])

    def test_write_to_file(self, eval_result):
        exporter = CsvExporter()
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            exporter.write(eval_result, f.name)
            content = Path(f.name).read_text()
            assert "scope" in content
            assert "overall" in content


class TestEvaluateCLIExport:
    def test_export_csv_alongside_markdown(self, eval_result):
        """--export .csv should produce a side-car CSV next to the .md report."""
        from click.testing import CliRunner
        from safetybench.cli import cli

        runner = CliRunner()
        with runner.isolated_filesystem():
            # Generate minimal synthetic data file
            from safetybench.generators.synthetic import GeneratorConfig, SyntheticDataGenerator
            df = SyntheticDataGenerator(GeneratorConfig(n_samples=500, seed=0)).generate()
            df.to_csv("data.csv", index=False)

            result = runner.invoke(cli, [
                "evaluate", "--data", "data.csv",
                "--output", "report.md",
                "--no-ci",
                "--export", "metrics.csv",
            ])
            assert result.exit_code == 0, result.output
            assert Path("report.md").exists()
            assert Path("metrics.csv").exists()
            rows = list(csv.DictReader(open("metrics.csv")))
            assert any(r["scope"] == "overall" for r in rows)

    def test_export_json_alongside_html(self, eval_result):
        """--export .json should produce a side-car JSON next to the .html report."""
        from click.testing import CliRunner
        from safetybench.cli import cli

        runner = CliRunner()
        with runner.isolated_filesystem():
            from safetybench.generators.synthetic import GeneratorConfig, SyntheticDataGenerator
            df = SyntheticDataGenerator(GeneratorConfig(n_samples=500, seed=0)).generate()
            df.to_csv("data.csv", index=False)

            result = runner.invoke(cli, [
                "evaluate", "--data", "data.csv",
                "--output", "report.html",
                "--no-ci",
                "--export", "metrics.json",
            ])
            assert result.exit_code == 0, result.output
            assert Path("report.html").exists()
            assert Path("metrics.json").exists()
            data = json.loads(Path("metrics.json").read_text())
            assert "metrics" in data
