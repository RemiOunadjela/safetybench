"""Command-line interface for safetybench."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
import pandas as pd

from safetybench.evaluation.runner import EvaluationConfig, EvaluationRunner
from safetybench.generators.synthetic import GeneratorConfig, SyntheticDataGenerator
from safetybench.reports.csv_export import CsvExporter
from safetybench.reports.html import HTMLReportGenerator
from safetybench.reports.markdown import MarkdownReportGenerator


@click.group()
@click.version_option(package_name="safetybench")
def cli() -> None:
    """safetybench: Benchmarking Content Moderation at Scale."""


@cli.command()
@click.option(
    "--data", required=True, type=click.Path(exists=True),
    help="Path to evaluation data (CSV or Parquet).",
)
@click.option("--threshold", default=0.5, type=float, help="Decision threshold.")
@click.option("--output", default="report.md", type=click.Path(), help="Output path.")
@click.option(
    "--format", "fmt", type=click.Choice(["md", "html", "json"]),
    default=None, help="Output format. Inferred from extension if omitted.",
)
@click.option("--title", default="Moderation Model Evaluation", help="Report title.")
@click.option("--no-ci", is_flag=True, help="Skip confidence interval computation.")
@click.option(
    "--export", "export_path", default=None, type=click.Path(),
    help="Also export raw metrics to a side-car file (.csv or .json).",
)
@click.option(
    "--verbose", is_flag=True,
    help="Show detailed metric breakdowns: inline CI bounds and latency percentiles.",
)
def evaluate(
    data: str,
    threshold: float,
    output: str,
    fmt: str | None,
    title: str,
    no_ci: bool,
    export_path: str | None,
    verbose: bool,
) -> None:
    """Evaluate a moderation model on labeled data."""
    df = _load_data(data)

    config = EvaluationConfig(
        threshold=threshold,
        compute_ci=not no_ci,
    )
    runner = EvaluationRunner(config)
    result = runner.evaluate(df)

    out_path = Path(output)
    if fmt is None:
        fmt = out_path.suffix.lstrip(".")
        if fmt not in ("md", "html", "json"):
            fmt = "md"

    if fmt == "json":
        out_path.write_text(json.dumps(result.to_dict(), indent=2, default=str))
    elif fmt == "html":
        HTMLReportGenerator().write(result, out_path, title=title)
    else:
        MarkdownReportGenerator().write(result, out_path, title=title, verbose=verbose)

    click.echo(f"Report written to {out_path}")

    if export_path:
        exp_path = Path(export_path)
        exp_fmt = exp_path.suffix.lstrip(".")
        if exp_fmt == "csv":
            CsvExporter().write(result, exp_path)
        else:
            exp_path.write_text(json.dumps(result.to_dict(), indent=2, default=str))
        click.echo(f"Data exported to {exp_path}")


@cli.command()
@click.option(
    "--reports", required=True, multiple=True,
    type=click.Path(exists=True), help="JSON report files to compare.",
)
@click.option(
    "--output", default="comparison.md", type=click.Path(),
    help="Output comparison report.",
)
def compare(reports: tuple[str, ...], output: str) -> None:
    """Compare evaluation results from multiple runs."""
    results = {}
    for path in reports:
        data = json.loads(Path(path).read_text())
        name = Path(path).stem
        results[name] = data.get("metrics", {})

    rows = []
    for name, metrics in results.items():
        row = {"model": name}
        row.update(metrics)
        rows.append(row)

    df = pd.DataFrame(rows).set_index("model")

    lines = [
        "# Model Comparison",
        "",
        df.to_markdown(),
        "",
    ]

    Path(output).write_text("\n".join(lines))
    click.echo(f"Comparison written to {output}")


@cli.command()
@click.option(
    "--categories", default="hate_speech,harassment,spam",
    help="Comma-separated violation categories.",
)
@click.option("--markets", default="us", help="Comma-separated market codes.")
@click.option("--n", "n_samples", default=10_000, type=int, help="Number of samples to generate.")
@click.option("--output", default="synthetic_data.csv", type=click.Path(), help="Output file path.")
@click.option("--seed", default=42, type=int, help="Random seed.")
@click.option("--violation-rate", default=0.05, type=float, help="Base violation rate.")
def generate(
    categories: str,
    markets: str,
    n_samples: int,
    output: str,
    seed: int,
    violation_rate: float,
) -> None:
    """Generate synthetic moderation data."""
    config = GeneratorConfig(
        n_samples=n_samples,
        categories=categories.split(","),
        markets=markets.split(","),
        seed=seed,
        violation_rate=violation_rate,
    )
    gen = SyntheticDataGenerator(config)
    df = gen.generate()

    out_path = Path(output)
    if out_path.suffix == ".parquet":
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)

    click.echo(f"Generated {len(df):,} samples -> {out_path}")


def _load_data(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".parquet":
        return pd.read_parquet(p)
    elif p.suffix == ".csv":
        df = pd.read_csv(p)
        for col in ("created_at", "actioned_at"):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col])
        return df
    else:
        click.echo(f"Unsupported file format: {p.suffix}", err=True)
        sys.exit(1)
