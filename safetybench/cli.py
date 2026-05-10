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
    """safetybench: Benchmarking Content Moderation at Scale.

    \b
    Quick start:
      safetybench generate --n 50000 --output data.csv
      safetybench evaluate --data data.csv --output report.md
      safetybench compare --reports run1.json --reports run2.json
    """


@cli.command(
    epilog=(
        "Examples:\n\n"
        "  # Markdown report at default threshold\n"
        "  safetybench evaluate --data scores.csv --output report.md\n\n"
        "  # Stricter threshold, skip CI, export raw metrics to CSV\n"
        "  safetybench evaluate --data scores.csv --threshold 0.7 --no-ci --export metrics.csv\n\n"
        "  # HTML report with verbose CI bounds and latency percentiles\n"
        "  safetybench evaluate --data scores.parquet --output report.html --verbose\n"
    ),
)
@click.option(
    "--data", required=True, type=click.Path(exists=True),
    help="Labeled CSV or Parquet file. Required columns: is_violation, model_score, flagged, actioned.",
)
@click.option(
    "--threshold", default=0.5, type=float,
    help="Decision threshold for binary classification (0.0–1.0). Default: 0.5.",
)
@click.option(
    "--output", default="report.md", type=click.Path(),
    help="Report output path. Format inferred from extension: .md, .html, or .json.",
)
@click.option(
    "--format", "fmt", type=click.Choice(["md", "html", "json"]),
    default=None, help="Override output format instead of inferring from extension.",
)
@click.option("--title", default="Moderation Model Evaluation", help="Report title.")
@click.option(
    "--no-ci", is_flag=True,
    help="Skip bootstrap confidence intervals (faster on large datasets).",
)
@click.option(
    "--export", "export_path", default=None, type=click.Path(),
    help="Also write raw metric values to a sidecar file (.csv or .json).",
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


@cli.command(
    epilog=(
        "Examples:\n\n"
        "  # Compare two model versions saved as JSON reports\n"
        "  safetybench compare --reports model_v1.json --reports model_v2.json\n\n"
        "  # Compare three runs and write to a named file\n"
        "  safetybench compare --reports a.json --reports b.json --reports c.json --output comparison.md\n"
    ),
)
@click.option(
    "--reports", required=True, multiple=True,
    type=click.Path(exists=True),
    help="JSON evaluation report file. Pass once per model (e.g. --reports v1.json --reports v2.json).",
)
@click.option(
    "--output", default="comparison.md", type=click.Path(),
    help="Output path for the comparison table (Markdown).",
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


@cli.command(
    epilog=(
        "Examples:\n\n"
        "  # 10k samples for a single US market\n"
        "  safetybench generate --n 10000 --output data.csv\n\n"
        "  # 100k multi-market dataset with higher violation rate\n"
        "  safetybench generate --n 100000 --markets us,br,de,in --violation-rate 0.08 --output big.parquet\n\n"
        "  # Specific categories for a focused evaluation\n"
        "  safetybench generate --categories csam,terrorism,self_harm --n 5000 --seed 7\n"
    ),
)
@click.option(
    "--categories", default="hate_speech,harassment,spam",
    help="Comma-separated violation categories (e.g. hate_speech,harassment,spam).",
)
@click.option(
    "--markets", default="us",
    help="Comma-separated market codes (e.g. us,br,de,in). Controls sample distribution.",
)
@click.option(
    "--n", "n_samples", default=10_000, type=int,
    help="Total number of content items to generate.",
)
@click.option(
    "--output", default="synthetic_data.csv", type=click.Path(),
    help="Output file path. Use .csv for CSV or .parquet for Parquet.",
)
@click.option("--seed", default=42, type=int, help="Random seed for reproducibility.")
@click.option(
    "--violation-rate", default=0.05, type=float,
    help="Base violation rate across the dataset (0.0–1.0). Default: 0.05.",
)
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
