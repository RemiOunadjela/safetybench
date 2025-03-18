# safetybench

[![CI](https://github.com/RemiOunadjela/safetybench/actions/workflows/ci.yml/badge.svg)](https://github.com/RemiOunadjela/safetybench/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Benchmarking Content Moderation at Scale**

A Python framework for evaluating and benchmarking content safety models against industry-standard Trust & Safety metrics.

---

## Why safetybench?

Content moderation teams at scale care about metrics that generic ML evaluation tools don't cover: proactive detection rate, zero-view violation rate, time-to-action percentiles, appeal overturn rates, and cross-market performance parity. These are the metrics that show up in transparency reports and regulatory filings, not just F1 and AUC.

`safetybench` fills that gap. It provides a single framework to:

- **Evaluate** moderation models using T&S-specific metrics with statistical rigor
- **Compare** model versions with significance testing (bootstrap CI, McNemar's test)
- **Generate** realistic synthetic moderation data across violation categories and markets
- **Report** results in Markdown or HTML, ready for stakeholder review

If you've built content moderation systems and been frustrated stitching together ad-hoc evaluation scripts, this is for you.

## Quick Start

### Installation

```bash
pip install safetybench
```

For development:

```bash
git clone https://github.com/RemiOunadjela/safetybench.git
cd safetybench
pip install -e ".[dev]"
```

### Generate Synthetic Data

```python
from safetybench.generators import SyntheticDataGenerator
from safetybench.generators.synthetic import GeneratorConfig

config = GeneratorConfig(
    n_samples=50_000,
    markets=["us", "br", "in"],
    violation_rate=0.05,
    model_auc=0.93,
)
generator = SyntheticDataGenerator(config)
df = generator.generate()
```

### Evaluate a Model

```python
from safetybench.evaluation import EvaluationRunner
from safetybench.evaluation.runner import EvaluationConfig

config = EvaluationConfig(threshold=0.5, compute_ci=True)
runner = EvaluationRunner(config)
result = runner.evaluate(df)

print(result.summary())
```

### Compare Two Models

```python
from safetybench.evaluation import ModelComparator

df["score_v2"] = improved_model.predict(df)  # your model here

comparator = ModelComparator()
comparison = comparator.compare(df, {"baseline": "model_score", "v2": "score_v2"})
print(comparison.summary_table)
```

### Generate Reports

```python
from safetybench.reports import MarkdownReportGenerator, HTMLReportGenerator

MarkdownReportGenerator().write(result, "report.md")
HTMLReportGenerator().write(result, "report.html")
```

### CLI

```bash
# Generate synthetic data
safetybench generate --categories hate_speech,spam,harassment --markets us,br --n 50000

# Evaluate
safetybench evaluate --data synthetic_data.csv --threshold 0.5 --output report.html

# Compare runs
safetybench compare --reports run1.json --reports run2.json --output comparison.md
```

## Metrics

### Detection Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **Proactive Detection Rate** | `\|proactive_catches\| / \|violations\|` | Fraction of violations caught by the model before any user report |
| **Zero-View Violation Rate** | `\|zero_view_actioned\| / \|violations\|` | Fraction of violations removed before a single view |
| **Action Rate** | `\|actioned\| / \|reviewed\|` | Fraction of reviewed content that received a moderation action |

### Quality Metrics

| Metric | Formula | Description |
|--------|---------|-------------|
| **False Positive Rate** | `FP / (FP + TN)` | Rate of clean content incorrectly actioned, at a given threshold |
| **Appeal Overturn Rate** | `\|overturned\| / \|appealed\|` | Fraction of appeals resulting in decision reversal |
| **Precision / Recall** | Standard definitions | Computed at configurable thresholds |

### Latency Metrics

| Metric | Description |
|--------|-------------|
| **Median Time-to-Action** | Median seconds from content creation to moderation action |
| **TTA Percentiles** | p50, p90, p95, p99 time-to-action |

### Statistical Testing

| Test | Use Case |
|------|----------|
| **Bootstrap CI** | Confidence intervals for any metric |
| **McNemar's Test** | Comparing error rates of two classifiers on the same data |
| **Permutation Test** | Non-parametric comparison of two groups |

## Development

```bash
# Run tests
pytest --cov=safetybench -q

# Lint
ruff check safetybench/ tests/
```

## License

MIT
