"""Tests for the evaluation engine."""

import numpy as np
import pandas as pd
import pytest

from safetybench.evaluation.comparator import ModelComparator
from safetybench.evaluation.runner import EvaluationConfig, EvaluationResult, EvaluationRunner
from safetybench.evaluation.thresholds import ThresholdAnalyzer
from safetybench.generators.synthetic import GeneratorConfig, SyntheticDataGenerator


@pytest.fixture
def sample_data():
    config = GeneratorConfig(n_samples=2000, markets=["us", "br"], seed=42)
    return SyntheticDataGenerator(config).generate()


@pytest.fixture
def simple_data():
    """Minimal dataset for targeted tests."""
    rng = np.random.default_rng(0)
    n = 500
    labels = rng.random(n) < 0.1
    scores = np.where(labels, rng.beta(8, 2, n), rng.beta(2, 8, n))
    base = pd.Timestamp("2025-01-01")
    created = pd.Series([base] * n)
    tta = rng.exponential(60, n)

    return pd.DataFrame({
        "is_violation": labels,
        "model_score": scores,
        "flagged": scores >= 0.5,
        "actioned": scores >= 0.5,
        "user_reported": np.zeros(n, dtype=bool),
        "view_count": rng.integers(0, 100, n),
        "created_at": created,
        "actioned_at": [
            base + pd.Timedelta(seconds=float(tta[i])) if scores[i] >= 0.5 else pd.NaT
            for i in range(n)
        ],
        "appealed": rng.random(n) < 0.02,
        "overturned": np.zeros(n, dtype=bool),
        "category": np.where(labels, rng.choice(["hate_speech", "spam"], n), "none"),
        "market": rng.choice(["us", "br"], n),
    })


class TestEvaluationRunner:
    def test_evaluate_returns_result(self, sample_data):
        runner = EvaluationRunner()
        result = runner.evaluate(sample_data)

        assert isinstance(result, EvaluationResult)
        assert "action_rate" in result.metrics
        assert "precision" in result.metrics
        assert "recall" in result.metrics

    def test_core_metrics_range(self, sample_data):
        runner = EvaluationRunner()
        result = runner.evaluate(sample_data)

        for key, val in result.metrics.items():
            if "time" in key or "tta" in key:
                continue
            assert 0.0 <= val <= 1.0, f"{key} = {val} out of range"

    def test_per_category(self, sample_data):
        runner = EvaluationRunner()
        result = runner.evaluate(sample_data)
        assert len(result.per_category) > 0

    def test_per_market(self, sample_data):
        runner = EvaluationRunner()
        result = runner.evaluate(sample_data)
        assert "us" in result.per_market
        assert "br" in result.per_market

    def test_confidence_intervals(self, simple_data):
        config = EvaluationConfig(compute_ci=True, bootstrap_samples=500)
        runner = EvaluationRunner(config)
        result = runner.evaluate(simple_data)

        assert "action_rate" in result.confidence_intervals
        est, lo, hi = result.confidence_intervals["action_rate"]
        assert lo <= est <= hi

    def test_no_ci(self, simple_data):
        config = EvaluationConfig(compute_ci=False)
        runner = EvaluationRunner(config)
        result = runner.evaluate(simple_data)
        assert len(result.confidence_intervals) == 0

    def test_to_dict(self, simple_data):
        runner = EvaluationRunner(EvaluationConfig(compute_ci=False))
        result = runner.evaluate(simple_data)
        d = result.to_dict()
        assert "metrics" in d
        assert "metadata" in d

    def test_summary_dataframe(self, simple_data):
        runner = EvaluationRunner(EvaluationConfig(compute_ci=False))
        result = runner.evaluate(simple_data)
        summary = result.summary()
        assert isinstance(summary, pd.DataFrame)
        assert "metric" in summary.columns
        assert "value" in summary.columns


class TestThresholdAnalyzer:
    def test_analyze(self):
        rng = np.random.default_rng(42)
        n = 1000
        labels = rng.random(n) < 0.1
        scores = np.where(labels, rng.beta(8, 2, n), rng.beta(2, 8, n))

        analyzer = ThresholdAnalyzer(n_points=20)
        df = analyzer.analyze(scores, labels)

        assert len(df) == 20
        assert set(df.columns) == {"threshold", "precision", "recall", "fpr", "f1"}

    def test_optimal_f1(self):
        rng = np.random.default_rng(42)
        labels = rng.random(500) < 0.2
        scores = np.where(labels, rng.beta(8, 2, 500), rng.beta(2, 8, 500))

        analyzer = ThresholdAnalyzer()
        point = analyzer.optimal_threshold(scores, labels, strategy="f1")
        assert 0.0 < point.threshold < 1.0
        assert point.f1 > 0

    def test_optimal_youden(self):
        rng = np.random.default_rng(42)
        labels = rng.random(500) < 0.2
        scores = np.where(labels, rng.beta(8, 2, 500), rng.beta(2, 8, 500))

        analyzer = ThresholdAnalyzer()
        point = analyzer.optimal_threshold(scores, labels, strategy="youden")
        assert 0.0 < point.threshold < 1.0

    def test_recall_at_fpr(self):
        rng = np.random.default_rng(42)
        labels = rng.random(500) < 0.2
        scores = np.where(labels, rng.beta(8, 2, 500), rng.beta(2, 8, 500))

        analyzer = ThresholdAnalyzer()
        point = analyzer.optimal_threshold(scores, labels, strategy="recall_at_fpr", max_fpr=0.05)
        assert point.fpr <= 0.05

    def test_auc(self):
        rng = np.random.default_rng(42)
        labels = rng.random(500) < 0.2
        scores = np.where(labels, rng.beta(8, 2, 500), rng.beta(2, 8, 500))

        analyzer = ThresholdAnalyzer()
        auc = analyzer.auc(scores, labels)
        assert 0.8 < auc <= 1.0

    def test_category_thresholds(self):
        rng = np.random.default_rng(42)
        n = 1000
        labels = rng.random(n) < 0.2
        scores = np.where(labels, rng.beta(8, 2, n), rng.beta(2, 8, n))
        cats = pd.Series(np.where(labels, rng.choice(["a", "b"], n), "none"))

        analyzer = ThresholdAnalyzer()
        results = analyzer.category_thresholds(scores, labels, cats)
        assert "a" in results or "b" in results


class TestModelComparator:
    def test_compare_two_models(self, simple_data):
        df = simple_data.copy()
        rng = np.random.default_rng(99)
        df["score_v1"] = df["model_score"]
        df["score_v2"] = np.clip(df["model_score"] + rng.normal(0, 0.1, len(df)), 0, 1)

        comparator = ModelComparator()
        result = comparator.compare(df, {"model_v1": "score_v1", "model_v2": "score_v2"})

        assert "model_v1" in result.results
        assert "model_v2" in result.results
        assert len(result.pairwise_tests) == 1
        assert isinstance(result.summary_table, pd.DataFrame)

    def test_winner(self, simple_data):
        df = simple_data.copy()
        df["score_good"] = df["model_score"]
        df["score_bad"] = np.zeros(len(df))

        comparator = ModelComparator()
        result = comparator.compare(df, {"good": "score_good", "bad": "score_bad"})

        winner = result.winner(metric="precision")
        assert winner == "good"
