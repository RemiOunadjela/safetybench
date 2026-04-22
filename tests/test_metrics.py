"""Tests for metrics modules."""

import numpy as np
import pandas as pd
import pytest

from safetybench.metrics.detection import (
    action_rate,
    detection_rate_by_category,
    proactive_detection_rate,
    zero_view_violation_rate,
)
from safetybench.metrics.latency import (
    median_time_to_action,
    time_to_action_by_category,
    time_to_action_percentiles,
)
from safetybench.metrics.quality import (
    appeal_overturn_rate,
    false_positive_rate_at_threshold,
    false_positive_rate_curve,
    precision_at_k,
    precision_recall_at_thresholds,
    recall_at_k,
)
from safetybench.metrics.statistical import bootstrap_ci, mcnemar_test, permutation_test


class TestDetectionMetrics:
    def test_pdr_perfect(self):
        flagged = np.array([True, True, True, False, False])
        reported = np.array([False, False, False, False, False])
        violations = np.array([True, True, True, False, False])
        assert proactive_detection_rate(flagged, reported, violations) == 1.0

    def test_pdr_none_proactive(self):
        flagged = np.array([False, False, False])
        reported = np.array([True, True, True])
        violations = np.array([True, True, True])
        assert proactive_detection_rate(flagged, reported, violations) == 0.0

    def test_pdr_partial(self):
        flagged = np.array([True, False, False, False])
        reported = np.array([False, True, False, False])
        violations = np.array([True, True, True, False])
        # 1 proactive out of 3 violations
        assert abs(proactive_detection_rate(flagged, reported, violations) - 1 / 3) < 1e-9

    def test_pdr_no_violations(self):
        flagged = np.array([True, False])
        reported = np.array([False, False])
        violations = np.array([False, False])
        assert proactive_detection_rate(flagged, reported, violations) == 0.0

    def test_action_rate(self):
        actioned = np.array([True, True, False, False])
        reviewed = np.array([True, True, True, False])
        assert abs(action_rate(actioned, reviewed) - 2 / 3) < 1e-9

    def test_action_rate_none_reviewed(self):
        actioned = np.array([False])
        reviewed = np.array([False])
        assert action_rate(actioned, reviewed) == 0.0

    def test_zero_view_violation_rate(self):
        views = np.array([0, 0, 5, 0])
        actioned = np.array([True, True, True, False])
        violations = np.array([True, True, True, True])
        # 2 zero-view catches out of 4 violations
        assert abs(zero_view_violation_rate(views, actioned, violations) - 0.5) < 1e-9

    def test_detection_rate_by_category(self):
        preds = np.array([True, True, False, True])
        labels = np.array([True, True, True, True])
        cats = pd.Series(["hate", "hate", "spam", "spam"])
        result = detection_rate_by_category(preds, labels, cats)
        assert result["hate"] == 1.0
        assert result["spam"] == 0.5


class TestQualityMetrics:
    def test_fpr_at_threshold(self):
        scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
        labels = np.array([True, True, False, False, False])
        fpr = false_positive_rate_at_threshold(scores, labels, 0.5)
        assert fpr == 0.0

        fpr_low = false_positive_rate_at_threshold(scores, labels, 0.15)
        assert abs(fpr_low - 2 / 3) < 1e-9

    def test_fpr_no_negatives(self):
        scores = np.array([0.9, 0.8])
        labels = np.array([True, True])
        assert false_positive_rate_at_threshold(scores, labels, 0.5) == 0.0

    def test_fpr_curve(self):
        scores = np.random.default_rng(42).random(100)
        labels = np.random.default_rng(42).choice([True, False], 100)
        thresholds, fpr_vals = false_positive_rate_curve(scores, labels)
        assert len(thresholds) == 50
        assert len(fpr_vals) == 50
        assert fpr_vals[0] >= fpr_vals[-1]  # higher threshold -> lower FPR

    def test_appeal_overturn_rate(self):
        appealed = np.array([True, True, True, False, False])
        overturned = np.array([True, False, False, False, False])
        assert abs(appeal_overturn_rate(appealed, overturned) - 1 / 3) < 1e-9

    def test_appeal_overturn_none(self):
        appealed = np.array([False, False])
        overturned = np.array([False, False])
        assert appeal_overturn_rate(appealed, overturned) == 0.0

    def test_precision_recall_manual(self):
        scores = np.array([0.9, 0.8, 0.4, 0.3])
        labels = np.array([True, False, True, False])
        result = precision_recall_at_thresholds(scores, labels, np.array([0.5]))
        assert result["precision"][0] == 0.5  # 1 TP, 1 FP
        assert result["recall"][0] == 0.5  # 1 TP, 1 FN

    def test_precision_recall_auto_thresholds(self):
        rng = np.random.default_rng(0)
        scores = rng.random(200)
        labels = rng.choice([True, False], 200)
        result = precision_recall_at_thresholds(scores, labels)
        assert len(result["precision"]) == len(result["thresholds"])


class TestRankedMetrics:
    """Tests for precision_at_k and recall_at_k."""

    def test_precision_at_k_perfect(self):
        # Top-3 are all positives
        scores = np.array([0.9, 0.8, 0.7, 0.3, 0.2])
        labels = np.array([True, True, True, False, False])
        assert precision_at_k(scores, labels, k=3) == pytest.approx(1.0)

    def test_precision_at_k_none(self):
        # Top-2 are all negatives
        scores = np.array([0.9, 0.8, 0.3, 0.2])
        labels = np.array([False, False, True, True])
        assert precision_at_k(scores, labels, k=2) == pytest.approx(0.0)

    def test_precision_at_k_mixed(self):
        # Top-4: scores [0.9, 0.8, 0.7, 0.6], labels [True, False, True, False] -> 2/4
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.1])
        labels = np.array([True, False, True, False, True])
        assert precision_at_k(scores, labels, k=4) == pytest.approx(0.5)

    def test_precision_at_k_clamps_to_n(self):
        # k larger than array length should still work
        scores = np.array([0.9, 0.5])
        labels = np.array([True, False])
        assert precision_at_k(scores, labels, k=100) == pytest.approx(0.5)

    def test_precision_at_k_zero(self):
        scores = np.array([0.9, 0.5])
        labels = np.array([True, False])
        assert precision_at_k(scores, labels, k=0) == 0.0

    def test_precision_at_k_empty(self):
        assert precision_at_k(np.array([]), np.array([]), k=5) == 0.0

    def test_recall_at_k_perfect(self):
        # All 3 positives are in top-3
        scores = np.array([0.9, 0.8, 0.7, 0.2, 0.1])
        labels = np.array([True, True, True, False, False])
        assert recall_at_k(scores, labels, k=3) == pytest.approx(1.0)

    def test_recall_at_k_none(self):
        # Top-2 contain no positives; all positives are ranked lower
        scores = np.array([0.9, 0.8, 0.3, 0.2])
        labels = np.array([False, False, True, True])
        assert recall_at_k(scores, labels, k=2) == pytest.approx(0.0)

    def test_recall_at_k_partial(self):
        # 4 positives total; top-2 captures 2 of them -> 0.5
        scores = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
        labels = np.array([True, True, True, True, False])
        assert recall_at_k(scores, labels, k=2) == pytest.approx(0.5)

    def test_recall_at_k_no_positives(self):
        scores = np.array([0.9, 0.5, 0.2])
        labels = np.array([False, False, False])
        assert recall_at_k(scores, labels, k=2) == 0.0

    def test_recall_at_k_zero(self):
        scores = np.array([0.9, 0.5])
        labels = np.array([True, False])
        assert recall_at_k(scores, labels, k=0) == 0.0

    def test_recall_at_k_empty(self):
        assert recall_at_k(np.array([]), np.array([]), k=5) == 0.0

    def test_precision_recall_at_k_tradeoff(self):
        # As k increases, recall should non-decrease and precision can vary
        scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
        labels = np.array([True, True, False, True, False, False])
        recalls = [recall_at_k(scores, labels, k=i) for i in range(1, 7)]
        # recall must be non-decreasing
        assert all(recalls[i] <= recalls[i + 1] for i in range(len(recalls) - 1))


class TestLatencyMetrics:
    def setup_method(self):
        base = pd.Timestamp("2025-01-01")
        self.created = pd.Series([base, base, base, base])
        self.actioned = pd.Series([
            base + pd.Timedelta(seconds=10),
            base + pd.Timedelta(seconds=20),
            base + pd.Timedelta(seconds=30),
            pd.NaT,
        ])

    def test_median_tta(self):
        result = median_time_to_action(self.created, self.actioned)
        assert result == 20.0

    def test_percentiles(self):
        result = time_to_action_percentiles(
            self.created, self.actioned, percentiles=[50.0, 100.0]
        )
        assert result["p50"] == 20.0
        assert result["p100"] == 30.0

    def test_by_category(self):
        cats = pd.Series(["a", "a", "b", "b"])
        result = time_to_action_by_category(self.created, self.actioned, cats)
        assert result["a"] == 15.0  # median of 10, 20

    def test_empty(self):
        created = pd.Series(dtype="datetime64[ns]")
        actioned = pd.Series(dtype="datetime64[ns]")
        result = median_time_to_action(created, actioned)
        assert np.isnan(result)


class TestStatisticalMetrics:
    def test_bootstrap_ci_mean(self):
        data = np.random.default_rng(42).normal(5.0, 1.0, size=1000)
        est, lo, hi = bootstrap_ci(data, seed=42)
        assert 4.5 < est < 5.5
        assert lo < est < hi

    def test_bootstrap_ci_covers_true(self):
        data = np.random.default_rng(0).normal(0.0, 1.0, size=500)
        _, lo, hi = bootstrap_ci(data, confidence=0.99, seed=0)
        assert lo < 0.0 < hi

    def test_mcnemar_identical(self):
        preds = np.array([True, False, True, False, True])
        labels = np.array([True, False, True, False, True])
        result = mcnemar_test(preds, preds, labels)
        assert result["p_value"] == 1.0

    def test_mcnemar_different(self):
        rng = np.random.default_rng(42)
        labels = rng.choice([True, False], 500)
        preds_a = labels.copy()
        preds_b = rng.choice([True, False], 500)
        result = mcnemar_test(preds_a, preds_b, labels)
        assert result["p_value"] < 0.05

    def test_permutation_test(self):
        rng = np.random.default_rng(42)
        data_a = rng.normal(0, 1, 100)
        data_b = rng.normal(0, 1, 100)
        result = permutation_test(
            np.mean(data_a), np.mean(data_b), data_a, data_b, seed=42, n_permutations=1000
        )
        assert "p_value" in result
        assert "observed_diff" in result


class TestEdgeCasesEmptyInputs:
    """Metric functions given zero-element arrays should return sensible defaults."""

    def test_proactive_detection_rate_empty(self):
        empty = np.array([], dtype=bool)
        assert proactive_detection_rate(empty, empty, empty) == 0.0

    def test_action_rate_empty(self):
        empty = np.array([], dtype=bool)
        assert action_rate(empty, empty) == 0.0

    def test_zero_view_violation_rate_empty(self):
        empty_bool = np.array([], dtype=bool)
        empty_int = np.array([], dtype=np.int64)
        assert zero_view_violation_rate(empty_int, empty_bool, empty_bool) == 0.0

    def test_detection_rate_by_category_empty(self):
        result = detection_rate_by_category(
            np.array([], dtype=bool),
            np.array([], dtype=bool),
            pd.Series([], dtype=str),
        )
        assert result == {}

    def test_false_positive_rate_at_threshold_empty(self):
        empty_f = np.array([], dtype=float)
        empty_b = np.array([], dtype=bool)
        assert false_positive_rate_at_threshold(empty_f, empty_b, 0.5) == 0.0

    def test_false_positive_rate_curve_empty(self):
        empty_f = np.array([], dtype=float)
        empty_b = np.array([], dtype=bool)
        thresholds, fpr_vals = false_positive_rate_curve(empty_f, empty_b)
        assert len(thresholds) == 50
        assert (fpr_vals == 0.0).all()

    def test_appeal_overturn_rate_empty(self):
        empty = np.array([], dtype=bool)
        assert appeal_overturn_rate(empty, empty) == 0.0

    def test_precision_recall_empty(self):
        empty_f = np.array([], dtype=float)
        empty_b = np.array([], dtype=bool)
        result = precision_recall_at_thresholds(empty_f, empty_b, np.array([0.5]))
        assert result["precision"][0] == 0.0
        assert result["recall"][0] == 0.0

    def test_median_tta_empty(self):
        empty_ts = pd.Series(dtype="datetime64[ns]")
        assert np.isnan(median_time_to_action(empty_ts, empty_ts))

    def test_tta_percentiles_empty(self):
        empty_ts = pd.Series(dtype="datetime64[ns]")
        result = time_to_action_percentiles(empty_ts, empty_ts)
        assert all(np.isnan(v) for v in result.values())


class TestEdgeCasesSingleSample:
    """Metric functions given a single-element array should return exact values."""

    def test_proactive_detection_rate_single_caught(self):
        assert proactive_detection_rate(
            np.array([True]), np.array([False]), np.array([True])
        ) == 1.0

    def test_proactive_detection_rate_single_missed(self):
        assert proactive_detection_rate(
            np.array([False]), np.array([False]), np.array([True])
        ) == 0.0

    def test_action_rate_single_actioned(self):
        assert action_rate(np.array([True]), np.array([True])) == 1.0

    def test_action_rate_single_not_actioned(self):
        assert action_rate(np.array([False]), np.array([True])) == 0.0

    def test_zero_view_violation_rate_single_zero_views(self):
        assert zero_view_violation_rate(
            np.array([0]), np.array([True]), np.array([True])
        ) == 1.0

    def test_zero_view_violation_rate_single_nonzero_views(self):
        assert zero_view_violation_rate(
            np.array([5]), np.array([True]), np.array([True])
        ) == 0.0

    def test_detection_rate_by_category_single(self):
        result = detection_rate_by_category(
            np.array([True]), np.array([True]), pd.Series(["hate"])
        )
        assert result["hate"] == 1.0

    def test_false_positive_rate_single_flagged_negative(self):
        # 1 negative, flagged → FPR = 1.0
        assert false_positive_rate_at_threshold(
            np.array([0.9]), np.array([False]), 0.5
        ) == 1.0

    def test_false_positive_rate_single_positive_only(self):
        # No negatives → FPR = 0.0
        assert false_positive_rate_at_threshold(
            np.array([0.9]), np.array([True]), 0.5
        ) == 0.0

    def test_appeal_overturn_rate_single_overturned(self):
        assert appeal_overturn_rate(np.array([True]), np.array([True])) == 1.0

    def test_appeal_overturn_rate_single_upheld(self):
        assert appeal_overturn_rate(np.array([True]), np.array([False])) == 0.0

    def test_precision_recall_single_true_positive(self):
        result = precision_recall_at_thresholds(
            np.array([0.9]), np.array([True]), np.array([0.5])
        )
        assert result["precision"][0] == pytest.approx(1.0)
        assert result["recall"][0] == pytest.approx(1.0)

    def test_precision_recall_single_false_negative(self):
        result = precision_recall_at_thresholds(
            np.array([0.1]), np.array([True]), np.array([0.5])
        )
        assert result["precision"][0] == pytest.approx(0.0)
        assert result["recall"][0] == pytest.approx(0.0)

    def test_median_tta_single(self):
        base = pd.Timestamp("2025-01-01")
        created = pd.Series([base])
        actioned = pd.Series([base + pd.Timedelta(seconds=42)])
        assert median_time_to_action(created, actioned) == pytest.approx(42.0)

    def test_tta_percentiles_single(self):
        base = pd.Timestamp("2025-01-01")
        created = pd.Series([base])
        actioned = pd.Series([base + pd.Timedelta(seconds=60)])
        result = time_to_action_percentiles(created, actioned, percentiles=[50.0, 99.0])
        assert result["p50"] == pytest.approx(60.0)
        assert result["p99"] == pytest.approx(60.0)

    def test_bootstrap_ci_single(self):
        # All bootstrap resamples equal the one value, so CI collapses to a point.
        data = np.array([3.14])
        est, lo, hi = bootstrap_ci(data, seed=0)
        assert est == pytest.approx(3.14)
        assert lo == pytest.approx(3.14)
        assert hi == pytest.approx(3.14)

    def test_mcnemar_single_both_correct(self):
        result = mcnemar_test(
            np.array([True]), np.array([True]), np.array([True])
        )
        assert result["p_value"] == 1.0
