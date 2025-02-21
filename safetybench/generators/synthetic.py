"""Synthetic data generation for content moderation benchmarking.

Generates realistic moderation pipeline data including model scores,
ground-truth labels, timestamps, view counts, and user reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from safetybench.generators.distributions import VIOLATION_CATEGORIES, get_distribution


@dataclass
class GeneratorConfig:
    """Configuration for synthetic data generation."""

    n_samples: int = 10_000
    violation_rate: float = 0.05
    categories: list[str] = field(default_factory=lambda: list(VIOLATION_CATEGORIES))
    markets: list[str] = field(default_factory=lambda: ["us"])
    model_auc: float = 0.92
    proactive_rate: float = 0.85
    appeal_rate: float = 0.03
    overturn_rate: float = 0.15
    median_tta_seconds: float = 120.0
    start_date: str = "2025-01-01"
    end_date: str = "2025-03-31"
    seed: int | None = 42


class SyntheticDataGenerator:
    """Generates synthetic content moderation datasets.

    The generator produces data that mirrors real moderation pipeline outputs:
    each row represents a piece of content with associated model scores,
    ground-truth labels, action timestamps, and appeal outcomes.
    """

    def __init__(self, config: GeneratorConfig | None = None):
        self.config = config or GeneratorConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def generate(self) -> pd.DataFrame:
        """Generate a complete moderation dataset.

        Returns:
            DataFrame with columns: content_id, market, category, is_violation,
            model_score, flagged, actioned, user_reported, view_count,
            created_at, actioned_at, appealed, overturned.
        """
        frames = []
        samples_per_market = self.config.n_samples // len(self.config.markets)

        for market in self.config.markets:
            df = self._generate_market(market, samples_per_market)
            frames.append(df)

        result = pd.concat(frames, ignore_index=True)
        result["content_id"] = [f"c_{i:08d}" for i in range(len(result))]
        return result

    def _generate_market(self, market: str, n: int) -> pd.DataFrame:
        dist = get_distribution(market)
        available_cats = [c for c in self.config.categories if c in dist]
        weights = np.array([dist[c] for c in available_cats])
        weights /= weights.sum()

        is_violation = self.rng.random(n) < self.config.violation_rate

        categories = np.where(
            is_violation,
            self.rng.choice(available_cats, size=n, p=weights),
            "none",
        )

        model_score = self._generate_scores(is_violation)

        threshold = self._auc_to_threshold(self.config.model_auc, model_score, is_violation)
        flagged = model_score >= threshold

        # Proactive vs user-reported
        user_reported = np.zeros(n, dtype=bool)
        violation_idx = np.where(is_violation & ~flagged)[0]
        if len(violation_idx) > 0:
            report_prob = 0.4
            user_reported[violation_idx] = self.rng.random(len(violation_idx)) < report_prob

        proactive_mask = flagged & is_violation & ~user_reported
        total_violations = is_violation.sum()
        current_pdr = proactive_mask.sum() / total_violations if total_violations > 0 else 0

        # Adjust some missed violations to be proactively caught
        if current_pdr < self.config.proactive_rate and total_violations > 0:
            missed = np.where(is_violation & ~flagged & ~user_reported)[0]
            needed = int((self.config.proactive_rate - current_pdr) * total_violations)
            if len(missed) > 0:
                to_flag = self.rng.choice(missed, size=min(needed, len(missed)), replace=False)
                flagged[to_flag] = True
                model_score[to_flag] = self.rng.uniform(threshold, 1.0, size=len(to_flag))

        actioned = flagged | user_reported

        # Timestamps
        start = pd.Timestamp(self.config.start_date)
        end = pd.Timestamp(self.config.end_date)
        total_seconds = int((end - start).total_seconds())
        offsets = self.rng.integers(0, total_seconds, size=n)
        created_at = pd.Series([start + pd.Timedelta(seconds=int(s)) for s in offsets])

        tta_seconds = self.rng.exponential(self.config.median_tta_seconds, size=n)
        actioned_at = pd.Series([
            created_at.iloc[i] + pd.Timedelta(seconds=float(tta_seconds[i]))
            if actioned[i] else pd.NaT
            for i in range(n)
        ])

        # View counts: zero-view for very fast actions, otherwise lognormal
        view_counts = np.zeros(n, dtype=np.int64)
        actioned_idx = np.where(actioned)[0]
        if len(actioned_idx) > 0:
            fast_mask = tta_seconds[actioned_idx] < 30
            slow_idx = actioned_idx[~fast_mask]
            if len(slow_idx) > 0:
                view_counts[slow_idx] = self.rng.lognormal(
                    mean=3.0, sigma=2.0, size=len(slow_idx)
                ).astype(np.int64)

        # Appeals
        appealed = np.zeros(n, dtype=bool)
        actioned_idx_list = np.where(actioned)[0]
        if len(actioned_idx_list) > 0:
            appealed[actioned_idx_list] = (
                self.rng.random(len(actioned_idx_list)) < self.config.appeal_rate
            )

        overturned = np.zeros(n, dtype=bool)
        appeal_idx = np.where(appealed)[0]
        if len(appeal_idx) > 0:
            # Higher overturn rate for false positives
            for idx in appeal_idx:
                if is_violation[idx]:
                    overturned[idx] = self.rng.random() < (self.config.overturn_rate * 0.3)
                else:
                    overturned[idx] = self.rng.random() < self.config.overturn_rate

        return pd.DataFrame({
            "market": market,
            "category": categories,
            "is_violation": is_violation,
            "model_score": model_score,
            "flagged": flagged,
            "actioned": actioned,
            "user_reported": user_reported,
            "view_count": view_counts,
            "created_at": created_at,
            "actioned_at": actioned_at,
            "appealed": appealed,
            "overturned": overturned,
        })

    def _generate_scores(self, is_violation: np.ndarray) -> np.ndarray:
        """Generate model scores with separation controlled by target AUC."""
        n = len(is_violation)
        scores = np.empty(n)

        pos_mask = is_violation
        neg_mask = ~is_violation

        # Use beta distributions to control separation
        sep = self.config.model_auc
        alpha_pos = 2 + 8 * sep
        beta_pos = 2 + 8 * (1 - sep)
        alpha_neg = 2 + 8 * (1 - sep)
        beta_neg = 2 + 8 * sep

        scores[pos_mask] = self.rng.beta(alpha_pos, beta_pos, size=pos_mask.sum())
        scores[neg_mask] = self.rng.beta(alpha_neg, beta_neg, size=neg_mask.sum())

        return np.clip(scores, 0, 1)

    def _auc_to_threshold(
        self, target_auc: float, scores: np.ndarray, labels: np.ndarray
    ) -> float:
        """Find a reasonable threshold given the score distribution."""
        from sklearn.metrics import roc_curve

        fpr, tpr, thresholds = roc_curve(labels, scores)
        # Youden's J statistic
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        return float(thresholds[best_idx])

    def generate_temporal(
        self,
        n_periods: int = 12,
        period: str = "W",
    ) -> pd.DataFrame:
        """Generate data with temporal structure for trend analysis.

        Args:
            n_periods: Number of time periods.
            period: Pandas frequency string ('W' for weekly, 'M' for monthly).

        Returns:
            DataFrame with an additional 'period' column.
        """
        original_n = self.config.n_samples
        per_period = original_n // n_periods

        frames = []
        base_date = pd.Timestamp(self.config.start_date)

        for i in range(n_periods):
            period_start = base_date + pd.DateOffset(**{_freq_to_offset(period): i})
            period_end = base_date + pd.DateOffset(**{_freq_to_offset(period): i + 1})

            self.config.n_samples = per_period
            self.config.start_date = str(period_start.date())
            self.config.end_date = str(period_end.date())

            # Slight drift in violation rate over time
            drift = 1 + 0.02 * i
            original_vr = self.config.violation_rate
            self.config.violation_rate = min(0.5, original_vr * drift)

            df = self.generate()
            df["period"] = i
            df["period_start"] = period_start
            frames.append(df)

            self.config.violation_rate = original_vr

        self.config.n_samples = original_n
        self.config.start_date = str(base_date.date())

        return pd.concat(frames, ignore_index=True)


def _freq_to_offset(freq: str) -> str:
    mapping = {"W": "weeks", "M": "months", "D": "days"}
    return mapping.get(freq, "weeks")
