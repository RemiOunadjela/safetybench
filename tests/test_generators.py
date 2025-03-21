"""Tests for synthetic data generators."""

import numpy as np
import pandas as pd

from safetybench.generators.distributions import (
    MARKET_DISTRIBUTIONS,
    VIOLATION_CATEGORIES,
    get_distribution,
)
from safetybench.generators.synthetic import GeneratorConfig, SyntheticDataGenerator


class TestDistributions:
    def test_all_markets_sum_to_one(self):
        for market, dist in MARKET_DISTRIBUTIONS.items():
            total = sum(dist.values())
            assert abs(total - 1.0) < 1e-6, f"{market} sums to {total}"

    def test_all_categories_covered(self):
        for market, dist in MARKET_DISTRIBUTIONS.items():
            for cat in VIOLATION_CATEGORIES:
                assert cat in dist, f"{cat} missing from {market}"

    def test_fallback_distribution(self):
        dist = get_distribution("zz")
        assert len(dist) == len(VIOLATION_CATEGORIES)
        total = sum(dist.values())
        assert abs(total - 1.0) < 1e-6

    def test_known_market(self):
        dist = get_distribution("us")
        assert dist == MARKET_DISTRIBUTIONS["us"]


class TestSyntheticGenerator:
    def test_basic_generation(self):
        config = GeneratorConfig(n_samples=1000, seed=42)
        gen = SyntheticDataGenerator(config)
        df = gen.generate()

        assert len(df) >= 900  # allow rounding from market splits
        assert "content_id" in df.columns
        assert "is_violation" in df.columns
        assert "model_score" in df.columns
        assert "flagged" in df.columns
        assert "actioned" in df.columns

    def test_violation_rate_approx(self):
        config = GeneratorConfig(n_samples=10_000, violation_rate=0.10, seed=0)
        gen = SyntheticDataGenerator(config)
        df = gen.generate()

        actual_rate = df["is_violation"].mean()
        assert 0.05 < actual_rate < 0.20

    def test_multi_market(self):
        config = GeneratorConfig(
            n_samples=3000,
            markets=["us", "br", "mx"],
            seed=42,
        )
        gen = SyntheticDataGenerator(config)
        df = gen.generate()

        markets = df["market"].unique()
        assert set(markets) == {"us", "br", "mx"}

    def test_categories_match_config(self):
        config = GeneratorConfig(
            n_samples=5000,
            categories=["hate_speech", "spam"],
            seed=42,
        )
        gen = SyntheticDataGenerator(config)
        df = gen.generate()

        violation_cats = set(df[df["is_violation"]]["category"].unique())
        assert violation_cats <= {"hate_speech", "spam"}

    def test_timestamps_present(self):
        config = GeneratorConfig(n_samples=500, seed=42)
        gen = SyntheticDataGenerator(config)
        df = gen.generate()

        assert np.issubdtype(df["created_at"].dtype, np.datetime64)
        actioned_rows = df[df["actioned"]]
        assert actioned_rows["actioned_at"].notna().all()

    def test_appeal_columns(self):
        config = GeneratorConfig(n_samples=5000, appeal_rate=0.10, seed=42)
        gen = SyntheticDataGenerator(config)
        df = gen.generate()

        assert "appealed" in df.columns
        assert "overturned" in df.columns
        # overturned only among appealed
        assert not (df["overturned"] & ~df["appealed"]).any()

    def test_view_counts_non_negative(self):
        config = GeneratorConfig(n_samples=1000, seed=42)
        gen = SyntheticDataGenerator(config)
        df = gen.generate()

        assert (df["view_count"] >= 0).all()

    def test_deterministic_with_seed(self):
        config = GeneratorConfig(n_samples=500, seed=99)
        df1 = SyntheticDataGenerator(config).generate()

        config2 = GeneratorConfig(n_samples=500, seed=99)
        df2 = SyntheticDataGenerator(config2).generate()

        pd.testing.assert_frame_equal(df1, df2)

    def test_temporal_generation(self):
        config = GeneratorConfig(n_samples=1200, seed=42)
        gen = SyntheticDataGenerator(config)
        df = gen.generate_temporal(n_periods=4, period="W")

        assert "period" in df.columns
        assert "period_start" in df.columns
        assert df["period"].nunique() == 4
