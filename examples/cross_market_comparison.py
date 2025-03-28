"""Cross-market comparison: evaluate the same model across different markets."""

import numpy as np

from safetybench.evaluation.comparator import ModelComparator
from safetybench.evaluation.runner import EvaluationConfig
from safetybench.evaluation.thresholds import ThresholdAnalyzer
from safetybench.generators.synthetic import GeneratorConfig, SyntheticDataGenerator


def main():
    # Generate data across multiple markets
    config = GeneratorConfig(
        n_samples=30_000,
        markets=["us", "br", "mx", "in", "de"],
        violation_rate=0.06,
        model_auc=0.92,
        seed=42,
    )
    generator = SyntheticDataGenerator(config)
    df = generator.generate()

    print(f"Dataset: {len(df):,} samples across {df['market'].nunique()} markets\n")

    # Per-market threshold optimization
    analyzer = ThresholdAnalyzer(n_points=50)
    for market in df["market"].unique():
        mdf = df[df["market"] == market]
        scores = mdf["model_score"].values
        labels = mdf["is_violation"].values

        optimal = analyzer.optimal_threshold(scores, labels, strategy="f1")
        auc = analyzer.auc(scores, labels)

        print(
            f"[{market.upper()}] AUC={auc:.3f}  "
            f"Optimal threshold={optimal.threshold:.3f}  "
            f"F1={optimal.f1:.3f}  "
            f"Precision={optimal.precision:.3f}  "
            f"Recall={optimal.recall:.3f}"
        )

    # Simulate two model versions for comparison
    rng = np.random.default_rng(99)
    df["score_v1"] = df["model_score"]
    df["score_v2"] = np.clip(df["model_score"] * 1.05 + rng.normal(0, 0.02, len(df)), 0, 1)

    comparator = ModelComparator(EvaluationConfig(threshold=0.5))
    result = comparator.compare(df, {"baseline_v1": "score_v1", "improved_v2": "score_v2"})

    print("\n--- Model Comparison ---")
    print(result.summary_table[["precision", "recall", "fpr", "action_rate"]].to_string())

    for pair, test in result.pairwise_tests.items():
        sig = "significant" if test["p_value"] < 0.05 else "not significant"
        print(f"\n{pair}: p={test['p_value']:.4f} ({sig})")

    winner = result.winner(metric="recall")
    print(f"\nWinner by recall: {winner}")


if __name__ == "__main__":
    main()
