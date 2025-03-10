"""Basic evaluation example: generate data, evaluate, produce a report."""

from safetybench.evaluation.runner import EvaluationConfig, EvaluationRunner
from safetybench.generators.synthetic import GeneratorConfig, SyntheticDataGenerator
from safetybench.reports.markdown import MarkdownReportGenerator


def main():
    # Generate synthetic moderation data
    gen_config = GeneratorConfig(
        n_samples=50_000,
        markets=["us"],
        violation_rate=0.05,
        model_auc=0.94,
        seed=42,
    )
    generator = SyntheticDataGenerator(gen_config)
    df = generator.generate()

    print(f"Generated {len(df):,} samples with {df['is_violation'].sum():,} violations")

    # Run evaluation
    eval_config = EvaluationConfig(
        threshold=0.5,
        bootstrap_samples=2_000,
        compute_ci=True,
    )
    runner = EvaluationRunner(eval_config)
    result = runner.evaluate(df)

    # Print summary
    print("\n--- Evaluation Summary ---")
    summary = result.summary()
    print(summary.to_string(index=False))

    # Write report
    report_gen = MarkdownReportGenerator()
    report_gen.write(result, "evaluation_report.md")
    print("\nReport written to evaluation_report.md")


if __name__ == "__main__":
    main()
