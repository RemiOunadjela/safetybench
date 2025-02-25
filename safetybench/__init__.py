"""safetybench: Benchmarking Content Moderation at Scale."""

__version__ = "0.1.0"

from safetybench.evaluation.runner import EvaluationRunner
from safetybench.generators.synthetic import SyntheticDataGenerator

__all__ = ["EvaluationRunner", "SyntheticDataGenerator", "__version__"]
