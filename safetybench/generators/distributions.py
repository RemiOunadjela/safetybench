"""Market-specific violation category distributions.

These distributions are synthetic approximations inspired by publicly available
transparency reports. They do not reflect any proprietary data.
"""

from __future__ import annotations

VIOLATION_CATEGORIES: list[str] = [
    "hate_speech",
    "harassment",
    "dangerous_acts",
    "csam",
    "violent_extremism",
    "self_harm",
    "spam",
    "misinformation",
]

# Relative prevalence weights per market. These are normalized to probabilities
# at generation time.  The shape of these distributions varies across markets
# due to differences in regulatory focus, cultural norms, and attack surfaces.

MARKET_DISTRIBUTIONS: dict[str, dict[str, float]] = {
    "us": {
        "hate_speech": 0.20,
        "harassment": 0.22,
        "dangerous_acts": 0.07,
        "csam": 0.04,
        "violent_extremism": 0.06,
        "self_harm": 0.09,
        "spam": 0.21,
        "misinformation": 0.11,
    },
    "br": {
        "hate_speech": 0.14,
        "harassment": 0.17,
        "dangerous_acts": 0.09,
        "csam": 0.05,
        "violent_extremism": 0.04,
        "self_harm": 0.07,
        "spam": 0.30,
        "misinformation": 0.14,
    },
    "in": {
        "hate_speech": 0.25,
        "harassment": 0.14,
        "dangerous_acts": 0.06,
        "csam": 0.04,
        "violent_extremism": 0.07,
        "self_harm": 0.05,
        "spam": 0.22,
        "misinformation": 0.17,
    },
    "id": {
        "hate_speech": 0.16,
        "harassment": 0.12,
        "dangerous_acts": 0.07,
        "csam": 0.04,
        "violent_extremism": 0.09,
        "self_harm": 0.05,
        "spam": 0.28,
        "misinformation": 0.19,
    },
    "de": {
        "hate_speech": 0.22,
        "harassment": 0.18,
        "dangerous_acts": 0.05,
        "csam": 0.04,
        "violent_extremism": 0.08,
        "self_harm": 0.07,
        "spam": 0.20,
        "misinformation": 0.16,
    },
    "jp": {
        "hate_speech": 0.10,
        "harassment": 0.22,
        "dangerous_acts": 0.05,
        "csam": 0.07,
        "violent_extremism": 0.03,
        "self_harm": 0.12,
        "spam": 0.28,
        "misinformation": 0.13,
    },
}


def get_distribution(market: str) -> dict[str, float]:
    """Return the violation distribution for a given market code.

    Falls back to a uniform distribution over all categories if the market
    is not in the predefined set.
    """
    if market in MARKET_DISTRIBUTIONS:
        return MARKET_DISTRIBUTIONS[market]

    n = len(VIOLATION_CATEGORIES)
    return {cat: 1.0 / n for cat in VIOLATION_CATEGORIES}
