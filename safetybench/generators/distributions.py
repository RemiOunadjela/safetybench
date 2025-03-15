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
    "nudity",
    "regulated_goods",
]

# Relative prevalence weights per market. These are normalized to probabilities
# at generation time.  The shape of these distributions varies across markets
# due to differences in regulatory focus, cultural norms, and attack surfaces.

MARKET_DISTRIBUTIONS: dict[str, dict[str, float]] = {
    "us": {
        "hate_speech": 0.18,
        "harassment": 0.20,
        "dangerous_acts": 0.06,
        "csam": 0.03,
        "violent_extremism": 0.05,
        "self_harm": 0.08,
        "spam": 0.22,
        "misinformation": 0.10,
        "nudity": 0.05,
        "regulated_goods": 0.03,
    },
    "br": {
        "hate_speech": 0.12,
        "harassment": 0.15,
        "dangerous_acts": 0.08,
        "csam": 0.04,
        "violent_extremism": 0.03,
        "self_harm": 0.06,
        "spam": 0.28,
        "misinformation": 0.12,
        "nudity": 0.08,
        "regulated_goods": 0.04,
    },
    "mx": {
        "hate_speech": 0.10,
        "harassment": 0.14,
        "dangerous_acts": 0.10,
        "csam": 0.03,
        "violent_extremism": 0.04,
        "self_harm": 0.05,
        "spam": 0.30,
        "misinformation": 0.11,
        "nudity": 0.07,
        "regulated_goods": 0.06,
    },
    "in": {
        "hate_speech": 0.22,
        "harassment": 0.12,
        "dangerous_acts": 0.05,
        "csam": 0.03,
        "violent_extremism": 0.06,
        "self_harm": 0.04,
        "spam": 0.20,
        "misinformation": 0.18,
        "nudity": 0.04,
        "regulated_goods": 0.06,
    },
    "id": {
        "hate_speech": 0.14,
        "harassment": 0.10,
        "dangerous_acts": 0.06,
        "csam": 0.03,
        "violent_extremism": 0.08,
        "self_harm": 0.04,
        "spam": 0.25,
        "misinformation": 0.15,
        "nudity": 0.09,
        "regulated_goods": 0.06,
    },
    "de": {
        "hate_speech": 0.20,
        "harassment": 0.16,
        "dangerous_acts": 0.04,
        "csam": 0.03,
        "violent_extremism": 0.07,
        "self_harm": 0.06,
        "spam": 0.18,
        "misinformation": 0.14,
        "nudity": 0.06,
        "regulated_goods": 0.06,
    },
    "jp": {
        "hate_speech": 0.08,
        "harassment": 0.18,
        "dangerous_acts": 0.04,
        "csam": 0.05,
        "violent_extremism": 0.02,
        "self_harm": 0.10,
        "spam": 0.25,
        "misinformation": 0.08,
        "nudity": 0.14,
        "regulated_goods": 0.06,
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
