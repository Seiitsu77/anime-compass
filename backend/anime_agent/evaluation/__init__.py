"""Personalized offline evaluation utilities.

This package is intentionally separate from the conversational benchmark in
``scripts/evaluate_recommender.py``.  The latter protects catalog and agent
behavior; this package measures held-out, per-user ranking quality.
"""

from .metrics import RankingMetrics, ranking_metrics
from .split import FeedbackConfig, SplitConfig, SplitStore, UserSplit

__all__ = [
    "FeedbackConfig",
    "RankingMetrics",
    "SplitConfig",
    "SplitStore",
    "UserSplit",
    "ranking_metrics",
]
