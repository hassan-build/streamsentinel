"""
Baseline detectors for dissertation evaluation.

Three baselines, one per research question:
  - RuleBasedDetector       : "Does ML help?"
  - RandomForestBaseline    : "Does the graph structure help?"
  - build_unimodal_gnn      : "Does text fusion help?"
"""

from evaluation.baselines.random_forest import (
    RF_FEATURES,
    RandomForestBaseline,
    RandomForestConfig,
)
from evaluation.baselines.rule_based import RuleBasedConfig, RuleBasedDetector
from evaluation.baselines.unimodal_gnn import build_unimodal_gnn

__all__ = [
    "RuleBasedDetector", "RuleBasedConfig",
    "RandomForestBaseline", "RandomForestConfig", "RF_FEATURES",
    "build_unimodal_gnn",
]
