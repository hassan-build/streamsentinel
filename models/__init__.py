"""
StreamSentinel — models package.

Public API:
  - GNNEncoder, FinBERTEncoder, FusionModule, AnomalyScorer (components)
  - FullPipeline (composed end-to-end model)
  - ANOMALY_CLASSES, NUM_CLASSES, NORMAL_CLASS_IDX (label constants)
"""

from models.anomaly_scorer import (
    ANOMALY_CLASSES,
    NORMAL_CLASS_IDX,
    NUM_CLASSES,
    AdaptiveCUSUM,
    AnomalyScorer,
    AnomalyScorerConfig,
)
from models.finbert_encoder import FinBERTEncoder, FinBERTEncoderConfig
from models.full_pipeline import FullPipeline, FullPipelineConfig
from models.fusion_module import FusionModule, FusionModuleConfig
from models.gnn_encoder import GNNEncoder, GNNEncoderConfig

__all__ = [
    "GNNEncoder", "GNNEncoderConfig",
    "FinBERTEncoder", "FinBERTEncoderConfig",
    "FusionModule", "FusionModuleConfig",
    "AnomalyScorer", "AnomalyScorerConfig",
    "AdaptiveCUSUM",
    "FullPipeline", "FullPipelineConfig",
    "ANOMALY_CLASSES", "NUM_CLASSES", "NORMAL_CLASS_IDX",
]
