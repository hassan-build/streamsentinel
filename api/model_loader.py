"""
api/model_loader.py
===================
Constructs a FullPipeline from `config.yaml` and loads the trained
checkpoint at `checkpoints/best_model.pt` (or whatever path is
provided).

Mirrors the pipeline-construction code from
`evaluation/run_evaluation.py` and `explainability/run_explainability.py`
— factored here so the API doesn't reach into other modules' internals.
"""

from __future__ import annotations

from pathlib import Path

import torch

from logger import get_logger
from config_loader import load_config
from models.anomaly_scorer import AnomalyScorerConfig
from models.finbert_encoder import FinBERTEncoderConfig
from models.full_pipeline import FullPipeline, FullPipelineConfig
from models.fusion_module import FusionModuleConfig
from models.gnn_encoder import GNNEncoderConfig


log = get_logger(__name__)


def build_pipeline_from_config() -> FullPipeline:
    """Construct an un-initialised FullPipeline from `config.yaml`.

    Weights are random until `load_checkpoint()` runs.
    """
    cfg_all = load_config()
    m = cfg_all["models"]
    base_cfg = FullPipelineConfig(
        gnn=GNNEncoderConfig(
            hidden_channels=m["gnn"]["hidden_channels"],
            num_layers=m["gnn"]["num_layers"],
            heads=m["gnn"]["heads"],
            dropout=m["gnn"]["dropout"],
            output_dim=m["gnn"]["output_dim"],
        ),
        finbert=FinBERTEncoderConfig(
            model_name=m["finbert"]["model_name"],
            max_length=m["finbert"]["max_length"],
            output_dim=m["finbert"]["output_dim"],
            device="cpu",
            cache_dir=m["finbert"].get("cache_dir"),
            allow_offline_fallback=True,
        ),
        fusion=FusionModuleConfig(
            gnn_dim=m["fusion"]["gnn_dim"],
            text_dim=m["fusion"]["text_dim"],
            num_heads=m["fusion"]["cross_attention_heads"],
            fusion_dim=m["fusion"]["fusion_hidden_dim"],
            dropout=m["fusion"]["dropout"],
            output_dim=m["fusion"]["output_dim"],
        ),
        scorer=AnomalyScorerConfig(
            input_dim=m["fusion"]["output_dim"],
            fixed_threshold=m["anomaly_scorer"].get("fixed_threshold", 0.5),
            cusum_k=m["anomaly_scorer"]["cusum_k"],
            cusum_h=m["anomaly_scorer"]["cusum_h"],
        ),
    )
    return FullPipeline(base_cfg)


def load_checkpoint(
    pipeline: FullPipeline,
    checkpoint_path: Path,
    strict: bool = False,
) -> bool:
    """
    Load a saved checkpoint into a FullPipeline.

    Parameters
    ----------
    pipeline : FullPipeline
        Constructed via `build_pipeline_from_config()`.
    checkpoint_path : Path
        File written by `models/train.py`. Must contain a `model_state`
        key (the default output format).
    strict : bool
        Forward to `nn.Module.load_state_dict(strict=...)`. We default
        to False so that minor ablation-related state-key changes don't
        crash startup.

    Returns
    -------
    bool : True if loaded successfully.
    """
    if not checkpoint_path.exists():
        log.warning(
            f"Checkpoint not found at {checkpoint_path}; serving with "
            "random weights. Predictions will be uninformative."
        )
        return False
    try:
        ckpt = torch.load(checkpoint_path, map_location="cpu",
                          weights_only=False)
    except Exception as exc:  # noqa: BLE001
        log.error(f"Checkpoint failed to load: {exc}")
        return False

    state = ckpt.get("model_state", ckpt)
    try:
        pipeline.load_state_dict(state, strict=strict)
        log.info(f"Loaded checkpoint from {checkpoint_path}")
        return True
    except Exception as exc:  # noqa: BLE001
        log.error(f"State-dict load failed: {exc}")
        return False
