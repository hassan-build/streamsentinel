"""
models/full_pipeline.py
=======================
Composes the four neural components into a single `nn.Module`.

This is the artifact that:
  - `train.py` optimises end-to-end
  - `api/fastapi_service.py` loads for live inference
  - `evaluation/ablation.py` toggles flags on for each ablation run

Ablation flags (each maps to one dissertation evaluation row):
  - `use_text`            : if False, FinBERT branch is skipped (no_llm)
  - `use_dynamic_graph`   : (informational only here; toggled in the
                            graph updater itself before construction)
  - `use_adaptive_cusum`  : if False, scorer uses fixed threshold
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import torch
import torch.nn as nn
from torch_geometric.data import Data

from models.anomaly_scorer import AnomalyScorer, AnomalyScorerConfig
from models.finbert_encoder import FinBERTEncoder, FinBERTEncoderConfig
from models.fusion_module import FusionModule, FusionModuleConfig
from models.gnn_encoder import GNNEncoder, GNNEncoderConfig


@dataclass
class FullPipelineConfig:
    """Top-level configuration. Each sub-config controls one module."""
    gnn: GNNEncoderConfig = field(default_factory=GNNEncoderConfig)
    finbert: FinBERTEncoderConfig = field(default_factory=FinBERTEncoderConfig)
    fusion: FusionModuleConfig = field(default_factory=FusionModuleConfig)
    scorer: AnomalyScorerConfig = field(default_factory=AnomalyScorerConfig)

    # Top-level ablation flags
    use_text: bool = True
    use_adaptive_cusum: bool = True

    def __post_init__(self) -> None:
        """Cross-check inter-module dimensions."""
        if self.gnn.output_dim != self.fusion.gnn_dim:
            raise ValueError(
                f"gnn.output_dim ({self.gnn.output_dim}) must equal "
                f"fusion.gnn_dim ({self.fusion.gnn_dim})"
            )
        if self.finbert.output_dim != self.fusion.text_dim:
            raise ValueError(
                f"finbert.output_dim ({self.finbert.output_dim}) must equal "
                f"fusion.text_dim ({self.fusion.text_dim})"
            )
        if self.fusion.output_dim != self.scorer.input_dim:
            raise ValueError(
                f"fusion.output_dim ({self.fusion.output_dim}) must equal "
                f"scorer.input_dim ({self.scorer.input_dim})"
            )
        # Propagate the ablation flag down into the scorer config.
        self.scorer.use_adaptive_cusum = self.use_adaptive_cusum


class FullPipeline(nn.Module):
    """End-to-end StreamSentinel model."""

    def __init__(self, config: FullPipelineConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or FullPipelineConfig()
        self.gnn = GNNEncoder(self.cfg.gnn)
        self.finbert = FinBERTEncoder(self.cfg.finbert)
        self.fusion = FusionModule(self.cfg.fusion)
        self.scorer = AnomalyScorer(self.cfg.scorer)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        graph: Data,
        headlines: Sequence[str] | None = None,
    ) -> torch.Tensor:
        """
        Run the full pipeline.

        Parameters
        ----------
        graph : torch_geometric.data.Data
            Graph from `GraphBuilder.build()`.
        headlines : sequence of str, optional
            News headlines for the current window. Ignored if
            `cfg.use_text=False` (no_llm ablation).

        Returns
        -------
        Tensor [N, num_classes]
            Per-node class logits. Apply softmax to get probabilities.
        """
        z_graph = self.gnn(graph)

        if self.cfg.use_text and headlines is not None:
            z_text = self.finbert(headlines)
            # Move text embedding to the same device as the graph tensors.
            z_text = z_text.to(z_graph.device)
        else:
            z_text = None   # fusion falls back to pure-graph path

        z_fused = self.fusion(z_graph, z_text)
        logits = self.scorer(z_fused)
        return logits

    # ------------------------------------------------------------------
    # Prediction with anomaly decision
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(
        self,
        graph: Data,
        headlines: Sequence[str] | None = None,
    ) -> dict[str, torch.Tensor | list[bool]]:
        """
        Run inference and produce final alarms.

        Returns a dict with:
          - logits      : Tensor [N, num_classes]
          - probs       : Tensor [N, num_classes]
          - p_anomalous : Tensor [N]
          - alarms      : list[bool] of length N
        """
        logits = self.forward(graph, headlines)
        probs, p_anomalous, alarms = self.scorer.predict(
            # Re-derive z_fused via fusion forward — cheaper than caching.
            # Predict calls forward internally over `self.scorer`.
            z_fused=self._fuse_only(graph, headlines)
        )
        return {
            "logits": logits.detach(),
            "probs": probs,
            "p_anomalous": p_anomalous,
            "alarms": alarms,
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _fuse_only(
        self, graph: Data, headlines: Sequence[str] | None
    ) -> torch.Tensor:
        """Compute fused embedding (no scorer). Reusable for predict()."""
        z_graph = self.gnn(graph)
        z_text = (
            self.finbert(headlines).to(z_graph.device)
            if (self.cfg.use_text and headlines is not None) else None
        )
        return self.fusion(z_graph, z_text)

    def reset_streaming_state(self) -> None:
        """Reset all stateful components (CUSUM, BERT cache)."""
        self.scorer.reset_state()
        self.finbert.clear_cache()

    def trainable_parameters(self):
        """Yield only the parameters that should be optimised.

        FinBERT is frozen — yielding only the GNN, fusion, and scorer
        weights avoids creating spurious gradient buffers in AdamW.
        """
        for module in (self.gnn, self.fusion, self.scorer):
            yield from module.parameters()
