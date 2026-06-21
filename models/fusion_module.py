"""
models/fusion_module.py
=======================
Cross-attention fusion of graph node embeddings with text embedding.

Why cross-attention?
--------------------
The naive baseline would be to concatenate the text embedding to every
node's features and run a final MLP. This gives every asset *the same*
text signal — which is incorrect: different stocks react to different
news. A headline about "TSLA recall" is highly relevant to TSLA, less
to AAPL, almost irrelevant to SPY.

Cross-attention lets each graph node *query* the text embedding for the
relevant slice. We use the standard Transformer cross-attention:

    Q = node_embeddings    (from GNN)
    K = V = text_embedding (from FinBERT)

The output is a `[N, fusion_out_dim]` tensor — one fused vector per
node — which the anomaly scorer turns into per-node logits.

Architecture
------------
    z_graph: [N, gnn_dim]   z_text: [1, text_dim]
        │                       │
        ▼                       ▼
    Linear(gnn_dim->d)      Linear(text_dim->d)
        │                       │
        │  Q                    │  K, V (broadcast)
        ▼                       ▼
    MultiheadAttention(d, heads)
        │
        ▼
    [N, d]
        │
        ▼
    Residual + LayerNorm
        │
        ▼
    Linear(d -> fusion_out_dim) + LayerNorm
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class FusionModuleConfig:
    """Configuration for `FusionModule`.

    Attributes
    ----------
    gnn_dim : int
        Dimensionality of the GNN encoder's output. Default 64.
    text_dim : int
        Dimensionality of the text encoder's output. Default 768
        (FinBERT/BERT-base hidden size).
    fusion_dim : int
        Internal projection dim used inside the attention block.
        Default 256.
    output_dim : int
        Final per-node fused embedding dim. Default 128.
    num_heads : int
        Number of attention heads. Default 8 (matches BERT base config).
    dropout : float
        Dropout on attention scores and after fusion projection.
    """
    gnn_dim: int = 64
    text_dim: int = 768
    fusion_dim: int = 256
    output_dim: int = 128
    num_heads: int = 8
    dropout: float = 0.2

    def __post_init__(self) -> None:
        if self.fusion_dim % self.num_heads != 0:
            raise ValueError(
                f"fusion_dim ({self.fusion_dim}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


class FusionModule(nn.Module):
    """Cross-attention fusion of GNN and text embeddings."""

    def __init__(self, config: FusionModuleConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or FusionModuleConfig()
        c = self.cfg

        # Project Q (graph) and K, V (text) to the same fusion dim.
        self.q_proj = nn.Linear(c.gnn_dim, c.fusion_dim)
        self.kv_proj = nn.Linear(c.text_dim, c.fusion_dim)

        # Cross-attention. batch_first=True because PyG node tensors
        # are layed out as [N, F] and we treat N as the sequence dim.
        self.attention = nn.MultiheadAttention(
            embed_dim=c.fusion_dim,
            num_heads=c.num_heads,
            dropout=c.dropout,
            batch_first=True,
        )

        # Norm + output projection.
        self.norm_attn = nn.LayerNorm(c.fusion_dim)
        self.output_proj = nn.Sequential(
            nn.Linear(c.fusion_dim, c.fusion_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.fusion_dim, c.output_dim),
        )
        self.norm_out = nn.LayerNorm(c.output_dim)

    def forward(
        self,
        z_graph: torch.Tensor,
        z_text: torch.Tensor | None = None,
        return_attention: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse a per-node graph embedding with a single text embedding.

        Parameters
        ----------
        z_graph : Tensor [N, gnn_dim]
            Per-node embedding from `GNNEncoder`.
        z_text : Tensor [1, text_dim], optional
            Single text embedding from `FinBERTEncoder`. If None, fusion
            falls back to a pure-graph path (skip-connection through
            self-attention identity), supporting the `no_llm` ablation.
        return_attention : bool
            If True, also return the per-node attention weights for
            visualisation. Default False.

        Returns
        -------
        Tensor [N, output_dim]   (and Tensor [N, 1] if return_attention)
            The fused per-node embedding.
        """
        # Project Q from graph nodes; shape [1, N, fusion_dim] (batch dim of 1).
        q = self.q_proj(z_graph).unsqueeze(0)        # [1, N, D]

        if z_text is None:
            # Ablation path: use the projected graph embedding as both
            # Q and K, V. Effectively self-attention with no text.
            kv = q
        else:
            # Project text to same dim. Broadcast over N nodes via the
            # natural [1, 1, D] shape (MHA broadcasts K/V across queries).
            kv = self.kv_proj(z_text).unsqueeze(0)   # [1, 1, D]

        # Cross-attend.
        attn_out, attn_weights = self.attention(
            query=q, key=kv, value=kv,
            need_weights=return_attention,
            average_attn_weights=True,
        )  # attn_out: [1, N, D]

        # Residual + norm.
        fused = self.norm_attn(attn_out + q)         # [1, N, D]
        fused = fused.squeeze(0)                     # [N, D]

        out = self.output_proj(fused)                # [N, output_dim]
        out = self.norm_out(out)

        if return_attention:
            # attn_weights shape: [1, N, kv_seq_len]
            return out, attn_weights.squeeze(0)
        return out
