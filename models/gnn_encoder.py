"""
models/gnn_encoder.py
=====================
Graph Attention Network (GAT) encoder for the StreamSentinel asset graph.

We use GAT (Veličković et al. 2018) over GCN or GraphSAGE because the
attention coefficients are intrinsically interpretable: for any given
node prediction we can ask "which neighbour did the model attend to?"
This is the foundation for the attention-visualisation chapter of the
dissertation explainability evaluation.

Architecture
------------
    x -> Linear(input_dim -> hidden) -> ReLU
       -> GATConv block × num_layers   (each: GAT + LayerNorm + residual)
       -> Linear(hidden -> output_dim)
       -> LayerNorm

Each GAT block uses multi-head attention. The residual connection is
applied AFTER concatenating the head outputs and projecting back to
`hidden`, so the dimensions line up without an extra projection layer.

The encoder consumes a `torch_geometric.data.Data` object produced by
`graph.GraphBuilder` and emits a `[num_nodes, output_dim]` tensor that
the fusion module later combines with the text embedding.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATConv


@dataclass
class GNNEncoderConfig:
    """Configuration for `GNNEncoder`.

    Attributes
    ----------
    input_dim : int
        Dimensionality of node features (from `graph.NODE_FEATURE_DIM`).
        Default 10 matches the GraphBuilder schema.
    hidden_channels : int
        Channels in each hidden GAT layer. Default 128.
    num_layers : int
        Number of GAT blocks. Default 3. >3 risks over-smoothing on
        small graphs (5–10 nodes is our typical case).
    heads : int
        Number of attention heads per GAT layer. Default 4.
        Outputs of the heads are concatenated then projected back.
    output_dim : int
        Dimensionality of the final per-node embedding. Default 64.
    dropout : float
        Dropout applied to attention coefficients and after each block.
        Default 0.3.
    edge_dim : int
        Edge feature dimensionality (from `graph.EDGE_FEATURE_DIM`).
        Default 2 (correlation + sign). Set to None to ignore edges.
    """
    input_dim: int = 10
    hidden_channels: int = 128
    num_layers: int = 3
    heads: int = 4
    output_dim: int = 64
    dropout: float = 0.3
    edge_dim: int | None = 2

    def __post_init__(self) -> None:
        if self.num_layers < 1:
            raise ValueError(f"num_layers must be >= 1, got {self.num_layers}")
        if self.heads < 1:
            raise ValueError(f"heads must be >= 1, got {self.heads}")
        if self.hidden_channels % self.heads != 0:
            raise ValueError(
                f"hidden_channels ({self.hidden_channels}) must be divisible "
                f"by heads ({self.heads})"
            )
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")


class GNNEncoder(nn.Module):
    """Multi-layer GAT encoder with residual connections and LayerNorm."""

    def __init__(self, config: GNNEncoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or GNNEncoderConfig()
        c = self.cfg

        # Input projection: raw node features -> hidden_channels
        self.input_proj = nn.Linear(c.input_dim, c.hidden_channels)

        # Per-head dimension: heads are concatenated to produce hidden_channels.
        per_head = c.hidden_channels // c.heads

        # Stack of GAT blocks. Each block:
        #   1. GATConv (with multi-head attention)
        #   2. LayerNorm
        #   3. residual add
        #   4. dropout
        self.gat_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(c.num_layers):
            self.gat_layers.append(
                GATConv(
                    in_channels=c.hidden_channels,
                    out_channels=per_head,
                    heads=c.heads,
                    concat=True,           # concat heads -> per_head*heads
                    dropout=c.dropout,
                    edge_dim=c.edge_dim,
                    add_self_loops=False,  # we add them in GraphBuilder
                )
            )
            self.norms.append(nn.LayerNorm(c.hidden_channels))

        # Output projection
        self.output_proj = nn.Linear(c.hidden_channels, c.output_dim)
        self.output_norm = nn.LayerNorm(c.output_dim)

    def forward(self, data: Data) -> torch.Tensor:
        """
        Run the encoder.

        Parameters
        ----------
        data : torch_geometric.data.Data
            With `x: [N, input_dim]`, `edge_index: [2, E]`, optional
            `edge_attr: [E, edge_dim]`, and optionally `batch: [N]`
            for batched graphs.

        Returns
        -------
        torch.Tensor
            Per-node embedding of shape `[N, output_dim]`. When the
            input represents a batch of B graphs, N is the total
            number of nodes across the batch (PyG's standard layout).
        """
        x = data.x
        edge_index = data.edge_index
        edge_attr = getattr(data, "edge_attr", None)

        # Initial projection.
        h = F.relu(self.input_proj(x))

        # GAT blocks with residual + norm.
        for gat, norm in zip(self.gat_layers, self.norms):
            residual = h
            h_new = gat(h, edge_index, edge_attr=edge_attr)
            # GATConv with concat=True returns shape [N, per_head * heads]
            # which equals hidden_channels by construction.
            h_new = F.dropout(F.elu(h_new), p=self.cfg.dropout,
                              training=self.training)
            h = norm(residual + h_new)

        # Output projection.
        out = self.output_proj(h)
        out = self.output_norm(out)
        return out

    @torch.no_grad()
    def get_attention_weights(
        self, data: Data, layer_idx: int = -1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Extract attention weights from a single GAT layer for visualisation.

        Used by `explainability/attention_visualiser.py` to produce the
        attention heatmaps included in the dissertation.

        Parameters
        ----------
        data : Data
            Graph to run.
        layer_idx : int
            Which GAT layer to extract from. Default -1 = last layer.

        Returns
        -------
        (edge_index_used, attention_weights)
            edge_index_used : LongTensor [2, E']
                The edge index PyG used internally (may include self-loops).
            attention_weights : Tensor [E', heads]
                Per-edge, per-head attention scores in [0, 1].
        """
        was_training = self.training
        self.eval()
        try:
            x = F.relu(self.input_proj(data.x))
            edge_attr = getattr(data, "edge_attr", None)

            # Walk to the requested layer.
            target_idx = layer_idx if layer_idx >= 0 else (
                len(self.gat_layers) + layer_idx
            )

            for i, (gat, norm) in enumerate(zip(self.gat_layers, self.norms)):
                if i == target_idx:
                    # Request attention coefficients from PyG.
                    out, (ei_used, alpha) = gat(
                        x, data.edge_index, edge_attr=edge_attr,
                        return_attention_weights=True,
                    )
                    return ei_used.detach(), alpha.detach()
                residual = x
                h_new = gat(x, data.edge_index, edge_attr=edge_attr)
                h_new = F.elu(h_new)
                x = norm(residual + h_new)
            raise IndexError(
                f"layer_idx {layer_idx} out of range for "
                f"{len(self.gat_layers)} layers"
            )
        finally:
            self.train(was_training)
