"""
explainability/attention_visualiser.py
======================================
Renders GAT attention heatmaps for the dissertation.

The `models.GNNEncoder` exposes `get_attention_weights()`, which returns
per-edge, per-head attention scores. This module turns those into:

  1. **Per-prediction heatmaps.** For a single inference event, an
     N × N matrix showing how each node attended to every other node.
     One PNG per representative anomaly case.
  2. **Aggregate heatmap.** Mean attention across many predictions —
     shows the *learned market structure*.

Attention as explanation: caveats
---------------------------------
Recent work (Jain & Wallace 2019; Bilodeau et al. 2024) shows attention
weights can be manipulated without changing model predictions. We use
attention as a **behavioural probe**, NOT a causal claim, and report
SHAP attributions in parallel as the more rigorous trust evidence. The
attention heatmaps are intended for the dissertation Discussion to
show that the GAT *learns* sensible cross-asset structure — they are
not the headline interpretability result.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch_geometric.data import Data

from models.full_pipeline import FullPipeline


@dataclass
class AttentionResult:
    """Output of `AttentionVisualiser.compute_attention_matrix`.

    Attributes
    ----------
    matrix : np.ndarray
        Shape [n_nodes, n_nodes]. matrix[i, j] is the attention weight
        from query node i to key node j, averaged across heads.
    symbols : tuple[str, ...]
        Node labels for plotting.
    layer_idx : int
        Which GAT layer was probed.
    head_strategy : str
        "mean" (averaged across heads) or "max" or "head_<k>".
    """
    matrix: np.ndarray
    symbols: tuple[str, ...]
    layer_idx: int
    head_strategy: str = "mean"


class AttentionVisualiser:
    """Extracts and plots attention weights from the GAT encoder."""

    def __init__(self, pipeline: FullPipeline) -> None:
        self.pipeline = pipeline
        self.pipeline.eval()

    # ------------------------------------------------------------------
    # Extraction
    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_attention_matrix(
        self,
        graph: Data,
        symbols: Sequence[str],
        layer_idx: int = -1,
        head_strategy: str = "mean",
    ) -> AttentionResult:
        """
        Extract a single N × N attention matrix from the GAT encoder.

        Parameters
        ----------
        graph : Data
            One graph snapshot.
        symbols : sequence of str
            Symbol names in canonical order; used as axis labels.
        layer_idx : int
            Which GAT layer (-1 = last).
        head_strategy : str
            "mean" averages across heads (default); "max" takes the
            element-wise maximum; "head_<k>" selects head k.

        Returns
        -------
        AttentionResult
        """
        edge_index, alpha = self.pipeline.gnn.get_attention_weights(
            graph, layer_idx=layer_idx
        )
        # alpha shape: [n_edges, n_heads]
        if head_strategy == "mean":
            attn = alpha.mean(dim=-1).cpu().numpy()
        elif head_strategy == "max":
            attn = alpha.max(dim=-1).values.cpu().numpy()
        elif head_strategy.startswith("head_"):
            k = int(head_strategy.split("_")[1])
            attn = alpha[:, k].cpu().numpy()
        else:
            raise ValueError(f"Unknown head_strategy: {head_strategy}")

        n = len(symbols)
        matrix = np.zeros((n, n), dtype=np.float64)
        # PyG convention: edge_index[0] = source (key), edge_index[1] = target (query).
        # GAT semantics: target node attends to source -> matrix[target, source] = weight.
        srcs = edge_index[0].cpu().numpy()
        dsts = edge_index[1].cpu().numpy()
        for src, dst, w in zip(srcs, dsts, attn):
            if 0 <= dst < n and 0 <= src < n:
                matrix[int(dst), int(src)] += float(w)

        return AttentionResult(
            matrix=matrix,
            symbols=tuple(symbols),
            layer_idx=layer_idx,
            head_strategy=head_strategy,
        )

    @torch.no_grad()
    def aggregate_attention(
        self,
        graphs: list[Data],
        symbols: Sequence[str],
        layer_idx: int = -1,
    ) -> AttentionResult:
        """
        Average the attention matrix across many graph snapshots.

        Reveals the persistent cross-asset structure the GAT has learned
        — for the dissertation, this typically shows e.g. SPY attending
        strongly to all S&P constituents.
        """
        n = len(symbols)
        agg = np.zeros((n, n), dtype=np.float64)
        count = 0
        for g in graphs:
            result = self.compute_attention_matrix(
                g, symbols, layer_idx=layer_idx, head_strategy="mean"
            )
            agg += result.matrix
            count += 1
        if count > 0:
            agg /= count
        return AttentionResult(
            matrix=agg,
            symbols=tuple(symbols),
            layer_idx=layer_idx,
            head_strategy=f"mean_over_{count}_graphs",
        )

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    @staticmethod
    def plot_heatmap(
        result: AttentionResult,
        out_path: Path,
        title: str | None = None,
    ) -> None:
        """Render an N × N attention heatmap PNG."""
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        n = len(result.symbols)
        fig, ax = plt.subplots(figsize=(5 + 0.3 * n, 4 + 0.3 * n))
        im = ax.imshow(result.matrix, cmap="viridis", aspect="auto")
        fig.colorbar(im, ax=ax, label="attention weight")

        ax.set_xticks(range(n)); ax.set_yticks(range(n))
        ax.set_xticklabels(result.symbols, rotation=45, ha="right")
        ax.set_yticklabels(result.symbols)
        ax.set_xlabel("Key (attended TO)")
        ax.set_ylabel("Query (attending FROM)")
        ax.set_title(title or (
            f"GAT attention (layer={result.layer_idx}, "
            f"head_strategy={result.head_strategy})"
        ))

        # Annotate cells with values for small graphs.
        if n <= 10:
            for i in range(n):
                for j in range(n):
                    val = result.matrix[i, j]
                    ax.text(j, i, f"{val:.2f}",
                            ha="center", va="center",
                            color="white" if val < result.matrix.max() / 2
                            else "black",
                            fontsize=8)

        fig.tight_layout()
        fig.savefig(out_path, dpi=120)
        plt.close(fig)
