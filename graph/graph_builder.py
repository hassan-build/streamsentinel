"""
graph/graph_builder.py
======================
Stateless graph constructor.

Given a window of recent order book snapshots across N symbols, produces
a PyTorch Geometric `Data` object representing the cross-asset graph at
that point in time:

  - Nodes: the N tracked symbols
  - Node features: 10-d vector computed from each symbol's recent history
  - Edges: pairs of symbols whose log-return correlation exceeds threshold
  - Edge features: [correlation_value, sign]

This module is intentionally pure: same input -> same output, no internal
state. The dynamic_graph_updater wraps this for streaming use.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data


# Indices into the 10-d node feature vector. Defined as module constants
# so other modules (training, SHAP, dashboards) can reference them by name.
FEATURE_NAMES: tuple[str, ...] = (
    "log_return_1",
    "log_return_5",
    "spread_bps",
    "trade_imbalance",
    "order_cancel_rate",
    "depth_imbalance_top5",
    "depth_weighted_price_dev",
    "volatility_rolling",
    "mid_zscore",
    "cancel_rate_zscore",
)
NODE_FEATURE_DIM: int = len(FEATURE_NAMES)
EDGE_FEATURE_DIM: int = 2


@dataclass
class GraphBuilderConfig:
    """Configuration for `GraphBuilder`.

    Attributes
    ----------
    symbols : sequence of str
        Tracked tickers (fixed at construction).
    edge_threshold : float
        Minimum absolute Pearson correlation to add an edge in [0, 1].
        0.0 yields a complete graph; 1.0 yields self-loops only.
    window_size : int
        Number of snapshots PER SYMBOL used to compute features and
        correlations. 300 at 100 ms step = 30 s lookback.
    add_self_loops : bool
        Add (i, i) edges for every node. Helps GNNs preserve self info.
    correlation_min_samples : int
        Minimum overlap (in snapshots) required before computing a
        correlation. Below this, the pair is treated as zero correlation.
    eps : float
        Numerical stability epsilon used in z-scoring and division.
    """
    symbols: Sequence[str]
    edge_threshold: float = 0.3
    window_size: int = 300
    add_self_loops: bool = True
    correlation_min_samples: int = 30
    eps: float = 1e-9

    def __post_init__(self) -> None:
        if not self.symbols:
            raise ValueError("symbols must be non-empty")
        if len(set(self.symbols)) != len(self.symbols):
            raise ValueError(f"symbols contain duplicates: {self.symbols}")
        if not 0.0 <= self.edge_threshold <= 1.0:
            raise ValueError(
                f"edge_threshold must be in [0, 1], got {self.edge_threshold}"
            )
        if self.window_size < 2:
            raise ValueError(f"window_size must be >= 2, got {self.window_size}")
        if self.correlation_min_samples < 2:
            raise ValueError(
                "correlation_min_samples must be >= 2, got "
                f"{self.correlation_min_samples}"
            )


class GraphBuilder:
    """Stateless builder: order book window -> PyG `Data` object."""

    def __init__(self, config: GraphBuilderConfig) -> None:
        self.cfg = config
        # Stable mapping symbol -> node index used everywhere downstream.
        self.symbol_to_idx: dict[str, int] = {
            s: i for i, s in enumerate(config.symbols)
        }
        self.n_nodes: int = len(config.symbols)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def build(self, window_df: pd.DataFrame) -> Data:
        """
        Build a graph from a window of order book snapshots.

        Parameters
        ----------
        window_df : pd.DataFrame
            Columns must include: timestamp, symbol, mid_price,
            spread_bps, trade_imbalance, order_cancel_rate, and the
            per-level columns bidsize_l1..l5, asksize_l1..l5,
            bid_l1..l5, ask_l1..l5. Multiple timestamps × symbols.

        Returns
        -------
        torch_geometric.data.Data
            With attributes:
              - x          : Tensor [n_nodes, NODE_FEATURE_DIM]
              - edge_index : LongTensor [2, n_edges]
              - edge_attr  : Tensor [n_edges, EDGE_FEATURE_DIM]
              - num_nodes  : int
              - symbols    : list[str] (extra; preserves ordering)
        """
        self._validate_input(window_df)

        # 1. Build per-symbol time series and per-node features.
        returns_by_symbol, node_features = self._compute_node_features(
            window_df
        )

        # 2. Build correlation matrix and derive edges.
        edge_index, edge_attr = self._build_edges(returns_by_symbol)

        # 3. Assemble PyG Data.
        data = Data(
            x=torch.tensor(node_features, dtype=torch.float32),
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=self.n_nodes,
        )
        # Attach symbol list (PyG allows arbitrary extra attributes).
        data.symbols = list(self.cfg.symbols)
        return data

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _validate_input(self, df: pd.DataFrame) -> None:
        """Raise ValueError if the input doesn't satisfy our schema."""
        required = {
            "timestamp", "symbol", "mid_price", "spread_bps",
            "trade_imbalance", "order_cancel_rate",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"window_df missing required columns: {sorted(missing)}"
            )
        # Per-level columns: enforce top-5 presence as a minimum.
        for prefix in ("bid_l", "ask_l", "bidsize_l", "asksize_l"):
            for lvl in range(1, 6):
                col = f"{prefix}{lvl}"
                if col not in df.columns:
                    raise ValueError(f"window_df missing column: {col}")

    def _compute_node_features(
        self, window_df: pd.DataFrame
    ) -> tuple[dict[str, np.ndarray], np.ndarray]:
        """
        Compute per-symbol log-returns and the [n_nodes, F] node feature matrix.

        Returns
        -------
        returns_by_symbol : dict[symbol -> log_return_1 array]
            Used downstream for correlation calculation.
        node_features : np.ndarray of shape [n_nodes, NODE_FEATURE_DIM]
            Float32 features in the canonical FEATURE_NAMES order.
        """
        node_features = np.zeros(
            (self.n_nodes, NODE_FEATURE_DIM), dtype=np.float32
        )
        returns_by_symbol: dict[str, np.ndarray] = {}

        for symbol, node_idx in self.symbol_to_idx.items():
            sym_df = window_df[window_df["symbol"] == symbol]
            if sym_df.empty:
                # No data for this symbol -> features remain zero,
                # and an all-zero returns vector means no edges to/from it.
                returns_by_symbol[symbol] = np.zeros(2, dtype=np.float64)
                continue

            # Newest snapshot last — required for tail-based features.
            sym_df = sym_df.sort_values("timestamp")

            mids = sym_df["mid_price"].to_numpy(dtype=np.float64)
            cancels = sym_df["order_cancel_rate"].to_numpy(dtype=np.float64)

            # Log-returns (1-step). prepend 0 so length == len(mids).
            log_returns = np.diff(np.log(np.maximum(mids, self.cfg.eps)),
                                  prepend=np.log(max(mids[0], self.cfg.eps)))
            returns_by_symbol[symbol] = log_returns

            # ----- Per-feature computations ----------------------------
            feat = np.zeros(NODE_FEATURE_DIM, dtype=np.float64)

            # 0. log_return over 1 step (most recent)
            feat[0] = float(log_returns[-1])

            # 1. log_return over last 5 steps
            if len(mids) >= 6:
                feat[1] = float(np.log(mids[-1] / max(mids[-6], self.cfg.eps)))
            else:
                feat[1] = float(np.log(
                    mids[-1] / max(mids[0], self.cfg.eps)
                ))

            # 2. spread_bps (current)
            feat[2] = float(sym_df["spread_bps"].iloc[-1])

            # 3. trade_imbalance (current)
            feat[3] = float(sym_df["trade_imbalance"].iloc[-1])

            # 4. order_cancel_rate (current)
            feat[4] = float(cancels[-1])

            # 5. depth_imbalance over top-5 levels
            last = sym_df.iloc[-1]
            bidsize_top5 = sum(float(last[f"bidsize_l{i}"]) for i in range(1, 6))
            asksize_top5 = sum(float(last[f"asksize_l{i}"]) for i in range(1, 6))
            total = bidsize_top5 + asksize_top5
            feat[5] = (
                (bidsize_top5 - asksize_top5) / total if total > 0 else 0.0
            )

            # 6. depth-weighted price deviation from mid
            #    DWP = sum(size_i * price_i) / sum(size_i)  over L1..L5 both sides
            num = 0.0
            den = 0.0
            for i in range(1, 6):
                bp, bs = float(last[f"bid_l{i}"]), float(last[f"bidsize_l{i}"])
                ap, asz = float(last[f"ask_l{i}"]), float(last[f"asksize_l{i}"])
                num += bp * bs + ap * asz
                den += bs + asz
            dwp = num / den if den > 0 else float(mids[-1])
            mid = float(mids[-1])
            feat[6] = (dwp - mid) / max(mid, self.cfg.eps)

            # 7. rolling volatility (std of 1-step log returns over window)
            if len(log_returns) >= 2:
                feat[7] = float(np.std(log_returns, ddof=0))
            else:
                feat[7] = 0.0

            # 8. mid z-score: (current_mid - window_mean) / window_std
            mu_mid, sd_mid = float(np.mean(mids)), float(np.std(mids, ddof=0))
            feat[8] = (mids[-1] - mu_mid) / max(sd_mid, self.cfg.eps)

            # 9. cancel-rate z-score
            mu_c, sd_c = float(np.mean(cancels)), float(np.std(cancels, ddof=0))
            feat[9] = (cancels[-1] - mu_c) / max(sd_c, self.cfg.eps)

            node_features[node_idx] = feat.astype(np.float32)

        return returns_by_symbol, node_features

    def _build_edges(
        self, returns_by_symbol: dict[str, np.ndarray]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the correlation matrix across symbols and emit edges where
        |corr| >= threshold. Returns PyG-style (edge_index, edge_attr).

        Edges are undirected; we emit both (i, j) and (j, i) to satisfy
        message-passing assumptions in PyG's MessagePassing layers.
        """
        n = self.n_nodes
        edge_sources: list[int] = []
        edge_targets: list[int] = []
        edge_attrs: list[list[float]] = []

        # Cache aligned-length returns. If two symbols have different
        # observation counts, we truncate to the shorter one before corr.
        for i, sym_i in enumerate(self.cfg.symbols):
            r_i = returns_by_symbol.get(sym_i, np.zeros(2))
            for j in range(i + 1, n):
                sym_j = self.cfg.symbols[j]
                r_j = returns_by_symbol.get(sym_j, np.zeros(2))

                k = min(len(r_i), len(r_j))
                if k < self.cfg.correlation_min_samples:
                    continue   # insufficient overlap

                a = r_i[-k:]
                b = r_j[-k:]
                # Pearson correlation; guard against zero variance.
                std_a = np.std(a, ddof=0)
                std_b = np.std(b, ddof=0)
                if std_a < self.cfg.eps or std_b < self.cfg.eps:
                    continue
                corr = float(
                    np.mean((a - a.mean()) * (b - b.mean())) / (std_a * std_b)
                )
                if not np.isfinite(corr):
                    continue
                if abs(corr) < self.cfg.edge_threshold:
                    continue

                sign = 1.0 if corr > 0 else (-1.0 if corr < 0 else 0.0)
                # Undirected edge: emit both directions for GNN message passing.
                edge_sources.extend([i, j])
                edge_targets.extend([j, i])
                edge_attrs.extend([[corr, sign], [corr, sign]])

        # Optional self-loops.
        if self.cfg.add_self_loops:
            for i in range(n):
                edge_sources.append(i)
                edge_targets.append(i)
                edge_attrs.append([1.0, 1.0])

        if not edge_sources:
            # Empty graph: still need correctly-shaped tensors so PyG
            # doesn't choke when batching with non-empty graphs.
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float32)
        else:
            edge_index = torch.tensor(
                [edge_sources, edge_targets], dtype=torch.long
            )
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float32)

        return edge_index, edge_attr
