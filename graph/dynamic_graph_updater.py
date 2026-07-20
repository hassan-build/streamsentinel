"""
graph/dynamic_graph_updater.py
==============================
Stateful wrapper around `GraphBuilder` for streaming use.

The updater maintains a rolling buffer of the most recent snapshots per
symbol. As new data streams in, it decides when to emit a fresh graph
(throttled by `update_interval_ms`) and produces it on demand.


"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Sequence

import pandas as pd
import torch
from torch_geometric.data import Data

from graph.graph_builder import GraphBuilder, GraphBuilderConfig


@dataclass
class DynamicGraphUpdaterConfig:
    """Configuration for `DynamicGraphUpdater`.

    Attributes
    ----------
    symbols : sequence of str
        Tracked tickers, in canonical order.
    edge_threshold : float
        Same semantics as `GraphBuilderConfig.edge_threshold`.
    window_size : int
        Snapshots per symbol kept in the rolling buffer.
    update_interval_ms : int
        Minimum gap (ms) between successive graph emissions.
        New snapshots are still ingested between emissions.
    add_self_loops : bool
        Same as GraphBuilderConfig.
    """
    symbols: Sequence[str]
    edge_threshold: float = 0.3
    window_size: int = 300
    update_interval_ms: int = 1000
    add_self_loops: bool = True


class DynamicGraphUpdater:
    """
    Rolling-window graph maintainer for streaming inference.

    Lifecycle
    ---------
        updater = DynamicGraphUpdater(config)
        for snapshot in stream:
            updater.ingest(snapshot)                    # always
            if updater.should_emit(snapshot["timestamp"]):
                graph = updater.current_graph()         # heavy compute
                send_to_gnn(graph)
    """

    def __init__(self, config: DynamicGraphUpdaterConfig) -> None:
        self.cfg = config
        builder_cfg = GraphBuilderConfig(
            symbols=list(config.symbols),
            edge_threshold=config.edge_threshold,
            window_size=config.window_size,
            add_self_loops=config.add_self_loops,
        )
        self.builder = GraphBuilder(builder_cfg)

        # Per-symbol rolling buffer. We store row dicts rather than a
        # giant DataFrame so appending is O(1) — DataFrames copy on append.
        self._buffers: dict[str, Deque[dict[str, Any]]] = {
            s: deque(maxlen=config.window_size) for s in config.symbols
        }
        self._last_emit_ts_ms: int | None = None

        # Ablation hooks: when frozen, edges are reused across updates.
        self._frozen: bool = False
        self._frozen_edge_index: torch.Tensor | None = None
        self._frozen_edge_attr: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ingest(self, snapshot: dict[str, Any]) -> None:
        """
        Append a new snapshot to the rolling buffer for its symbol.

        Parameters
        ----------
        snapshot : dict
            Must contain at least: timestamp, symbol, mid_price,
            spread_bps, trade_imbalance, order_cancel_rate, and the
            per-level bid/ask price+size columns (l1..l5).
            Snapshots for unknown symbols are silently dropped.
        """
        sym = snapshot.get("symbol")
        if sym not in self._buffers:
            return  # silently ignore foreign symbols
        self._buffers[sym].append(snapshot)

    def should_emit(self, current_ts_ms: int) -> bool:
        """
        Return True if enough time has passed since the last emission to
        warrant computing a fresh graph.

        We do NOT compute the graph here — building is expensive and
        callers may want to skip emissions during high load.
        """
        if self._last_emit_ts_ms is None:
            return True
        return (current_ts_ms - self._last_emit_ts_ms
                ) >= self.cfg.update_interval_ms

    def current_graph(self, timestamp_ms: int | None = None) -> Data:
        """
        Build and return the latest PyG graph from the rolling buffer.

        Parameters
        ----------
        timestamp_ms : int, optional
            The current timestamp to record as the last emission time.
            If None, derived from the newest snapshot in the buffer.

        Returns
        -------
        torch_geometric.data.Data
            See `GraphBuilder.build`. When frozen, `edge_index` and
            `edge_attr` are replaced by the snapshot taken at freeze time.
        """
        df = self._buffer_to_df()
        data = self.builder.build(df)

        # If frozen, override the edges with the stored snapshot.
        if self._frozen and self._frozen_edge_index is not None:
            data.edge_index = self._frozen_edge_index.clone()
            data.edge_attr = (
                self._frozen_edge_attr.clone()
                if self._frozen_edge_attr is not None
                else data.edge_attr
            )

        # Record emission time.
        if timestamp_ms is not None:
            self._last_emit_ts_ms = int(timestamp_ms)
        else:
            latest = self._latest_timestamp()
            if latest is not None:
                self._last_emit_ts_ms = latest

        return data

    def freeze(self) -> None:
        """
        Lock the current edge structure for the static-graph ablation.

        Builds one graph immediately to capture the current topology,
        then re-uses those edges for all subsequent `current_graph`
        calls. Node features continue to refresh.
        """
        df = self._buffer_to_df()
        data = self.builder.build(df)
        self._frozen_edge_index = data.edge_index.clone()
        self._frozen_edge_attr = data.edge_attr.clone()
        self._frozen = True

    def unfreeze(self) -> None:
        """Return to fully-dynamic edges."""
        self._frozen = False
        self._frozen_edge_index = None
        self._frozen_edge_attr = None

    @property
    def is_frozen(self) -> bool:
        """Whether the updater is currently in static-graph mode."""
        return self._frozen

    def buffer_size(self, symbol: str) -> int:
        """Return the number of snapshots currently buffered for `symbol`."""
        return len(self._buffers.get(symbol, deque()))

    def reset(self) -> None:
        """
        Drop all buffered state and frozen edges.

        Useful when restarting the stream after an outage or when
        beginning a new evaluation run.
        """
        for buf in self._buffers.values():
            buf.clear()
        self._last_emit_ts_ms = None
        self.unfreeze()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _buffer_to_df(self) -> pd.DataFrame:
        """Concatenate the per-symbol buffers into a single DataFrame."""
        rows: list[dict[str, Any]] = []
        for buf in self._buffers.values():
            rows.extend(buf)
        if not rows:
            # Return an empty DataFrame with the expected columns so the
            # builder's validation still passes.
            cols = [
                "timestamp", "symbol", "mid_price", "spread_bps",
                "trade_imbalance", "order_cancel_rate",
            ]
            for prefix in ("bid_l", "ask_l", "bidsize_l", "asksize_l"):
                for lvl in range(1, 6):
                    cols.append(f"{prefix}{lvl}")
            return pd.DataFrame(columns=cols)
        return pd.DataFrame(rows)

    def _latest_timestamp(self) -> int | None:
        """Find the newest timestamp across all per-symbol buffers."""
        latest: int | None = None
        for buf in self._buffers.values():
            if not buf:
                continue
            ts = int(buf[-1].get("timestamp", 0))
            if latest is None or ts > latest:
                latest = ts
        return latest
