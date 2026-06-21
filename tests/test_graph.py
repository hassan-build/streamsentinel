"""
tests/test_graph.py
===================
Test suite for the graph construction module.

Verifies:
  - GraphBuilder produces valid PyG Data objects
  - Node and edge features have correct shapes
  - Self-loops present when configured
  - Edge index is symmetric (undirected)
  - Threshold extremes behave correctly (1.0 -> empty/self-loops only,
    0.0 -> complete graph)
  - DynamicGraphUpdater throttles emissions at the configured interval
  - freeze() / unfreeze() preserve edge structure across updates
  - Reproducibility: same input -> identical output
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Data

from graph import (
    EDGE_FEATURE_DIM,
    FEATURE_NAMES,
    NODE_FEATURE_DIM,
    DynamicGraphUpdater,
    DynamicGraphUpdaterConfig,
    GraphBuilder,
    GraphBuilderConfig,
)


# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------

SYMBOLS = ["AAPL", "MSFT", "TSLA"]


def _make_window_df(
    n_steps: int = 100,
    symbols: list[str] | None = None,
    seed: int = 42,
    correlated: bool = True,
) -> pd.DataFrame:
    """
    Build a synthetic window DataFrame matching the schema produced by
    the synthetic anomaly injector.

    If `correlated=True`, all symbols share a common return shock plus
    independent noise — should produce strong positive edges.
    If False, symbols are fully independent.
    """
    syms = symbols or SYMBOLS
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    common_shock = rng.normal(0.0, 0.01, n_steps)

    for sym_idx, sym in enumerate(syms):
        price = 100.0 + sym_idx * 50  # different price levels
        sym_returns = (
            0.7 * common_shock + 0.3 * rng.normal(0.0, 0.01, n_steps)
            if correlated
            else rng.normal(0.0, 0.01, n_steps)
        )
        for t in range(n_steps):
            price *= float(np.exp(sym_returns[t]))
            row = {
                "timestamp": t * 100,
                "symbol": sym,
                "mid_price": float(price),
                "spread_bps": float(rng.uniform(1, 5)),
                "trade_imbalance": float(rng.uniform(-0.5, 0.5)),
                "order_cancel_rate": float(rng.uniform(10, 50)),
            }
            # Top-5 levels on both sides
            tick = 0.01
            for lvl in range(1, 6):
                row[f"bid_l{lvl}"] = price - lvl * tick
                row[f"ask_l{lvl}"] = price + lvl * tick
                row[f"bidsize_l{lvl}"] = float(rng.uniform(100, 500))
                row[f"asksize_l{lvl}"] = float(rng.uniform(100, 500))
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# GraphBuilderConfig validation
# ---------------------------------------------------------------------------

class TestGraphBuilderConfig:
    def test_empty_symbols_rejected(self):
        with pytest.raises(ValueError):
            GraphBuilderConfig(symbols=[])

    def test_duplicate_symbols_rejected(self):
        with pytest.raises(ValueError):
            GraphBuilderConfig(symbols=["AAPL", "AAPL"])

    def test_bad_threshold_rejected(self):
        with pytest.raises(ValueError):
            GraphBuilderConfig(symbols=SYMBOLS, edge_threshold=1.5)
        with pytest.raises(ValueError):
            GraphBuilderConfig(symbols=SYMBOLS, edge_threshold=-0.1)

    def test_bad_window_size_rejected(self):
        with pytest.raises(ValueError):
            GraphBuilderConfig(symbols=SYMBOLS, window_size=1)


# ---------------------------------------------------------------------------
# GraphBuilder
# ---------------------------------------------------------------------------

class TestGraphBuilder:
    def test_returns_pyg_data(self):
        cfg = GraphBuilderConfig(symbols=SYMBOLS)
        builder = GraphBuilder(cfg)
        df = _make_window_df()
        data = builder.build(df)
        assert isinstance(data, Data)

    def test_node_count_matches_symbols(self):
        cfg = GraphBuilderConfig(symbols=SYMBOLS)
        builder = GraphBuilder(cfg)
        data = builder.build(_make_window_df())
        assert data.x.shape == (len(SYMBOLS), NODE_FEATURE_DIM)
        assert data.num_nodes == len(SYMBOLS)

    def test_feature_names_match_dimension(self):
        assert len(FEATURE_NAMES) == NODE_FEATURE_DIM

    def test_edge_attr_shape_correct(self):
        cfg = GraphBuilderConfig(symbols=SYMBOLS, edge_threshold=0.0)
        builder = GraphBuilder(cfg)
        data = builder.build(_make_window_df())
        assert data.edge_attr.shape[1] == EDGE_FEATURE_DIM
        assert data.edge_index.shape[0] == 2
        assert data.edge_index.shape[1] == data.edge_attr.shape[0]

    def test_self_loops_added_by_default(self):
        cfg = GraphBuilderConfig(symbols=SYMBOLS, edge_threshold=1.0,
                                 add_self_loops=True)
        builder = GraphBuilder(cfg)
        data = builder.build(_make_window_df())
        # With threshold=1.0 nothing else qualifies. Only self-loops.
        n = len(SYMBOLS)
        # All edges must be self-loops i->i
        sources = data.edge_index[0].tolist()
        targets = data.edge_index[1].tolist()
        assert len(sources) == n
        assert all(s == t for s, t in zip(sources, targets))

    def test_no_self_loops_when_disabled(self):
        cfg = GraphBuilderConfig(symbols=SYMBOLS, edge_threshold=1.0,
                                 add_self_loops=False)
        builder = GraphBuilder(cfg)
        data = builder.build(_make_window_df())
        # With threshold=1.0 AND no self-loops, edge index is empty.
        assert data.edge_index.shape[1] == 0
        assert data.edge_attr.shape[0] == 0

    def test_threshold_zero_complete_graph(self):
        cfg = GraphBuilderConfig(symbols=SYMBOLS, edge_threshold=0.0,
                                 add_self_loops=False)
        builder = GraphBuilder(cfg)
        data = builder.build(_make_window_df(correlated=True))
        n = len(SYMBOLS)
        # Complete graph (undirected): n*(n-1) directed edges total.
        expected = n * (n - 1)
        assert data.edge_index.shape[1] == expected

    def test_edge_index_symmetric(self):
        """For every (i, j) edge, (j, i) must also exist."""
        cfg = GraphBuilderConfig(symbols=SYMBOLS, edge_threshold=0.0,
                                 add_self_loops=False)
        builder = GraphBuilder(cfg)
        data = builder.build(_make_window_df(correlated=True))
        pairs = {
            (int(s), int(t))
            for s, t in zip(data.edge_index[0], data.edge_index[1])
        }
        for s, t in pairs:
            assert (t, s) in pairs, f"missing reverse edge ({t}, {s})"

    def test_correlated_data_produces_edges(self):
        cfg = GraphBuilderConfig(symbols=SYMBOLS, edge_threshold=0.3,
                                 add_self_loops=False)
        builder = GraphBuilder(cfg)
        data = builder.build(_make_window_df(correlated=True))
        # Strongly correlated symbols should produce at least one edge pair.
        assert data.edge_index.shape[1] >= 2

    def test_independent_data_few_edges(self):
        cfg = GraphBuilderConfig(symbols=SYMBOLS, edge_threshold=0.5,
                                 add_self_loops=False)
        builder = GraphBuilder(cfg)
        # Independent symbols + high threshold -> few/no edges.
        data = builder.build(_make_window_df(correlated=False, seed=1))
        # Allow at most one chance correlation pair (2 directed edges).
        assert data.edge_index.shape[1] <= 2

    def test_features_are_finite(self):
        cfg = GraphBuilderConfig(symbols=SYMBOLS)
        builder = GraphBuilder(cfg)
        data = builder.build(_make_window_df())
        assert torch.isfinite(data.x).all(), "node features contain non-finite values"
        assert torch.isfinite(data.edge_attr).all(), "edge features contain non-finite values"

    def test_reproducible(self):
        """Same input -> identical output."""
        cfg = GraphBuilderConfig(symbols=SYMBOLS)
        builder = GraphBuilder(cfg)
        df = _make_window_df(seed=42)
        data1 = builder.build(df)
        data2 = builder.build(df)
        torch.testing.assert_close(data1.x, data2.x)
        assert torch.equal(data1.edge_index, data2.edge_index)
        torch.testing.assert_close(data1.edge_attr, data2.edge_attr)

    def test_missing_symbol_handled(self):
        """If one symbol has no data, node features are zero, no crash."""
        cfg = GraphBuilderConfig(symbols=SYMBOLS)
        builder = GraphBuilder(cfg)
        df = _make_window_df(symbols=["AAPL", "MSFT"])  # missing TSLA
        data = builder.build(df)
        tsla_idx = SYMBOLS.index("TSLA")
        assert torch.allclose(data.x[tsla_idx], torch.zeros(NODE_FEATURE_DIM))

    def test_input_validation_missing_column(self):
        cfg = GraphBuilderConfig(symbols=SYMBOLS)
        builder = GraphBuilder(cfg)
        df = _make_window_df().drop(columns=["spread_bps"])
        with pytest.raises(ValueError, match="missing"):
            builder.build(df)


# ---------------------------------------------------------------------------
# DynamicGraphUpdater
# ---------------------------------------------------------------------------

class TestDynamicGraphUpdater:
    def _ingest_window(
        self, updater: DynamicGraphUpdater, n_steps: int = 50, seed: int = 42
    ) -> None:
        """Pump a synthetic window of snapshots through the updater."""
        df = _make_window_df(n_steps=n_steps, seed=seed)
        for _, row in df.iterrows():
            updater.ingest(row.to_dict())

    def test_emits_first_time(self):
        cfg = DynamicGraphUpdaterConfig(symbols=SYMBOLS, window_size=50)
        updater = DynamicGraphUpdater(cfg)
        assert updater.should_emit(0) is True

    def test_throttles_within_interval(self):
        cfg = DynamicGraphUpdaterConfig(
            symbols=SYMBOLS, window_size=50, update_interval_ms=1000
        )
        updater = DynamicGraphUpdater(cfg)
        self._ingest_window(updater, n_steps=50)
        # Compute and "emit" at timestamp 5000
        updater.current_graph(timestamp_ms=5000)
        # 500 ms later -> should NOT emit
        assert updater.should_emit(5500) is False
        # 1000 ms later -> should emit
        assert updater.should_emit(6000) is True

    def test_buffer_size_caps_at_window(self):
        cfg = DynamicGraphUpdaterConfig(symbols=SYMBOLS, window_size=30)
        updater = DynamicGraphUpdater(cfg)
        self._ingest_window(updater, n_steps=100)
        for sym in SYMBOLS:
            assert updater.buffer_size(sym) == 30

    def test_unknown_symbol_silently_ignored(self):
        cfg = DynamicGraphUpdaterConfig(symbols=SYMBOLS, window_size=50)
        updater = DynamicGraphUpdater(cfg)
        snap = {
            "timestamp": 0, "symbol": "GOOGL", "mid_price": 100.0,
            "spread_bps": 1.0, "trade_imbalance": 0.0,
            "order_cancel_rate": 10.0,
        }
        for lvl in range(1, 6):
            snap[f"bid_l{lvl}"] = 100.0
            snap[f"ask_l{lvl}"] = 100.01
            snap[f"bidsize_l{lvl}"] = 100.0
            snap[f"asksize_l{lvl}"] = 100.0
        updater.ingest(snap)
        assert updater.buffer_size("AAPL") == 0  # nothing for known symbol

    def test_freeze_preserves_edge_index(self):
        cfg = DynamicGraphUpdaterConfig(
            symbols=SYMBOLS, window_size=50, edge_threshold=0.3
        )
        updater = DynamicGraphUpdater(cfg)
        self._ingest_window(updater, n_steps=50, seed=1)
        updater.freeze()
        graph_a = updater.current_graph(timestamp_ms=5000)
        edges_a = graph_a.edge_index.clone()

        # Ingest more (different) data and rebuild.
        self._ingest_window(updater, n_steps=50, seed=2)
        graph_b = updater.current_graph(timestamp_ms=10000)
        edges_b = graph_b.edge_index.clone()

        assert torch.equal(edges_a, edges_b), (
            "frozen edge_index changed across updates"
        )

    def test_unfreeze_allows_edges_to_change(self):
        cfg = DynamicGraphUpdaterConfig(
            symbols=SYMBOLS, window_size=50, edge_threshold=0.3
        )
        updater = DynamicGraphUpdater(cfg)
        self._ingest_window(updater, n_steps=50, seed=1)
        updater.freeze()
        edges_frozen = updater.current_graph(timestamp_ms=5000).edge_index.clone()
        updater.unfreeze()
        assert updater.is_frozen is False
        # After unfreezing, calling current_graph returns a freshly
        # computed graph that does NOT carry over the frozen edges.
        # Whether the topology matches by coincidence is possible but
        # the internal frozen state must be cleared:
        assert updater._frozen_edge_index is None

    def test_features_update_when_frozen(self):
        """Frozen mode locks edges but node features should still refresh."""
        cfg = DynamicGraphUpdaterConfig(
            symbols=SYMBOLS, window_size=50, edge_threshold=0.3
        )
        updater = DynamicGraphUpdater(cfg)
        self._ingest_window(updater, n_steps=50, seed=1)
        updater.freeze()
        x_a = updater.current_graph(timestamp_ms=5000).x.clone()

        self._ingest_window(updater, n_steps=50, seed=2)
        x_b = updater.current_graph(timestamp_ms=10000).x.clone()

        # Node features should be different (new data was ingested).
        assert not torch.allclose(x_a, x_b), (
            "node features did not update under frozen mode"
        )

    def test_reset_clears_state(self):
        cfg = DynamicGraphUpdaterConfig(symbols=SYMBOLS, window_size=50)
        updater = DynamicGraphUpdater(cfg)
        self._ingest_window(updater, n_steps=50)
        updater.current_graph(timestamp_ms=5000)
        updater.freeze()

        updater.reset()
        assert all(updater.buffer_size(s) == 0 for s in SYMBOLS)
        assert updater.is_frozen is False
        assert updater.should_emit(0) is True  # no prior emission
