"""
tests/test_explainability.py
============================
Tests for the explainability module.

Verifies:
  - SHAP explainer produces correctly-shaped attribution arrays
  - Per-class aggregation respects label boundaries
  - Attention extraction works and produces N x N matrix
  - Aggregate attention averages correctly across graphs
  - Plots write to disk without crashing
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from graph import (
    FEATURE_NAMES,
    GraphBuilder,
    GraphBuilderConfig,
    NODE_FEATURE_DIM,
)
from models.full_pipeline import FullPipeline, FullPipelineConfig
from models.gnn_encoder import GNNEncoderConfig
from models.finbert_encoder import FinBERTEncoderConfig
from models.fusion_module import FusionModuleConfig
from models.anomaly_scorer import AnomalyScorerConfig, NUM_CLASSES

from explainability.attention_visualiser import (
    AttentionResult,
    AttentionVisualiser,
)
from explainability.shap_explainer import (
    SHAPExplainer,
    SHAPExplainerConfig,
)


SYMBOLS = ["AAPL", "MSFT", "TSLA"]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _tiny_pipeline() -> FullPipeline:
    """A small but valid FullPipeline for tests."""
    cfg = FullPipelineConfig(
        gnn=GNNEncoderConfig(input_dim=NODE_FEATURE_DIM,
                             hidden_channels=16, num_layers=2,
                             heads=2, output_dim=16),
        finbert=FinBERTEncoderConfig(
            model_name="x", allow_offline_fallback=True
        ),
        fusion=FusionModuleConfig(gnn_dim=16, text_dim=768,
                                  fusion_dim=32, num_heads=2,
                                  output_dim=16),
        scorer=AnomalyScorerConfig(input_dim=16),
    )
    return FullPipeline(cfg)


def _make_df(n_per_sym: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for sym_idx, sym in enumerate(SYMBOLS):
        price = 100.0 + sym_idx * 50
        for t in range(n_per_sym):
            price *= float(np.exp(rng.normal(0, 0.001)))
            label = int(rng.choice([0, 0, 0, 0, 1, 3, 5]))
            row = dict(timestamp=t * 100, symbol=sym, mid_price=price,
                       spread_bps=rng.uniform(1, 4),
                       trade_imbalance=rng.uniform(-0.3, 0.3),
                       order_cancel_rate=rng.uniform(15, 35),
                       label=label, anomaly_severity=0.0, injection_id="")
            for lvl in range(1, 11):
                row[f"bid_l{lvl}"] = price - lvl * 0.01
                row[f"ask_l{lvl}"] = price + lvl * 0.01
                row[f"bidsize_l{lvl}"] = float(rng.uniform(50, 500))
                row[f"asksize_l{lvl}"] = float(rng.uniform(50, 500))
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# SHAPExplainer
# ---------------------------------------------------------------------------

class TestSHAPExplainer:
    def test_requires_background_first(self):
        pipeline = _tiny_pipeline()
        gb = GraphBuilder(GraphBuilderConfig(
            symbols=SYMBOLS, window_size=30,
            correlation_min_samples=10,
        ))
        explainer = SHAPExplainer(pipeline, gb)
        with pytest.raises(RuntimeError, match="background"):
            explainer.explain(_make_df(), SYMBOLS, window_size=30)

    def test_fit_background_then_explain(self):
        pipeline = _tiny_pipeline()
        gb = GraphBuilder(GraphBuilderConfig(
            symbols=SYMBOLS, window_size=20,
            correlation_min_samples=10,
        ))
        explainer = SHAPExplainer(
            pipeline, gb,
            SHAPExplainerConfig(
                n_background_samples=5,
                n_kernel_samples=20,
                n_trials=2,
            ),
        )
        df = _make_df(n_per_sym=80)
        explainer.fit_background(df, SYMBOLS, window_size=20)
        result = explainer.explain(df, SYMBOLS, window_size=20,
                                   n_samples=3, n_trials=2)
        assert result.attributions.shape == (
            3, len(SYMBOLS), NODE_FEATURE_DIM, NUM_CLASSES
        )
        assert result.sample_labels.shape == (3, len(SYMBOLS))
        assert len(result.top_k_per_trial) == 3
        # Each trial is a top-k list.
        for sample_trials in result.top_k_per_trial:
            assert len(sample_trials) == 2
            for trial in sample_trials:
                assert all(0 <= idx < NODE_FEATURE_DIM for idx in trial)

    def test_per_class_attribution_shape(self):
        pipeline = _tiny_pipeline()
        gb = GraphBuilder(GraphBuilderConfig(
            symbols=SYMBOLS, window_size=20,
            correlation_min_samples=10,
        ))
        explainer = SHAPExplainer(
            pipeline, gb,
            SHAPExplainerConfig(
                n_background_samples=4,
                n_kernel_samples=15,
                n_trials=1,
            ),
        )
        df = _make_df(n_per_sym=80)
        explainer.fit_background(df, SYMBOLS, window_size=20)
        result = explainer.explain(df, SYMBOLS, window_size=20,
                                   n_samples=2, n_trials=1)
        report = result.per_class_attribution()
        assert "n_samples" in report.columns
        for name in FEATURE_NAMES:
            assert name in report.columns
        assert len(report) == NUM_CLASSES


# ---------------------------------------------------------------------------
# AttentionVisualiser
# ---------------------------------------------------------------------------

class TestAttentionVisualiser:
    def _graph(self) -> "torch_geometric.data.Data":
        gb = GraphBuilder(GraphBuilderConfig(
            symbols=SYMBOLS, window_size=20,
            correlation_min_samples=10,
            edge_threshold=0.0,   # ensure non-trivial edges
        ))
        return gb.build(_make_df(n_per_sym=40))

    def test_compute_returns_nxn(self):
        pipeline = _tiny_pipeline()
        av = AttentionVisualiser(pipeline)
        result = av.compute_attention_matrix(self._graph(), SYMBOLS)
        assert isinstance(result, AttentionResult)
        n = len(SYMBOLS)
        assert result.matrix.shape == (n, n)
        assert result.symbols == tuple(SYMBOLS)

    def test_aggregate_attention(self):
        pipeline = _tiny_pipeline()
        av = AttentionVisualiser(pipeline)
        graphs = [self._graph() for _ in range(3)]
        agg = av.aggregate_attention(graphs, SYMBOLS)
        n = len(SYMBOLS)
        assert agg.matrix.shape == (n, n)
        assert agg.head_strategy.startswith("mean_over_")

    def test_plot_writes_png(self, tmp_path):
        pipeline = _tiny_pipeline()
        av = AttentionVisualiser(pipeline)
        result = av.compute_attention_matrix(self._graph(), SYMBOLS)
        out = tmp_path / "subdir" / "attn.png"
        av.plot_heatmap(result, out)
        assert out.exists()
        assert out.stat().st_size > 0

    def test_head_strategy_options(self):
        pipeline = _tiny_pipeline()
        av = AttentionVisualiser(pipeline)
        g = self._graph()
        r_mean = av.compute_attention_matrix(g, SYMBOLS, head_strategy="mean")
        r_max = av.compute_attention_matrix(g, SYMBOLS, head_strategy="max")
        r_head0 = av.compute_attention_matrix(g, SYMBOLS, head_strategy="head_0")
        assert r_mean.head_strategy == "mean"
        assert r_max.head_strategy == "max"
        assert r_head0.head_strategy == "head_0"

    def test_unknown_head_strategy_rejected(self):
        pipeline = _tiny_pipeline()
        av = AttentionVisualiser(pipeline)
        with pytest.raises(ValueError, match="Unknown head_strategy"):
            av.compute_attention_matrix(self._graph(), SYMBOLS,
                                        head_strategy="bogus")
