"""
tests/test_evaluation.py
========================
Test suite for the evaluation module.

Verifies:
  - Each baseline trains + predicts in the documented shape
  - Each metric returns values in its valid range
  - Bootstrap CIs are well-formed (point inside [low, high])
  - SHAP consistency variance behaves correctly on synthetic data
  - The ablation runner trains a small pipeline end-to-end
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evaluation.baselines import (
    RandomForestBaseline,
    RuleBasedDetector,
    build_unimodal_gnn,
)
from evaluation.metrics import (
    auroc,
    bootstrap_metric,
    confusion,
    f1_macro,
    latency_summary,
    per_class_report,
    pr_auc,
    precision_binary,
    recall_binary,
    shap_consistency_variance,
    throughput_events_per_second,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_df(n_per_sym: int = 200, seed: int = 42,
                       anomaly_rate: float = 0.2) -> pd.DataFrame:
    """Build a minimal DataFrame matching the synthetic injector schema."""
    rng = np.random.default_rng(seed)
    rows = []
    for sym_idx, sym in enumerate(["AAPL", "MSFT", "TSLA"]):
        price = 100.0 + sym_idx * 50
        for t in range(n_per_sym):
            price *= float(np.exp(rng.normal(0, 0.001)))
            label = (int(rng.choice([1, 3, 5])) if rng.uniform() < anomaly_rate
                     else 0)
            sev = float(rng.uniform()) if label > 0 else 0.0
            row = dict(timestamp=t * 100, symbol=sym, mid_price=price,
                       spread_bps=rng.uniform(1, 5),
                       trade_imbalance=rng.uniform(-0.5, 0.5),
                       order_cancel_rate=rng.uniform(10, 80),
                       label=label, anomaly_severity=sev, injection_id="x")
            for lvl in range(1, 11):
                row[f"bid_l{lvl}"] = price - lvl * 0.01
                row[f"ask_l{lvl}"] = price + lvl * 0.01
                row[f"bidsize_l{lvl}"] = float(rng.uniform(50, 500))
                row[f"asksize_l{lvl}"] = float(rng.uniform(50, 500))
            rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

class TestRuleBased:
    def test_fit_returns_self(self):
        det = RuleBasedDetector()
        assert det.fit(_make_synthetic_df()) is det

    def test_predict_proba_shape(self):
        df = _make_synthetic_df()
        det = RuleBasedDetector().fit(df)
        proba = det.predict_proba(df)
        assert proba.shape == (len(df), 2)

    def test_proba_rows_sum_to_one(self):
        df = _make_synthetic_df()
        det = RuleBasedDetector().fit(df)
        proba = det.predict_proba(df)
        sums = proba.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-9)

    def test_predict_binary_values(self):
        df = _make_synthetic_df()
        det = RuleBasedDetector().fit(df)
        preds = det.predict(df)
        assert set(np.unique(preds).tolist()).issubset({0, 1})

    def test_missing_columns_rejected(self):
        df = _make_synthetic_df().drop(columns=["spread_bps"])
        det = RuleBasedDetector().fit(df)
        with pytest.raises(ValueError, match="Missing columns"):
            det.predict_proba(df)


class TestRandomForest:
    def test_fit_and_predict(self):
        df = _make_synthetic_df()
        rf = RandomForestBaseline().fit(df)
        preds = rf.predict(df)
        assert preds.shape == (len(df),)

    def test_predict_proba_shape(self):
        df = _make_synthetic_df()
        rf = RandomForestBaseline().fit(df)
        proba = rf.predict_proba(df)
        assert proba.shape[0] == len(df)
        # Class count depends on labels present in train.
        assert proba.shape[1] >= 1

    def test_proba_rows_sum_to_one(self):
        df = _make_synthetic_df()
        rf = RandomForestBaseline().fit(df)
        proba = rf.predict_proba(df)
        sums = proba.sum(axis=1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-6)

    def test_classes_attribute(self):
        df = _make_synthetic_df()
        rf = RandomForestBaseline().fit(df)
        assert hasattr(rf, "classes_")
        assert len(rf.classes_) >= 1


class TestUnimodalGNN:
    def test_use_text_is_false(self):
        pipeline = build_unimodal_gnn()
        assert pipeline.cfg.use_text is False


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_auroc_perfect(self):
        y_true = np.array([0, 0, 1, 1])
        y_score = np.array([0.1, 0.2, 0.8, 0.9])
        est = auroc(y_true, y_score, n_bootstrap=50)
        assert est.point == 1.0

    def test_auroc_random(self):
        rng = np.random.default_rng(0)
        y_true = rng.integers(0, 2, size=500)
        y_score = rng.uniform(size=500)
        est = auroc(y_true, y_score, n_bootstrap=50)
        # Should be near 0.5 with wide CI.
        assert 0.3 < est.point < 0.7

    def test_ci_contains_point_estimate_for_pr_auc(self):
        rng = np.random.default_rng(1)
        y_true = rng.integers(0, 2, size=400)
        y_score = rng.uniform(size=400)
        est = pr_auc(y_true, y_score, n_bootstrap=100)
        # Bootstrap CI should bracket the point estimate (modulo edge cases).
        assert est.ci_low <= est.point <= est.ci_high or np.isnan(est.ci_low)

    def test_f1_macro_perfect(self):
        y_true = np.array([0, 1, 2, 3, 4, 5])
        y_pred = y_true.copy()
        est = f1_macro(y_true, y_pred, n_bootstrap=50)
        assert est.point == pytest.approx(1.0, abs=1e-9)

    def test_precision_recall_ranges(self):
        rng = np.random.default_rng(2)
        y_true = rng.integers(0, 2, size=200)
        y_pred = rng.integers(0, 2, size=200)
        p = precision_binary(y_true, y_pred, n_bootstrap=50)
        r = recall_binary(y_true, y_pred, n_bootstrap=50)
        assert 0.0 <= p.point <= 1.0
        assert 0.0 <= r.point <= 1.0

    def test_latency_percentiles_ordered(self):
        samples = np.array([1.0, 2.0, 3.0, 10.0, 100.0])
        lat = latency_summary(samples)
        assert lat.p50 <= lat.p95 <= lat.p99
        assert lat.n == 5

    def test_latency_empty(self):
        lat = latency_summary(np.array([]))
        assert lat.n == 0
        assert lat.p50 == 0.0

    def test_throughput(self):
        assert throughput_events_per_second(100, 2.0) == 50.0
        assert throughput_events_per_second(100, 0.0) == 0.0

    def test_per_class_report_shape(self):
        rng = np.random.default_rng(0)
        n_classes = 6
        y_true = rng.integers(0, n_classes, size=300)
        y_pred = rng.integers(0, n_classes, size=300)
        out = per_class_report(y_true, y_pred, n_classes)
        assert set(out.keys()) == set(range(n_classes))
        for c, metrics in out.items():
            assert {"precision", "recall", "f1", "support"} <= set(metrics)

    def test_confusion_matrix_shape(self):
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 2, 2])
        cm = confusion(y_true, y_pred, 3)
        assert cm.shape == (3, 3)

    def test_bootstrap_handles_undefined_metric(self):
        """Single-class input should not crash bootstrap_metric."""
        y_true = np.zeros(50, dtype=int)
        y_score = np.random.default_rng(0).uniform(size=50)
        est = bootstrap_metric(
            lambda yt, ys: float(np.mean(yt == 0)),
            y_true, y_score, n_bootstrap=20,
        )
        # No exception; point estimate computed.
        assert est.point == 1.0


class TestShapConsistency:
    def test_zero_variance_when_rankings_identical(self):
        rankings = [[0, 1, 2, 3, 4] for _ in range(10)]
        v = shap_consistency_variance(rankings, k=5)
        assert v == 0.0

    def test_high_variance_when_rankings_differ(self):
        rankings = [
            [0, 1, 2, 3, 4],
            [4, 3, 2, 1, 0],
            [2, 0, 4, 1, 3],
            [1, 4, 0, 3, 2],
        ]
        v = shap_consistency_variance(rankings, k=5)
        assert v > 0.5

    def test_raises_when_ranking_too_short(self):
        with pytest.raises(ValueError):
            shap_consistency_variance([[0, 1]], k=5)


# ---------------------------------------------------------------------------
# Ablation runner (smoke)
# ---------------------------------------------------------------------------

class TestAblationRunner:
    def test_runs_smoke(self, tmp_path):
        """End-to-end test that the ablation runner trains + evaluates."""
        from evaluation.ablation import train_and_evaluate_ablation, AblationSpec
        from models.full_pipeline import FullPipelineConfig
        from models.gnn_encoder import GNNEncoderConfig
        from models.finbert_encoder import FinBERTEncoderConfig
        from models.fusion_module import FusionModuleConfig
        from models.anomaly_scorer import AnomalyScorerConfig

        # Tiny but big enough dataset to yield real windows on each split.
        df = _make_synthetic_df(n_per_sym=300)
        train_df = df.iloc[: int(0.7 * len(df))]
        val_df = df.iloc[int(0.7 * len(df)): int(0.85 * len(df))]
        test_df = df.iloc[int(0.85 * len(df)):]

        base = FullPipelineConfig(
            gnn=GNNEncoderConfig(input_dim=10, hidden_channels=16,
                                 num_layers=2, heads=2, output_dim=16),
            finbert=FinBERTEncoderConfig(
                model_name="x", allow_offline_fallback=True
            ),
            fusion=FusionModuleConfig(gnn_dim=16, text_dim=768,
                                      fusion_dim=32, num_heads=2,
                                      output_dim=16),
            scorer=AnomalyScorerConfig(input_dim=16),
        )

        metrics = train_and_evaluate_ablation(
            spec=AblationSpec(name="smoke", description="test",
                              use_text=False),
            base_config=base,
            train_df=train_df, val_df=val_df, test_df=test_df,
            symbols=["AAPL", "MSFT", "TSLA"],
            epochs=1, batch_size=2, window_size=20, stride=5,
            max_train_batches=5, max_eval_batches=5,
        )
        assert metrics.model_name == "smoke"
        # Either a real number or NaN (small dataset may have one class).
        assert isinstance(metrics.auroc.point, float)
        assert metrics.n_samples > 0
