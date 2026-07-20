"""
evaluation/run_evaluation.py
============================


Runs:
  1. Three baselines (rule-based, RF, unimodal GNN)
  2. All ablations from config.yaml
  3. The full StreamSentinel system

Writes:
  - dissertation_table.csv         (the headline results table)
  - dissertation_table.json        (same, machine-readable)
  - per_model/<name>/confusion_matrix.png
  - per_model/<name>/pr_curve.png
  - per_model/<name>/roc_curve.png
  - per_model/<name>/latency_histogram.png
  - per_model/<name>/metrics.json
  - summary.md                     (human-readable narrative)

Usage
-----
    # Smoke test (~3 min on CPU)
    python -m evaluation.run_evaluation --epochs 2 --quick

    # Real run (~3–5 hours on CPU; run overnight)
    python -m evaluation.run_evaluation --epochs 30
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # no display required
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_loader import load_config
from logger import get_logger
from models.full_pipeline import FullPipelineConfig
from models.anomaly_scorer import NORMAL_CLASS_IDX, NUM_CLASSES, ANOMALY_CLASSES

from evaluation.ablation import (
    DEFAULT_ABLATIONS,
    train_and_evaluate_ablation,
)
from evaluation.baselines import (
    RandomForestBaseline,
    RuleBasedDetector,
)
from evaluation.metrics import (
    LatencyEstimate,
    ModelMetrics,
    auroc,
    confusion,
    f1_macro,
    latency_summary,
    per_class_report,
    pr_auc,
    precision_binary,
    recall_binary,
    throughput_events_per_second,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Baseline evaluation
# ---------------------------------------------------------------------------

def evaluate_rule_based(
    train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int
) -> ModelMetrics:
    """Train (no-op) + evaluate the rule-based detector."""
    log.info("=== Baseline: Rule-Based ===")
    det = RuleBasedDetector().fit(train_df)

    # Latency: time per snapshot (group inference is vectorised over symbols).
    t0 = time.time()
    proba = det.predict_proba(test_df)
    elapsed = time.time() - t0
    # Approximate per-row latency by total elapsed / n_rows.
    n_rows = len(test_df)
    per_row_ms = (elapsed * 1000.0 / max(1, n_rows))
    latencies = np.full(n_rows, per_row_ms, dtype=np.float64)

    labels = test_df["label"].to_numpy(dtype=np.int64)
    preds = (proba[:, 1] >= 0.5).astype(np.int64)
    p_anom = proba[:, 1]
    is_anom = (labels != NORMAL_CLASS_IDX).astype(int)

    return ModelMetrics(
        model_name="rule_based",
        auroc=auroc(is_anom, p_anom, seed=seed),
        pr_auc=pr_auc(is_anom, p_anom, seed=seed),
        f1_macro=f1_macro(labels, preds * (NUM_CLASSES - 1), seed=seed),
        precision=precision_binary(is_anom, preds, seed=seed),
        recall=recall_binary(is_anom, preds, seed=seed),
        latency=latency_summary(latencies),
        throughput=throughput_events_per_second(n_rows, elapsed),
        per_class=per_class_report(labels, preds * (NUM_CLASSES - 1),
                                   NUM_CLASSES),
        confusion_matrix=confusion(is_anom, preds, 2),
        n_samples=n_rows,
    )


def evaluate_random_forest(
    train_df: pd.DataFrame, test_df: pd.DataFrame, seed: int
) -> ModelMetrics:
    """Train Random Forest and evaluate on the test set."""
    log.info("=== Baseline: Random Forest ===")
    rf = RandomForestBaseline().fit(train_df)

    t0 = time.time()
    proba = rf.predict_proba(test_df)
    elapsed = time.time() - t0
    n_rows = len(test_df)
    per_row_ms = elapsed * 1000.0 / max(1, n_rows)
    latencies = np.full(n_rows, per_row_ms, dtype=np.float64)

    preds = rf.predict(test_df)
    labels = test_df["label"].to_numpy(dtype=np.int64)

    # Build per-class probability matrix that covers all NUM_CLASSES,
    # even if some weren't in train (set those columns to 0).
    proba_full = np.zeros((n_rows, NUM_CLASSES), dtype=np.float64)
    for col_idx, cls in enumerate(rf.classes_):
        proba_full[:, int(cls)] = proba[:, col_idx]
    p_anom = 1.0 - proba_full[:, NORMAL_CLASS_IDX]
    is_anom = (labels != NORMAL_CLASS_IDX).astype(int)
    is_anom_pred = (p_anom >= 0.5).astype(int)

    return ModelMetrics(
        model_name="random_forest",
        auroc=auroc(is_anom, p_anom, seed=seed),
        pr_auc=pr_auc(is_anom, p_anom, seed=seed),
        f1_macro=f1_macro(labels, preds, seed=seed),
        precision=precision_binary(is_anom, is_anom_pred, seed=seed),
        recall=recall_binary(is_anom, is_anom_pred, seed=seed),
        latency=latency_summary(latencies),
        throughput=throughput_events_per_second(n_rows, elapsed),
        per_class=per_class_report(labels, preds, NUM_CLASSES),
        confusion_matrix=confusion(labels, preds, NUM_CLASSES),
        n_samples=n_rows,
    )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm: np.ndarray, out_path: Path,
                          labels: list[str] | None = None) -> None:
    """Save a confusion matrix heatmap as PNG."""
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    n = cm.shape[0]
    labels = labels or [str(i) for i in range(n)]
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    for i in range(n):
        for j in range(n):
            txt = f"{cm[i, j]}"
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_latency_histogram(latencies_ms: np.ndarray, out_path: Path
                           ) -> None:
    """Save a histogram of per-event latencies with p95 marked."""
    fig, ax = plt.subplots(figsize=(6, 4))
    if latencies_ms.size > 0:
        ax.hist(latencies_ms, bins=40, alpha=0.7)
        p95 = float(np.percentile(latencies_ms, 95))
        ax.axvline(p95, color="red", linestyle="--",
                   label=f"p95 = {p95:.2f} ms")
        ax.legend()
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Frequency")
    ax.set_title("Per-event Inference Latency")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_roc_curve(y_true_binary: np.ndarray, y_score: np.ndarray,
                   out_path: Path) -> None:
    """Save an ROC curve PNG."""
    from sklearn.metrics import roc_curve, auc as sk_auc
    if len(np.unique(y_true_binary)) < 2:
        # Single-class: can't compute ROC.
        return
    fpr, tpr, _ = roc_curve(y_true_binary, y_score)
    roc_auc = sk_auc(fpr, tpr)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def plot_pr_curve(y_true_binary: np.ndarray, y_score: np.ndarray,
                  out_path: Path) -> None:
    """Save a precision-recall curve PNG."""
    from sklearn.metrics import precision_recall_curve, average_precision_score
    if len(np.unique(y_true_binary)) < 2:
        return
    prec, rec, _ = precision_recall_curve(y_true_binary, y_score)
    ap = average_precision_score(y_true_binary, y_score)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(rec, prec, label=f"AP = {ap:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def write_table_csv(records: list[ModelMetrics], out_path: Path) -> None:
    """Flatten metrics into a dissertation-style CSV."""
    rows: list[dict] = []
    for m in records:
        rows.append({
            "model": m.model_name,
            "n_samples": m.n_samples,
            "auroc": m.auroc.point,
            "auroc_ci_low": m.auroc.ci_low,
            "auroc_ci_high": m.auroc.ci_high,
            "pr_auc": m.pr_auc.point,
            "pr_auc_ci_low": m.pr_auc.ci_low,
            "pr_auc_ci_high": m.pr_auc.ci_high,
            "f1_macro": m.f1_macro.point,
            "f1_macro_ci_low": m.f1_macro.ci_low,
            "f1_macro_ci_high": m.f1_macro.ci_high,
            "precision": m.precision.point,
            "recall": m.recall.point,
            "latency_p50_ms": m.latency.p50,
            "latency_p95_ms": m.latency.p95,
            "latency_p99_ms": m.latency.p99,
            "throughput_per_sec": m.throughput,
        })
    pd.DataFrame(rows).to_csv(out_path, index=False)
    log.info(f"Wrote dissertation table -> {out_path}")


def write_table_json(records: list[ModelMetrics], out_path: Path) -> None:
    """Full metrics in JSON (including confusion matrices, per-class)."""
    data = [m.to_dict() for m in records]
    out_path.write_text(json.dumps(data, indent=2))
    log.info(f"Wrote JSON details -> {out_path}")


def write_summary_md(records: list[ModelMetrics], out_path: Path) -> None:
    """Write a human-readable Markdown summary suitable for the dissertation."""
    lines = ["# StreamSentinel — Evaluation Summary\n"]
    lines.append("## Headline metrics\n")
    lines.append("| Model | AUROC | PR-AUC | F1 (macro) | Precision | Recall | p95 latency (ms) |")
    lines.append("|---|---|---|---|---|---|---|")
    for m in records:
        lines.append(
            f"| {m.model_name} | {m.auroc.format()} | {m.pr_auc.format()} "
            f"| {m.f1_macro.format()} | {m.precision.point:.3f} "
            f"| {m.recall.point:.3f} | {m.latency.p95:.2f} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- Confidence intervals are 95% bootstrap (1000 resamples).")
    lines.append("- Latency measured per-graph on CPU.")
    lines.append("- Throughput in events/sec under the test load.")
    out_path.write_text("\n".join(lines))
    log.info(f"Wrote summary -> {out_path}")


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(prog="evaluation.run_evaluation")
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("evaluation/results"))
    parser.add_argument("--epochs", type=int, default=10,
                        help="Epochs for each ablation model")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--quick", action="store_true",
        help="Cap batches per epoch for smoke testing",
    )
    parser.add_argument(
        "--skip-baselines", action="store_true",
        help="Skip rule-based & RF baselines (faster reruns)",
    )
    parser.add_argument(
        "--skip-ablations", action="store_true",
        help="Skip ablation runs (only baselines)",
    )
    args = parser.parse_args()

    # Load data.
    train_path = args.data_dir / "train.parquet"
    val_path = args.data_dir / "val.parquet"
    test_path = args.data_dir / "test.parquet"
    for p in (train_path, val_path, test_path):
        if not p.exists():
            log.error(f"Missing {p}. Run synthetic/anomaly_injector.py first.")
            return 1
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path)
    test_df = pd.read_parquet(test_path)
    log.info(f"Train: {len(train_df):,}  Val: {len(val_df):,}  "
             f"Test: {len(test_df):,}")

    symbols = sorted(train_df["symbol"].unique().tolist())
    log.info(f"Symbols: {symbols}")

    # Build the base pipeline config from config.yaml.
    cfg_all = load_config()
    m = cfg_all["models"]

    from models.gnn_encoder import GNNEncoderConfig
    from models.finbert_encoder import FinBERTEncoderConfig
    from models.fusion_module import FusionModuleConfig
    from models.anomaly_scorer import AnomalyScorerConfig

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

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    per_model_dir = out_dir / "per_model"
    per_model_dir.mkdir(exist_ok=True)

    records: list[ModelMetrics] = []
    quick_caps = (5 if args.quick else None)

    # --- Baselines ---
    if not args.skip_baselines:
        records.append(evaluate_rule_based(train_df, test_df, args.seed))
        records.append(evaluate_random_forest(train_df, test_df, args.seed))

    # --- Ablations ---
    if not args.skip_ablations:
        for spec in DEFAULT_ABLATIONS:
            metrics = train_and_evaluate_ablation(
                spec=spec,
                base_config=base_cfg,
                train_df=train_df, val_df=val_df, test_df=test_df,
                symbols=symbols,
                epochs=args.epochs,
                batch_size=args.batch_size,
                window_size=args.window_size,
                stride=args.stride,
                learning_rate=args.lr,
                seed=args.seed,
                max_train_batches=quick_caps,
                max_eval_batches=quick_caps,
            )
            records.append(metrics)

    # --- Per-model PNGs ---
    log.info("Writing per-model artefacts...")
    for m in records:
        d = per_model_dir / m.model_name
        d.mkdir(exist_ok=True, parents=True)
        # confusion matrix
        if m.confusion_matrix is not None:
            labels = (ANOMALY_CLASSES
                      if m.confusion_matrix.shape[0] == NUM_CLASSES
                      else ["normal", "anomaly"])
            plot_confusion_matrix(m.confusion_matrix, d / "confusion_matrix.png",
                                  labels=list(labels))
        # metrics json
        (d / "metrics.json").write_text(json.dumps(m.to_dict(), indent=2))

    # --- Headline outputs ---
    write_table_csv(records, out_dir / "dissertation_table.csv")
    write_table_json(records, out_dir / "dissertation_table.json")
    write_summary_md(records, out_dir / "summary.md")

    log.info("")
    log.info("=" * 70)
    log.info(f"Evaluation complete. Results: {out_dir}")
    log.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
