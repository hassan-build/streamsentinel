"""
explainability/run_explainability.py
====================================
One-command CLI that produces every explainability artefact for the
dissertation:

  - per_class_attributions.csv / .png  (SHAP, aggregated per anomaly type)
  - attention_aggregate.png            (mean GAT attention over test set)
  - attention_examples/*.png           (per-prediction heatmaps for cases)
  - consistency.json                   (SHAP variance across n_trials)
  - summary.md                         (narrative for dissertation chapter)

Usage
-----
    # Fast smoke (~5 min on CPU)
    python -m explainability.run_explainability --n-samples 50 --n-trials 3

    # Dissertation-final run (longer)
    python -m explainability.run_explainability --full
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_loader import load_config
from logger import get_logger
from graph import GraphBuilder, GraphBuilderConfig
from models.anomaly_scorer import ANOMALY_CLASSES, NORMAL_CLASS_IDX
from models.full_pipeline import FullPipeline, FullPipelineConfig
from evaluation.metrics import shap_consistency_variance

from explainability.attention_visualiser import AttentionVisualiser
from explainability.shap_explainer import (
    SHAPExplainer,
    SHAPExplainerConfig,
    SHAPResult,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline construction (mirrors evaluation/run_evaluation.py)
# ---------------------------------------------------------------------------

def _build_pipeline() -> FullPipeline:
    """Construct a FullPipeline from config.yaml."""
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
    return FullPipeline(base_cfg)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def plot_per_class_attributions(df: pd.DataFrame, out_path: Path,
                                feature_names: list[str]) -> None:
    """Grouped bar chart: one row per class, one bar per feature."""
    fig, ax = plt.subplots(figsize=(11, 5))
    # Drop the "n_samples" column for plotting (kept in CSV).
    plot_df = df[feature_names].copy()
    # Replace numeric class index with readable name.
    plot_df.index = [
        ANOMALY_CLASSES[i] if i < len(ANOMALY_CLASSES) else str(i)
        for i in plot_df.index
    ]
    plot_df.plot(kind="bar", ax=ax, width=0.8)
    ax.set_xlabel("True anomaly class")
    ax.set_ylabel("Mean |SHAP value|")
    ax.set_title("SHAP attribution by class and feature")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    plt.xticks(rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_summary_md(
    out_path: Path,
    shap_df: pd.DataFrame,
    consistency_var: float,
    n_samples: int,
    n_trials: int,
    feature_names: list[str],
) -> None:
    """Write a dissertation-ready narrative summary."""
    lines = ["# StreamSentinel — Explainability Summary\n"]
    lines.append("## Methodology\n")
    lines.append(
        f"KernelSHAP attributions computed on {n_samples} test windows, "
        f"each repeated {n_trials} times with different background "
        "samples to estimate consistency."
    )
    lines.append("")
    lines.append("## Top features per class (mean |SHAP value|)\n")
    lines.append("| Class | Top 3 features |")
    lines.append("|---|---|")
    for idx, row in shap_df.iterrows():
        cls_name = (ANOMALY_CLASSES[idx]
                    if idx < len(ANOMALY_CLASSES) else str(idx))
        top3 = row[feature_names].sort_values(ascending=False).head(3)
        triples = ", ".join(f"{n} ({v:.3f})" for n, v in top3.items())
        lines.append(f"| {cls_name} | {triples} |")
    lines.append("")
    lines.append("## Consistency\n")
    lines.append(
        f"SHAP top-k consistency variance: **{consistency_var:.4f}**"
    )
    lines.append(
        "(0 = perfectly stable across trials; higher = less stable.)\n"
    )
    lines.append("## Files in this directory\n")
    lines.append("- `per_class_attributions.csv` — full table.")
    lines.append("- `per_class_attributions.png` — bar chart.")
    lines.append("- `attention_aggregate.png` — averaged GAT attention.")
    lines.append("- `attention_examples/` — per-prediction heatmaps.")
    lines.append("- `consistency.json` — raw consistency data.")
    out_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(prog="explainability.run_explainability")
    parser.add_argument("--data-dir", type=Path,
                        default=Path("data/synthetic"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("explainability/outputs"))
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/best_model.pt"))
    parser.add_argument("--n-samples", type=int, default=50,
                        help="Number of test windows to explain")
    parser.add_argument("--n-trials", type=int, default=3,
                        help="SHAP repetitions per sample for consistency")
    parser.add_argument("--n-attention-examples", type=int, default=5)
    parser.add_argument("--window-size", type=int, default=60)
    parser.add_argument("--n-kernel-samples", type=int, default=100)
    parser.add_argument("--full", action="store_true",
                        help="Use dissertation-final defaults")
    parser.add_argument("--no-shap", action="store_true",
                        help="Skip SHAP, attention only")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.full:
        args.n_samples = max(args.n_samples, 200)
        args.n_trials = max(args.n_trials, 10)
        args.n_attention_examples = max(args.n_attention_examples, 10)
        args.n_kernel_samples = max(args.n_kernel_samples, 200)

    # --- Load data ---
    test_path = args.data_dir / "test.parquet"
    train_path = args.data_dir / "train.parquet"
    if not test_path.exists():
        log.error(f"Missing {test_path}. Generate synthetic data first.")
        return 1
    test_df = pd.read_parquet(test_path)
    train_df = pd.read_parquet(train_path) if train_path.exists() else test_df

    symbols = sorted(test_df["symbol"].unique().tolist())
    log.info(f"Symbols: {symbols}")
    log.info(f"Test rows: {len(test_df):,}")
    log.info(f"Settings: n_samples={args.n_samples}, "
             f"n_trials={args.n_trials}, "
             f"n_kernel_samples={args.n_kernel_samples}")

    # --- Load trained model ---
    pipeline = _build_pipeline()
    if args.checkpoint.exists():
        ckpt = torch.load(args.checkpoint, map_location="cpu",
                          weights_only=False)
        try:
            pipeline.load_state_dict(ckpt["model_state"], strict=False)
            log.info(f"Loaded checkpoint from {args.checkpoint}")
        except Exception as exc:
            log.warning(f"Checkpoint load failed: {exc}. Using random weights.")
    else:
        log.warning(
            f"No checkpoint at {args.checkpoint}. Running with random weights "
            "— results are illustrative only."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    feature_names: list[str] = []  # populated after SHAP runs (uses FEATURE_NAMES)

    # ----- SHAP -----
    consistency_var: float = float("nan")
    if not args.no_shap:
        log.info("Running SHAP analysis...")
        t0 = time.time()

        gb = GraphBuilder(GraphBuilderConfig(
            symbols=list(symbols),
            window_size=args.window_size,
            correlation_min_samples=min(args.window_size, 30),
        ))
        shap_cfg = SHAPExplainerConfig(
            n_background_samples=30,
            n_kernel_samples=args.n_kernel_samples,
            n_trials=args.n_trials,
            seed=args.seed,
        )
        explainer = SHAPExplainer(pipeline, gb, shap_cfg)
        explainer.fit_background(train_df, symbols, args.window_size)

        try:
            result: SHAPResult = explainer.explain(
                test_df, symbols, args.window_size,
                n_samples=args.n_samples,
                n_trials=args.n_trials,
            )
        except Exception as exc:
            log.error(f"SHAP analysis failed: {exc}")
            log.warning("Continuing with attention-only outputs.")
            result = None  # type: ignore

        if result is not None:
            feature_names = list(result.feature_names)
            shap_df = result.per_class_attribution()
            shap_df.to_csv(args.output_dir / "per_class_attributions.csv")
            plot_per_class_attributions(
                shap_df, args.output_dir / "per_class_attributions.png",
                feature_names,
            )
            log.info(
                f"SHAP attributions written -> "
                f"{args.output_dir / 'per_class_attributions.csv'}"
            )

            # Consistency variance.
            if args.n_trials > 1 and result.top_k_per_trial:
                per_sample_vars: list[float] = []
                for rankings in result.top_k_per_trial:
                    try:
                        v = shap_consistency_variance(
                            rankings, k=min(5, len(rankings[0]))
                        )
                        per_sample_vars.append(v)
                    except Exception:
                        continue
                if per_sample_vars:
                    consistency_var = float(np.mean(per_sample_vars))
                    (args.output_dir / "consistency.json").write_text(
                        json.dumps({
                            "mean_variance": consistency_var,
                            "per_sample_variance": per_sample_vars,
                            "n_samples": args.n_samples,
                            "n_trials": args.n_trials,
                        }, indent=2)
                    )
                    log.info(
                        f"Consistency variance: {consistency_var:.4f} "
                        f"(over {len(per_sample_vars)} samples)"
                    )
        elapsed = time.time() - t0
        log.info(f"SHAP analysis complete in {elapsed:.1f}s.")
    else:
        # Use default feature names if SHAP was skipped.
        from graph import FEATURE_NAMES
        feature_names = list(FEATURE_NAMES)

    # ----- Attention -----
    log.info("Computing attention heatmaps...")
    av = AttentionVisualiser(pipeline)

    # Build a small batch of graphs for aggregation + examples.
    gb = GraphBuilder(GraphBuilderConfig(
        symbols=list(symbols),
        window_size=args.window_size,
        correlation_min_samples=min(args.window_size, 30),
    ))
    unique_ts = test_df["timestamp"].drop_duplicates().to_numpy()
    rng = np.random.default_rng(args.seed)
    n_attn_graphs = min(args.n_samples, len(unique_ts) - args.window_size)
    end_idxs = rng.choice(
        np.arange(args.window_size, len(unique_ts)),
        size=n_attn_graphs, replace=False,
    )

    graphs: list = []
    labelled_examples: list[tuple] = []  # (graph, true_class, end_ts)
    labels_by_ts = (
        test_df.set_index(["timestamp", "symbol"])["label"].to_dict()
    )
    for end_idx in end_idxs:
        ts_window = unique_ts[end_idx - args.window_size: end_idx]
        end_ts = int(ts_window[-1])
        window_df = test_df[test_df["timestamp"].isin(ts_window)]
        g = gb.build(window_df)
        graphs.append(g)
        # If this window has at least one anomaly node, save it as an example.
        true_classes = [
            int(labels_by_ts.get((end_ts, s), 0)) for s in symbols
        ]
        if any(c != NORMAL_CLASS_IDX for c in true_classes):
            anomaly_idx = next(
                (i for i, c in enumerate(true_classes)
                 if c != NORMAL_CLASS_IDX), 0
            )
            labelled_examples.append(
                (g, true_classes[anomaly_idx], end_ts, symbols[anomaly_idx])
            )

    # Aggregate.
    if graphs:
        agg = av.aggregate_attention(graphs, symbols)
        av.plot_heatmap(
            agg, args.output_dir / "attention_aggregate.png",
            title=f"GAT attention averaged over {len(graphs)} test windows",
        )
        log.info(
            f"Aggregate attention -> "
            f"{args.output_dir / 'attention_aggregate.png'}"
        )

    # Per-example heatmaps.
    ex_dir = args.output_dir / "attention_examples"
    ex_dir.mkdir(exist_ok=True)
    for i, (g, cls, ts, sym) in enumerate(
        labelled_examples[: args.n_attention_examples]
    ):
        cls_name = (ANOMALY_CLASSES[cls] if cls < len(ANOMALY_CLASSES)
                    else str(cls))
        result = av.compute_attention_matrix(g, symbols)
        av.plot_heatmap(
            result,
            ex_dir / f"example_{i + 1:03d}_{cls_name}.png",
            title=f"Attention | {sym} = {cls_name} | t = {ts}",
        )
    log.info(
        f"Wrote {min(len(labelled_examples), args.n_attention_examples)} "
        "per-prediction examples"
    )

    # ----- Summary -----
    if not args.no_shap and 'shap_df' in locals():
        write_summary_md(
            args.output_dir / "summary.md",
            shap_df,
            consistency_var,
            n_samples=args.n_samples,
            n_trials=args.n_trials,
            feature_names=feature_names,
        )

    log.info("")
    log.info("=" * 70)
    log.info(f"Explainability artefacts written to: {args.output_dir}")
    log.info("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
