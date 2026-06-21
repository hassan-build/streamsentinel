"""
models/train.py
===============
End-to-end training script for StreamSentinel.

Loads the synthetic training/validation Parquet files (produced by
`synthetic/anomaly_injector.py`), constructs graph windows on-the-fly,
trains the FullPipeline with AdamW + cosine LR + class-balanced
cross-entropy, logs everything to MLflow, and saves the best
checkpoint by validation AUROC.

Usage
-----
    python -m models.train --data-dir data/synthetic --epochs 30 --batch-size 32

For a quick smoke test:
    python -m models.train --data-dir data/synthetic --epochs 1 --max-train-batches 5
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset

# Make the script runnable from the repo root via `python -m models.train`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_loader import load_config  # noqa: E402
from logger import get_logger          # noqa: E402
from graph import GraphBuilder, GraphBuilderConfig  # noqa: E402
from models.anomaly_scorer import NORMAL_CLASS_IDX, NUM_CLASSES  # noqa: E402
from models.full_pipeline import FullPipeline, FullPipelineConfig  # noqa: E402

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GraphWindowDataset(Dataset):
    """
    Yields (graph, labels) pairs sampled from a Parquet file.

    Each item:
      - graph: PyG Data built from a window of `window_size` snapshots
        per symbol ending at a randomly-sampled timestamp.
      - labels: LongTensor [num_symbols] of class IDs (0..5) — the
        label of the LAST snapshot per symbol in the window.

    Design notes
    ------------
    * We sample one window per index call. Number of items = len(timestamps)
      / `stride`. Larger stride = faster epochs, less overlap.
    * We construct the graph on-the-fly because storing every graph is
      both memory-expensive and prevents stride tuning.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        symbols: list[str],
        window_size: int,
        stride: int,
        graph_builder: GraphBuilder,
    ) -> None:
        super().__init__()
        self.df = df.sort_values("timestamp").reset_index(drop=True)
        self.symbols = symbols
        self.window_size = window_size
        self.stride = stride
        self.builder = graph_builder

        # Compute unique sorted timestamps. Each timestamp corresponds
        # to (up to) one snapshot per symbol. Item i ends at
        # unique_ts[ window_size + i*stride - 1 ].
        self._unique_ts = (
            self.df["timestamp"].drop_duplicates().sort_values().to_numpy()
        )
        self._n_items = max(0, (len(self._unique_ts) - window_size) // stride)

        # Index by (timestamp, symbol) for fast row lookup of labels.
        self._labels_idx = (
            self.df.set_index(["timestamp", "symbol"])["label"].to_dict()
        )

    def __len__(self) -> int:
        return self._n_items

    def __getitem__(self, idx: int):
        end_ts_idx = self.window_size + idx * self.stride - 1
        start_ts_idx = end_ts_idx - self.window_size + 1
        ts_window = self._unique_ts[start_ts_idx: end_ts_idx + 1]
        end_ts = int(ts_window[-1])

        window_df = self.df[self.df["timestamp"].isin(ts_window)]
        graph = self.builder.build(window_df)

        # Last-snapshot label per symbol.
        labels = torch.tensor(
            [
                self._labels_idx.get((end_ts, sym), 0)
                for sym in self.symbols
            ],
            dtype=torch.long,
        )
        return graph, labels


def collate_one_graph(batch):
    """
    Collate function: a "batch" here is a list of (graph, labels) pairs.

    We treat each graph as its own forward pass rather than packing them
    into a PyG Batch object — this simplifies the loss computation and
    keeps memory tiny for our 5-node graphs.

    Returns
    -------
    list[(graph, labels)] of length batch_size.
    """
    return batch


# ---------------------------------------------------------------------------
# Training core
# ---------------------------------------------------------------------------

def compute_class_weights(labels: np.ndarray, num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights for the unbalanced synthetic data.

    Following sklearn's "balanced" convention:
        w_c = n_total / (num_classes * n_c)

    Classes absent from the training set get weight 1.0 to avoid div-by-zero.
    """
    counts = np.bincount(labels.astype(int), minlength=num_classes)
    weights = np.ones(num_classes, dtype=np.float32)
    n_total = counts.sum()
    for c in range(num_classes):
        if counts[c] > 0:
            weights[c] = n_total / (num_classes * counts[c])
    return torch.tensor(weights, dtype=torch.float32)


def train_one_epoch(
    pipeline: FullPipeline,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run one training epoch and return metrics."""
    pipeline.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    for batch_idx, items in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch_loss = 0.0
        optimiser.zero_grad()
        for graph, labels in items:
            graph = graph.to(device)
            labels = labels.to(device)
            logits = pipeline(graph, headlines=None)
            loss = criterion(logits, labels)
            batch_loss = batch_loss + loss
            with torch.no_grad():
                pred = logits.argmax(dim=-1)
                total_correct += int((pred == labels).sum().item())
                total_samples += int(labels.numel())

        # Average loss over items in the (logical) batch.
        batch_loss = batch_loss / max(1, len(items))
        batch_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(pipeline.trainable_parameters()), max_norm=1.0
        )
        optimiser.step()

        total_loss += float(batch_loss.item()) * len(items)

    n_batches = max_batches if max_batches is not None else len(loader)
    return {
        "train_loss": total_loss / max(1, n_batches),
        "train_acc": total_correct / max(1, total_samples),
    }


@torch.no_grad()
def evaluate(
    pipeline: FullPipeline,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float]:
    """Run validation/test and return metrics."""
    pipeline.eval()
    all_probs: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    total_loss = 0.0
    n_items = 0

    for batch_idx, items in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        for graph, labels in items:
            graph = graph.to(device)
            labels = labels.to(device)
            logits = pipeline(graph, headlines=None)
            loss = criterion(logits, labels)
            total_loss += float(loss.item())
            n_items += 1
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

    if not all_probs:
        return {"val_loss": float("nan")}

    probs = np.concatenate(all_probs, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    p_anomalous = 1.0 - probs[:, NORMAL_CLASS_IDX]
    is_anomaly = (labels != NORMAL_CLASS_IDX).astype(int)
    preds = probs.argmax(axis=-1)

    # Compute AUROC only when both classes are present.
    auroc = (
        float(roc_auc_score(is_anomaly, p_anomalous))
        if is_anomaly.sum() > 0 and is_anomaly.sum() < len(is_anomaly)
        else float("nan")
    )
    f1_macro = float(f1_score(labels, preds, average="macro", zero_division=0))
    precision = float(precision_score(
        is_anomaly, (p_anomalous >= 0.5).astype(int), zero_division=0
    ))
    recall = float(recall_score(
        is_anomaly, (p_anomalous >= 0.5).astype(int), zero_division=0
    ))

    return {
        "val_loss": total_loss / max(1, n_items),
        "val_auroc": auroc,
        "val_f1_macro": f1_macro,
        "val_precision": precision,
        "val_recall": recall,
        "val_n_samples": int(len(labels)),
    }


# ---------------------------------------------------------------------------
# Top-level training entrypoint
# ---------------------------------------------------------------------------

def build_pipeline_from_yaml() -> FullPipeline:
    """Build a FullPipeline from `config.yaml`."""
    cfg_all = load_config()
    m = cfg_all["models"]

    from models.gnn_encoder import GNNEncoderConfig
    from models.finbert_encoder import FinBERTEncoderConfig
    from models.fusion_module import FusionModuleConfig
    from models.anomaly_scorer import AnomalyScorerConfig

    pipeline_cfg = FullPipelineConfig(
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
            device="cpu",   # fine for inference; training doesn't use it
            cache_dir=m["finbert"].get("cache_dir"),
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
    return FullPipeline(pipeline_cfg)


def main() -> int:
    parser = argparse.ArgumentParser(prog="models.train")
    parser.add_argument("--data-dir", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of windows per backprop step")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--window-size", type=int, default=60,
                        help="Snapshots per symbol per window")
    parser.add_argument("--stride", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--symbols", type=str, default=None,
                        help="Comma-separated tickers (default: config)")
    parser.add_argument(
        "--checkpoint-dir", type=Path, default=Path("checkpoints"))
    parser.add_argument("--mlflow", action="store_true",
                        help="Log to MLflow (set MLFLOW_TRACKING_URI)")
    parser.add_argument("--max-train-batches", type=int, default=None,
                        help="Cap batches per epoch (for smoke tests)")
    parser.add_argument("--max-val-batches", type=int, default=None)
    args = parser.parse_args()

    # Reproducibility.
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Device: {device}")

    # --- Load data ---
    train_path = args.data_dir / "train.parquet"
    val_path = args.data_dir / "val.parquet"
    if not train_path.exists():
        log.error(f"Training data not found at {train_path}. "
                  f"Run synthetic/anomaly_injector.py first.")
        return 1
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path) if val_path.exists() else None
    log.info(f"Train rows: {len(train_df):,}"
             + (f"  Val rows: {len(val_df):,}" if val_df is not None else ""))

    symbols = (
        args.symbols.split(",") if args.symbols
        else sorted(train_df["symbol"].unique().tolist())
    )
    log.info(f"Symbols: {symbols}")

    # --- Datasets and loaders ---
    gb = GraphBuilder(GraphBuilderConfig(
        symbols=symbols, window_size=args.window_size,
        correlation_min_samples=min(args.window_size, 30),
    ))
    train_ds = GraphWindowDataset(
        train_df, symbols, args.window_size, args.stride, gb
    )
    val_ds = (
        GraphWindowDataset(val_df, symbols, args.window_size,
                           args.stride, gb)
        if val_df is not None else None
    )
    log.info(f"Train windows: {len(train_ds):,}"
             + (f"  Val windows: {len(val_ds):,}"
                if val_ds is not None else ""))

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_one_graph, num_workers=0,
    )
    val_loader = (
        DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                   collate_fn=collate_one_graph, num_workers=0)
        if val_ds is not None else None
    )

    # --- Model, optimiser, loss ---
    pipeline = build_pipeline_from_yaml().to(device)
    log.info(f"Pipeline parameters (trainable): "
             f"{sum(p.numel() for p in pipeline.trainable_parameters()):,}")

    cls_weights = compute_class_weights(
        train_df["label"].to_numpy(), NUM_CLASSES
    ).to(device)
    log.info(f"Class weights: {cls_weights.tolist()}")
    criterion = nn.CrossEntropyLoss(weight=cls_weights)

    optimiser = AdamW(
        pipeline.trainable_parameters(),
        lr=args.lr, weight_decay=args.weight_decay,
    )
    scheduler = CosineAnnealingLR(optimiser, T_max=max(1, args.epochs))

    # --- MLflow (optional) ---
    mlflow_run = None
    if args.mlflow:
        import mlflow
        cfg_all = load_config()
        mlflow.set_tracking_uri(cfg_all["mlflow"]["tracking_uri"])
        mlflow.set_experiment(cfg_all["mlflow"]["experiment_name"])
        mlflow_run = mlflow.start_run()
        mlflow.log_params({
            "epochs": args.epochs, "batch_size": args.batch_size,
            "lr": args.lr, "weight_decay": args.weight_decay,
            "window_size": args.window_size, "stride": args.stride,
            "seed": args.seed,
            "symbols": ",".join(symbols),
            "n_train_windows": len(train_ds),
            "n_val_windows": len(val_ds) if val_ds is not None else 0,
        })

    # --- Training loop ---
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_auroc: float = -1.0
    best_path = args.checkpoint_dir / "best_model.pt"
    history: list[dict[str, float]] = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            pipeline, train_loader, optimiser, criterion, device,
            max_batches=args.max_train_batches,
        )
        scheduler.step()

        val_metrics = (
            evaluate(pipeline, val_loader, criterion, device,
                     max_batches=args.max_val_batches)
            if val_loader is not None else {}
        )

        epoch_metrics = {
            "epoch": epoch,
            "lr": float(optimiser.param_groups[0]["lr"]),
            "elapsed_sec": time.time() - t0,
            **train_metrics,
            **val_metrics,
        }
        history.append(epoch_metrics)
        log.info(
            f"epoch {epoch:>3}/{args.epochs} | "
            f"loss={train_metrics['train_loss']:.4f} "
            f"acc={train_metrics['train_acc']:.3f} | "
            + (f"val_loss={val_metrics.get('val_loss', float('nan')):.4f} "
               f"val_auroc={val_metrics.get('val_auroc', float('nan')):.3f} "
               f"val_f1={val_metrics.get('val_f1_macro', float('nan')):.3f}"
               if val_metrics else "no validation")
            + f" | {epoch_metrics['elapsed_sec']:.1f}s"
        )

        if mlflow_run is not None:
            import mlflow
            mlflow.log_metrics(
                {k: v for k, v in epoch_metrics.items()
                 if isinstance(v, (int, float)) and not np.isnan(v)},
                step=epoch,
            )

        # Track best-by-val-AUROC checkpoint.
        if val_metrics.get("val_auroc", float("-inf")) > best_auroc:
            best_auroc = val_metrics["val_auroc"]
            torch.save({
                "epoch": epoch,
                "model_state": pipeline.state_dict(),
                "config": asdict(pipeline.cfg.gnn),
                "val_auroc": best_auroc,
            }, best_path)
            log.info(f"  saved checkpoint -> {best_path}")

    # Persist training history.
    history_path = args.checkpoint_dir / "history.json"
    history_path.write_text(json.dumps(history, indent=2))
    log.info(f"History saved -> {history_path}")
    log.info(f"Best val_auroc: {best_auroc:.4f}")

    if mlflow_run is not None:
        import mlflow
        mlflow.log_artifact(str(best_path))
        mlflow.log_artifact(str(history_path))
        mlflow.end_run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
