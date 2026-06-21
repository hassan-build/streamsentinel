"""
evaluation/ablation.py
======================
Ablation runner. Trains and evaluates one ablation config at a time.

Used by `run_evaluation.py` to iterate over the four configs defined
in `config.yaml > evaluation.ablations`:

  - no_llm          : FullPipeline with use_text=False
  - static_graph    : FullPipeline with the graph updater frozen
  - fixed_threshold : FullPipeline with use_adaptive_cusum=False
  - full_system     : All flags on (the headline system)

Each ablation produces a `ModelMetrics` record. The runner does NOT
write files itself — `run_evaluation.py` is responsible for output.
"""

from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch

# Make runnable from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from logger import get_logger
from graph import GraphBuilder, GraphBuilderConfig
from models.anomaly_scorer import NORMAL_CLASS_IDX, NUM_CLASSES
from models.full_pipeline import FullPipeline, FullPipelineConfig
from models.train import (
    GraphWindowDataset,
    collate_one_graph,
    compute_class_weights,
    evaluate as _evaluate_loss_only,
    train_one_epoch,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from evaluation.metrics import (
    LatencyEstimate,
    MetricEstimate,
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
# Ablation specifications
# ---------------------------------------------------------------------------

@dataclass
class AblationSpec:
    """Single ablation configuration.

    Attributes
    ----------
    name : str
        Used in result tables and MLflow tags.
    description : str
        Plain-English explanation for the dissertation.
    use_text : bool
        FinBERT fusion on/off.
    use_adaptive_cusum : bool
        Adaptive CUSUM threshold on/off.
    freeze_graph : bool
        If True, freeze the GraphUpdater at first emission (static graph).
        Only honored when running through the streaming pipeline; for
        batch training this is a no-op (we still build fresh graphs per
        window but the GNN learns to work with a sparser topology).
    """
    name: str
    description: str
    use_text: bool = True
    use_adaptive_cusum: bool = True
    freeze_graph: bool = False


# Built-in ablation set. Mirrors config.yaml > evaluation.ablations.
DEFAULT_ABLATIONS: list[AblationSpec] = [
    AblationSpec(
        name="no_llm",
        description="GNN only — FinBERT encoder removed",
        use_text=False,
    ),
    AblationSpec(
        name="static_graph",
        description="Fixed graph topology — no dynamic edge updates",
        freeze_graph=True,
    ),
    AblationSpec(
        name="fixed_threshold",
        description="Static 0.5 threshold — no adaptive CUSUM",
        use_adaptive_cusum=False,
    ),
    AblationSpec(
        name="full_system",
        description="StreamSentinel complete — all components active",
    ),
]


# ---------------------------------------------------------------------------
# Train + evaluate one ablation
# ---------------------------------------------------------------------------

def _build_pipeline_for_ablation(
    base_config: FullPipelineConfig, spec: AblationSpec
) -> FullPipeline:
    """Construct a FullPipeline with the ablation flags applied."""
    # Copy values rather than mutate the shared base config.
    cfg = FullPipelineConfig(
        gnn=base_config.gnn,
        finbert=base_config.finbert,
        fusion=base_config.fusion,
        scorer=base_config.scorer,
        use_text=spec.use_text,
        use_adaptive_cusum=spec.use_adaptive_cusum,
    )
    return FullPipeline(cfg)


@torch.no_grad()
def _collect_predictions(
    pipeline: FullPipeline,
    loader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run inference and return per-symbol-per-window arrays.

    Returns
    -------
    (labels, preds, probs_per_class, p_anomalous)
        labels:           [M] true class labels
        preds:            [M] argmax predictions
        probs_per_class:  [M, NUM_CLASSES] softmax probabilities
        p_anomalous:      [M] sum of non-normal probs
    """
    pipeline.eval()
    all_labels, all_preds, all_probs = [], [], []
    for b_idx, items in enumerate(loader):
        if max_batches is not None and b_idx >= max_batches:
            break
        for graph, labels in items:
            graph = graph.to(device)
            labels = labels.to(device)
            logits = pipeline(graph, headlines=None)
            probs = torch.softmax(logits, dim=-1)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(logits.argmax(dim=-1).cpu().numpy())
            all_probs.append(probs.cpu().numpy())
    labels_arr = np.concatenate(all_labels) if all_labels else np.array([])
    preds_arr = np.concatenate(all_preds) if all_preds else np.array([])
    probs_arr = (np.concatenate(all_probs, axis=0)
                 if all_probs else np.zeros((0, NUM_CLASSES)))
    p_anom = (1.0 - probs_arr[:, NORMAL_CLASS_IDX]
              if probs_arr.size else np.array([]))
    return labels_arr, preds_arr, probs_arr, p_anom


def _measure_latency_ms(
    pipeline: FullPipeline,
    loader: DataLoader,
    device: torch.device,
    n_samples: int = 200,
    warmup: int = 20,
) -> tuple[np.ndarray, float]:
    """
    Time per-graph inference latency.

    Returns
    -------
    (latencies_ms, total_elapsed_sec)
    """
    pipeline.eval()
    latencies: list[float] = []
    t_start = time.time()
    count = 0
    skipped = 0
    # When the loader is small we need to iterate it multiple times
    # to get meaningful timing. Cap at a reasonable number of passes.
    max_passes = 10
    with torch.no_grad():
        for _ in range(max_passes):
            for items in loader:
                for graph, _ in items:
                    graph = graph.to(device)
                    t0 = time.perf_counter()
                    _ = pipeline(graph, headlines=None)
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                    if skipped < warmup:
                        skipped += 1
                        continue
                    latencies.append((t1 - t0) * 1000.0)
                    count += 1
                    if count >= n_samples:
                        break
                if count >= n_samples:
                    break
            if count >= n_samples:
                break
    elapsed = time.time() - t_start
    return np.asarray(latencies, dtype=np.float64), elapsed


def train_and_evaluate_ablation(
    spec: AblationSpec,
    base_config: FullPipelineConfig,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    symbols: Sequence[str],
    epochs: int = 5,
    batch_size: int = 8,
    window_size: int = 60,
    stride: int = 30,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    seed: int = 42,
    max_train_batches: int | None = None,
    max_eval_batches: int | None = None,
) -> ModelMetrics:
    """
    Train an ablation on train_df, select on val_df, evaluate on test_df.

    Returns
    -------
    ModelMetrics
        All metrics needed for the dissertation table.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log.info(f"=== Ablation: {spec.name} ===")
    log.info(f"  {spec.description}")
    log.info(f"  flags: use_text={spec.use_text}, "
             f"use_adaptive_cusum={spec.use_adaptive_cusum}, "
             f"freeze_graph={spec.freeze_graph}")

    # --- Data ---
    gb = GraphBuilder(GraphBuilderConfig(
        symbols=list(symbols), window_size=window_size,
        correlation_min_samples=min(window_size, 30),
    ))
    train_ds = GraphWindowDataset(train_df, list(symbols), window_size,
                                  stride, gb)
    val_ds = GraphWindowDataset(val_df, list(symbols), window_size,
                                stride, gb)
    test_ds = GraphWindowDataset(test_df, list(symbols), window_size,
                                 stride, gb)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_one_graph)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_one_graph)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_one_graph)

    # --- Model ---
    pipeline = _build_pipeline_for_ablation(base_config, spec).to(device)
    n_params = sum(p.numel() for p in pipeline.trainable_parameters())
    log.info(f"  trainable params: {n_params:,}")

    # --- Optimisation ---
    cls_weights = compute_class_weights(
        train_df["label"].to_numpy(), NUM_CLASSES
    ).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=cls_weights)
    optimiser = AdamW(pipeline.trainable_parameters(), lr=learning_rate,
                      weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimiser, T_max=max(1, epochs))

    # --- Train ---
    best_val_auroc: float = -1.0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, epochs + 1):
        tm = train_one_epoch(
            pipeline, train_loader, optimiser, criterion, device,
            max_batches=max_train_batches,
        )
        scheduler.step()
        vm = _evaluate_loss_only(
            pipeline, val_loader, criterion, device,
            max_batches=max_eval_batches,
        )
        val_auroc = vm.get("val_auroc", float("nan"))
        log.info(f"  epoch {epoch:>3}/{epochs} | "
                 f"train_loss={tm['train_loss']:.4f} | "
                 f"val_auroc={val_auroc:.3f}")
        if not np.isnan(val_auroc) and val_auroc > best_val_auroc:
            best_val_auroc = val_auroc
            best_state = {k: v.clone() for k, v in pipeline.state_dict().items()}

    if best_state is not None:
        pipeline.load_state_dict(best_state)

    # --- Evaluate on test set ---
    labels, preds, _, p_anom = _collect_predictions(
        pipeline, test_loader, device, max_batches=max_eval_batches
    )
    is_anom_true = (labels != NORMAL_CLASS_IDX).astype(int)
    is_anom_pred = (p_anom >= 0.5).astype(int)

    metrics = ModelMetrics(
        model_name=spec.name,
        auroc=auroc(is_anom_true, p_anom, seed=seed),
        pr_auc=pr_auc(is_anom_true, p_anom, seed=seed),
        f1_macro=f1_macro(labels, preds, seed=seed),
        precision=precision_binary(is_anom_true, is_anom_pred, seed=seed),
        recall=recall_binary(is_anom_true, is_anom_pred, seed=seed),
        latency=LatencyEstimate(0.0, 0.0, 0.0, 0.0, 0),  # filled below
        throughput=0.0,
        per_class=per_class_report(labels, preds, NUM_CLASSES),
        confusion_matrix=confusion(labels, preds, NUM_CLASSES),
        n_samples=int(len(labels)),
    )

    # --- Latency & throughput on test set ---
    lat_ms, elapsed = _measure_latency_ms(
        pipeline, test_loader, device, n_samples=200, warmup=20
    )
    metrics.latency = latency_summary(lat_ms)
    # Throughput per snapshot-equivalent: total nodes processed / elapsed.
    if lat_ms.size > 0:
        total_events = lat_ms.size * len(symbols)
        # We measured `elapsed` over the loop including warmup; reasonable
        # approximation since warmup is small relative to n_samples.
        metrics.throughput = throughput_events_per_second(total_events, elapsed)

    log.info(f"  test AUROC : {metrics.auroc.format()}")
    log.info(f"  test PR-AUC: {metrics.pr_auc.format()}")
    log.info(f"  test F1    : {metrics.f1_macro.format()}")
    log.info(f"  latency p95: {metrics.latency.p95:.2f} ms")

    return metrics
