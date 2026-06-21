"""
explainability/shap_explainer.py
================================
SHAP attribution for StreamSentinel predictions.

We use **KernelSHAP** (Lundberg & Lee 2017) rather than DeepSHAP because:
  1. Our pipeline includes a graph-message-passing layer that DeepSHAP's
     tensor-rewriting approach doesn't handle cleanly.
  2. KernelSHAP is model-agnostic — it treats the model as a black box,
     which avoids any tight coupling to the internal architecture.

Attribution is computed at the **named node-feature level**: the 10
features defined in `graph/graph_builder.py: FEATURE_NAMES`. We do NOT
attribute at the edge level (would require ~25 attributions per sample
for a 5-node graph) or at the text-token level (out of scope for this
dissertation).

Consistency analysis
--------------------
A single SHAP run depends on the randomly-sampled "background" data
(the reference baseline against which counterfactuals are constructed).
Running SHAP n times with different background samples gives n
attribution vectors per sample. If the top-k features are consistent
across runs, the explanation is trustworthy; if they vary wildly, the
explanation is noisy.

We report the consistency variance via
`evaluation.metrics.shap_consistency_variance`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from graph import FEATURE_NAMES, GraphBuilder, NODE_FEATURE_DIM
from models.full_pipeline import FullPipeline


@dataclass
class SHAPExplainerConfig:
    """Configuration for `SHAPExplainer`.

    Attributes
    ----------
    n_background_samples : int
        Number of windows to draw as the SHAP background dataset.
        Larger = more accurate baseline, slower SHAP.
    n_kernel_samples : int
        KernelSHAP coalition samples per attribution. Default 100.
        Lundberg & Lee suggest 2*n_features + 2048; we use 100 because
        n_features = 10 (small) and we run many samples.
    n_trials : int
        Number of trials per sample for consistency variance. 1 disables.
    top_k : int
        Top-K features to track for the consistency metric.
    seed : int
        PRNG seed for background-sample selection and KernelSHAP itself.
    """
    n_background_samples: int = 30
    n_kernel_samples: int = 100
    n_trials: int = 3
    top_k: int = 5
    seed: int = 42


@dataclass
class SHAPResult:
    """SHAP attribution result for one or more samples.

    Attributes
    ----------
    attributions : np.ndarray
        Shape `[n_samples, n_nodes, n_features, n_classes]`. The SHAP
        value of feature `f` of node `i` for class `c` of sample `s`.
    sample_labels : np.ndarray
        Shape `[n_samples, n_nodes]`. True class labels per node.
    sample_predictions : np.ndarray
        Shape `[n_samples, n_nodes]`. Argmax predicted class per node.
    top_k_per_trial : list[list[list[int]]]
        For each sample, for each trial, the top-k attributed feature
        indices (collapsed across nodes by absolute mean). Used by
        `shap_consistency_variance`.
    feature_names : tuple[str, ...]
        Mapping from feature index to human-readable name.
    """
    attributions: np.ndarray
    sample_labels: np.ndarray
    sample_predictions: np.ndarray
    top_k_per_trial: list[list[list[int]]] = field(default_factory=list)
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def per_class_attribution(self) -> pd.DataFrame:
        """
        Aggregate mean |SHAP value| per (class, feature) pair.

        Returns
        -------
        pd.DataFrame
            Rows = classes (0..NUM_CLASSES-1), columns = feature names,
            values = mean absolute SHAP attribution across samples
            whose TRUE label is that class.
        """
        n_samples, n_nodes, n_features, n_classes = self.attributions.shape
        out = np.zeros((n_classes, n_features), dtype=np.float64)
        counts = np.zeros(n_classes, dtype=np.int64)

        for s in range(n_samples):
            for i in range(n_nodes):
                label = int(self.sample_labels[s, i])
                # |SHAP value| for the predicted class on this sample's
                # node — we average these across all (sample, node) with
                # the same true label.
                # Attribution shape per sample: [n_nodes, n_features, n_classes]
                attrib_for_class = np.abs(self.attributions[s, i, :, label])
                out[label] += attrib_for_class
                counts[label] += 1

        # Avoid div-by-zero for classes with no samples.
        counts_safe = np.maximum(counts, 1)
        out = out / counts_safe[:, None]

        df = pd.DataFrame(
            out,
            columns=list(self.feature_names),
        )
        df.index.name = "class_idx"
        df["n_samples"] = counts
        return df


class SHAPExplainer:
    """SHAP-based explainer for the FullPipeline.

    Usage
    -----
        explainer = SHAPExplainer(pipeline, graph_builder, config)
        explainer.fit_background(train_df, symbols)
        result = explainer.explain(test_df, symbols, n_samples=50)
        result.per_class_attribution()      # pd.DataFrame
        result.top_k_per_trial              # list for consistency variance
    """

    def __init__(
        self,
        pipeline: FullPipeline,
        graph_builder: GraphBuilder,
        config: SHAPExplainerConfig | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.gb = graph_builder
        self.cfg = config or SHAPExplainerConfig()
        self.pipeline.eval()
        self._background: np.ndarray | None = None
        self._rng = np.random.default_rng(self.cfg.seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def fit_background(self, train_df: pd.DataFrame,
                       symbols: Sequence[str], window_size: int) -> None:
        """
        Draw a sample of clean (label=0) windows to use as SHAP's background.

        SHAP estimates "what would the prediction be if this feature were
        absent?" by substituting the feature with values from the
        background distribution. A representative baseline is critical
        for meaningful attributions.
        """
        # Use only label=0 windows so SHAP's "no anomaly" baseline is genuine.
        clean_df = train_df[train_df["label"] == 0]
        if len(clean_df) == 0:
            # Fallback: use everything.
            clean_df = train_df

        # Sample timestamps and build node features at each.
        unique_ts = clean_df["timestamp"].drop_duplicates().to_numpy()
        if len(unique_ts) < window_size + 1:
            raise ValueError(
                f"Not enough background data ({len(unique_ts)} timestamps) "
                f"for window_size={window_size}"
            )

        chosen_end_idxs = self._rng.choice(
            len(unique_ts) - window_size,
            size=min(self.cfg.n_background_samples,
                     len(unique_ts) - window_size),
            replace=False,
        )

        bg_rows: list[np.ndarray] = []
        for end_idx in chosen_end_idxs:
            ts_window = unique_ts[end_idx: end_idx + window_size]
            window_df = clean_df[clean_df["timestamp"].isin(ts_window)]
            graph = self.gb.build(window_df)
            # Background "row" is the flat node-feature matrix [N, F]
            # flattened to a single vector of length N*F.
            bg_rows.append(graph.x.detach().cpu().numpy().flatten())

        self._background = np.stack(bg_rows, axis=0)

    def explain(
        self,
        sample_df: pd.DataFrame,
        symbols: Sequence[str],
        window_size: int,
        n_samples: int = 50,
        n_trials: int | None = None,
    ) -> SHAPResult:
        """
        Compute SHAP attributions for a sample of test windows.

        Parameters
        ----------
        sample_df : pd.DataFrame
            Test data in the synthetic-injector schema.
        symbols : sequence of str
            Tickers in the canonical order.
        window_size : int
            Snapshots per symbol per window.
        n_samples : int
            Number of test windows to explain.
        n_trials : int, optional
            Number of repeat trials per sample for consistency.
            Defaults to `self.cfg.n_trials`.

        Returns
        -------
        SHAPResult
        """
        if self._background is None:
            raise RuntimeError(
                "Call fit_background() before explain()."
            )

        n_trials = n_trials if n_trials is not None else self.cfg.n_trials
        n_symbols = len(symbols)
        n_classes = self._infer_n_classes()

        # Pick `n_samples` random end-points across the test set.
        unique_ts = sample_df["timestamp"].drop_duplicates().to_numpy()
        valid_end_idxs = np.arange(window_size, len(unique_ts))
        if len(valid_end_idxs) < n_samples:
            n_samples = len(valid_end_idxs)
        chosen = self._rng.choice(valid_end_idxs, size=n_samples,
                                  replace=False)

        attributions = np.zeros(
            (n_samples, n_symbols, NODE_FEATURE_DIM, n_classes),
            dtype=np.float32,
        )
        labels_arr = np.zeros((n_samples, n_symbols), dtype=np.int64)
        preds_arr = np.zeros((n_samples, n_symbols), dtype=np.int64)
        top_k_per_trial: list[list[list[int]]] = []

        labels_by_ts = (
            sample_df.set_index(["timestamp", "symbol"])["label"].to_dict()
        )

        for i, end_idx in enumerate(chosen):
            ts_window = unique_ts[end_idx - window_size: end_idx]
            end_ts = int(ts_window[-1])
            window_df = sample_df[sample_df["timestamp"].isin(ts_window)]
            graph = self.gb.build(window_df)

            # Record ground truth and prediction.
            for j, sym in enumerate(symbols):
                labels_arr[i, j] = int(labels_by_ts.get((end_ts, sym), 0))
            with torch.no_grad():
                logits = self.pipeline(graph, headlines=None)
                preds = logits.argmax(dim=-1).cpu().numpy()
            preds_arr[i] = preds

            # Run KernelSHAP n_trials times.
            sample_top_k: list[list[int]] = []
            for trial in range(n_trials):
                attrib = self._kernel_shap_one(
                    graph=graph,
                    trial_seed=int(self.cfg.seed + 1000 * trial + i),
                )
                # attrib shape: [n_symbols, n_features, n_classes]
                attributions[i] += attrib / n_trials  # average across trials

                # Track top-k for consistency variance — collapse nodes
                # by absolute mean to a single feature ranking per trial.
                per_feature_score = np.mean(np.abs(attrib), axis=(0, 2))
                top_k = list(np.argsort(-per_feature_score)[: self.cfg.top_k])
                sample_top_k.append([int(x) for x in top_k])
            top_k_per_trial.append(sample_top_k)

        return SHAPResult(
            attributions=attributions,
            sample_labels=labels_arr,
            sample_predictions=preds_arr,
            top_k_per_trial=top_k_per_trial,
        )

    # ------------------------------------------------------------------
    # Internal: KernelSHAP for a single graph
    # ------------------------------------------------------------------
    def _kernel_shap_one(self, graph: Data, trial_seed: int) -> np.ndarray:
        """
        Run KernelSHAP attribution on one graph snapshot.

        Returns
        -------
        np.ndarray
            Shape `[n_symbols, n_features, n_classes]`.
        """
        import shap

        n_symbols = graph.x.shape[0]
        n_features = graph.x.shape[1]
        x_flat = graph.x.detach().cpu().numpy().flatten()  # [N*F]

        # Pick a fresh subset of the background per trial for variance.
        rng = np.random.default_rng(trial_seed)
        idx = rng.choice(
            len(self._background),
            size=min(self.cfg.n_background_samples, len(self._background)),
            replace=False,
        )
        background = self._background[idx]

        # The prediction function for SHAP: takes a flattened input
        # vector or batch and returns per-class logits aggregated to
        # the graph level (sum across nodes — SHAP needs one scalar per
        # class for the explanation to make sense).
        def predict_fn(x_batch: np.ndarray) -> np.ndarray:
            # x_batch: [batch, N*F]
            batch_logits: list[np.ndarray] = []
            for row in x_batch:
                xb = torch.from_numpy(
                    row.reshape(n_symbols, n_features)
                ).float().to(graph.x.device)
                # Reuse the same edge_index / edge_attr; only x changes.
                g_modified = Data(
                    x=xb,
                    edge_index=graph.edge_index,
                    edge_attr=graph.edge_attr,
                    num_nodes=n_symbols,
                )
                with torch.no_grad():
                    logits = self.pipeline(g_modified, headlines=None)
                # Sum logits across nodes -> [n_classes]
                batch_logits.append(logits.sum(dim=0).cpu().numpy())
            return np.stack(batch_logits, axis=0)

        # KernelExplainer is the model-agnostic option. n_samples is
        # the number of coalitions SHAP evaluates.
        explainer = shap.KernelExplainer(predict_fn, background)
        shap_vals = explainer.shap_values(
            x_flat[None, :],
            nsamples=self.cfg.n_kernel_samples,
            l1_reg="num_features(10)",
            silent=True,
        )
        # shap_vals can come back in two shapes depending on SHAP version:
        # (a) Modern (>=0.42): single ndarray [batch, features, classes]
        # (b) Legacy: list[n_classes] of [batch, features]
        if isinstance(shap_vals, list):
            arr = np.stack(shap_vals, axis=-1)   # [batch, features, classes]
            arr = arr[0]                          # [features, classes]
        else:
            arr = shap_vals[0]                    # [features, classes]
        # Reshape to [n_symbols, n_features, n_classes]
        n_classes = arr.shape[-1]
        return arr.reshape(n_symbols, n_features, n_classes)

    def _infer_n_classes(self) -> int:
        """Run a tiny forward pass to discover num_classes."""
        with torch.no_grad():
            # Build a dummy graph that matches the pipeline's expected
            # dimensions. We just need the output shape.
            dummy_x = torch.zeros(1, NODE_FEATURE_DIM)
            dummy_ei = torch.tensor([[0], [0]], dtype=torch.long)
            dummy_ea = torch.zeros(1, 2)
            g = Data(x=dummy_x, edge_index=dummy_ei, edge_attr=dummy_ea,
                     num_nodes=1)
            logits = self.pipeline(g, headlines=None)
            return int(logits.shape[-1])
