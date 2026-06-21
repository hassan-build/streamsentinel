"""
evaluation/baselines/random_forest.py
=====================================
Random Forest baseline — a strong tabular ML benchmark with NO graph
structure and NO text input.

This baseline answers the dissertation question:
    "Does the graph structure (and the GNN) buy us anything over a
    strong off-the-shelf tabular model?"

If StreamSentinel's GNN doesn't significantly beat the Random Forest,
the entire graph architecture is overengineered. The RF therefore acts
as the **structural justification** for the GNN.

Design notes
------------
- We use the per-snapshot features that are independent of the graph:
  spread_bps, trade_imbalance, order_cancel_rate, top-5 size sums,
  log-returns over short and medium windows. These are exactly the
  features a non-graph practitioner would engineer.
- Class imbalance is handled via `class_weight='balanced'` — the same
  inverse-frequency convention used in `models/train.py`.
- The forest is deterministic (random_state=42) for reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


# Features the RF consumes. Mirrors what models/gnn_encoder uses on a
# per-symbol basis but lacks the cross-asset graph structure.
RF_FEATURES: tuple[str, ...] = (
    "spread_bps",
    "trade_imbalance",
    "order_cancel_rate",
    "bidsize_l1", "bidsize_l2", "bidsize_l3", "bidsize_l4", "bidsize_l5",
    "asksize_l1", "asksize_l2", "asksize_l3", "asksize_l4", "asksize_l5",
    "log_return_1",
    "log_return_5",
    "depth_imbalance_top5",
)


@dataclass
class RandomForestConfig:
    """Configuration for `RandomForestBaseline`.

    Defaults are taken from sklearn's standard tuning for tabular
    classification on moderately-imbalanced data.
    """
    n_estimators: int = 200
    max_depth: int | None = 20
    min_samples_split: int = 10
    min_samples_leaf: int = 5
    n_jobs: int = -1
    random_state: int = 42


class RandomForestBaseline:
    """Sklearn RandomForestClassifier wrapped to consume StreamSentinel
    Parquet rows directly.

    Public interface mirrors sklearn (fit/predict/predict_proba) so that
    `evaluation/metrics.py` can score it identically to any other model.
    """

    def __init__(self, config: RandomForestConfig | None = None) -> None:
        self.cfg = config or RandomForestConfig()
        self._clf = RandomForestClassifier(
            n_estimators=self.cfg.n_estimators,
            max_depth=self.cfg.max_depth,
            min_samples_split=self.cfg.min_samples_split,
            min_samples_leaf=self.cfg.min_samples_leaf,
            n_jobs=self.cfg.n_jobs,
            random_state=self.cfg.random_state,
            class_weight="balanced",
        )

    # ------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------
    @staticmethod
    def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
        """Compute derived features the RF needs (log-returns, depth imb.).

        Per-symbol calculation to avoid leakage across assets.
        """
        out_chunks: list[pd.DataFrame] = []
        for sym, g in df.groupby("symbol"):
            g = g.sort_values("timestamp").copy()
            mid = g["mid_price"].to_numpy(dtype=np.float64)
            log_ret_1 = np.diff(
                np.log(np.maximum(mid, 1e-9)), prepend=0.0
            )
            # Pandas rolling for the 5-step log return.
            log_mid = pd.Series(np.log(np.maximum(mid, 1e-9)))
            log_ret_5 = (log_mid - log_mid.shift(5)).fillna(0).to_numpy()

            bid5 = sum(g[f"bidsize_l{i}"] for i in range(1, 6))
            ask5 = sum(g[f"asksize_l{i}"] for i in range(1, 6))
            total = bid5 + ask5
            depth_imb = np.where(total > 0, (bid5 - ask5) / total, 0.0)

            g["log_return_1"] = log_ret_1
            g["log_return_5"] = log_ret_5
            g["depth_imbalance_top5"] = depth_imb
            out_chunks.append(g)
        return pd.concat(out_chunks).sort_index()

    def _build_X(self, df: pd.DataFrame) -> np.ndarray:
        """Engineer features and return a 2-D float matrix."""
        enriched = self._engineer_features(df)
        X = enriched[list(RF_FEATURES)].to_numpy(dtype=np.float32)
        # Replace any NaN/inf from edge cases with zeros — RF can't ingest them.
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        return X

    # ------------------------------------------------------------------
    # sklearn-style interface
    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame, y: np.ndarray | None = None
            ) -> "RandomForestBaseline":
        """Train on the Parquet rows.

        Parameters
        ----------
        df : pd.DataFrame
            Output schema from `synthetic/anomaly_injector.py`. Must
            include the `label` column unless `y` is provided.
        y : np.ndarray, optional
            Explicit labels. If omitted, uses `df['label']`.
        """
        X = self._build_X(df)
        y_arr = (y if y is not None else df["label"].to_numpy(dtype=np.int64))
        self._clf.fit(X, y_arr)
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """Return per-class probabilities."""
        X = self._build_X(df)
        return self._clf.predict_proba(X)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return per-row argmax class predictions."""
        X = self._build_X(df)
        return self._clf.predict(X).astype(np.int64)

    @property
    def classes_(self) -> np.ndarray:
        """Class labels seen during fit (for column lookups)."""
        return self._clf.classes_
