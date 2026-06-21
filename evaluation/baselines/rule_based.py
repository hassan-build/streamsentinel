"""
evaluation/baselines/rule_based.py
==================================
Rule-based threshold detector.

This baseline answers the dissertation question:
    "Does a learned ML model actually improve over a naive threshold rule?"

It is intentionally simple: no training, no parameters fit on data. We
compute a small set of hand-crafted z-scores from the order book features
and flag a snapshot as anomalous if ANY of them exceeds a multiplier.

Heuristics encoded
------------------
  1. Spread spike       : spread_bps z-score > k  (catches flash crashes
                          and liquidity shocks)
  2. Cancel-rate spike  : cancel_rate z-score > k (catches spoofing and
                          layering — their signature is mass cancellation)
  3. Imbalance extreme  : |trade_imbalance| > 0.7 (catches coordinated
                          trading via sustained one-sided pressure)
  4. Price velocity     : |1-step log-return| > k * rolling_std
                          (catches flash crashes during the drop)

A snapshot's "anomaly probability" is the count of rules triggered
divided by the total number of rules. Hard binary decisions are made
when any rule fires.

Reference: this is broadly the form used in Cao et al. (2014) and the
standard "supervised + interpretable baseline" referenced in Lewis (2024).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class RuleBasedConfig:
    """Configuration for `RuleBasedDetector`.

    Attributes
    ----------
    z_threshold : float
        Number of standard deviations above the rolling mean to flag
        a feature as extreme.
    imbalance_threshold : float
        Absolute trade_imbalance above which we flag the snapshot.
    rolling_window : int
        Snapshots to use for computing rolling mean and std.
    """
    z_threshold: float = 3.0
    imbalance_threshold: float = 0.7
    rolling_window: int = 100


class RuleBasedDetector:
    """Non-ML baseline anomaly detector.

    This detector requires no `fit()` — it is fully defined by its
    config. We still expose a `fit()` no-op so the interface matches
    sklearn-style baselines (useful in the evaluation harness).
    """

    def __init__(self, config: RuleBasedConfig | None = None) -> None:
        self.cfg = config or RuleBasedConfig()
        self._is_fitted: bool = False

    def fit(self, X: pd.DataFrame, y: np.ndarray | None = None) -> "RuleBasedDetector":
        """No-op fit. Returns self for sklearn-style chaining."""
        self._is_fitted = True
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Score each row's probability of being anomalous.

        Parameters
        ----------
        X : pd.DataFrame
            Must contain columns: timestamp, symbol, mid_price,
            spread_bps, trade_imbalance, order_cancel_rate.

        Returns
        -------
        np.ndarray of shape (n_samples, 2)
            Column 0: P(normal); Column 1: P(anomalous).
            Probabilities are derived from the fraction of rules fired.
        """
        required = {"timestamp", "symbol", "mid_price", "spread_bps",
                    "trade_imbalance", "order_cancel_rate"}
        missing = required - set(X.columns)
        if missing:
            raise ValueError(f"Missing columns: {sorted(missing)}")

        n = len(X)
        if n == 0:
            return np.zeros((0, 2), dtype=np.float64)

        # Rule counts per row in [0, 4].
        rule_hits = np.zeros(n, dtype=np.int32)

        # Apply rules per symbol — z-scores are per-asset.
        # We use pd.concat to preserve original row order.
        per_sym_results: list[pd.DataFrame] = []
        for sym, g in X.groupby("symbol"):
            sub = g.sort_values("timestamp").copy()
            w = max(2, min(self.cfg.rolling_window, len(sub)))

            # Rule 1: spread z-score
            spread = sub["spread_bps"].to_numpy(dtype=np.float64)
            spread_mean = pd.Series(spread).rolling(w, min_periods=2).mean().to_numpy()
            spread_std = pd.Series(spread).rolling(w, min_periods=2).std().to_numpy()
            spread_z = np.where(
                spread_std > 1e-9, (spread - spread_mean) / spread_std, 0.0
            )
            sub["_rule_spread"] = (spread_z > self.cfg.z_threshold).astype(int)

            # Rule 2: cancel-rate z-score
            cancel = sub["order_cancel_rate"].to_numpy(dtype=np.float64)
            cancel_mean = pd.Series(cancel).rolling(w, min_periods=2).mean().to_numpy()
            cancel_std = pd.Series(cancel).rolling(w, min_periods=2).std().to_numpy()
            cancel_z = np.where(
                cancel_std > 1e-9, (cancel - cancel_mean) / cancel_std, 0.0
            )
            sub["_rule_cancel"] = (cancel_z > self.cfg.z_threshold).astype(int)

            # Rule 3: imbalance extreme
            sub["_rule_imbalance"] = (
                sub["trade_imbalance"].abs() > self.cfg.imbalance_threshold
            ).astype(int)

            # Rule 4: price velocity (1-step log return)
            mid = sub["mid_price"].to_numpy(dtype=np.float64)
            log_ret = np.diff(np.log(np.maximum(mid, 1e-9)), prepend=0.0)
            ret_std = pd.Series(log_ret).rolling(w, min_periods=2).std().to_numpy()
            ret_z = np.where(
                ret_std > 1e-9, np.abs(log_ret) / ret_std, 0.0
            )
            sub["_rule_velocity"] = (ret_z > self.cfg.z_threshold).astype(int)

            per_sym_results.append(sub)

        # Re-merge in original order.
        merged = pd.concat(per_sym_results)
        merged = merged.reindex(X.index)
        rule_hits = (
            merged["_rule_spread"].fillna(0)
            + merged["_rule_cancel"].fillna(0)
            + merged["_rule_imbalance"].fillna(0)
            + merged["_rule_velocity"].fillna(0)
        ).to_numpy(dtype=np.float64)

        p_anomalous = rule_hits / 4.0
        proba = np.column_stack([1.0 - p_anomalous, p_anomalous])
        return proba

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return hard 0/1 anomaly decisions."""
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(np.int64)
