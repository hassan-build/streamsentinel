"""
evaluation/metrics.py
=====================
Metrics library for the dissertation evaluation.

Every metric reports a **bootstrap 95% confidence interval** in addition
to the point estimate. Single point estimates with no uncertainty are not
credible at the first-class level; the bootstrap CI is the standard
way to attach statistical rigour without making distributional
assumptions.

Functions in this file are pure: given the same inputs and seed, they
return the same outputs. All randomness flows from an explicit `seed`
argument.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# Default number of bootstrap resamples. 1000 is industry-standard for
# 95% CIs; large enough that the CI itself has low variance, small
# enough that one evaluation runs in seconds.
DEFAULT_N_BOOTSTRAP: int = 1000


@dataclass
class MetricEstimate:
    """A point estimate with its bootstrap 95% confidence interval.

    Attributes
    ----------
    point : float
        The metric computed on the full sample.
    ci_low : float
        2.5th percentile of the bootstrap distribution.
    ci_high : float
        97.5th percentile of the bootstrap distribution.
    n : int
        Number of samples behind the estimate.
    """
    point: float
    ci_low: float
    ci_high: float
    n: int

    def to_dict(self) -> dict[str, float | int]:
        """Serialise for CSV/JSON output."""
        return {
            "point": float(self.point),
            "ci_low": float(self.ci_low),
            "ci_high": float(self.ci_high),
            "n": int(self.n),
        }

    def format(self, precision: int = 3) -> str:
        """Human-readable formatting: '0.873 ± [0.851, 0.895]'."""
        return f"{self.point:.{precision}f} ± [{self.ci_low:.{precision}f}, {self.ci_high:.{precision}f}]"


# ---------------------------------------------------------------------------
# Bootstrap utility
# ---------------------------------------------------------------------------

def bootstrap_metric(
    metric_fn: Callable[[np.ndarray, np.ndarray], float],
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 42,
    ci_percentiles: tuple[float, float] = (2.5, 97.5),
) -> MetricEstimate:
    """
    Compute a metric and its bootstrap CI by resampling with replacement.

    Parameters
    ----------
    metric_fn : Callable
        Takes (y_true, y_score) and returns a single float.
    y_true : np.ndarray
        Ground-truth labels.
    y_score : np.ndarray
        Predicted scores or labels (whatever metric_fn expects).
    n_bootstrap : int
        Number of resamples. 1000 is the standard.
    seed : int
        PRNG seed for reproducibility.
    ci_percentiles : tuple[float, float]
        Lower and upper percentiles for the CI bounds.

    Returns
    -------
    MetricEstimate
        Point estimate and 95% CI.
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n = len(y_true)
    if n == 0:
        return MetricEstimate(float("nan"), float("nan"), float("nan"), 0)

    # Point estimate on full sample. Catch errors gracefully so a single
    # ill-defined metric (e.g. AUROC with one class) doesn't kill the run.
    try:
        point = float(metric_fn(y_true, y_score))
    except (ValueError, ZeroDivisionError):
        point = float("nan")

    rng = np.random.default_rng(seed)
    values: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        try:
            values.append(float(metric_fn(y_true[idx], y_score[idx])))
        except (ValueError, ZeroDivisionError):
            # Skip resamples that produced an undefined metric (e.g.
            # only one class present in the resample).
            continue

    if not values:
        return MetricEstimate(point, float("nan"), float("nan"), n)

    arr = np.asarray(values)
    return MetricEstimate(
        point=point,
        ci_low=float(np.percentile(arr, ci_percentiles[0])),
        ci_high=float(np.percentile(arr, ci_percentiles[1])),
        n=n,
    )


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def auroc(
    y_true_binary: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 42,
) -> MetricEstimate:
    """Area Under the ROC Curve with bootstrap CI."""
    return bootstrap_metric(
        lambda yt, ys: roc_auc_score(yt, ys),
        y_true_binary, y_score, n_bootstrap, seed,
    )


def pr_auc(
    y_true_binary: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 42,
) -> MetricEstimate:
    """Precision-Recall AUC (average precision). Better than AUROC under
    class imbalance, which is our case (95% normal)."""
    return bootstrap_metric(
        lambda yt, ys: average_precision_score(yt, ys),
        y_true_binary, y_score, n_bootstrap, seed,
    )


def f1_macro(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 42,
) -> MetricEstimate:
    """Macro-averaged F1 across all classes. Favours systems that catch
    minority classes (spoofing, coordinated trading)."""
    return bootstrap_metric(
        lambda yt, yp: f1_score(yt, yp, average="macro", zero_division=0),
        y_true, y_pred, n_bootstrap, seed,
    )


def precision_binary(
    y_true_binary: np.ndarray,
    y_pred_binary: np.ndarray,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 42,
) -> MetricEstimate:
    """Binary precision: of those flagged, how many were truly anomalous?"""
    return bootstrap_metric(
        lambda yt, yp: precision_score(yt, yp, zero_division=0),
        y_true_binary, y_pred_binary, n_bootstrap, seed,
    )


def recall_binary(
    y_true_binary: np.ndarray,
    y_pred_binary: np.ndarray,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = 42,
) -> MetricEstimate:
    """Binary recall: of true anomalies, how many were caught?"""
    return bootstrap_metric(
        lambda yt, yp: recall_score(yt, yp, zero_division=0),
        y_true_binary, y_pred_binary, n_bootstrap, seed,
    )


def per_class_report(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> dict[int, dict[str, float]]:
    """Per-class precision/recall/F1 without bootstrap (point estimates
    only; rare-class CIs are usually too wide to be meaningful).

    Returns NaN for every metric if input arrays are empty — sklearn
    raises on empty input, but downstream reporting code needs to
    survive that case (e.g. an ablation that produced no test windows).
    """
    out: dict[int, dict[str, float]] = {}
    n = len(y_true)
    for c in range(n_classes):
        if n == 0:
            out[c] = {"precision": float("nan"), "recall": float("nan"),
                      "f1": float("nan"), "support": 0}
            continue
        yt = (y_true == c).astype(int)
        yp = (y_pred == c).astype(int)
        out[c] = {
            "precision": float(precision_score(yt, yp, zero_division=0)),
            "recall": float(recall_score(yt, yp, zero_division=0)),
            "f1": float(f1_score(yt, yp, zero_division=0)),
            "support": int(yt.sum()),
        }
    return out


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
              ) -> np.ndarray:
    """Standard confusion matrix as [n_classes, n_classes] ndarray.

    Returns a zero matrix if inputs are empty.
    """
    if len(y_true) == 0:
        return np.zeros((n_classes, n_classes), dtype=np.int64)
    return confusion_matrix(y_true, y_pred, labels=list(range(n_classes)))


# ---------------------------------------------------------------------------
# Latency / throughput
# ---------------------------------------------------------------------------

@dataclass
class LatencyEstimate:
    """Latency percentiles in milliseconds."""
    p50: float
    p95: float
    p99: float
    mean: float
    n: int

    def to_dict(self) -> dict[str, float | int]:
        return {"p50": float(self.p50), "p95": float(self.p95),
                "p99": float(self.p99), "mean": float(self.mean),
                "n": int(self.n)}


def latency_summary(samples_ms: np.ndarray) -> LatencyEstimate:
    """Compute p50/p95/p99/mean from an array of per-event latencies (ms)."""
    samples = np.asarray(samples_ms, dtype=np.float64)
    if len(samples) == 0:
        return LatencyEstimate(0.0, 0.0, 0.0, 0.0, 0)
    return LatencyEstimate(
        p50=float(np.percentile(samples, 50)),
        p95=float(np.percentile(samples, 95)),
        p99=float(np.percentile(samples, 99)),
        mean=float(np.mean(samples)),
        n=len(samples),
    )


def throughput_events_per_second(total_events: int, elapsed_seconds: float
                                 ) -> float:
    """events/sec — used for sustained-load throughput numbers."""
    if elapsed_seconds <= 0:
        return 0.0
    return float(total_events) / float(elapsed_seconds)


# ---------------------------------------------------------------------------
# SHAP consistency variance
# ---------------------------------------------------------------------------

def shap_consistency_variance(top_k_rankings: list[list[int]],
                              k: int = 5) -> float:
    """
    Variance of the top-k SHAP feature rankings across repeated trials.

    Parameters
    ----------
    top_k_rankings : list of lists
        Each inner list is the top-k feature indices for one trial.
    k : int
        How many top features to compare.

    Returns
    -------
    float
        Mean variance across rank positions. 0.0 = perfectly stable;
        higher = more variable.

    Notes
    -----
    For each rank position 1..k we compute the variance of the indices
    that appeared there across trials, then average. This is the measure
    used in Lundberg & Lee (2017) Appendix B for explanation stability.
    """
    if not top_k_rankings:
        return float("nan")
    if any(len(r) < k for r in top_k_rankings):
        raise ValueError(
            f"all rankings must have length >= k={k}; "
            f"got min length {min(len(r) for r in top_k_rankings)}"
        )
    arr = np.asarray([r[:k] for r in top_k_rankings], dtype=np.float64)
    return float(np.mean(np.var(arr, axis=0)))


# ---------------------------------------------------------------------------
# Combined per-model metrics dict
# ---------------------------------------------------------------------------

@dataclass
class ModelMetrics:
    """All metrics for one model — what goes into the dissertation table."""
    model_name: str
    auroc: MetricEstimate
    pr_auc: MetricEstimate
    f1_macro: MetricEstimate
    precision: MetricEstimate
    recall: MetricEstimate
    latency: LatencyEstimate
    throughput: float
    per_class: dict[int, dict[str, float]] = field(default_factory=dict)
    confusion_matrix: np.ndarray | None = None
    n_samples: int = 0

    def to_dict(self) -> dict:
        """Serialise to a flat dict suitable for CSV/JSON."""
        out: dict = {
            "model_name": self.model_name,
            "n_samples": int(self.n_samples),
            "auroc": self.auroc.to_dict(),
            "pr_auc": self.pr_auc.to_dict(),
            "f1_macro": self.f1_macro.to_dict(),
            "precision": self.precision.to_dict(),
            "recall": self.recall.to_dict(),
            "latency_ms": self.latency.to_dict(),
            "throughput_per_sec": float(self.throughput),
            "per_class": self.per_class,
        }
        if self.confusion_matrix is not None:
            out["confusion_matrix"] = self.confusion_matrix.tolist()
        return out
