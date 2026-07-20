"""StreamSentinel  evaluation package.

Public API:
  - ModelMetrics, MetricEstimate, LatencyEstimate (data classes)
  - bootstrap_metric, auroc, pr_auc, f1_macro, precision_binary, recall_binary
  - latency_summary, throughput_events_per_second
  - shap_consistency_variance
  - per_class_report, confusion
  - DEFAULT_ABLATIONS, AblationSpec, train_and_evaluate_ablation
"""

from evaluation.ablation import (
    DEFAULT_ABLATIONS,
    AblationSpec,
    train_and_evaluate_ablation,
)
from evaluation.metrics import (
    DEFAULT_N_BOOTSTRAP,
    LatencyEstimate,
    MetricEstimate,
    ModelMetrics,
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

__all__ = [
    "ModelMetrics", "MetricEstimate", "LatencyEstimate",
    "DEFAULT_N_BOOTSTRAP",
    "bootstrap_metric", "auroc", "pr_auc", "f1_macro",
    "precision_binary", "recall_binary",
    "latency_summary", "throughput_events_per_second",
    "shap_consistency_variance",
    "per_class_report", "confusion",
    "DEFAULT_ABLATIONS", "AblationSpec", "train_and_evaluate_ablation",
]
