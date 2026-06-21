"""
models/anomaly_scorer.py
========================
Final scoring head + adaptive CUSUM thresholding.

This module turns the fused per-node embedding into:
  1. **Class logits** — 6-way softmax over {normal, spoofing, layering,
     flash_crash, coordinated, liquidity_shock} for classification loss.
  2. **Binary anomaly decision** — derived from the logits via either a
     fixed threshold or the adaptive CUSUM detector.

Why CUSUM?
----------
Cumulative Sum (Page 1954) is a statistical change-point detector that
tracks the running deviation of a signal from its mean and triggers an
alarm when the deviation crosses a decision bound. Unlike a fixed
threshold, CUSUM **adapts to regime shifts**: if the baseline anomaly
score drifts upward (e.g. volatility regime change), CUSUM's running
mean drifts with it, and only *unusual* spikes trigger alarms.

Formally we run the one-sided CUSUM on the per-node "anomalous prob":

    p_t = sum over class c != normal of softmax(logits)[c]

    S_t = max(0, S_{t-1} + p_t - mu_t - k)
    alarm if S_t >= h

where `mu_t` is an exponentially-weighted moving average of `p_t` and
`k`, `h` are the allowance and decision threshold respectively. See
Lai (1995) for theoretical properties.

Setting the `use_adaptive_cusum` flag to False bypasses CUSUM and uses
a fixed `threshold` on `p_t`. This is the `fixed_threshold` ablation.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


# The 6 anomaly classes. Order MUST match `synthetic.injectors` labels
# AND `config.yaml > models.anomaly_scorer.anomaly_classes`.
ANOMALY_CLASSES: tuple[str, ...] = (
    "normal",
    "spoofing",
    "layering",
    "flash_crash",
    "coordinated_trading",
    "liquidity_shock",
)
NUM_CLASSES: int = len(ANOMALY_CLASSES)
NORMAL_CLASS_IDX: int = 0


@dataclass
class AnomalyScorerConfig:
    """Configuration for `AnomalyScorer`.

    Attributes
    ----------
    input_dim : int
        Dimensionality of fused embedding from `FusionModule`. Default 128.
    hidden_dim : int
        MLP hidden width. Default 64.
    num_classes : int
        Output classes. Default 6 (= NUM_CLASSES).
    dropout : float
        Dropout in the MLP. Default 0.3.
    use_adaptive_cusum : bool
        If True, use adaptive CUSUM. If False, use fixed threshold.
    cusum_k : float
        CUSUM allowance parameter. Higher = less sensitive.
    cusum_h : float
        CUSUM decision threshold. Higher = fewer alarms.
    cusum_ema_alpha : float
        EWMA smoothing factor for the running baseline. 0 = constant,
        1 = no smoothing (use immediate value).
    fixed_threshold : float
        Used only when `use_adaptive_cusum=False`. Probability above
        which a node is flagged anomalous.
    """
    input_dim: int = 128
    hidden_dim: int = 64
    num_classes: int = NUM_CLASSES
    dropout: float = 0.3
    use_adaptive_cusum: bool = True
    cusum_k: float = 0.05
    cusum_h: float = 1.0
    cusum_ema_alpha: float = 0.05
    fixed_threshold: float = 0.5

    def __post_init__(self) -> None:
        if self.num_classes < 2:
            raise ValueError(f"num_classes must be >= 2, got {self.num_classes}")
        if not 0.0 <= self.cusum_ema_alpha <= 1.0:
            raise ValueError(
                f"cusum_ema_alpha must be in [0, 1], got {self.cusum_ema_alpha}"
            )
        if self.cusum_k < 0:
            raise ValueError(f"cusum_k must be >= 0, got {self.cusum_k}")
        if self.cusum_h <= 0:
            raise ValueError(f"cusum_h must be > 0, got {self.cusum_h}")


class AdaptiveCUSUM:
    """Per-node adaptive CUSUM change-point detector.

    Maintains one EWMA + CUSUM accumulator per node index. State is
    indexed by node ID so streaming inference over a fixed asset set
    is straightforward.
    """

    def __init__(self, k: float, h: float, ema_alpha: float) -> None:
        self.k = k
        self.h = h
        self.ema_alpha = ema_alpha
        # node_id -> (ewma_mean, cusum_sum)
        self._state: dict[int, tuple[float, float]] = {}

    def step(self, node_id: int, p_anomalous: float) -> tuple[bool, float, float]:
        """
        Update detector for one node with the latest anomaly probability.

        Parameters
        ----------
        node_id : int
            Stable identifier for the node. Use the symbol's index in
            `cfg.symbols` for streaming use.
        p_anomalous : float
            Total probability mass on non-normal classes, in [0, 1].

        Returns
        -------
        (alarm, ewma_mean, cusum_sum)
            alarm : True if CUSUM crossed `h` on this step.
            ewma_mean, cusum_sum : updated internal state (for logging).
        """
        prev_mean, prev_cusum = self._state.get(node_id, (p_anomalous, 0.0))

        # EWMA update of the running baseline.
        new_mean = (
            self.ema_alpha * p_anomalous
            + (1.0 - self.ema_alpha) * prev_mean
        )
        # CUSUM update.
        new_cusum = max(0.0, prev_cusum + p_anomalous - new_mean - self.k)
        alarm = new_cusum >= self.h
        if alarm:
            # Reset accumulator after alarm to detect subsequent events.
            new_cusum = 0.0

        self._state[node_id] = (new_mean, new_cusum)
        return alarm, new_mean, new_cusum

    def reset(self, node_id: int | None = None) -> None:
        """Reset state for one node, or all nodes if node_id is None."""
        if node_id is None:
            self._state.clear()
        else:
            self._state.pop(node_id, None)

    def state(self, node_id: int) -> tuple[float, float] | None:
        """Read current (ewma_mean, cusum_sum) for a node."""
        return self._state.get(node_id)


class AnomalyScorer(nn.Module):
    """MLP classifier head with adaptive thresholding."""

    def __init__(self, config: AnomalyScorerConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or AnomalyScorerConfig()
        c = self.cfg

        # 2-layer MLP classification head.
        self.mlp = nn.Sequential(
            nn.Linear(c.input_dim, c.hidden_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden_dim, c.hidden_dim),
            nn.GELU(),
            nn.Dropout(c.dropout),
            nn.Linear(c.hidden_dim, c.num_classes),
        )

        # CUSUM lives outside the nn.Module graph (stateful, non-learnable).
        self.cusum = AdaptiveCUSUM(
            k=c.cusum_k, h=c.cusum_h, ema_alpha=c.cusum_ema_alpha
        )

    def forward(self, z_fused: torch.Tensor) -> torch.Tensor:
        """
        Run the classifier head.

        Parameters
        ----------
        z_fused : Tensor [N, input_dim]
            Per-node fused embedding from `FusionModule`.

        Returns
        -------
        Tensor [N, num_classes]
            Raw logits. Use `cross_entropy` directly during training.
        """
        return self.mlp(z_fused)

    @torch.no_grad()
    def predict(
        self, z_fused: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[bool]]:
        """
        Convenience prediction: logits -> class probs + CUSUM/threshold alarms.

        Parameters
        ----------
        z_fused : Tensor [N, input_dim]
            Per-node fused embedding.

        Returns
        -------
        probs : Tensor [N, num_classes]
            Softmax probabilities.
        p_anomalous : Tensor [N]
            Sum of non-normal probabilities per node, in [0, 1].
        alarms : list[bool] of length N
            Whether each node tripped the anomaly detector.
        """
        was_training = self.training
        self.eval()
        try:
            logits = self.forward(z_fused)
            probs = F.softmax(logits, dim=-1)
            p_anomalous = 1.0 - probs[:, NORMAL_CLASS_IDX]

            alarms: list[bool] = []
            if self.cfg.use_adaptive_cusum:
                for node_id, p in enumerate(p_anomalous.tolist()):
                    alarm, _, _ = self.cusum.step(node_id, float(p))
                    alarms.append(alarm)
            else:
                alarms = (p_anomalous >= self.cfg.fixed_threshold).tolist()

            return probs.detach(), p_anomalous.detach(), alarms
        finally:
            self.train(was_training)

    def reset_state(self) -> None:
        """Reset CUSUM. Call between evaluation runs / on stream restart."""
        self.cusum.reset()
