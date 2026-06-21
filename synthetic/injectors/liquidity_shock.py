"""
synthetic/injectors/liquidity_shock.py
======================================
Injects liquidity shock patterns into clean L2 sequences.

Liquidity shock definition (per Easley, López de Prado & O'Hara 2012)
---------------------------------------------------------------------
A sudden withdrawal of market depth in which a substantial fraction
of resting orders at deeper book levels disappear simultaneously. The
top of the book may remain visually intact, masking the fact that
true available liquidity has collapsed.

Liquidity shocks are not always intentionally manipulative — they
occur naturally during news events and risk-off transitions — but
they are operationally important to detect because they invalidate
the assumptions of execution algorithms (notably TWAP/VWAP/POV).

Implementation
--------------
Within a chosen window of `duration_ms`:
  1. Choose `depth_reduction_pct` (50–95%).
  2. Apply the reduction multiplicatively to levels 2..n on BOTH sides
     (top level kept intact — characteristic "hidden" liquidity drain).
  3. Spike the cancel rate proportionally to the reduction.
  4. Hold for the full duration, then snap back instantly.
"""

from __future__ import annotations

import numpy as np

from synthetic.base_market import OrderBookSnapshot
from synthetic.injectors.base import (
    LABEL_LIQUIDITY_SHOCK,
    AnomalyInjector,
    InjectionResult,
)


class LiquidityShockInjector(AnomalyInjector):
    """Inject a depth-withdrawal liquidity shock."""

    LABEL = LABEL_LIQUIDITY_SHOCK
    NAME = "liquidity_shock"

    def inject(
        self, base_snapshots: list[OrderBookSnapshot]
    ) -> InjectionResult:
        n = len(base_snapshots)
        if n == 0:
            raise ValueError("Cannot inject into an empty snapshot list.")

        # ------------------------------------------------------------------
        # 1. Sample parameters
        # ------------------------------------------------------------------
        reduction_pct = self._uniform_float(
            tuple(self.params["depth_reduction_pct"])
        )
        duration_ms = self._uniform_int(tuple(self.params["duration_ms"]))

        step_ms = max(
            1,
            (base_snapshots[1].timestamp_ms - base_snapshots[0].timestamp_ms)
            if n >= 2 else 100,
        )
        duration_snaps = max(1, duration_ms // step_ms)
        start_idx = int(self.rng.integers(0, max(1, n - duration_snaps)))
        end_idx = min(n, start_idx + duration_snaps)

        keep_fraction = 1.0 - reduction_pct / 100.0
        severity = float(np.clip(reduction_pct / 100.0, 0.0, 1.0))

        # ------------------------------------------------------------------
        # 2. Apply
        # ------------------------------------------------------------------
        out_snapshots: list[OrderBookSnapshot] = []
        labels = np.zeros(n, dtype=np.int8)
        severities = np.zeros(n, dtype=np.float64)

        for i, snap in enumerate(base_snapshots):
            cloned = self._clone(snap)

            if start_idx <= i < end_idx:
                # Wipe out 50–95% of depth at levels 2..n on both sides.
                # Top level (index 0) preserved — that's the hallmark.
                if len(cloned.bid_sizes) > 1:
                    cloned.bid_sizes[1:] *= keep_fraction
                if len(cloned.ask_sizes) > 1:
                    cloned.ask_sizes[1:] *= keep_fraction

                # Mass cancellation -> cancel rate explodes.
                cloned.order_cancel_rate = float(
                    cloned.order_cancel_rate * (4.0 + 8.0 * severity)
                )

                labels[i] = self.LABEL
                severities[i] = severity

            out_snapshots.append(cloned)

        return InjectionResult(
            snapshots=out_snapshots,
            labels=labels,
            severities=severities,
            injection_id=self._new_injection_id(),
            params={
                "depth_reduction_pct": float(reduction_pct),
                "duration_ms": int(duration_ms),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "duration_snaps": int(duration_snaps),
            },
        )
