"""
synthetic/injectors/spoofing.py
===============================
Injects spoofing patterns into clean L2 sequences.

Spoofing definition (per Dodd-Frank §747 and Lee et al. 2013)
-------------------------------------------------------------
A trader places a large visible order with the intent to cancel it
before execution, creating false price pressure to push the market in
a direction beneficial to a smaller order they have on the opposite
side. The hallmark is a large size that appears, sits briefly, and
vanishes without trading.

Implementation
--------------
Within a randomly chosen window inside `base_snapshots`:
  1. Pick a side (bid/ask) uniformly at random.
  2. Multiply the size at the best level on that side by 5–20×.
  3. Hold the inflated size for 10–200 ms.
  4. Restore the size (cancellation) and elevate the cancel_rate signal.

Label
-----
Snapshots inside the spoof's lifetime are labelled 1 (spoofing).
Severity scales with the size multiplier used.
"""

from __future__ import annotations

import numpy as np

from synthetic.base_market import OrderBookSnapshot
from synthetic.injectors.base import (
    LABEL_SPOOFING,
    AnomalyInjector,
    InjectionResult,
)


class SpoofingInjector(AnomalyInjector):
    """Inject a spoofing event into a clean L2 sequence."""

    LABEL = LABEL_SPOOFING
    NAME = "spoofing"

    def inject(
        self, base_snapshots: list[OrderBookSnapshot]
    ) -> InjectionResult:
        n = len(base_snapshots)
        if n == 0:
            raise ValueError("Cannot inject into an empty snapshot list.")

        # ------------------------------------------------------------------
        # 1. Sample injection parameters within configured ranges.
        # ------------------------------------------------------------------
        size_mult = self._uniform_float(
            tuple(self.params["order_size_multiplier"])
        )
        cancel_delay_ms = self._uniform_int(
            tuple(self.params["cancel_delay_ms"])
        )
        side = self.rng.choice(["bid", "ask"])

        # Convert cancel_delay (ms) into number of snapshots given the
        # step size encoded in the input timestamps.
        if n >= 2:
            step_ms = max(1, base_snapshots[1].timestamp_ms
                         - base_snapshots[0].timestamp_ms)
        else:
            step_ms = 100
        spoof_duration_snaps = max(1, cancel_delay_ms // step_ms)

        # Place the spoof somewhere it has room to start AND end
        # within the window.
        start_idx = int(self.rng.integers(0, max(1, n - spoof_duration_snaps)))
        end_idx = min(n, start_idx + spoof_duration_snaps)

        # ------------------------------------------------------------------
        # 2. Build mutated snapshot list.
        # ------------------------------------------------------------------
        out_snapshots: list[OrderBookSnapshot] = []
        labels = np.zeros(n, dtype=np.int8)
        severities = np.zeros(n, dtype=np.float64)

        # Normalise severity into [0, 1] over the configured multiplier range.
        mult_lo, mult_hi = self.params["order_size_multiplier"]
        severity = float((size_mult - mult_lo) / max(1e-9, mult_hi - mult_lo))
        severity = float(np.clip(severity, 0.0, 1.0))

        for i, snap in enumerate(base_snapshots):
            cloned = self._clone(snap)

            if start_idx <= i < end_idx:
                # Inside spoof lifetime — inflate the size at level 1.
                if side == "bid":
                    cloned.bid_sizes[0] *= size_mult
                else:
                    cloned.ask_sizes[0] *= size_mult

                # The market is being skewed by the visible imbalance.
                imbalance_sign = 1.0 if side == "bid" else -1.0
                cloned.trade_imbalance = float(np.clip(
                    cloned.trade_imbalance + 0.4 * imbalance_sign * severity,
                    -1.0, 1.0,
                ))

                labels[i] = self.LABEL
                severities[i] = severity

            elif i == end_idx:
                # Cancel snapshot: the spoof has just disappeared. Spike
                # the cancel rate so downstream features can pick it up.
                cloned.order_cancel_rate = float(
                    cloned.order_cancel_rate * (3.0 + 5.0 * severity)
                )
                # Still labelled spoofing because the cancellation is the
                # smoking gun of the manipulation.
                labels[i] = self.LABEL
                severities[i] = severity

            out_snapshots.append(cloned)

        return InjectionResult(
            snapshots=out_snapshots,
            labels=labels,
            severities=severities,
            injection_id=self._new_injection_id(),
            params={
                "size_multiplier": size_mult,
                "cancel_delay_ms": cancel_delay_ms,
                "side": str(side),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "duration_snaps": int(spoof_duration_snaps),
            },
        )
