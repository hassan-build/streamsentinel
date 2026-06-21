"""
synthetic/injectors/layering.py
===============================
Injects layering patterns into clean L2 sequences.

Layering definition (per FCA Market Watch #57, 2017)
----------------------------------------------------
A multi-level variant of spoofing. The trader places several fake
orders simultaneously at consecutive price levels on one side of the
book, creating the illusion of multi-level support or resistance.
None of the fake orders are intended to be filled; all are cancelled.

Implementation
--------------
Within a randomly chosen window:
  1. Pick a side (bid/ask) uniformly at random.
  2. Inflate sizes at levels 1..n (n drawn from `n_levels` range).
  3. Each level's inflation is sampled within `price_spread_bps`.
  4. Hold for `lifetime_ms`, then cancel all at once.

Layering's signature differs from spoofing in three ways the GNN
should learn to pick up:
  - Multi-level inflation, not just level 1.
  - Sustained over a longer period (100 ms – 2 s vs 10–200 ms).
  - Larger cancel-rate spike at the end (all levels cancel together).
"""

from __future__ import annotations

import numpy as np

from synthetic.base_market import OrderBookSnapshot
from synthetic.injectors.base import (
    LABEL_LAYERING,
    AnomalyInjector,
    InjectionResult,
)


class LayeringInjector(AnomalyInjector):
    """Inject a layering (multi-level spoofing) event."""

    LABEL = LABEL_LAYERING
    NAME = "layering"

    def inject(
        self, base_snapshots: list[OrderBookSnapshot]
    ) -> InjectionResult:
        n = len(base_snapshots)
        if n == 0:
            raise ValueError("Cannot inject into an empty snapshot list.")

        # ------------------------------------------------------------------
        # 1. Sample parameters
        # ------------------------------------------------------------------
        n_levels_inject = self._uniform_int(tuple(self.params["n_levels"]))
        price_spread_bps = self._uniform_float(
            tuple(self.params["price_spread_bps"])
        )
        lifetime_ms = self._uniform_int(tuple(self.params["lifetime_ms"]))
        side = self.rng.choice(["bid", "ask"])

        # Compute window length in snapshots from the input step.
        step_ms = max(
            1,
            (base_snapshots[1].timestamp_ms - base_snapshots[0].timestamp_ms)
            if n >= 2 else 100,
        )
        duration_snaps = max(1, lifetime_ms // step_ms)
        start_idx = int(self.rng.integers(0, max(1, n - duration_snaps)))
        end_idx = min(n, start_idx + duration_snaps)

        # Cap the number of levels actually mutated by the book depth.
        max_book_levels = len(base_snapshots[0].bid_prices)
        n_levels_inject = min(n_levels_inject, max_book_levels)

        # ------------------------------------------------------------------
        # 2. Build size multipliers per level (decay outward from top)
        # ------------------------------------------------------------------
        # Top-level multiplier is highest; deeper levels are inflated
        # less. Spread parameter controls how fast the multiplier decays.
        decay_factor = max(1.0, 10.0 / max(1e-6, price_spread_bps))
        per_level_mult = np.array([
            1.0 + (decay_factor / (1.0 + i)) for i in range(n_levels_inject)
        ])

        # Severity scaled by total inflation across all touched levels.
        severity = float(np.clip(per_level_mult.sum() / 20.0, 0.0, 1.0))

        # ------------------------------------------------------------------
        # 3. Mutate the snapshot sequence
        # ------------------------------------------------------------------
        out_snapshots: list[OrderBookSnapshot] = []
        labels = np.zeros(n, dtype=np.int8)
        severities = np.zeros(n, dtype=np.float64)

        for i, snap in enumerate(base_snapshots):
            cloned = self._clone(snap)

            if start_idx <= i < end_idx:
                if side == "bid":
                    cloned.bid_sizes[:n_levels_inject] *= per_level_mult
                else:
                    cloned.ask_sizes[:n_levels_inject] *= per_level_mult

                # Stronger imbalance shift than spoofing because more
                # levels are skewed.
                sign = 1.0 if side == "bid" else -1.0
                cloned.trade_imbalance = float(np.clip(
                    cloned.trade_imbalance + 0.6 * sign * severity,
                    -1.0, 1.0,
                ))
                labels[i] = self.LABEL
                severities[i] = severity

            elif i == end_idx:
                # All layered orders cancel at once — much bigger cancel
                # rate spike than spoofing (signature feature).
                cloned.order_cancel_rate = float(
                    cloned.order_cancel_rate * (5.0 + 10.0 * severity)
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
                "n_levels_inject": int(n_levels_inject),
                "price_spread_bps": float(price_spread_bps),
                "lifetime_ms": int(lifetime_ms),
                "side": str(side),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "duration_snaps": int(duration_snaps),
            },
        )
