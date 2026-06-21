"""
synthetic/injectors/flash_crash.py
==================================
Injects flash crash patterns into clean L2 sequences.

Flash crash definition (per Kirilenko et al. 2017)
--------------------------------------------------
A sudden, severe price decline that recovers within a short window.
The defining features of the 2010 flash crash were:
  - 6% drop in the E-mini S&P futures within ~5 minutes,
  - Spreads widening by an order of magnitude during the drop,
  - Liquidity (depth) collapsing on the buy side,
  - Partial price recovery in the subsequent ~15 minutes.

Implementation
--------------
Within a randomly chosen window of duration `duration_ms`:
  1. Linearly drop the fair price by `price_drop_pct`.
  2. Apply the same drop to the entire L2 book.
  3. Widen the spread by 5–20x normal.
  4. Reduce bid-side depth by 70–95% (asymmetric — liquidity flees the
     buy side first because falling prices imply sellers dominate).
  5. After the crash bottom, linearly recover by `recovery_ratio` of
     the drop over an equal duration.

Note on scale
-------------
We scale all parameters down by ~1/600 so a "flash crash" event fits
inside a typical 5–10 second injection window. This preserves the
*pattern* shape that detectors must learn, without requiring 5-minute
windows that would balloon dataset size.
"""

from __future__ import annotations

import numpy as np

from synthetic.base_market import OrderBookSnapshot
from synthetic.injectors.base import (
    LABEL_FLASH_CRASH,
    AnomalyInjector,
    InjectionResult,
)


class FlashCrashInjector(AnomalyInjector):
    """Inject a flash crash + partial recovery pattern."""

    LABEL = LABEL_FLASH_CRASH
    NAME = "flash_crash"

    def inject(
        self, base_snapshots: list[OrderBookSnapshot]
    ) -> InjectionResult:
        n = len(base_snapshots)
        if n == 0:
            raise ValueError("Cannot inject into an empty snapshot list.")

        # ------------------------------------------------------------------
        # 1. Sample parameters
        # ------------------------------------------------------------------
        drop_pct = self._uniform_float(tuple(self.params["price_drop_pct"]))
        duration_ms = self._uniform_int(tuple(self.params["duration_ms"]))
        recovery_ratio = self._uniform_float(
            tuple(self.params["recovery_ratio"])
        )

        step_ms = max(
            1,
            (base_snapshots[1].timestamp_ms - base_snapshots[0].timestamp_ms)
            if n >= 2 else 100,
        )
        # Split the window: half crash, half recovery.
        crash_snaps = max(1, (duration_ms // step_ms) // 2)
        recover_snaps = crash_snaps
        total_snaps = crash_snaps + recover_snaps

        if total_snaps >= n:
            # Pattern doesn't fit in window — squeeze it.
            crash_snaps = max(1, n // 2)
            recover_snaps = max(1, n - crash_snaps - 1)
            total_snaps = crash_snaps + recover_snaps

        start_idx = int(self.rng.integers(0, max(1, n - total_snaps)))

        # Severity drives both label confidence and downstream feature noise.
        drop_lo, drop_hi = self.params["price_drop_pct"]
        severity = float(np.clip(
            (drop_pct - drop_lo) / max(1e-9, drop_hi - drop_lo), 0.0, 1.0
        ))

        # ------------------------------------------------------------------
        # 2. Build crash trajectory: linear down, linear up.
        # ------------------------------------------------------------------
        crash_traj = np.linspace(0.0, drop_pct / 100.0, crash_snaps)
        recover_traj = np.linspace(
            drop_pct / 100.0,
            drop_pct / 100.0 * (1.0 - recovery_ratio),
            recover_snaps,
        )
        trajectory = np.concatenate([crash_traj, recover_traj])  # fraction down

        # ------------------------------------------------------------------
        # 3. Apply to each snapshot in the window
        # ------------------------------------------------------------------
        out_snapshots: list[OrderBookSnapshot] = []
        labels = np.zeros(n, dtype=np.int8)
        severities = np.zeros(n, dtype=np.float64)

        for i, snap in enumerate(base_snapshots):
            cloned = self._clone(snap)
            rel = i - start_idx

            if 0 <= rel < len(trajectory):
                pct_down = float(trajectory[rel])
                price_mult = 1.0 - pct_down

                # Move the entire book down by the same fraction.
                cloned.bid_prices *= price_mult
                cloned.ask_prices *= price_mult

                # Widen the spread. We do this by pushing the best ask
                # outward by (5 + 15*severity) extra ticks.
                # Detect tick size from neighbouring levels.
                if len(cloned.ask_prices) >= 2:
                    tick = float(
                        cloned.ask_prices[1] - cloned.ask_prices[0]
                    )
                else:
                    tick = 0.01
                spread_widen_ticks = 5.0 + 15.0 * severity * pct_down / max(
                    1e-9, drop_pct / 100.0
                )
                cloned.ask_prices += tick * spread_widen_ticks
                cloned.bid_prices -= tick * spread_widen_ticks * 0.5

                # Buy-side depth evaporates more aggressively than ask side.
                depth_kill = 1.0 - (0.7 + 0.25 * severity) * (
                    pct_down / max(1e-9, drop_pct / 100.0)
                )
                cloned.bid_sizes *= max(0.05, depth_kill)
                cloned.ask_sizes *= max(0.3, 1.0 - 0.4 * pct_down)

                # Strong sell pressure shows in the imbalance signal.
                cloned.trade_imbalance = float(np.clip(
                    cloned.trade_imbalance - 0.7 * severity, -1.0, 1.0
                ))

                # Crash-time cancel rate is elevated.
                cloned.order_cancel_rate = float(
                    cloned.order_cancel_rate * (1.0 + 3.0 * severity)
                )

                labels[i] = self.LABEL
                severities[i] = severity * (pct_down / max(
                    1e-9, drop_pct / 100.0
                ))

            out_snapshots.append(cloned)

        return InjectionResult(
            snapshots=out_snapshots,
            labels=labels,
            severities=severities,
            injection_id=self._new_injection_id(),
            params={
                "price_drop_pct": float(drop_pct),
                "duration_ms": int(duration_ms),
                "recovery_ratio": float(recovery_ratio),
                "start_idx": int(start_idx),
                "crash_snaps": int(crash_snaps),
                "recover_snaps": int(recover_snaps),
            },
        )
