"""
synthetic/injectors/coordinated_trading.py
==========================================
Injects coordinated trading patterns into clean L2 sequences.

Coordinated trading definition (per Pirrong 2018)
-------------------------------------------------
Multiple accounts placing same-direction orders within a narrow time
window to amplify their collective market impact while masking the
fact that a single trader is behind them. This pattern looks like a
sudden burst of unconnected small trades.

Implementation
--------------
Within a chosen window:
  1. Pick a side (bid/ask) uniformly at random — this is the side the
     accounts are "buying into" (e.g. picking up bid liquidity to
     drive the price down).
  2. Over a sync window of `sync_window_ms`, eat through `n_accounts`
     consecutive size depletions at the affected side.
  3. Each depletion is small enough on its own that no single account
     looks suspicious, but their clustering is the giveaway.
  4. Boost the trade_imbalance toward the coordinated direction.

Signal for the detector
-----------------------
This is the most subtle anomaly in our set. The order book itself
moves only slightly per snapshot, but `trade_imbalance` exhibits an
unusual sustained skew, and the cancel rate stays normal (no
cancellations — these are real fills). The GNN's correlation across
related symbols is what should catch coordinated trading.
"""

from __future__ import annotations

import numpy as np

from synthetic.base_market import OrderBookSnapshot
from synthetic.injectors.base import (
    LABEL_COORDINATED,
    AnomalyInjector,
    InjectionResult,
)


class CoordinatedTradingInjector(AnomalyInjector):
    """Inject a coordinated multi-account trading burst."""

    LABEL = LABEL_COORDINATED
    NAME = "coordinated_trading"

    def inject(
        self, base_snapshots: list[OrderBookSnapshot]
    ) -> InjectionResult:
        n = len(base_snapshots)
        if n == 0:
            raise ValueError("Cannot inject into an empty snapshot list.")

        # ------------------------------------------------------------------
        # 1. Sample parameters
        # ------------------------------------------------------------------
        n_accounts = self._uniform_int(tuple(self.params["n_accounts"]))
        sync_window_ms = self._uniform_int(
            tuple(self.params["sync_window_ms"])
        )
        direction_cfg = self.params.get("direction", "random")
        if direction_cfg == "random":
            side = self.rng.choice(["bid", "ask"])
        else:
            side = direction_cfg

        step_ms = max(
            1,
            (base_snapshots[1].timestamp_ms - base_snapshots[0].timestamp_ms)
            if n >= 2 else 100,
        )
        duration_snaps = max(1, sync_window_ms // step_ms)
        start_idx = int(self.rng.integers(0, max(1, n - duration_snaps)))
        end_idx = min(n, start_idx + duration_snaps)

        # Per-snapshot size to remove: spread the total impact across
        # the sync window. Larger n_accounts -> bigger total impact.
        per_step_depletion = 0.05 + 0.02 * n_accounts
        severity = float(np.clip(n_accounts / 10.0, 0.0, 1.0))

        # ------------------------------------------------------------------
        # 2. Apply
        # ------------------------------------------------------------------
        out_snapshots: list[OrderBookSnapshot] = []
        labels = np.zeros(n, dtype=np.int8)
        severities = np.zeros(n, dtype=np.float64)

        for i, snap in enumerate(base_snapshots):
            cloned = self._clone(snap)

            if start_idx <= i < end_idx:
                # Coordinated buys consume bid-side liquidity OR coordinated
                # sells consume ask-side liquidity at the top of book.
                # (The semantics: someone is hitting that side aggressively.)
                if side == "bid":
                    cloned.bid_sizes[0] *= (1.0 - per_step_depletion)
                    sign = -1.0   # selling pressure
                else:
                    cloned.ask_sizes[0] *= (1.0 - per_step_depletion)
                    sign = 1.0    # buying pressure

                cloned.trade_imbalance = float(np.clip(
                    cloned.trade_imbalance + sign * 0.5 * severity,
                    -1.0, 1.0,
                ))

                # Cancel rate does NOT spike — these are real trades, not
                # cancellations. This asymmetry vs spoofing/layering is
                # what the model should learn to distinguish them.
                labels[i] = self.LABEL
                severities[i] = severity

            out_snapshots.append(cloned)

        return InjectionResult(
            snapshots=out_snapshots,
            labels=labels,
            severities=severities,
            injection_id=self._new_injection_id(),
            params={
                "n_accounts": int(n_accounts),
                "sync_window_ms": int(sync_window_ms),
                "side": str(side),
                "start_idx": int(start_idx),
                "end_idx": int(end_idx),
                "per_step_depletion": float(per_step_depletion),
            },
        )
