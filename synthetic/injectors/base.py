"""
synthetic/injectors/base.py
===========================
Abstract base class for all anomaly injectors.

Every injector takes a sequence of clean `OrderBookSnapshot` objects
(from BaseMarketSimulator) and returns a mutated sequence that contains
a specific manipulation pattern. The injector also reports per-snapshot
labels so the dataset assembler can record ground truth.

Design philosophy
-----------------
- Injectors NEVER mutate input snapshots in place. They construct new
  snapshots so the base sequence remains reusable.
- Each injector owns its own PRNG (seeded from a master seed) so that
  changing one injector's behaviour cannot perturb others — important
  when running ablation studies.
- All parameter ranges are uniform-sampled between the [low, high] bounds
  specified in `config.yaml`. This keeps the dissertation evaluation
  transparent: any reviewer can read the config and reproduce.
"""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from synthetic.base_market import OrderBookSnapshot


# Integer label conventions used everywhere in StreamSentinel.
# These match the order in config.yaml -> models.anomaly_scorer.anomaly_classes.
LABEL_NORMAL: int = 0
LABEL_SPOOFING: int = 1
LABEL_LAYERING: int = 2
LABEL_FLASH_CRASH: int = 3
LABEL_COORDINATED: int = 4
LABEL_LIQUIDITY_SHOCK: int = 5


@dataclass
class InjectionResult:
    """
    Output of a single anomaly injection over a window of snapshots.

    Attributes
    ----------
    snapshots : list[OrderBookSnapshot]
        Mutated snapshots, one per input timestamp.
    labels : np.ndarray
        Integer label for each snapshot (0 = normal, 1..5 = anomaly type).
        Even within the injection window, some leading/trailing snapshots
        may carry the normal label (0) if the manipulation hasn't started
        or has already ended at that millisecond.
    severities : np.ndarray
        Per-snapshot anomaly severity in [0, 1]. Useful for soft labels.
    injection_id : str
        UUID4 tying all rows of this event together for traceability.
    params : dict
        The exact parameter values used to inject this anomaly. Logged
        so the examiner can audit any single event by ID.
    """
    snapshots: list[OrderBookSnapshot]
    labels: np.ndarray
    severities: np.ndarray
    injection_id: str
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.snapshots)
        if len(self.labels) != n:
            raise ValueError(
                f"labels length {len(self.labels)} != n_snapshots {n}"
            )
        if len(self.severities) != n:
            raise ValueError(
                f"severities length {len(self.severities)} != n_snapshots {n}"
            )


class AnomalyInjector(ABC):
    """
    Abstract base for all anomaly injectors.

    Subclasses implement `inject()`, which takes a clean sequence and
    returns an `InjectionResult` with mutations and labels applied.
    """

    #: Class-level label identifier. Subclasses must override.
    LABEL: int = LABEL_NORMAL

    #: Human-readable name; used in logging and Parquet metadata.
    NAME: str = "abstract"

    def __init__(self, params: dict[str, Any], seed: int = 0) -> None:
        """
        Parameters
        ----------
        params : dict
            The injector-specific config block from config.yaml
            (e.g. config["synthetic"]["scenarios"]["spoofing"]).
        seed : int
            PRNG seed for this injector instance. Each injector type
            should be seeded with a derived value (e.g. master_seed +
            label) so that toggling one injector doesn't perturb others.
        """
        self.params = params
        self.rng = np.random.default_rng(seed)

    @abstractmethod
    def inject(
        self, base_snapshots: list[OrderBookSnapshot]
    ) -> InjectionResult:
        """
        Inject the manipulation pattern into a sequence of clean snapshots.

        Parameters
        ----------
        base_snapshots : list[OrderBookSnapshot]
            The clean L2 sequence into which to inject the anomaly.
            MUST NOT be mutated in place.

        Returns
        -------
        InjectionResult
            Mutated snapshots, labels, severities, and injection metadata.
        """
        ...

    # ------------------------------------------------------------------
    # Helpers shared by concrete injectors
    # ------------------------------------------------------------------
    def _new_injection_id(self) -> str:
        """Generate a UUID4 for traceability."""
        return str(uuid.uuid4())

    def _uniform_int(self, lo_hi: tuple[int, int]) -> int:
        """Sample an integer uniformly in the inclusive range [lo, hi]."""
        lo, hi = int(lo_hi[0]), int(lo_hi[1])
        return int(self.rng.integers(lo, hi + 1))

    def _uniform_float(self, lo_hi: tuple[float, float]) -> float:
        """Sample a float uniformly in [lo, hi]."""
        lo, hi = float(lo_hi[0]), float(lo_hi[1])
        return float(self.rng.uniform(lo, hi))

    @staticmethod
    def _clone(snap: OrderBookSnapshot) -> OrderBookSnapshot:
        """
        Return a deep copy of a snapshot whose arrays can be safely mutated.

        We can't use `dataclasses.replace` directly because the numpy
        arrays would still be shared by reference.
        """
        return OrderBookSnapshot(
            timestamp_ms=snap.timestamp_ms,
            symbol=snap.symbol,
            bid_prices=snap.bid_prices.copy(),
            ask_prices=snap.ask_prices.copy(),
            bid_sizes=snap.bid_sizes.copy(),
            ask_sizes=snap.ask_sizes.copy(),
            trade_imbalance=snap.trade_imbalance,
            order_cancel_rate=snap.order_cancel_rate,
        )
