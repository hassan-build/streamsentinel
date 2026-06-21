"""
synthetic/base_market.py
========================
Generates "clean" Level 2 order book sequences with no manipulation
present. This is the baseline market state on top of which the anomaly
injectors operate.

Model
-----
Fair value follows a geometric Brownian motion (GBM):

    dS_t = mu * S_t * dt + sigma * S_t * dW_t

Bid and ask prices oscillate around the fair value with a half-spread
drawn from a log-normal distribution. Depth at each level decays
exponentially from the top of the book, with random per-level noise to
mimic real market microstructure (Cont & Stoikov, 2010).

The simulator is deterministic given the random seed — important for
dissertation reproducibility.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

# Default model parameters — calibrated to liquid US equities (e.g. AAPL).
# Values come from a 1-week mean over October 2024 IEX data, then rounded.
DEFAULT_INITIAL_PRICE: float = 150.0          # USD
DEFAULT_DRIFT: float = 0.0                    # mu (annualised)
DEFAULT_VOLATILITY: float = 0.20              # sigma (annualised)
DEFAULT_HALF_SPREAD_BPS: float = 1.5          # mean half-spread, basis points
DEFAULT_DEPTH_LEVELS: int = 10                # L2 depth on each side
DEFAULT_TOP_LEVEL_SIZE: float = 500.0         # shares at the inside quote
DEFAULT_DEPTH_DECAY: float = 0.35             # exponential decay across levels
DEFAULT_TICK_SIZE: float = 0.01               # USD; the price grid


@dataclass
class OrderBookSnapshot:
    """
    A single L2 order book snapshot at a point in time.

    Attributes
    ----------
    timestamp_ms : int
        Millisecond epoch (UTC) at which this snapshot was captured.
    symbol : str
        Ticker symbol, e.g. "AAPL".
    bid_prices : np.ndarray
        Array of length `n_levels`; bid_prices[0] is the best bid.
    ask_prices : np.ndarray
        Array of length `n_levels`; ask_prices[0] is the best ask.
    bid_sizes : np.ndarray
        Array of length `n_levels`; quantities on the bid side.
    ask_sizes : np.ndarray
        Array of length `n_levels`; quantities on the ask side.
    trade_imbalance : float
        Buy-volume minus sell-volume over the prior 100 ms window,
        normalised by total volume in [-1, 1].
    order_cancel_rate : float
        Cancellation events per second over the prior 100 ms window.
    """
    timestamp_ms: int
    symbol: str
    bid_prices: np.ndarray
    ask_prices: np.ndarray
    bid_sizes: np.ndarray
    ask_sizes: np.ndarray
    trade_imbalance: float = 0.0
    order_cancel_rate: float = 0.0

    @property
    def mid_price(self) -> float:
        """Midpoint between best bid and best ask."""
        return float(0.5 * (self.bid_prices[0] + self.ask_prices[0]))

    @property
    def spread_bps(self) -> float:
        """Bid-ask spread in basis points (1 bp = 0.01%)."""
        spread = float(self.ask_prices[0] - self.bid_prices[0])
        return 10_000.0 * spread / self.mid_price if self.mid_price > 0 else 0.0


@dataclass
class BaseMarketConfig:
    """Configuration for the BaseMarketSimulator.

    All fields have defaults calibrated to liquid US equities. Override
    only the ones you need to change.
    """
    symbol: str = "AAPL"
    initial_price: float = DEFAULT_INITIAL_PRICE
    drift: float = DEFAULT_DRIFT
    volatility: float = DEFAULT_VOLATILITY
    half_spread_bps: float = DEFAULT_HALF_SPREAD_BPS
    n_levels: int = DEFAULT_DEPTH_LEVELS
    top_level_size: float = DEFAULT_TOP_LEVEL_SIZE
    depth_decay: float = DEFAULT_DEPTH_DECAY
    tick_size: float = DEFAULT_TICK_SIZE
    step_ms: int = 100                         # snapshot frequency
    seed: int = 42

    # Trade/cancel feature noise — independent of the order book itself.
    base_imbalance_std: float = 0.15
    base_cancel_rate_mean: float = 25.0        # cancels/sec under normal flow
    base_cancel_rate_std: float = 8.0

    def __post_init__(self) -> None:
        """Validate configuration immediately at construction."""
        if self.n_levels < 1:
            raise ValueError(f"n_levels must be >= 1, got {self.n_levels}")
        if self.tick_size <= 0:
            raise ValueError(f"tick_size must be > 0, got {self.tick_size}")
        if self.step_ms < 1:
            raise ValueError(f"step_ms must be >= 1 ms, got {self.step_ms}")
        if self.volatility < 0:
            raise ValueError(f"volatility must be >= 0, got {self.volatility}")


class BaseMarketSimulator:
    """
    Geometric Brownian motion driven L2 order book simulator.

    Produces a sequence of `OrderBookSnapshot` objects representing
    normal (non-manipulated) market evolution. The state is fully
    reproducible given the seed in the config.

    Usage
    -----
        cfg = BaseMarketConfig(symbol="AAPL", seed=42)
        sim = BaseMarketSimulator(cfg)
        for snapshot in sim.run(n_steps=1000):
            print(snapshot.mid_price)
    """

    # Annualisation constant for stepping GBM at sub-second resolution.
    # 252 trading days/year × 6.5 hours × 3600 seconds/hour = 5,896,800 sec.
    _SECONDS_PER_TRADING_YEAR: float = 252 * 6.5 * 3600

    def __init__(self, config: BaseMarketConfig | None = None) -> None:
        """
        Parameters
        ----------
        config : BaseMarketConfig, optional
            Simulator parameters. If None, defaults are used.
        """
        self.cfg = config or BaseMarketConfig()
        self._rng = np.random.default_rng(self.cfg.seed)
        self._current_price: float = self.cfg.initial_price
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run(self, n_steps: int, start_timestamp_ms: int | None = None
            ) -> Iterator[OrderBookSnapshot]:
        """
        Yield `n_steps` order book snapshots.

        Parameters
        ----------
        n_steps : int
            Number of snapshots to generate.
        start_timestamp_ms : int, optional
            Starting timestamp in millisecond epoch. Defaults to 0 for
            test reproducibility.

        Yields
        ------
        OrderBookSnapshot
            One snapshot per `cfg.step_ms` milliseconds.
        """
        if n_steps < 0:
            raise ValueError(f"n_steps must be >= 0, got {n_steps}")
        ts = start_timestamp_ms if start_timestamp_ms is not None else 0
        for _ in range(n_steps):
            self._advance_price()
            snap = self._build_snapshot(ts)
            yield snap
            ts += self.cfg.step_ms
            self._step_count += 1

    def snapshot(self, timestamp_ms: int) -> OrderBookSnapshot:
        """
        Generate a single snapshot at the given timestamp without
        advancing the simulator. Useful for one-off use inside injectors.
        """
        return self._build_snapshot(timestamp_ms)

    @property
    def current_price(self) -> float:
        """Return the current GBM fair value (mid-price proxy)."""
        return self._current_price

    # ------------------------------------------------------------------
    # Internal mechanics
    # ------------------------------------------------------------------
    def _advance_price(self) -> None:
        """
        Advance the fair value one step of GBM:

            S_{t+dt} = S_t * exp((mu - 0.5*sigma^2) * dt + sigma * sqrt(dt) * Z)

        where Z ~ N(0, 1). The log-Euler form is exact for GBM and avoids
        numerical drift over long runs.
        """
        dt = (self.cfg.step_ms / 1000.0) / self._SECONDS_PER_TRADING_YEAR
        z = self._rng.standard_normal()
        log_return = (
            (self.cfg.drift - 0.5 * self.cfg.volatility ** 2) * dt
            + self.cfg.volatility * np.sqrt(dt) * z
        )
        self._current_price *= float(np.exp(log_return))

    def _build_snapshot(self, timestamp_ms: int) -> OrderBookSnapshot:
        """
        Build an L2 snapshot consistent with the current fair value.

        Steps:
          1. Draw a half-spread from log-normal centred on `half_spread_bps`.
          2. Place best bid and ask symmetrically around fair value.
          3. Fill remaining levels at integer tick offsets.
          4. Populate sizes with exponential decay + multiplicative noise.
          5. Sample trade imbalance and cancel rate from configured noise.
        """
        cfg = self.cfg
        fair = self._current_price

        # --- prices ---
        half_spread_log = self._rng.normal(
            loc=np.log(cfg.half_spread_bps), scale=0.3
        )
        half_spread_bps = float(np.exp(half_spread_log))
        half_spread = fair * half_spread_bps / 10_000.0

        # Snap to the price grid (tick size)
        best_bid = self._round_to_tick(fair - half_spread)
        best_ask = self._round_to_tick(fair + half_spread)

        # Guarantee best_ask > best_bid even after rounding (avoids
        # zero-spread artifacts at very low volatility settings).
        if best_ask <= best_bid:
            best_ask = best_bid + cfg.tick_size

        # Fill out the book at tick increments outwards from the top.
        offsets = np.arange(cfg.n_levels) * cfg.tick_size
        bid_prices = best_bid - offsets
        ask_prices = best_ask + offsets

        # --- sizes ---
        # Exponential decay with small multiplicative log-normal noise.
        decay = np.exp(-cfg.depth_decay * np.arange(cfg.n_levels))
        bid_noise = self._rng.lognormal(mean=0.0, sigma=0.25,
                                        size=cfg.n_levels)
        ask_noise = self._rng.lognormal(mean=0.0, sigma=0.25,
                                        size=cfg.n_levels)
        bid_sizes = np.maximum(1.0, cfg.top_level_size * decay * bid_noise)
        ask_sizes = np.maximum(1.0, cfg.top_level_size * decay * ask_noise)

        # --- aggregate features ---
        trade_imbalance = float(
            np.clip(self._rng.normal(0.0, cfg.base_imbalance_std), -1.0, 1.0)
        )
        order_cancel_rate = float(
            max(0.0, self._rng.normal(
                cfg.base_cancel_rate_mean, cfg.base_cancel_rate_std
            ))
        )

        return OrderBookSnapshot(
            timestamp_ms=timestamp_ms,
            symbol=cfg.symbol,
            bid_prices=bid_prices.astype(np.float64),
            ask_prices=ask_prices.astype(np.float64),
            bid_sizes=bid_sizes.astype(np.float64),
            ask_sizes=ask_sizes.astype(np.float64),
            trade_imbalance=trade_imbalance,
            order_cancel_rate=order_cancel_rate,
        )

    def _round_to_tick(self, price: float) -> float:
        """Snap a continuous price to the nearest valid tick."""
        return round(price / self.cfg.tick_size) * self.cfg.tick_size
