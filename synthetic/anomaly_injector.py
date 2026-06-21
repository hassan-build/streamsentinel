"""
synthetic/anomaly_injector.py
=============================
Main CLI entrypoint for synthetic dataset generation.

Walks a simulated timeline, calling BaseMarketSimulator for normal
order book evolution and overlaying anomaly injectors with probability
`anomaly_rate`. Streams output to Parquet files, optionally split into
train/val/test sets.

Usage
-----
    python -m synthetic.anomaly_injector \
        --n-events 100000 \
        --output-dir data/synthetic \
        --seed 42

Run `python -m synthetic.anomaly_injector --help` for all options.

Design notes
------------
The assembler operates in "blocks" — chunks of N snapshots in which at
most ONE anomaly is injected. This keeps labels clean (no overlapping
anomaly windows) and makes the per-block injection probability map
directly onto the configured `anomaly_rate`.

For a 100 ms step size and 50-snapshot blocks (= 5 s/block):
  - 100,000 snapshots = 2,000 blocks = ~2.8 hours of simulated time
  - At 15% anomaly rate, ~300 anomaly events total
  - Generation time on a single core: ~5–15 seconds
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Allow `python -m synthetic.anomaly_injector` to find sibling modules
# when invoked from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_loader import load_config  # noqa: E402
from logger import get_logger          # noqa: E402
from synthetic.base_market import (    # noqa: E402
    BaseMarketConfig,
    BaseMarketSimulator,
    OrderBookSnapshot,
)
from synthetic.injectors import (      # noqa: E402
    INJECTOR_REGISTRY,
    AnomalyInjector,
    LABEL_NORMAL,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Snapshot -> row conversion
# ---------------------------------------------------------------------------

def _snapshot_to_row(
    snap: OrderBookSnapshot,
    label: int,
    severity: float,
    injection_id: str,
) -> dict[str, Any]:
    """
    Flatten one OrderBookSnapshot into a flat dict suitable for a
    pandas DataFrame row.

    Per-level fields are expanded into separate columns: bid_l1..bid_l10,
    ask_l1..ask_l10, bidsize_l1..bidsize_l10, asksize_l1..asksize_l10.
    """
    n_levels = len(snap.bid_prices)
    row: dict[str, Any] = {
        "timestamp": int(snap.timestamp_ms),
        "symbol": snap.symbol,
        "mid_price": float(snap.mid_price),
        "spread_bps": float(snap.spread_bps),
        "trade_imbalance": float(snap.trade_imbalance),
        "order_cancel_rate": float(snap.order_cancel_rate),
        "label": int(label),
        "anomaly_severity": float(severity),
        "injection_id": injection_id,
    }
    for i in range(n_levels):
        row[f"bid_l{i + 1}"] = float(snap.bid_prices[i])
        row[f"ask_l{i + 1}"] = float(snap.ask_prices[i])
        row[f"bidsize_l{i + 1}"] = float(snap.bid_sizes[i])
        row[f"asksize_l{i + 1}"] = float(snap.ask_sizes[i])
    return row


# ---------------------------------------------------------------------------
# Per-symbol generation
# ---------------------------------------------------------------------------

def _build_injectors(
    cfg_synthetic: dict[str, Any], master_seed: int
) -> dict[str, AnomalyInjector]:
    """
    Build one injector instance per enabled scenario in the config.

    Each injector is seeded deterministically as `master_seed + label`,
    so toggling one scenario can't perturb the PRNG stream of another.
    Critical for ablation studies — we need identical "normal" data
    across runs that disable different anomaly types.
    """
    injectors: dict[str, AnomalyInjector] = {}
    for name, params in cfg_synthetic["scenarios"].items():
        if not params.get("enabled", True):
            log.info(f"Skipping disabled scenario: {name}")
            continue
        if name not in INJECTOR_REGISTRY:
            log.warning(f"Unknown scenario in config: {name} (no injector)")
            continue
        cls = INJECTOR_REGISTRY[name]
        injectors[name] = cls(
            params=params,
            seed=master_seed + cls.LABEL,
        )
        log.info(f"Initialised injector: {name} (label={cls.LABEL})")
    return injectors


def generate_for_symbol(
    symbol: str,
    n_snapshots: int,
    anomaly_rate: float,
    injectors: dict[str, AnomalyInjector],
    base_cfg_overrides: dict[str, Any] | None = None,
    block_size: int = 50,
    master_seed: int = 42,
    symbol_seed_offset: int = 0,
) -> list[dict[str, Any]]:
    """
    Generate `n_snapshots` rows for one symbol.

    Parameters
    ----------
    symbol : str
        Ticker, e.g. "AAPL".
    n_snapshots : int
        Total snapshot count to generate for this symbol.
    anomaly_rate : float
        Probability that any given block contains an injected anomaly.
    injectors : dict[str, AnomalyInjector]
        Map from scenario name to injector instance.
    base_cfg_overrides : dict, optional
        Overrides for BaseMarketConfig (e.g. initial_price per symbol).
    block_size : int
        Snapshots per injection block (default 50 = 5 s at 100 ms step).
    master_seed : int
        Base PRNG seed.
    symbol_seed_offset : int
        Offset added to the seed for this symbol so that different
        symbols generate different price paths.

    Returns
    -------
    list[dict]
        Flat row dicts ready for pandas/pyarrow.
    """
    overrides = base_cfg_overrides or {}
    base_cfg = BaseMarketConfig(
        symbol=symbol,
        seed=master_seed + symbol_seed_offset,
        **overrides,
    )
    sim = BaseMarketSimulator(base_cfg)

    # Per-symbol PRNG for injection decisions (independent of base sim).
    decision_rng = np.random.default_rng(master_seed + 9001 + symbol_seed_offset)
    injector_names = list(injectors.keys())

    rows: list[dict[str, Any]] = []
    snaps_buffer: list[OrderBookSnapshot] = []
    ts_iter = sim.run(n_snapshots, start_timestamp_ms=0)

    snaps_done = 0
    while snaps_done < n_snapshots:
        # Pull up to block_size snapshots into the buffer.
        block_target = min(block_size, n_snapshots - snaps_done)
        snaps_buffer.clear()
        for _ in range(block_target):
            try:
                snaps_buffer.append(next(ts_iter))
            except StopIteration:
                break
        if not snaps_buffer:
            break

        # Decide whether to inject. Each block is one Bernoulli trial.
        if injector_names and decision_rng.uniform() < anomaly_rate:
            name = str(decision_rng.choice(injector_names))
            injector = injectors[name]
            try:
                result = injector.inject(snaps_buffer)
                snaps_for_rows = result.snapshots
                labels = result.labels
                severities = result.severities
                injection_id = result.injection_id
            except Exception as exc:
                # Robust to bad parameter combinations in custom configs.
                log.warning(
                    f"{name} injection failed on block {snaps_done}: {exc}"
                )
                snaps_for_rows = snaps_buffer
                labels = np.zeros(len(snaps_buffer), dtype=np.int8)
                severities = np.zeros(len(snaps_buffer), dtype=np.float64)
                injection_id = ""
        else:
            # Clean block.
            snaps_for_rows = snaps_buffer
            labels = np.zeros(len(snaps_buffer), dtype=np.int8)
            severities = np.zeros(len(snaps_buffer), dtype=np.float64)
            injection_id = ""

        for snap, lbl, sev in zip(snaps_for_rows, labels, severities):
            rows.append(_snapshot_to_row(
                snap=snap,
                label=int(lbl),
                severity=float(sev),
                injection_id=injection_id if lbl != LABEL_NORMAL else "",
            ))
        snaps_done += len(snaps_for_rows)

    return rows


# ---------------------------------------------------------------------------
# Top-level dataset assembly
# ---------------------------------------------------------------------------

def assemble_dataset(
    symbols: list[str],
    n_events: int,
    anomaly_rate: float,
    seed: int,
    injectors: dict[str, AnomalyInjector],
    block_size: int = 50,
) -> pd.DataFrame:
    """
    Generate the full dataset across all symbols and return as one
    DataFrame sorted by timestamp.
    """
    snapshots_per_symbol = n_events // max(1, len(symbols))
    log.info(
        f"Generating {n_events} snapshots across {len(symbols)} symbols "
        f"({snapshots_per_symbol} each)"
    )

    all_rows: list[dict[str, Any]] = []
    # Use slightly different initial prices per symbol so the dataset
    # reflects realistic cross-asset price levels.
    price_anchors = {
        "AAPL": 175.0, "MSFT": 420.0, "TSLA": 240.0,
        "SPY": 540.0, "NVDA": 120.0,
    }
    for i, symbol in enumerate(symbols):
        t0 = time.time()
        overrides = {"initial_price": price_anchors.get(symbol, 100.0)}
        rows = generate_for_symbol(
            symbol=symbol,
            n_snapshots=snapshots_per_symbol,
            anomaly_rate=anomaly_rate,
            injectors=injectors,
            base_cfg_overrides=overrides,
            block_size=block_size,
            master_seed=seed,
            symbol_seed_offset=i * 1000,   # large offset so paths diverge
        )
        elapsed = time.time() - t0
        log.info(
            f"  {symbol}: {len(rows)} rows in {elapsed:.2f}s "
            f"({len(rows) / max(elapsed, 1e-6):.0f} rows/sec)"
        )
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def write_splits(
    df: pd.DataFrame,
    output_dir: Path,
    splits: tuple[float, float, float],
) -> dict[str, Path]:
    """
    Write train/val/test Parquet files using a TIME-ORDERED split.

    Random splits leak temporal information across splits (a model can
    learn from "future" rows) — this is the most common evaluation
    mistake in time-series anomaly detection papers. We split by row
    order so train < val < test in time.
    """
    if abs(sum(splits) - 1.0) > 1e-6:
        raise ValueError(f"splits must sum to 1.0, got {sum(splits)}")

    n = len(df)
    n_train = int(splits[0] * n)
    n_val = int(splits[1] * n)
    # Whatever's left goes to test (handles rounding).
    train_df = df.iloc[:n_train]
    val_df = df.iloc[n_train: n_train + n_val]
    test_df = df.iloc[n_train + n_val:]

    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name, split_df in [("train", train_df), ("val", val_df),
                           ("test", test_df)]:
        path = output_dir / f"{name}.parquet"
        pq.write_table(pa.Table.from_pandas(split_df), path)
        paths[name] = path
        log.info(
            f"  {name}.parquet  rows={len(split_df):>7,}  "
            f"anomalies={(split_df['label'] != LABEL_NORMAL).sum():>5,}  "
            f"-> {path}"
        )
    return paths


def write_metadata(
    output_dir: Path,
    args: argparse.Namespace,
    df: pd.DataFrame,
    elapsed_sec: float,
    split_paths: dict[str, Path],
) -> Path:
    """
    Write a JSON metadata sidecar describing the dataset.

    This is what enables reproducibility: the metadata file records
    every parameter and version needed to regenerate the exact dataset.
    """
    label_counts = df["label"].value_counts().to_dict()
    label_counts = {int(k): int(v) for k, v in label_counts.items()}

    meta = {
        "n_rows_total": int(len(df)),
        "n_symbols": int(df["symbol"].nunique()),
        "symbols": sorted(df["symbol"].unique().tolist()),
        "label_counts": label_counts,
        "anomaly_rate_observed": float(
            (df["label"] != LABEL_NORMAL).mean()
        ),
        "args": {
            "n_events": int(args.n_events),
            "anomaly_rate_target": float(args.anomaly_rate),
            "seed": int(args.seed),
            "splits": list(args.splits),
            "block_size": int(args.block_size),
        },
        "split_paths": {k: str(v) for k, v in split_paths.items()},
        "generation_time_seconds": float(elapsed_sec),
        "schema_version": "1.0.0",
    }
    path = output_dir / "metadata.json"
    with open(path, "w") as fh:
        json.dump(meta, fh, indent=2)
    log.info(f"Metadata written to {path}")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    cfg = load_config()
    syn_cfg = cfg["synthetic"]
    src_cfg = cfg["data_sources"]["alpaca"]

    p = argparse.ArgumentParser(
        prog="synthetic.anomaly_injector",
        description="Generate labelled synthetic L2 anomaly dataset",
    )
    p.add_argument(
        "--n-events", type=int, default=100_000,
        help="Total snapshot count across all symbols (default: 100000)",
    )
    p.add_argument(
        "--output-dir", type=Path, default=Path("data/synthetic"),
        help="Where to write Parquet + metadata (default: data/synthetic)",
    )
    p.add_argument(
        "--seed", type=int, default=syn_cfg.get("seed", 42),
        help="Master PRNG seed (default from config.yaml)",
    )
    p.add_argument(
        "--symbols", type=str,
        default=",".join(src_cfg["symbols"]),
        help="Comma-separated tickers (default from config.yaml)",
    )
    p.add_argument(
        "--anomaly-rate", type=float,
        default=syn_cfg.get("anomaly_rate", 0.15),
        help="Per-block anomaly injection probability (default: 0.15)",
    )
    p.add_argument(
        "--splits", type=float, nargs=3, default=(0.7, 0.15, 0.15),
        metavar=("TRAIN", "VAL", "TEST"),
        help="Train/val/test fractions (default: 0.7 0.15 0.15)",
    )
    p.add_argument(
        "--block-size", type=int, default=50,
        help="Snapshots per injection block (default: 50)",
    )
    return p.parse_args()


def main() -> int:
    args = _parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    if not symbols:
        log.error("No symbols specified.")
        return 1

    log.info("=" * 70)
    log.info("StreamSentinel — Synthetic Anomaly Injection")
    log.info("=" * 70)
    log.info(f"  n_events      : {args.n_events:,}")
    log.info(f"  symbols       : {symbols}")
    log.info(f"  anomaly_rate  : {args.anomaly_rate}")
    log.info(f"  splits        : {args.splits}")
    log.info(f"  seed          : {args.seed}")
    log.info(f"  output_dir    : {args.output_dir}")

    cfg_synthetic = load_config()["synthetic"]
    injectors = _build_injectors(cfg_synthetic, master_seed=args.seed)
    if not injectors:
        log.error(
            "No injectors enabled. Enable at least one scenario in "
            "config.yaml > synthetic.scenarios.*.enabled."
        )
        return 1

    t0 = time.time()
    df = assemble_dataset(
        symbols=symbols,
        n_events=args.n_events,
        anomaly_rate=args.anomaly_rate,
        seed=args.seed,
        injectors=injectors,
        block_size=args.block_size,
    )
    elapsed = time.time() - t0

    log.info("")
    log.info(f"Total rows: {len(df):,}")
    log.info(f"Label distribution:")
    for lbl, cnt in df["label"].value_counts().sort_index().items():
        log.info(f"  label={int(lbl)}  count={int(cnt):>7,}  "
                 f"({cnt / len(df):.2%})")

    split_paths = write_splits(df, args.output_dir, tuple(args.splits))
    write_metadata(args.output_dir, args, df, elapsed, split_paths)

    log.info("")
    log.info(
        f"Done in {elapsed:.1f}s "
        f"({len(df) / max(elapsed, 1e-6):.0f} rows/sec)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
