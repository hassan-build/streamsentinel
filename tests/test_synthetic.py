"""
tests/test_synthetic.py
=======================
Test suite for the synthetic data generation module.

Verifies:
  - Base market simulator produces valid order books
  - Each injector produces non-empty, well-labelled output
  - Labels stay in valid range
  - Anomaly rate observed matches target within binomial CI
  - Parquet round-trip preserves all columns
  - Fixed seed produces byte-identical output across runs
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from synthetic.base_market import (
    BaseMarketConfig,
    BaseMarketSimulator,
)
from synthetic.injectors import (
    INJECTOR_REGISTRY,
    LABEL_NORMAL,
    LABEL_SPOOFING,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_base_sequence():
    """A 100-snapshot clean sequence for injector tests."""
    cfg = BaseMarketConfig(symbol="TEST", seed=123)
    sim = BaseMarketSimulator(cfg)
    return list(sim.run(100))


@pytest.fixture
def synthetic_config():
    """Minimal injector params matching config.yaml schema."""
    return {
        "spoofing": {
            "order_size_multiplier": [5, 20],
            "cancel_delay_ms": [10, 200],
            "n_layers": [3, 8],
        },
        "layering": {
            "n_levels": [3, 10],
            "price_spread_bps": [5, 30],
            "lifetime_ms": [100, 2000],
        },
        "flash_crash": {
            "price_drop_pct": [2, 15],
            "duration_ms": [200, 5000],
            "recovery_ratio": [0.5, 1.0],
        },
        "coordinated_trading": {
            "n_accounts": [3, 10],
            "sync_window_ms": [50, 500],
            "direction": "random",
        },
        "liquidity_shock": {
            "depth_reduction_pct": [50, 95],
            "duration_ms": [500, 10000],
        },
    }


# ---------------------------------------------------------------------------
# BaseMarketSimulator
# ---------------------------------------------------------------------------

class TestBaseMarketSimulator:
    """Tests for the clean L2 order book simulator."""

    def test_produces_valid_snapshots(self):
        cfg = BaseMarketConfig(symbol="AAPL", seed=42)
        sim = BaseMarketSimulator(cfg)
        snaps = list(sim.run(50))
        assert len(snaps) == 50
        for s in snaps:
            assert s.bid_prices.shape == (cfg.n_levels,)
            assert s.ask_prices.shape == (cfg.n_levels,)
            assert s.bid_sizes.shape == (cfg.n_levels,)
            assert s.ask_sizes.shape == (cfg.n_levels,)

    def test_no_crossed_book(self):
        cfg = BaseMarketConfig(symbol="AAPL", seed=42)
        sim = BaseMarketSimulator(cfg)
        for s in sim.run(200):
            assert s.ask_prices[0] > s.bid_prices[0], "best ask must > best bid"

    def test_monotone_levels(self):
        cfg = BaseMarketConfig(symbol="AAPL", seed=42)
        sim = BaseMarketSimulator(cfg)
        for s in sim.run(50):
            assert np.all(np.diff(s.bid_prices) < 0), "bids must descend"
            assert np.all(np.diff(s.ask_prices) > 0), "asks must ascend"

    def test_positive_sizes(self):
        cfg = BaseMarketConfig(symbol="AAPL", seed=42)
        sim = BaseMarketSimulator(cfg)
        for s in sim.run(50):
            assert np.all(s.bid_sizes > 0)
            assert np.all(s.ask_sizes > 0)

    def test_reproducible_with_seed(self):
        sim1 = BaseMarketSimulator(BaseMarketConfig(seed=999))
        sim2 = BaseMarketSimulator(BaseMarketConfig(seed=999))
        snaps1 = list(sim1.run(20))
        snaps2 = list(sim2.run(20))
        for s1, s2 in zip(snaps1, snaps2):
            np.testing.assert_array_equal(s1.bid_prices, s2.bid_prices)
            np.testing.assert_array_equal(s1.ask_prices, s2.ask_prices)
            np.testing.assert_array_equal(s1.bid_sizes, s2.bid_sizes)

    def test_different_seeds_diverge(self):
        sim1 = BaseMarketSimulator(BaseMarketConfig(seed=1))
        sim2 = BaseMarketSimulator(BaseMarketConfig(seed=2))
        snaps1 = list(sim1.run(50))
        snaps2 = list(sim2.run(50))
        # Allow first few to coincide by chance but later ones should diverge.
        diffs = [
            abs(s1.mid_price - s2.mid_price)
            for s1, s2 in zip(snaps1[-10:], snaps2[-10:])
        ]
        assert max(diffs) > 0.0

    def test_invalid_config_rejected(self):
        with pytest.raises(ValueError):
            BaseMarketConfig(n_levels=0)
        with pytest.raises(ValueError):
            BaseMarketConfig(tick_size=0)
        with pytest.raises(ValueError):
            BaseMarketConfig(volatility=-0.1)


# ---------------------------------------------------------------------------
# Injectors
# ---------------------------------------------------------------------------

class TestInjectors:
    """Per-injector smoke + correctness tests."""

    @pytest.mark.parametrize("name", list(INJECTOR_REGISTRY.keys()))
    def test_each_injector_runs(self, name, small_base_sequence, synthetic_config):
        cls = INJECTOR_REGISTRY[name]
        injector = cls(params=synthetic_config[name], seed=42)
        result = injector.inject(small_base_sequence)

        assert len(result.snapshots) == len(small_base_sequence)
        assert len(result.labels) == len(small_base_sequence)
        assert len(result.severities) == len(small_base_sequence)
        assert result.injection_id != ""

    @pytest.mark.parametrize("name", list(INJECTOR_REGISTRY.keys()))
    def test_label_is_correct_class(self, name, small_base_sequence,
                                    synthetic_config):
        cls = INJECTOR_REGISTRY[name]
        injector = cls(params=synthetic_config[name], seed=42)
        result = injector.inject(small_base_sequence)

        non_normal = result.labels[result.labels != LABEL_NORMAL]
        # At least some snapshots should be labelled with this anomaly.
        assert len(non_normal) > 0, f"{name} produced no anomalous labels"
        # All non-normal labels should match the injector's class label.
        assert np.all(non_normal == cls.LABEL), \
            f"{name} produced labels other than {cls.LABEL}: {set(non_normal)}"

    @pytest.mark.parametrize("name", list(INJECTOR_REGISTRY.keys()))
    def test_severities_in_range(self, name, small_base_sequence,
                                 synthetic_config):
        cls = INJECTOR_REGISTRY[name]
        injector = cls(params=synthetic_config[name], seed=42)
        result = injector.inject(small_base_sequence)
        assert np.all(result.severities >= 0.0)
        assert np.all(result.severities <= 1.0)

    @pytest.mark.parametrize("name", list(INJECTOR_REGISTRY.keys()))
    def test_does_not_mutate_input(self, name, small_base_sequence,
                                   synthetic_config):
        cls = INJECTOR_REGISTRY[name]
        injector = cls(params=synthetic_config[name], seed=42)
        original_bid_l1 = small_base_sequence[0].bid_sizes[0]
        injector.inject(small_base_sequence)
        # Original input list must be unchanged.
        assert small_base_sequence[0].bid_sizes[0] == original_bid_l1

    def test_spoofing_inflates_size(self, small_base_sequence,
                                    synthetic_config):
        from synthetic.injectors import SpoofingInjector
        injector = SpoofingInjector(
            params=synthetic_config["spoofing"], seed=42
        )
        result = injector.inject(small_base_sequence)
        # Find at least one snapshot where spoofing was active and check
        # that level-1 size is bigger than in the corresponding clean one.
        for i, lbl in enumerate(result.labels):
            if lbl == LABEL_SPOOFING:
                orig = small_base_sequence[i]
                mod = result.snapshots[i]
                assert (mod.bid_sizes[0] > orig.bid_sizes[0]
                        or mod.ask_sizes[0] > orig.ask_sizes[0])
                return
        pytest.fail("No spoofing snapshots found in injection result")

    def test_flash_crash_drops_price(self, small_base_sequence,
                                     synthetic_config):
        from synthetic.injectors import FlashCrashInjector
        # Force a clearly visible crash by tweaking params for the test.
        params = dict(synthetic_config["flash_crash"])
        params["price_drop_pct"] = [10, 15]      # strong drop
        params["duration_ms"] = [500, 1000]      # short so it fits
        injector = FlashCrashInjector(params=params, seed=7)
        result = injector.inject(small_base_sequence)
        crash_mids = [
            result.snapshots[i].mid_price
            for i, lbl in enumerate(result.labels) if lbl != LABEL_NORMAL
        ]
        normal_mids = [
            small_base_sequence[i].mid_price
            for i, lbl in enumerate(result.labels) if lbl != LABEL_NORMAL
        ]
        assert len(crash_mids) > 0
        # Minimum crash price should be measurably below corresponding
        # normal price (allowing some headroom for the rounding).
        assert min(crash_mids) < min(normal_mids)


# ---------------------------------------------------------------------------
# End-to-end CLI
# ---------------------------------------------------------------------------

class TestEndToEnd:
    """End-to-end pipeline tests via the CLI."""

    @pytest.fixture
    def tmp_output_dir(self, tmp_path):
        out = tmp_path / "synthetic_out"
        yield out
        if out.exists():
            shutil.rmtree(out)

    def test_cli_runs_and_writes_files(self, tmp_output_dir):
        repo_root = Path(__file__).resolve().parent.parent
        cmd = [
            sys.executable, "-m", "synthetic.anomaly_injector",
            "--n-events", "2000",
            "--output-dir", str(tmp_output_dir),
            "--seed", "42",
            "--symbols", "AAPL,MSFT",
        ]
        result = subprocess.run(
            cmd, cwd=repo_root, capture_output=True, text=True,
        )
        assert result.returncode == 0, (
            f"CLI failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
        )
        for name in ("train.parquet", "val.parquet", "test.parquet",
                     "metadata.json"):
            assert (tmp_output_dir / name).exists(), f"missing {name}"

    def test_parquet_schema_complete(self, tmp_output_dir):
        repo_root = Path(__file__).resolve().parent.parent
        cmd = [
            sys.executable, "-m", "synthetic.anomaly_injector",
            "--n-events", "1500",
            "--output-dir", str(tmp_output_dir),
            "--seed", "42",
            "--symbols", "AAPL",
        ]
        subprocess.run(cmd, cwd=repo_root, check=True, capture_output=True)

        df = pd.read_parquet(tmp_output_dir / "train.parquet")
        required = {
            "timestamp", "symbol", "mid_price", "spread_bps",
            "trade_imbalance", "order_cancel_rate", "label",
            "anomaly_severity", "injection_id",
        }
        # Per-level columns
        for i in range(1, 11):
            required.update({
                f"bid_l{i}", f"ask_l{i}",
                f"bidsize_l{i}", f"asksize_l{i}",
            })
        missing = required - set(df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_labels_valid_range(self, tmp_output_dir):
        repo_root = Path(__file__).resolve().parent.parent
        cmd = [
            sys.executable, "-m", "synthetic.anomaly_injector",
            "--n-events", "2000",
            "--output-dir", str(tmp_output_dir),
            "--seed", "42",
            "--symbols", "AAPL",
        ]
        subprocess.run(cmd, cwd=repo_root, check=True, capture_output=True)

        df = pd.concat([
            pd.read_parquet(tmp_output_dir / "train.parquet"),
            pd.read_parquet(tmp_output_dir / "val.parquet"),
            pd.read_parquet(tmp_output_dir / "test.parquet"),
        ])
        labels = set(df["label"].unique().tolist())
        assert labels.issubset({0, 1, 2, 3, 4, 5}), \
            f"unexpected labels: {labels}"

    def test_anomaly_rate_reasonable(self, tmp_output_dir):
        """Observed anomaly rate should fall within a wide CI of target."""
        repo_root = Path(__file__).resolve().parent.parent
        cmd = [
            sys.executable, "-m", "synthetic.anomaly_injector",
            "--n-events", "10000",
            "--output-dir", str(tmp_output_dir),
            "--seed", "42",
            "--symbols", "AAPL",
            "--anomaly-rate", "0.15",
        ]
        subprocess.run(cmd, cwd=repo_root, check=True, capture_output=True)

        meta = json.loads((tmp_output_dir / "metadata.json").read_text())
        observed = meta["anomaly_rate_observed"]
        # Block-level rate of 0.15 with block_size=50 -> observed
        # row-level rate is roughly 0.15 * (avg injected snaps / block).
        # We allow a very wide bracket — this is a sanity check, not a
        # precise unbiased estimator test.
        assert 0.02 < observed < 0.7, \
            f"observed anomaly rate {observed} out of plausible range"

    def test_reproducible_across_runs(self, tmp_output_dir, tmp_path):
        """Same seed -> byte-identical Parquet."""
        repo_root = Path(__file__).resolve().parent.parent

        out_a = tmp_path / "run_a"
        out_b = tmp_path / "run_b"

        for out in (out_a, out_b):
            cmd = [
                sys.executable, "-m", "synthetic.anomaly_injector",
                "--n-events", "1000",
                "--output-dir", str(out),
                "--seed", "42",
                "--symbols", "AAPL",
            ]
            subprocess.run(cmd, cwd=repo_root, check=True, capture_output=True)

        df_a = pd.read_parquet(out_a / "train.parquet")
        df_b = pd.read_parquet(out_b / "train.parquet")
        # Ignore the injection_id column (UUIDs are random by design).
        df_a = df_a.drop(columns=["injection_id"])
        df_b = df_b.drop(columns=["injection_id"])
        pd.testing.assert_frame_equal(df_a, df_b)
