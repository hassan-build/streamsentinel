# Synthetic Data Module

This module generates labelled Level 2 (L2) order book sequences with injected
market manipulation events. It is the **primary training and evaluation data
source** for StreamSentinel.

---

## Why synthetic data?

Three reasons, each addressing a known limitation of real-world data:

1. **Ground truth labels do not exist publicly.** SEC enforcement actions
   describe manipulators by name but never release tick-level data. Academic
   datasets like LOBSTER are unlabelled.

2. **Free-tier APIs only provide Level 1.** Alpaca's IEX feed gives the best
   bid and ask but no depth. Spoofing and layering manifest at deeper levels
   of the book, so L2 data is essential.

3. **Reproducibility.** A fixed PRNG seed regenerates the exact dataset on
   any machine — examiners can verify dissertation results bit-for-bit.

**Tradeoff:** The model may overfit to generator-specific artefacts. We
mitigate this by (a) basing each anomaly pattern on cited academic
definitions, (b) randomising every parameter within published ranges, and
(c) building the base "normal" market using a geometric Brownian motion
fair-value process — a standard model in market microstructure (Cont &
Stoikov, 2010).

---

## Architecture

```
┌──────────────────────────┐
│   BaseMarketSimulator    │   Geometric Brownian motion fair value
│   (base_market.py)       │   + exponential depth profile
└────────────┬─────────────┘
             │  clean L2 snapshots
             ▼
┌──────────────────────────┐
│   AnomalyInjector(s)     │   One class per manipulation type:
│   (injectors/*.py)       │   spoofing, layering, flash_crash,
│                          │   coordinated_trading, liquidity_shock
└────────────┬─────────────┘
             │  mutated snapshots + labels
             ▼
┌──────────────────────────┐
│   Dataset Assembler      │   Picks injector with prob = anomaly_rate
│   (anomaly_injector.py)  │   Streams to Parquet
└──────────────────────────┘
```

---

## Anomaly Definitions

Each definition cites the paper or regulatory document it follows.

### 1. Spoofing
Placing large visible orders with the intent to cancel before execution,
creating false price pressure. Definition follows Dodd-Frank §747 and
Lee, Eom & Park (2013).

**Pattern:** A large bid order (5–20× normal size) appears at a tight
price, sits for 10–200 ms, then is cancelled. Trades typically occur on
the opposite side during the spoof's lifetime.

### 2. Layering
Multi-level variant of spoofing. The trader places fake orders at multiple
price levels on one side of the book simultaneously. Definition follows
the FCA Market Watch #57 (2017).

**Pattern:** 3–10 large orders at consecutive price levels appear within
~100 ms, all on the same side, none filled, all cancelled within 100–2000 ms.

### 3. Flash Crash
Sudden price collapse (>2%) followed by partial or full recovery within
seconds. Definition follows Kirilenko et al. (2017) on the 2010 flash crash.

**Pattern:** Mid-price drops 2–15% in 200 ms – 5 s, then recovers 50–100%
of the drop. Spread widens dramatically; depth on the affected side
evaporates.

### 4. Coordinated Trading
Multiple accounts placing same-direction orders within a narrow time window.
Definition follows Pirrong (2018) on cross-account manipulation.

**Pattern:** 3–10 distinct order events arriving within 50–500 ms, all on
the same side, mimicking a single large hidden trade.

### 5. Liquidity Shock
Sudden withdrawal of market depth. Not always manipulative but useful as a
detection target. Definition follows Easley, López de Prado & O'Hara (2012).

**Pattern:** 50–95% of orders at depth levels 2–10 cancel within a 500 ms –
10 s window. Best bid and ask may remain visually unchanged.

---

## Output Schema

Parquet file with the following columns:

| Column | Type | Description |
|---|---|---|
| `timestamp` | int64 | Millisecond epoch (UTC) |
| `symbol` | string | Asset ticker, e.g. AAPL |
| `mid_price` | float64 | (best_bid + best_ask) / 2 |
| `spread_bps` | float64 | (ask - bid) / mid × 10000 |
| `bid_l1` … `bid_l10` | float64 | Top-10 bid prices |
| `ask_l1` … `ask_l10` | float64 | Top-10 ask prices |
| `bidsize_l1` … `bidsize_l10` | float64 | Sizes at each bid level |
| `asksize_l1` … `asksize_l10` | float64 | Sizes at each ask level |
| `trade_imbalance` | float64 | (buy_vol − sell_vol) / total_vol, 100 ms window |
| `order_cancel_rate` | float64 | Cancels per second, 100 ms window |
| `label` | int8 | 0=normal, 1=spoofing, 2=layering, 3=flash_crash, 4=coordinated, 5=liquidity_shock |
| `anomaly_severity` | float64 | 0.0–1.0; severity scaling factor used at injection |
| `injection_id` | string | UUID grouping all rows from one anomaly event |

---

## Usage

Generate a 100k-row dataset:

```bash
python -m synthetic.anomaly_injector --n-events 100000 --output-dir data/synthetic --seed 42
```

Per-flag detail:

| Flag | Default | Description |
|---|---|---|
| `--n-events` | 100000 | Total number of rows (snapshots) |
| `--output-dir` | `data/synthetic` | Where to write Parquet files |
| `--seed` | 42 | PRNG seed; same seed → identical dataset |
| `--symbols` | from config.yaml | Comma-separated tickers |
| `--anomaly-rate` | from config.yaml | Fraction of windows with injection |
| `--split` | `0.7,0.15,0.15` | Train/val/test fractions |

Splits are written as separate Parquet files:
```
data/synthetic/
├── train.parquet
├── val.parquet
└── test.parquet
```

---

## Testing

```bash
pytest tests/test_synthetic.py -v
```

The test suite verifies:
- Each injector produces non-empty output
- Label values are in {0..5}
- Anomaly rate matches the configured target within binomial confidence
- Parquet round-trip preserves all columns
- Fixed seed produces byte-identical output across runs

---

## References

- **Cont, R. & Stoikov, S. (2010).** *A stochastic model for order book
  dynamics.* Operations Research, 58(3), 549–563.
- **Lee, E. J., Eom, K. S. & Park, K. S. (2013).** *Microstructure-based
  manipulation: Strategic behavior and performance of spoofing traders.*
  Journal of Financial Markets, 16(2), 227–252.
- **FCA (2017).** *Market Watch 57.* Financial Conduct Authority.
- **Kirilenko, A., Kyle, A. S., Samadi, M. & Tuzun, T. (2017).** *The Flash
  Crash: High-frequency trading in an electronic market.* Journal of
  Finance, 72(3), 967–998.
- **Easley, D., López de Prado, M. & O'Hara, M. (2012).** *Flow toxicity
  and liquidity in a high-frequency world.* Review of Financial Studies,
  25(5), 1457–1493.
- **Pirrong, C. (2018).** *Manipulation of price-reporting systems.*
  Journal of Financial Markets, 39, 71–94.
