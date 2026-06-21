# StreamSentinel — Data Directory

This directory is intentionally **mostly empty** in the repository. Raw and
processed data files are excluded from version control via `.gitignore` to
avoid committing large binary files and proprietary market data.

---

## Directory Layout

```
data/
├── README.md            ← this file
├── delta_lake/          ← created at runtime by the Spark pipeline
│   ├── raw/
│   │   ├── ticks/
│   │   ├── orderbook/
│   │   └── news/
│   ├── features/
│   │   └── engineered/
│   └── results/
│       └── anomalies/
├── synthetic/           ← created by synthetic/anomaly_injector.py
│   ├── train/
│   ├── val/
│   └── test/
└── samples/             ← small (<1 MB) sample files committed for testing
    ├── sample_ticks.json
    ├── sample_orderbook.json
    └── sample_news.json
```

---

## Data Acquisition Guide

### 1. Live Market Data — Alpaca Markets (Free)

Alpaca provides free paper-trading websocket access for real-time tick data
and limited historical data.

```bash
# Set credentials in .env, then run:
python ingestion/kafka_producer.py --source alpaca --mode live
```

**Free tier limits:** IEX feed only (subset of trades). No Level 2 order
book on free tier — use synthetic L2 data for dissertations.

**Alternative:** Download historical minute bars via the REST API:
```python
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
```

---

### 2. Order Book Data

Real L2 order book data requires a paid Alpaca subscription or Polygon.io
Starter plan (≈$29/month).

**For dissertation use:** The `synthetic/anomaly_injector.py` module generates
realistic L2 order book snapshots with injected anomalies. These are the
primary training data source.

---

### 3. News Data — NewsAPI (Free Tier)

- 100 requests/day on the free tier
- 1-month lookback on free tier
- Set `NEWSAPI_KEY` in `.env`

```bash
python ingestion/kafka_producer.py --source newsapi --mode batch
```

---

### 4. GDELT (Free, Unlimited)

GDELT is a free global news event database updated every 15 minutes.
No API key required.

```bash
python ingestion/kafka_producer.py --source gdelt --mode live
```

---

## Synthetic Data (Primary Training Source)

Since ground-truth manipulation labels are scarce, synthetic data is the
**primary training and evaluation dataset**. The injector in
`synthetic/anomaly_injector.py` produces fully labelled datasets.

```bash
# Generate 100k labelled events with default scenario mix
python synthetic/anomaly_injector.py \
    --n-events 100000 \
    --output-dir data/synthetic \
    --seed 42
```

Output format: Parquet files with schema:
```
timestamp, symbol, features..., label, anomaly_type, injection_params
```

---

## Reproducibility

To regenerate all training data from scratch:

```bash
python synthetic/anomaly_injector.py --seed 42 --n-events 100000
```

The `--seed 42` flag guarantees identical data generation on any machine.
This seed is also logged to MLflow so every experiment is traceable to its
exact training data.
