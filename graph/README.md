# Graph Module

This module converts streaming order book features into **dynamic graph
structures** consumable by the GNN encoder. It is the bridge between the
tabular per-symbol features and the Graph Neural Network in the AI layer.

---

## Why model the market as a graph?

Financial markets exhibit strong cross-asset structure that flat feature
vectors cannot capture:

- **Co-movement.** Equities in the same sector move together; one stock
  leading another by milliseconds is a common signal.
- **Manipulation contagion.** Coordinated trading and layering attacks
  often span multiple correlated symbols simultaneously — the smoking
  gun is the synchrony, not the activity on any single asset.
- **Liquidity propagation.** A liquidity shock in SPY immediately
  affects every S&P 500 constituent; a model that sees each ticker in
  isolation will miss this entirely.

Representing the asset universe as a graph lets the GNN encoder attend
to neighbours during message passing, so per-symbol predictions can
incorporate context from related assets. This is the architectural
contribution evaluated against the unimodal-GNN baseline in the
ablation study (see `evaluation/ablation.py`).

---

## Two-layer architecture

```
┌──────────────────────────┐
│      GraphBuilder        │   Stateless: snapshot in -> Data object
│   (graph_builder.py)     │   Use for: training on historical data
└────────────┬─────────────┘
             │
             ▼
┌──────────────────────────┐
│   DynamicGraphUpdater    │   Stateful: rolling window of history
│ (dynamic_graph_updater.py)   Use for: live streaming inference
│                          │   Has freeze() for static-graph ablation
└──────────────────────────┘
```

`GraphBuilder` is a pure function — given a complete window of order
book history, it returns a PyG `Data` object. It is what the training
loop calls millions of times.

`DynamicGraphUpdater` wraps the builder with a rolling buffer of recent
history. New snapshots stream in; the updater emits an updated graph at
configurable intervals (default 1 second). This is what runs inside the
live inference service.

---

## Graph schema

### Nodes
One node per tracked symbol. The set of symbols is fixed at construction
(matches `config.yaml > data_sources.alpaca.symbols`).

### Node features (10-dimensional vector per node)

| Index | Feature | Description |
|---|---|---|
| 0 | `log_return_1` | Log-return of mid-price over last snapshot |
| 1 | `log_return_5` | Log-return of mid-price over last 5 snapshots |
| 2 | `spread_bps` | Bid-ask spread in basis points |
| 3 | `trade_imbalance` | Recent buy/sell volume imbalance, [-1, 1] |
| 4 | `order_cancel_rate` | Cancellations per second |
| 5 | `depth_imbalance_top5` | (∑bid_size - ∑ask_size) / (∑bid+ask), top 5 levels |
| 6 | `depth_weighted_price_dev` | (DWP - mid) / mid; DWP = price weighted by L1-L5 size |
| 7 | `volatility_rolling` | Rolling std of log_return_1 over the window |
| 8 | `mid_zscore` | Z-score of current mid vs window mean |
| 9 | `cancel_rate_zscore` | Z-score of current cancel rate vs window mean |

All node features are **standardised per symbol within each window** so
that different price scales (AAPL ≈ $175 vs SPY ≈ $540) don't dominate.

### Edges
**Undirected**, weighted, with self-loops added.

An edge `(i, j)` exists if the **Pearson correlation** of the two
symbols' `log_return_1` series over the rolling window exceeds the
configured `edge_threshold` (default 0.3, from `config.yaml > graph`).

### Edge features (2-dimensional vector per edge)

| Index | Feature | Description |
|---|---|---|
| 0 | `correlation` | Pearson correlation in [-1, 1] |
| 1 | `correlation_sign` | +1 if positive, -1 if negative, 0 if zero |

Self-loops have `correlation=1.0, sign=+1`.

---

## Static vs Dynamic graphs (ablation hook)

The dissertation's ablation study isolates the contribution of
**dynamic** graph updating. To support this without duplicating code:

```python
updater = DynamicGraphUpdater(config)
updater.freeze()       # graph topology + edges locked to current state
# subsequent .update() calls only refresh node features
updater.unfreeze()     # back to fully dynamic
```

When frozen, node features still update with each snapshot (so the GNN
still has fresh signals), but the edge structure is fixed to whatever
topology existed at the moment of `freeze()`. This matches the "static
graph" ablation in `config.yaml > evaluation.ablations.static_graph`.

---

## Usage

### Static use (training)

```python
import pandas as pd
from graph.graph_builder import GraphBuilder

# DataFrame columns: timestamp, symbol, mid_price, spread_bps, ...
# (the schema produced by synthetic/anomaly_injector.py)
df = pd.read_parquet("data/synthetic/train.parquet")

builder = GraphBuilder(
    symbols=["AAPL", "MSFT", "TSLA", "SPY", "NVDA"],
    edge_threshold=0.3,
    window_size=300,           # snapshots used for feature/correlation calc
)

# Build a graph from a sliding window ending at some timestamp.
window_df = df[df["timestamp"] <= some_ts].tail(300 * 5)  # 300 per symbol
data = builder.build(window_df)
# data is a torch_geometric.data.Data object:
#   data.x          -> [n_symbols, 10]    node features
#   data.edge_index -> [2, n_edges]       graph topology
#   data.edge_attr  -> [n_edges, 2]       edge features
```

### Dynamic use (streaming)

```python
from graph.dynamic_graph_updater import DynamicGraphUpdater

updater = DynamicGraphUpdater(
    symbols=["AAPL", "MSFT", "TSLA", "SPY", "NVDA"],
    edge_threshold=0.3,
    window_size=300,
    update_interval_ms=1000,
)

# As snapshots arrive from Kafka:
for snapshot_dict in kafka_consumer:
    updater.ingest(snapshot_dict)
    if updater.should_emit(snapshot_dict["timestamp"]):
        graph = updater.current_graph()
        yield graph                       # feed to GNN encoder
```

---

## Testing

```bash
pytest tests/test_graph.py -v
```

The test suite verifies:
- Output is a valid PyG `Data` object
- Node and edge counts match symbol set
- Self-loops are present
- Edge index is symmetric (undirected)
- Threshold=1.0 produces only self-loops; threshold=0.0 fully-connected
- Per-symbol z-scoring is correct
- Frozen updater preserves edge_index across updates
- Sliding window updates emit graphs at the configured interval

---

## Performance notes

A single graph build for 5 symbols × 300 snapshots takes <2 ms on CPU.
For 50 symbols it scales to ~20 ms — well within the 500 ms end-to-end
latency budget. The graph builder is intentionally pure Python + numpy
+ torch tensor construction; no CUDA needed at this stage.
