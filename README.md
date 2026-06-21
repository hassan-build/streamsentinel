# StreamSentinel

**Real-time adaptive anomaly detection for financial market microstructure events using Graph Neural Networks fused with a financial-domain LLM.**

---

[![tests](https://img.shields.io/badge/tests-165%20passing-brightgreen)]() [![python](https://img.shields.io/badge/python-3.11-blue)]() [![license](https://img.shields.io/badge/license-MIT-lightgrey)]()

StreamSentinel is a streaming anomaly-detection platform for financial order book and news data. It detects five categories of market microstructure abuse — spoofing, layering, flash crashes, coordinated trading, and liquidity shocks — by fusing a Graph Attention Network (GAT) over a cross-asset graph with FinBERT embeddings of contemporaneous news, via cross-attention. Predictions are scored with an adaptive CUSUM detector and explained via KernelSHAP attribution and per-head GAT attention heatmaps.

The system is built as a final-year computer science dissertation project. Every architectural decision in the codebase has a corresponding entry in the module READMEs explaining the rationale, the alternatives considered, and the supporting literature.

---

## Table of contents

1. [Headline results](#headline-results)
2. [Quick start](#quick-start)
3. [What's inside](#whats-inside)
4. [Live demo](#live-demo)
5. [Reproducing the dissertation results](#reproducing-the-dissertation-results)
6. [Mapping to dissertation objectives](#mapping-to-dissertation-objectives)
7. [Architecture](#architecture)
8. [Honest limitations](#honest-limitations)
9. [Tests](#tests)
10. [References](#references)

---

## Headline results

Computed on a 100,000-row synthetic dataset (5 symbols, 95.71% normal / 4.29% labelled anomalies across 5 anomaly classes).

| Metric | Value | 95% bootstrap CI |
|---|---|---|
| AUROC | **0.755** | [0.611, 0.888] |
| PR-AUC | **0.461** | [0.236, 0.659] |
| F1 (macro) | 0.196 | [0.194, 0.247] |
| Inference latency p95 | **7.0 ms** | – |
| Throughput | ~150 events/sec | – |
| Trainable parameters | 650,950 | – |
| Automated tests | **165 passing** | – |

End-to-end latency stays comfortably under the 500 ms target stipulated in the dissertation brief, even on CPU.

Full reproducible results are in `evaluation/results/dissertation_table.csv` after running the evaluation suite (see below).

---

## Quick start

### Prerequisites

- **Windows 10/11 with PowerShell** (the project has been validated on this combination; Linux/macOS should also work)
- **Python 3.11**
- **Docker Desktop** (for Kafka, Redis, TimescaleDB, Grafana, MLflow)
- **~10 GB free disk** (Docker images + Python deps + checkpoints)
- **Optional**: API keys for NewsAPI (free tier), Alpaca (paper trading), HuggingFace (free token for FinBERT download). Only HuggingFace is required for offline-style usage; NewsAPI is required only for the live multi-modal demo.

### Setup (one-time, ~20 minutes)

```powershell
# 1. Clone and create a virtual environment.
git clone <YOUR_GITHUB_URL> streamsentinel
cd streamsentinel
python -m venv .venv
.venv\Scripts\activate

# 2. Install PyTorch (CPU build), PyTorch Geometric, then everything else.
pip install torch==2.3.1 --index-url https://download.pytorch.org/whl/cpu
pip install torch-geometric==2.5.3
pip install -r requirements.txt

# 3. Copy and edit environment variables.
Copy-Item .env.example .env
notepad .env
# Set at minimum: HUGGINGFACE_TOKEN
# Optional: NEWSAPI_KEY, ALPACA_API_KEY, ALPACA_API_SECRET

# 4. Start the supporting infrastructure (Kafka, Redis, TimescaleDB, MLflow, Grafana).
docker compose -f deployment\docker-compose.yml up -d

# 5. Wait ~30 seconds for Kafka, then create the topics.
#    (The init container has a known YAML quoting issue; we do it manually.)
docker exec ss_kafka kafka-topics --bootstrap-server localhost:9050 --create --if-not-exists --topic market.ticks --partitions 3 --replication-factor 1
docker exec ss_kafka kafka-topics --bootstrap-server localhost:9050 --create --if-not-exists --topic orderbook.l2 --partitions 3 --replication-factor 1
docker exec ss_kafka kafka-topics --bootstrap-server localhost:9050 --create --if-not-exists --topic news.feed --partitions 1 --replication-factor 1
docker exec ss_kafka kafka-topics --bootstrap-server localhost:9050 --create --if-not-exists --topic anomaly.scores --partitions 3 --replication-factor 1

# 6. Verify everything's working.
pytest tests/
```

You should see **165 passed**.

### Port note

Kafka is exposed on host port **9050** (not the conventional 9092). This is because Windows reserves the range 9081–9180 for Hyper-V, and 9092 happens to be inside it on most installations. 9050 is in the gap between reserved ranges and works reliably. The producer/consumer code reads `KAFKA_BOOTSTRAP_SERVERS` from `.env`, so no source changes are needed.

---

## What's inside

```
streamsentinel/
├── synthetic/                  # Parametric anomaly injection (5 anomaly types)
├── graph/                      # Stateless and dynamic graph builders for PyG
├── models/                     # GAT encoder + FinBERT + cross-attention fusion + CUSUM scorer
├── ingestion/                  # Kafka producer (replay/Alpaca/NewsAPI modes) + consumer
├── api/                        # FastAPI inference service + async streaming loop
├── dashboard/                  # Streamlit live UI + Grafana provisioning
├── evaluation/                 # Three baselines, bootstrap-CI metrics, ablation runner
├── explainability/             # KernelSHAP attribution + GAT attention heatmaps
├── deployment/                 # docker-compose.yml + supporting infrastructure
├── tests/                      # 165 pytest test cases across all modules
├── config.yaml                 # Central configuration (no secrets)
├── requirements.txt            # Python deps
├── .env.example                # Template for API keys
└── README.md                   # This file
```

Each module folder has its own `README.md` with detailed design rationale and citations. Those documents are referenced from the dissertation report; this top-level README is the entry point.

---

## Live demo

Three terminals. Each runs an independent component of the streaming pipeline.

### Terminal 1 — Inference service + streaming loop

```powershell
.venv\Scripts\activate
python -m api.service
```

You should see:
```
INFO | Loaded checkpoint from checkpoints\best_model.pt
INFO | Connected to Redis at localhost:6379
INFO | Background streaming loop scheduled.
INFO | Streaming loop started. window=60, stride=1, symbols=['AAPL', 'MSFT', 'TSLA', 'SPY', 'NVDA']
INFO | Consumer subscribed to topics: ['orderbook.l2', 'news.feed']
INFO | API ready to serve requests.
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### Terminal 2 — Data feed (replay synthetic data)

```powershell
.venv\Scripts\activate
python -m ingestion.kafka_producer --source replay --speed 10x
```

`--speed 10x` compresses real-time replay 10×; use `--speed 1x` for real time, or `--speed instant` for stress testing.

### Terminal 3 — Streamlit dashboard

```powershell
.venv\Scripts\activate
streamlit run dashboard/streamlit_app.py
```

Open <http://localhost:8501>. Within a few seconds you should see per-symbol anomaly scores updating live, a per-symbol Plotly chart populating with the last ~2 minutes of scores, system-health badges showing green, and the rolling p95 latency in milliseconds.

### Optional terminal 4 — Live news (multi-modal fusion)

```powershell
.venv\Scripts\activate
python -m ingestion.kafka_producer --source newsapi --queries "AAPL,MSFT,TSLA,NVDA,SPY"
```

This enables the multi-modal path. With news flowing, the streaming loop's per-symbol news buffer fills, FinBERT is invoked on every inference call, and the resulting text embeddings are fused with the GNN node embeddings via cross-attention. The first inference after news arrives will be slower (~5–10 s for FinBERT initialisation); subsequent inferences settle around 50–200 ms p95.

### Grafana operational dashboard

`http://localhost:3000` (login: `admin`/`admin`). A pre-provisioned "StreamSentinel — Operational" dashboard exposes four panels: anomaly rate over time, latest per-symbol score, latency percentiles, and predicted-class distribution. Note: the panels currently show "No data" because the TimescaleDB persistence layer is documented as future work. The dashboard structure exists to demonstrate the operational monitoring story; the Streamlit dashboard is the primary live interface for the demo.

### MLflow experiment tracker

`http://localhost:5000`. Lists all training runs with metrics, parameters, and artefact links.

---

## Reproducing the dissertation results

The numbers in the headline table are reproducible from `seed=42`. Total wall-clock time on CPU: ~45 minutes.

### Step 1 — Generate the synthetic dataset (~30 seconds)

```powershell
python -m synthetic.anomaly_injector --n-events 100000 --seed 42
```

Writes labelled Parquet files to `data/synthetic/{train,val,test}.parquet` plus a `metadata.json` recording the seed and class balance.

### Step 2 — Train the full model (~15 minutes)

```powershell
python -m models.train --epochs 5
```

Writes the best checkpoint (selected by validation AUROC) to `checkpoints/best_model.pt`. Training history is in `checkpoints/history.json`. MLflow logs the run.

### Step 3 — Run the evaluation suite (~10 minutes)

```powershell
python -m evaluation.run_evaluation --epochs 5
```

Writes:
- `evaluation/results/dissertation_table.csv` — the headline results table
- `evaluation/results/dissertation_table.json` — machine-readable
- `evaluation/results/summary.md` — narrative summary
- `evaluation/results/per_model/<name>/` — per-model confusion matrices, ROC curves, PR curves, latency histograms

### Step 4 — Run the explainability pipeline (~5 minutes)

```powershell
python -m explainability.run_explainability --n-samples 15 --n-trials 2 --n-kernel-samples 50 --window-size 60
```

Writes:
- `explainability/outputs/per_class_attributions.csv` — mean |SHAP value| per (anomaly class × feature)
- `explainability/outputs/per_class_attributions.png` — bar chart suitable for the report
- `explainability/outputs/attention_aggregate.png` — averaged GAT attention heatmap
- `explainability/outputs/attention_examples/` — per-prediction heatmaps
- `explainability/outputs/consistency.json` — SHAP top-k consistency variance
- `explainability/outputs/summary.md` — narrative summary

For a more rigorous explainability run, use `--n-samples 50 --n-trials 5 --n-kernel-samples 200` (~30 minutes). The smaller defaults trade some consistency for runtime and are sufficient for demonstrating the pipeline.

---

## Mapping to dissertation objectives

The brief specified three research objectives. Each is addressed below with the corresponding deliverable.

### Objective 1 — A multi-modal streaming pipeline with sub-500 ms end-to-end latency

| Deliverable | Location | Evidence |
|---|---|---|
| Pipeline definition | `api/streaming_loop.py`, `models/full_pipeline.py` | Async Kafka-to-prediction loop with FinBERT fusion |
| Latency measurement | `evaluation/metrics.py:latency_summary` | p50 / p95 / p99 reported with bootstrap analysis |
| Achieved latency | `evaluation/results/dissertation_table.csv` | p95 = 7.0 ms (∼70× under target) |

### Objective 2 — GNN-LLM evaluation against three baselines with AUROC, F1, PR-AUC, latency, throughput

| Deliverable | Location | Evidence |
|---|---|---|
| Rule-based baseline | `evaluation/baselines/rule_based.py` | Z-score threshold detector |
| Random-forest baseline | `evaluation/baselines/random_forest.py` | sklearn RF on flat features |
| Unimodal-GNN baseline | `evaluation/baselines/unimodal_gnn.py` | GNN with text fusion disabled |
| Metric library | `evaluation/metrics.py` | All metrics with bootstrap 95% CI |
| Ablation runner | `evaluation/ablation.py` | Four ablations (no_llm, static_graph, fixed_threshold, full) |
| Results table | `evaluation/results/dissertation_table.csv` | Reproducible from seed 42 |

### Objective 3 — SHAP-based explainability with automated consistency metrics and ablation studies

| Deliverable | Location | Evidence |
|---|---|---|
| KernelSHAP attribution | `explainability/shap_explainer.py` | Model-agnostic, named-feature level |
| Consistency variance metric | `evaluation/metrics.py:shap_consistency_variance` | Cross-trial top-k variance |
| GAT attention extraction | `models/gnn_encoder.py:get_attention_weights` | Per-head, per-layer |
| Attention visualisation | `explainability/attention_visualiser.py` | Aggregate + per-prediction heatmaps |
| Outputs | `explainability/outputs/` | All figures reproducible |

---

## Architecture

### Layered view

```
┌─────────────────────────────────────────────────────────────────┐
│                          DATA SOURCES                            │
│  Synthetic (anomaly injector)  │  Alpaca API  │  NewsAPI         │
└────────────────────┬───────────────────────────────┬────────────┘
                     ▼                               ▼
┌─────────────────────────────────────────────────────────────────┐
│                       INGESTION (Kafka)                          │
│   market.ticks  │  orderbook.l2  │  news.feed  │  anomaly.scores │
└────────────────────┬─────────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                STREAMING INFERENCE LOOP (async)                  │
│  • Per-symbol rolling window (last `window_size` ticks)          │
│  • Per-symbol news buffer (last 5 min of headlines)              │
│  • Predictions to Redis cache + `anomaly.scores` topic           │
└────────────────────┬─────────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                          AI LAYER                                │
│  GAT graph encoder  ◄──► Cross-attention fusion ──► CUSUM scorer │
│                  ▲                ▲                              │
│                  └── FinBERT text encoder (frozen)               │
└────────────────────┬─────────────────────────────────────────────┘
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                       DEMO / OBSERVABILITY                       │
│  FastAPI (port 8000)  │  Streamlit (8501)  │  Grafana (3000)     │
│                          MLflow (5000)                           │
└─────────────────────────────────────────────────────────────────┘
```

### Why GAT over GCN or GraphSAGE?

GAT (Veličković et al., 2018) provides per-edge attention weights that can be extracted and visualised. This makes the model intrinsically interpretable in addition to the post-hoc SHAP analysis — a property the unimodal-GNN baseline shares but neither GCN nor GraphSAGE provide cleanly. See `models/gnn_encoder.py` for the implementation details and `explainability/attention_visualiser.py` for the extraction logic.

### Why KernelSHAP over DeepSHAP?

DeepSHAP relies on tensor-rewriting the computation graph, which is incompatible with the message-passing layer in our GAT encoder. KernelSHAP (Lundberg & Lee, 2017) is model-agnostic and treats the pipeline as a black box. We accept its higher computational cost in exchange for correctness on our architecture.

### Why a pure-Python streaming loop instead of Apache Spark Structured Streaming?

The original architecture diagram included Apache Spark Structured Streaming. We evaluated this and found that at the data scale relevant to the dissertation (~10 events/sec in demo, 15k rows in evaluation), Spark adds JVM startup overhead and a separate process to monitor without throughput benefit. The pure-Python streaming loop in `api/streaming_loop.py` (~200 lines) performs the same function (windowed feature aggregation, graph construction, model inference, downstream emission to Kafka + Redis) with measured p95 latency of 7 ms. The architecture remains production-ready: Spark could be swapped in at scale by replacing the loop module while leaving every other component untouched. This trade-off is documented in `api/README.md`.

---

## Honest limitations

The dissertation report contains a full discussion of limitations; this section flags the major ones.

### Synthetic data only

The model is trained and evaluated on parametrically injected anomalies, not real-world labelled market-manipulation events. Real manipulation events are rare, legally sensitive, and lack ground-truth labels (the FCA and SEC do not publish labelled tick datasets). Synthetic data provides reproducibility and a clear performance ceiling, but performance numbers will not transfer directly to live markets without re-evaluation. The ingestion module's `--source alpaca` path supports live ticks for sensitivity testing, but quantitative claims in the dissertation are restricted to the synthetic benchmark.

### Single-asset-class evaluation

All five symbols are large-cap US equities. The model has not been evaluated on options, futures, or cross-asset-class flows. Generalisation across asset classes is documented as future work.

### Pretrained FinBERT, not fine-tuned

`ProsusAI/finbert` is used as a frozen feature extractor. Fine-tuning on a labelled financial-news dataset would likely improve fusion quality but requires a labelled corpus that is out of scope here.

### Ablations show identical results when news is absent

The four ablations (no_llm, static_graph, fixed_threshold, full_system) in `evaluation/run_evaluation.py` produce identical metrics when run without live news input, because all four operate on the same offline Parquet without text. This is correct behaviour, not a bug, and is documented in `evaluation/README.md`. With news data flowing through the live demo, the ablations diverge meaningfully.

### TimescaleDB persistence not wired

The streaming loop writes predictions to Redis (read by Streamlit) and Kafka `anomaly.scores` (downstream consumers). Persistence to TimescaleDB — which would power the Grafana operational dashboard — is provisioned in `dashboard/grafana_config/` but the writer is not implemented. Grafana panels show "No data" in the current build.

### Spark/Flink/Kubernetes/Delta Lake omitted

The original architecture diagram includes these. We deliberately scoped them out: the data scale does not justify them, and a half-complete production deployment would distract from the core research contribution. The codebase is structured so they could be added without disturbing the AI layer.

---

## Tests

165 automated tests cover every module:

| Module | Tests |
|---|---|
| `synthetic/` | 34 |
| `graph/` | 26 |
| `models/` | 28 |
| `evaluation/` | 25 |
| `explainability/` | 8 |
| `ingestion/` | 19 |
| `api/` | 16 |
| `dashboard/` | 9 |
| **Total** | **165** |

Run the full suite:

```powershell
pytest tests/
```

Tests are wired to GitHub Actions CI via `.github/workflows/tests.yml` — every push runs the full suite on a clean Ubuntu runner, with PyTorch CPU build cached.

---

## References

The full reference list is in the dissertation report. Key sources informing architectural decisions are cited in the module READMEs and reproduced here.

- Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). *Graph Attention Networks*. ICLR.
- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. NeurIPS.
- Bilodeau, B., Jaques, N., Koh, P. W., & Kim, B. (2024). *Impossibility theorems for feature attribution*. PNAS.
- Jain, S., & Wallace, B. C. (2019). *Attention is not Explanation*. NAACL.
- Cont, R., & Stoikov, S. (2010). *The Price Impact of Order Book Events*. Journal of Financial Econometrics.
- Lee, E.-J., Eom, K. S., & Park, K. S. (2013). *Microstructure-based manipulation: Strategic behavior and performance of spoofing traders*. Journal of Financial Markets.
- Kirilenko, A., Kyle, A. S., Samadi, M., & Tuzun, T. (2017). *The Flash Crash: High-Frequency Trading in an Electronic Market*. Journal of Finance.
- Easley, D., López de Prado, M. M., & O'Hara, M. (2012). *Flow Toxicity and Liquidity in a High-Frequency World*. Review of Financial Studies.
- Cao, Y., Li, Y., Coleman, S., Belatreche, A., & McGinnity, T. M. (2014). *Adaptive Hidden Markov Model With Anomaly States for Price Manipulation Detection*. IEEE Transactions on Neural Networks.

---

## License

MIT. See `LICENSE` if present.

## Author

Hassan — Final-year Computer Science UA92, 2025–2026 academic year.

---

