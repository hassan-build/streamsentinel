# Evaluation Module

This module produces the **dissertation results chapter**: the tables,
plots, and numbers that quantitatively justify StreamSentinel's design
choices.

It contains:
  - Three **baseline detectors** for comparison
  - A comprehensive **metrics** library
  - An **ablation runner** that systematically disables each component
  - A top-level **CLI** (`run_evaluation.py`) that produces a single
    dissertation-ready CSV + JSON + PNGs

---

## Why each baseline?

Each baseline isolates a specific question.

### 1. Rule-Based Detector (`baselines/rule_based.py`)
A naive threshold over hand-engineered features (e.g. "flag if spread
in basis points exceeds 5σ from the rolling mean"). No training, no ML.

**Question it answers:** *Does a learned model actually help, or could
we get away with a simple threshold rule?* In the literature on market
manipulation detection (e.g. Cao et al. 2014), rule-based detectors
are the standard "is ML worth it?" baseline.

### 2. Random Forest (`baselines/random_forest.py`)
Sklearn's `RandomForestClassifier` trained on flattened per-snapshot
features — no graph structure, no news text. Strong tabular baseline.

**Question it answers:** *Does the graph structure (and the GNN) buy
us anything over a strong off-the-shelf tabular ML model?* If the GNN
doesn't beat the RF, the entire architecture is overengineered.

### 3. Unimodal GNN (`baselines/unimodal_gnn.py`)
The full GNN encoder + classifier, but **no FinBERT, no fusion**. Just
the graph branch end-to-end.

**Question it answers:** *Does fusing text with the graph (the headline
dissertation contribution) actually improve detection?* If the unimodal
GNN matches the full system, then the LLM fusion is dead weight.

These three baselines map directly to the three dissertation research
questions stated in `README.md`.

---

## Ablations

Defined declaratively in `config.yaml > evaluation.ablations`. The
ablation runner enumerates each and:
  1. Builds a `FullPipeline` with the relevant flags flipped
  2. Trains (or loads a pretrained checkpoint) on the train split
  3. Evaluates on the held-out test split
  4. Logs metrics to MLflow with the ablation name as a tag

| Ablation | Flag flipped | Tests hypothesis |
|---|---|---|
| `no_llm` | `use_text=False` | Does FinBERT fusion help? |
| `static_graph` | DynamicGraphUpdater frozen | Does dynamic graph updating help? |
| `fixed_threshold` | `use_adaptive_cusum=False` | Does adaptive CUSUM beat a fixed threshold? |
| `full_system` | All flags on | Baseline for comparison |

### Honest caveat

The `no_llm` and `static_graph` ablations only **differ in behaviour
from `full_system`** once the system is fed real news headlines and
sees a graph that genuinely evolves over time. In offline training
on Parquet files (where `headlines=None` and graphs are built from
the same windows for every ablation), these flags reduce to no-ops
and all four ablations train to identical weights.

The proper way to evaluate these ablations is:
  1. Train a single model on synthetic data (no headlines)
  2. Run inference on a stream that DOES include headlines and dynamic
     graph evolution (the `ingestion/` and `processing/` modules)
  3. Toggle the flags at inference time and compare metrics

This is the design the dissertation evaluation chapter will follow.
The runner supports it via `--skip-ablations` (train once) and then
running each flag combination at inference. We document this so the
examiner understands why ablation differences emerge at inference,
not at training time.

---

## Metrics

All implemented in `metrics.py`. See its docstrings for formulas.

### Classification quality
- **AUROC** — primary metric for the dissertation (binary anomaly vs not).
- **PR-AUC** — more informative than AUROC under heavy class imbalance.
- **F1-macro** — equal weight per class; favours systems that catch
  rare classes (spoofing, coordinated trading).
- **Per-class precision / recall** — required to discuss which anomaly
  types the model handles well or badly.

### Operational quality
- **Latency p50 / p95 / p99** — end-to-end inference time in ms.
- **Throughput** — events processed per second under sustained load.

### Trust quality
- **SHAP consistency variance** — repeat SHAP attribution n times on
  the same input with different background samples; report the variance
  of the top-k feature ranking. Lower variance = more reproducible
  explanations.

### Confidence
Every metric reports a **bootstrap 95% confidence interval** (default
1000 resamples). This is non-negotiable for first-class dissertation
marks — single point estimates with no uncertainty are not credible.

---

## Output

After running `python -m evaluation.run_evaluation`, you get:

```
evaluation/results/
├── dissertation_table.csv      ← The headline table
├── dissertation_table.json     ← Same data, machine-readable
├── per_model/
│   ├── full_system/
│   │   ├── confusion_matrix.png
│   │   ├── pr_curve.png
│   │   ├── roc_curve.png
│   │   ├── latency_histogram.png
│   │   └── metrics.json
│   ├── no_llm/...
│   ├── static_graph/...
│   ├── fixed_threshold/...
│   ├── rule_based/...
│   ├── random_forest/...
│   └── unimodal_gnn/...
└── summary.md                  ← Human-readable narrative
```

Every PNG can be dropped into your dissertation. The CSV is the
master table; the JSON is what `dashboard/streamlit_app.py` reads
to display live results.

---

## Usage

### Fast smoke test (~3 min on CPU)
```bash
python -m evaluation.run_evaluation --epochs 2 --quick
```

### Real evaluation (~3–5 hours on CPU; run overnight)
```bash
python -m evaluation.run_evaluation --epochs 30
```

### Single ablation
```bash
python -m evaluation.ablation --ablation no_llm --epochs 30
```

---

## Tests

```bash
pytest tests/test_evaluation.py -v
```

The suite verifies:
- Each baseline trains and predicts without error
- Each baseline produces outputs in the documented shape
- All metrics return values in their valid ranges
- Bootstrap CIs are computed correctly
- The ablation runner trains the configured set of models
- Output files are written and load back identically
