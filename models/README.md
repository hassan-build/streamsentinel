# Models — The AI Layer

This module implements the four neural components of StreamSentinel and
the training script that ties them together. It is the heart of the
dissertation contribution: a multi-modal architecture that fuses
**graph-structured order book features** with **text embeddings of
financial news** to detect market manipulation events.

---

## Architecture Overview

```
              ┌──────────────────────┐
              │   PyG Graph (Data)    │  ← from graph/graph_builder.py
              │   x:[N,10] ei:[2,E]   │
              └──────────┬────────────┘
                         │
                         ▼
              ┌──────────────────────┐
              │   GNNEncoder (GAT)   │  models/gnn_encoder.py
              │   per-node embedding │
              └──────────┬────────────┘
                         │  z_graph:[N, 64]
                         ▼
   ┌──────────────────────────────────────────────────┐
   │              FusionModule (cross-attn)            │  models/fusion_module.py
   │                                                   │
   │     Q = z_graph    K, V = z_text                  │
   │     output: per-node multi-modal embedding         │
   └──────────────────┬───────────────────────────────┘
                      │  z_fused:[N, 128]
                      ▼
              ┌──────────────────────┐
              │  AnomalyScorer head  │  models/anomaly_scorer.py
              │   6-class softmax    │
              │   + adaptive CUSUM   │
              └──────────────────────┘
                      ▲
                      │  z_text:[T, 768]
              ┌──────────────────────┐
              │  FinBERTEncoder      │  models/finbert_encoder.py
              │  recent news embedding│
              └──────────────────────┘
                      ▲
                      │  text input
              [news headlines (last 5 min)]
```

---

## Component-by-component

### 1. `gnn_encoder.py` — GAT
Multi-head Graph Attention Network (Veličković et al. 2018) that turns
a graph snapshot into a per-node embedding. We use GAT (not vanilla GCN
or GraphSAGE) because attention weights are inherently interpretable —
they tell us *which neighbour* mattered most for a given prediction.
This is invaluable for the SHAP/explainability chapter.

**Input:** `Data(x:[N,10], edge_index:[2,E], edge_attr:[E,2])`
**Output:** `[N, gnn_out_dim]` per-node embedding (default 64-d).
**Layers:** 3 GAT layers with residual connections + LayerNorm.

### 2. `finbert_encoder.py` — FinBERT wrapper
Thin wrapper around `ProsusAI/finbert` from HuggingFace. Tokenises a
batch of headlines, runs a forward pass, and pools the resulting
contextual embeddings into a single 768-d "market mood" vector.

**Pooling strategy:** mean of `[CLS]` tokens across all news items in
the window. We chose mean over max-pool because a single sensational
headline shouldn't dominate. We also cache the last embedding so we
don't re-run BERT every snapshot — BERT inference is the slowest part
of the pipeline.

**No fine-tuning required.** FinBERT is already pre-trained on financial
news. We freeze its weights and let the fusion module learn to use them.

### 3. `fusion_module.py` — Cross-Attention Fusion
The core dissertation contribution. Combines the GNN's per-node
embeddings with FinBERT's text embedding via **cross-attention**:
each graph node queries the text embedding for relevant information.

Why cross-attention vs concatenation? Concatenation gives every node
the same text vector, but different stocks react to different news.
Cross-attention lets each node *attend* to the most relevant parts
of the text representation, which is exactly the inductive bias we want.

**Input:** `(z_graph: [N, 64], z_text: [1, 768])`
**Output:** `z_fused: [N, 128]`

### 4. `anomaly_scorer.py` — Classifier + Adaptive CUSUM
A two-layer MLP head producing a 6-class softmax (normal + 5 anomaly
types). Wraps an **adaptive CUSUM** detector that turns the per-class
probabilities into a binary anomaly decision with a *drift-aware*
threshold.

Why CUSUM? A fixed threshold of e.g. 0.65 works for a static market
but fails when market regimes shift (e.g. volatility doubles overnight).
CUSUM tracks cumulative deviation from a running mean and triggers
when the deviation exceeds an automatic decision bound — adapting
naturally to drift. This is the **adaptive thresholding** ablation
hook in the dissertation evaluation.

### 5. `full_pipeline.py`
End-to-end nn.Module that wires all four components. Used by:
  - `train.py` for batch training,
  - `api/fastapi_service.py` for live inference,
  - `evaluation/ablation.py` for ablation runs.

### 6. `train.py` — Training Script
CLI entrypoint:

```bash
python -m models.train \
  --data-dir data/synthetic \
  --epochs 30 \
  --batch-size 32 \
  --lr 1e-3
```

Loads the train/val Parquet files, constructs graph windows in a
PyTorch DataLoader, runs an AdamW optimiser with cosine annealing,
logs everything to MLflow, and saves the best checkpoint to
`checkpoints/best_model.pt`.

---

## Why these design choices?

| Decision | Why |
|---|---|
| GAT (vs GCN/SAGE) | Attention weights = built-in interpretability for SHAP chapter |
| 3 GNN layers | 1 = neighbours only, 2 = good, 3 = good for 5–10 nodes. Deeper = oversmoothing. |
| Mean-pool [CLS] | Less sensitive to single headlines than max-pool |
| Frozen FinBERT | No labelled financial-news data to fine-tune on; out-of-the-box already good |
| Cross-attention fusion | Different stocks need different parts of the news — inductive bias |
| Adaptive CUSUM | Static threshold fails on regime shift; demanded by dissertation evaluation |
| AdamW + cosine LR | Standard for transformer-adjacent training; reproducible default |
| Class weights = balanced | Synthetic data is 95% normal; balanced weights compensate |

---

## Ablation Hooks (built-in, not bolt-on)

Each ablation in `config.yaml > evaluation.ablations` maps to a single
boolean flag on `FullPipeline`:

```python
pipeline = FullPipeline(config)
pipeline.use_text = False             # "no_llm" ablation
pipeline.use_dynamic_graph = False    # "static_graph" ablation
pipeline.use_adaptive_cusum = False   # "fixed_threshold" ablation
```

The evaluation runner toggles these flags between runs. No code is
duplicated across ablations — only flag combinations.

---

## Tests

```bash
pytest tests/test_models.py -v
```

The test suite verifies (32 tests):
  - Each model is constructable and has a runnable `forward`
  - Output shapes are exactly as documented
  - The full pipeline produces well-formed logits
  - CUSUM detects synthetic drift while ignoring stable signal
  - Ablation flags actually disable their respective components
  - `train.py` runs end-to-end on a tiny dataset (1-epoch smoke test)
  - Reproducibility: same seed -> identical loss curve
