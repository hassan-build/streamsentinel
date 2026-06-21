# Explainability Module

This module produces the **trust evidence** for the dissertation. It
answers three questions about every StreamSentinel prediction:

1. **Which input features drove this decision?** (SHAP attribution)
2. **Which neighbouring symbols did the GAT attend to?** (attention heatmaps)
3. **Are the explanations stable across repeated runs?** (consistency variance)

The third is non-obvious but matters: if running SHAP twice on the same
sample with different background data gives wildly different
attributions, the explanations are noise. Reporting variance alongside
the point estimates is what separates a dissertation's "we explained
the model" from "we trustworthily explained the model" (Bilodeau et al.
2024).

---

## What gets produced

Running `python -m explainability.run_explainability` writes to
`explainability/outputs/`:

```
outputs/
├── per_class_attributions.csv        # mean |SHAP value| per feature per class
├── per_class_attributions.png        # dissertation-ready bar chart
├── attention_aggregate.png           # mean attention across test set
├── attention_examples/
│   ├── example_001_spoofing.png      # one figure per representative case
│   └── ...
├── consistency.json                  # variance of top-k SHAP rankings
└── summary.md                        # narrative summary for the chapter
```

Every PNG can be dropped directly into the dissertation. The CSV is the
underlying data table.

---

## What's inside

| File | Purpose |
|---|---|
| `shap_explainer.py` | SHAP attribution + consistency analysis |
| `attention_visualiser.py` | GAT attention heatmap rendering |
| `run_explainability.py` | CLI: produces all figures with one command |

### `shap_explainer.py`

Uses **KernelSHAP** (Lundberg & Lee 2017), not DeepSHAP. Two reasons:

1. **Model-agnostic.** Our pipeline has a graph-message-passing layer
   that DeepSHAP's tensor-rewriting approach doesn't handle cleanly.
2. **Treats the model as a black box.** SHAP perturbs the input feature
   vector and observes how predictions change. This matches the
   inductive bias of "what would change if this feature were different?"
   that we want for the dissertation discussion.

Output: per-feature, per-class attribution scores. We attribute at the
**named node-feature level** (the 10 features defined in
`graph/graph_builder.py: FEATURE_NAMES`) — not at lower granularities,
because dissertation chapters benefit from a small number of
interpretable features over hundreds of opaque ones.

### `attention_visualiser.py`

The GAT encoder exposes `get_attention_weights()` which returns
per-edge, per-head attention scores. The visualiser produces:

- **Per-prediction heatmaps.** "AAPL was flagged spoofing because it
  attended strongly to SPY and MSFT in this window."
- **Aggregate heatmaps.** Mean attention across the full test set —
  shows the learned market structure.

This is unique to the GAT architecture — most GNN variants (GCN,
GraphSAGE) don't have explicit attention weights to extract. It's the
single strongest interpretability argument for choosing GAT over
alternatives, and dissertation examiners often ask about it.

### `run_explainability.py`

The one-command CLI. Loads a trained checkpoint (default:
`checkpoints/best_model.pt`), runs both SHAP and attention analyses on
the test set, and writes all the figures + tables.

```bash
# Fast smoke (~5 min on CPU)
python -m explainability.run_explainability --n-samples 50 --n-trials 3

# Dissertation-final run (~30-60 min)
python -m explainability.run_explainability --full
```

---

## Consistency metric

For each test sample we record SHAP attribution `n_trials` times with
different randomly-drawn background samples. For each rank position 1..k
in the top-k features, we compute the variance of the feature indices
that appeared there. The aggregate variance is reported as
**SHAP consistency variance** (lower = more stable).

This is the metric defined in `evaluation/metrics.py:
shap_consistency_variance` — we just feed it real data here.

---

## Honest limitations (cite these in the dissertation)

1. **KernelSHAP is approximate.** It samples coalitions of features; with
   small sample sizes, attributions vary. We report variance to make this
   visible.
2. **Attention weights are NOT explanations.** Recent work (Jain & Wallace
   2019; Bilodeau et al. 2024) shows attention can be manipulated without
   changing predictions. We treat attention as a *behavioural probe*
   rather than a causal claim, and report SHAP consistency alongside.
3. **Attribution to engineered features only.** We don't attribute at the
   raw-tick level. A model that uses a misleading proxy will look
   "well-explained" even when the underlying mechanism is wrong. This is
   a known limitation of post-hoc explanation.

These three caveats belong in the dissertation Discussion. They show
the examiner you've thought critically about explainability, which is
what differentiates a first-class submission.

---

## Tests

```bash
pytest tests/test_explainability.py -v
```

Verifies:
- SHAP explainer produces correctly-shaped attribution matrices
- Per-class aggregation respects label boundaries
- Attention extraction works on the trained GAT
- Consistency variance behaves as expected (low for stable input,
  high for noisy)
- The CLI runs end-to-end on a tiny model
