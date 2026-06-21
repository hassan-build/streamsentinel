"""
evaluation/baselines/unimodal_gnn.py
====================================
Unimodal GNN baseline — the full GNN encoder + classifier, with NO
FinBERT and NO fusion.

This baseline answers the dissertation question:
    "Does fusing text with the graph improve detection?"

If the unimodal GNN matches or exceeds the full StreamSentinel system,
the cross-attention fusion is dead weight. This is the most important
baseline for the dissertation — it directly isolates the headline
architectural contribution.

Implementation strategy
-----------------------
Rather than duplicate the GNN/scoring code, we re-use `FullPipeline`
with `use_text=False`. This guarantees the unimodal baseline shares
the exact same GNN encoder and classifier head as the full system —
the only difference is whether text is fused in.

This deliberately matches the `no_llm` ablation. The naming is
historical:
  - "no_llm ablation" = full system with text disabled
  - "unimodal GNN baseline" = same thing, named for the baselines chapter
We keep BOTH names because the dissertation refers to them in two
different contexts (baselines table vs ablation table).
"""

from __future__ import annotations

from typing import Sequence

from models.full_pipeline import FullPipeline, FullPipelineConfig


def build_unimodal_gnn(
    pipeline_config: FullPipelineConfig | None = None,
) -> FullPipeline:
    """
    Build a FullPipeline configured as a unimodal GNN baseline.

    All weights are randomly initialised; train it with
    `models/train.py --no-text` (or equivalent) before evaluating.

    Parameters
    ----------
    pipeline_config : FullPipelineConfig, optional
        Base config. The `use_text` flag is forced to False before
        constructing the pipeline. Provide your standard config so
        the GNN architecture matches the full system exactly.

    Returns
    -------
    FullPipeline
        With `use_text=False`. Forward passes ignore any headlines.
    """
    cfg = pipeline_config or FullPipelineConfig()
    cfg.use_text = False
    return FullPipeline(cfg)
