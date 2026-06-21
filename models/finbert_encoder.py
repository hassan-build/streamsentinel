"""
models/finbert_encoder.py
=========================
Thin wrapper around ProsusAI/FinBERT that encodes a batch of financial
news headlines into a single "market mood" embedding.

Design choices
--------------
- **Frozen weights.** We do NOT fine-tune FinBERT. The model was already
  pre-trained on financial corpora (Araci 2019), and we have no labelled
  news-to-manipulation pairs to fine-tune on. Treating it as a fixed
  feature extractor is the honest choice and the standard one in
  similar academic settings.

- **Mean pooling of [CLS] tokens.** Each headline produces one [CLS]
  embedding. We average them. Max-pool was considered but discarded
  because one sensational outlier headline shouldn't dominate the
  market-context vector.

- **Cached output.** FinBERT inference is the slowest part of the
  pipeline (~50 ms per batch on CPU). Since news doesn't arrive every
  100 ms, we cache the last embedding and re-use it until new text is
  fed in. This is critical for hitting the 500 ms end-to-end target.

- **Offline mode.** When the model can't be downloaded (no internet,
  HuggingFace down, etc.), the encoder transparently falls back to a
  random-but-deterministic embedding so the rest of the pipeline still
  trains. This is a development convenience, not a research claim —
  the dissertation evaluation always uses the real FinBERT.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn


@dataclass
class FinBERTEncoderConfig:
    """Configuration for `FinBERTEncoder`.

    Attributes
    ----------
    model_name : str
        HuggingFace model ID. Default `ProsusAI/finbert`.
    output_dim : int
        FinBERT hidden size. Default 768 (BERT-base).
    max_length : int
        Maximum tokens per headline. Default 128 — headlines are short.
    device : str
        "cuda" or "cpu". Falls back to cpu automatically.
    cache_dir : str | None
        Where HuggingFace caches the downloaded model. None = default.
    allow_offline_fallback : bool
        If True and the model can't load, fall back to a deterministic
        hash-based embedding. Useful for CI and offline development.
    """
    model_name: str = "ProsusAI/finbert"
    output_dim: int = 768
    max_length: int = 128
    device: str = "cpu"
    cache_dir: str | None = None
    allow_offline_fallback: bool = True


class FinBERTEncoder(nn.Module):
    """Encodes a list of headlines into a single `[1, output_dim]` vector."""

    EMPTY_TEXT_MARKER: str = "<no news>"

    def __init__(self, config: FinBERTEncoderConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or FinBERTEncoderConfig()

        # The HF model is loaded lazily on first use. This keeps unit
        # tests fast and avoids forcing every developer to download
        # ~440 MB before doing anything.
        self._hf_model: nn.Module | None = None
        self._hf_tokenizer = None
        self._using_fallback: bool = False

        # Embedding cache: maps content-hash -> last computed tensor.
        # When the same news batch is provided twice, we reuse.
        self._cache: dict[str, torch.Tensor] = {}

        # Freeze flag — set after the model is loaded so external
        # gradient zeroing still works correctly if the user opts in.
        self.freeze_weights: bool = True

    # ------------------------------------------------------------------
    # Lazy-load HF model
    # ------------------------------------------------------------------
    def _ensure_loaded(self) -> None:
        """Load FinBERT on first use; fall back to a deterministic stub
        if loading fails and `allow_offline_fallback` is True."""
        if self._hf_model is not None or self._using_fallback:
            return
        try:
            from transformers import AutoModel, AutoTokenizer
            self._hf_tokenizer = AutoTokenizer.from_pretrained(
                self.cfg.model_name, cache_dir=self.cfg.cache_dir
            )
            self._hf_model = AutoModel.from_pretrained(
                self.cfg.model_name, cache_dir=self.cfg.cache_dir
            )
            self._hf_model.to(self.cfg.device)
            if self.freeze_weights:
                for p in self._hf_model.parameters():
                    p.requires_grad_(False)
            self._hf_model.eval()
        except Exception as exc:
            if not self.cfg.allow_offline_fallback:
                raise
            # Soft fallback: deterministic hash embedding.
            self._using_fallback = True
            # Don't propagate the exception — but record it for debugging.
            self._fallback_reason = str(exc)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def forward(self, headlines: Sequence[str]) -> torch.Tensor:
        """
        Encode a batch of headlines into a single embedding.

        Parameters
        ----------
        headlines : sequence of str
            News headlines for the current window. May be empty — we
            return a zero-ish embedding plus the EMPTY_TEXT_MARKER hash
            so the rest of the pipeline still has a fixed-shape input.

        Returns
        -------
        torch.Tensor
            Embedding of shape `[1, output_dim]`. Has `requires_grad=False`
            because the encoder is frozen.
        """
        self._ensure_loaded()

        # Cache key on the joined hash of headlines (order-sensitive).
        joined = "\n".join(headlines) if headlines else self.EMPTY_TEXT_MARKER
        key = hashlib.sha1(joined.encode("utf-8")).hexdigest()
        if key in self._cache:
            return self._cache[key]

        if self._using_fallback:
            emb = self._fallback_embed(joined)
        else:
            emb = self._real_embed(headlines if headlines
                                   else [self.EMPTY_TEXT_MARKER])

        # Detach + non-trainable.
        emb = emb.detach()
        self._cache[key] = emb
        return emb

    def clear_cache(self) -> None:
        """Clear the embedding cache. Call when switching evaluation runs."""
        self._cache.clear()

    @property
    def is_offline(self) -> bool:
        """True if we are running in the deterministic-fallback mode."""
        return self._using_fallback

    # ------------------------------------------------------------------
    # Internal: real vs fallback embedding
    # ------------------------------------------------------------------
    def _real_embed(self, headlines: list[str]) -> torch.Tensor:
        """Run the real FinBERT forward pass."""
        assert self._hf_model is not None
        assert self._hf_tokenizer is not None
        with torch.no_grad():
            tokens = self._hf_tokenizer(
                list(headlines),
                padding=True,
                truncation=True,
                max_length=self.cfg.max_length,
                return_tensors="pt",
            )
            tokens = {k: v.to(self.cfg.device) for k, v in tokens.items()}
            outputs = self._hf_model(**tokens)
            # last_hidden_state: [batch, seq_len, hidden]
            # Take [CLS] = position 0 from each headline, then mean.
            cls_per_headline = outputs.last_hidden_state[:, 0, :]  # [B, H]
            pooled = cls_per_headline.mean(dim=0, keepdim=True)    # [1, H]
        return pooled

    def _fallback_embed(self, text: str) -> torch.Tensor:
        """Deterministic hash-based embedding for offline/CI use.

        We seed numpy from a SHA-1 of the input text, draw a normal
        vector, then unit-normalise. Same text -> same embedding.
        """
        import numpy as np
        h = hashlib.sha1(text.encode("utf-8")).digest()
        seed = int.from_bytes(h[:4], "big")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.cfg.output_dim).astype("float32")
        vec /= max(float((vec ** 2).sum() ** 0.5), 1e-9)
        return torch.from_numpy(vec).unsqueeze(0)   # [1, output_dim]
