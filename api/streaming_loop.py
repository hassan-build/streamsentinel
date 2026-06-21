"""
api/streaming_loop.py
=====================
The async background task that replaces Apache Spark in this dissertation.

Reads order-book snapshots from Kafka, maintains a per-symbol rolling
window, runs the FullPipeline once all symbols have enough data, and
writes scores to:

  1. Redis (key `latest:<symbol>`, TTL 5 min) — for instant dashboard reads
  2. Kafka topic `anomaly.scores` — for downstream/Delta-Lake persistence

For our data scale (≤10 events/sec in demo mode) a pure-Python loop
processes each prediction in <10 ms. Apache Spark would add JVM
overhead and a separate process to monitor without throughput benefit.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch

from logger import get_logger
from graph import GraphBuilder, GraphBuilderConfig
from models.anomaly_scorer import (
    ANOMALY_CLASSES,
    NORMAL_CLASS_IDX,
)
from models.full_pipeline import FullPipeline

from api.state import RedisClient, StatsBuffer

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class StreamingLoopConfig:
    """Streaming loop runtime parameters.

    Attributes
    ----------
    symbols : tuple[str, ...]
        Canonical symbol order.
    window_size : int
        How many ticks per symbol must accumulate before we run the GNN.
    stride : int
        Predict every `stride` ticks once the window is warm.
    correlation_min_samples : int
        Min observations for the GraphBuilder's correlation step.
    output_topic : str
        Kafka topic for anomaly scores.
    use_text : bool
        If True, headlines from `news.feed` are routed through FinBERT
        and fused with the GNN. If False, the loop ignores news records
        and calls the pipeline with `headlines=None` (faster, no text).
    news_window_seconds : int
        Discard news older than this when assembling per-symbol headlines.
    max_headlines_per_symbol : int
        Cap on the number of headlines passed to FinBERT per prediction.
    """
    symbols: tuple[str, ...]
    window_size: int = 60
    stride: int = 1
    correlation_min_samples: int = 30
    output_topic: str = "anomaly.scores"
    use_text: bool = False
    news_window_seconds: int = 300
    max_headlines_per_symbol: int = 5


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

class StreamingInferenceLoop:
    """
    Owns:
      - A per-symbol deque of recent ticks
      - A FullPipeline (shared with the synchronous /predict endpoint)
      - A StatsBuffer (so /stats reflects streaming activity too)

    Runs as `asyncio.create_task(loop.run_forever_async())`.
    """

    def __init__(
        self,
        pipeline: FullPipeline,
        graph_builder: GraphBuilder,
        config: StreamingLoopConfig,
        stats: StatsBuffer,
        redis_client: RedisClient,
        consumer: Any,
        producer: Any | None = None,
    ) -> None:
        self.pipeline = pipeline
        self.graph_builder = graph_builder
        self.config = config
        self.stats = stats
        self.redis = redis_client
        self.consumer = consumer
        self.producer = producer
        # Per-symbol rolling buffers, each holding the last `window_size+stride`
        # records (slight overhead so we can re-build dataframes safely).
        self._buffers: dict[str, list[dict[str, Any]]] = {
            s: [] for s in config.symbols
        }
        # Per-symbol news buffers. Each entry is (timestamp_ms, headline).
        # Old entries are discarded each time we read from the buffer.
        self._news_buffers: dict[str, list[tuple[int, str]]] = {
            s: [] for s in config.symbols
        }
        self._ticks_since_last_predict: int = 0
        self._shutdown = False

    # ------------------------------------------------------------------
    def request_shutdown(self) -> None:
        self._shutdown = True

    # ------------------------------------------------------------------
    async def run_forever_async(self, poll_timeout_s: float = 0.5) -> None:
        """The main loop. Yields to the asyncio scheduler regularly."""
        log.info(
            f"Streaming loop started. window={self.config.window_size}, "
            f"stride={self.config.stride}, "
            f"symbols={list(self.config.symbols)}"
        )
        while not self._shutdown:
            # Pull one batch's worth of messages.
            saw_record = False
            try:
                for rec in self.consumer.poll_forever(
                    poll_timeout_s=poll_timeout_s
                ):
                    if self._shutdown:
                        break
                    saw_record = True
                    self._handle_record(rec)
                    self.consumer.commit()
                    # Yield to the asyncio loop so HTTP requests get served.
                    await asyncio.sleep(0)
            except Exception as exc:  # noqa: BLE001
                log.warning(f"Loop iteration error: {exc}")
                await asyncio.sleep(1.0)

            # If poll_forever returned without yielding any records, yield
            # control to the event loop so shutdown signals + HTTP requests
            # can be serviced. Without this, an idle loop spin-blocks.
            if not saw_record:
                await asyncio.sleep(poll_timeout_s)

        log.info("Streaming loop shutting down.")

    # ------------------------------------------------------------------
    def _handle_record(self, rec: Any) -> None:
        """Route the record into the appropriate buffer; maybe predict."""
        value: dict[str, Any] = rec.value
        topic = getattr(rec, "topic", "") or ""

        # News records flow into the per-symbol news buffer.
        if topic.startswith("news") or "headline" in value:
            self._handle_news_record(value)
            return

        # Order-book records flow into the symbol buffer.
        sym = str(value.get("symbol", "?"))
        if sym not in self._buffers:
            # Unknown symbol — ignore.
            return
        self._buffers[sym].append(value)
        # Keep buffer bounded.
        keep = self.config.window_size + self.config.stride
        if len(self._buffers[sym]) > keep:
            self._buffers[sym] = self._buffers[sym][-keep:]

        # All symbols warm?
        if not all(len(buf) >= self.config.window_size
                   for buf in self._buffers.values()):
            return
        self._ticks_since_last_predict += 1
        if self._ticks_since_last_predict < self.config.stride:
            return
        self._ticks_since_last_predict = 0
        self._run_one_prediction()

    def _handle_news_record(self, value: dict[str, Any]) -> None:
        """Add a headline to the per-symbol news buffer for every
        symbol it mentions."""
        headline = str(value.get("headline", "")).strip()
        if not headline:
            return
        ts = int(value.get("timestamp", value.get("timestamp_ms", 0)))
        symbols = value.get("symbols") or []
        # Defensive: NewsAPI sometimes passes symbols as a string.
        if isinstance(symbols, str):
            symbols = [symbols]
        for sym in symbols:
            sym = str(sym).upper()
            if sym in self._news_buffers:
                self._news_buffers[sym].append((ts, headline))

    def _gather_headlines(self) -> list[str] | None:
        """
        Build the headline list for the current inference.

        Strategy: one representative headline per symbol (most recent),
        in canonical symbol order. Returns None if `use_text` is off OR
        no recent headlines exist for any symbol — in which case the
        pipeline will skip FinBERT entirely.
        """
        if not self.config.use_text:
            return None
        cutoff_ms = (time.time() - self.config.news_window_seconds) * 1000
        out: list[str] = []
        any_present = False
        for sym in self.config.symbols:
            # Filter out stale headlines.
            self._news_buffers[sym] = [
                (ts, h) for ts, h in self._news_buffers[sym]
                if ts >= cutoff_ms
            ]
            # Cap buffer length.
            if (len(self._news_buffers[sym])
                    > self.config.max_headlines_per_symbol):
                self._news_buffers[sym] = (
                    self._news_buffers[sym][
                        -self.config.max_headlines_per_symbol:
                    ]
                )
            if self._news_buffers[sym]:
                # Most recent first.
                latest = sorted(
                    self._news_buffers[sym], key=lambda x: -x[0]
                )[0][1]
                out.append(latest)
                any_present = True
            else:
                out.append("")
        return out if any_present else None

    # ------------------------------------------------------------------
    def _run_one_prediction(self) -> None:
        """Build a graph + run inference + cache results."""
        rows: list[dict[str, Any]] = []
        for sym, buf in self._buffers.items():
            window = buf[-self.config.window_size:]
            for r in window:
                # Avro decode produces dicts ready for the dataframe;
                # we just ensure label/severity are filled.
                row = dict(r)
                row.setdefault("label", 0)
                row.setdefault("anomaly_severity", 0.0)
                row.setdefault("injection_id", "")
                rows.append(row)
        df = pd.DataFrame(rows)

        # Build graph + forward pass.
        try:
            t0 = time.perf_counter()
            graph = self.graph_builder.build(df)
            headlines = self._gather_headlines()
            self.pipeline.eval()
            with torch.no_grad():
                logits = self.pipeline(graph, headlines=headlines)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            latency_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:  # noqa: BLE001
            log.warning(f"Inference failed: {exc}")
            return

        # Emit one record per symbol.
        n_anom = 0
        for i, sym in enumerate(self.config.symbols):
            class_probs = {
                ANOMALY_CLASSES[c]: float(probs[i, c])
                for c in range(probs.shape[1])
            }
            predicted_class = int(np.argmax(probs[i]))
            score = float(1.0 - probs[i, NORMAL_CLASS_IDX])
            is_anomaly = int(predicted_class != NORMAL_CLASS_IDX)
            n_anom += is_anomaly

            payload = {
                "symbol": sym,
                "timestamp_ms": int(time.time() * 1000),
                "anomaly_score": score,
                "predicted_class": ANOMALY_CLASSES[predicted_class],
                "predicted_class_idx": predicted_class,
                "class_probabilities": class_probs,
                "is_anomaly": bool(is_anomaly),
                "inference_latency_ms": float(latency_ms),
            }

            # Redis cache (instant dashboard reads).
            self.redis.set_latest(sym, payload)

            # Kafka score topic (downstream consumers).
            if self.producer is not None:
                try:
                    self.producer.produce(
                        topic=self.config.output_topic,
                        key=sym.encode("utf-8"),
                        value=json.dumps(payload).encode("utf-8"),
                    )
                except Exception as exc:  # noqa: BLE001
                    log.warning(f"Producer write failed: {exc}")

        self.stats.record(latency_ms, n_anom)
