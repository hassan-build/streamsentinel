"""
tests/test_api.py
=================
Tests for the FastAPI service and streaming loop.

i use FastAPI's TestClient with mocked Kafka + Redis so the tests run
without real infrastructure. The streaming-loop tests exercise the
real loop with a mock consumer that yields synthetic records.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
from fastapi.testclient import TestClient

from api.model_loader import build_pipeline_from_config
from api.service import create_app
from api.state import RedisClient, StatsBuffer
from api.streaming_loop import (
    StreamingInferenceLoop,
    StreamingLoopConfig,
)
from config_loader import load_config
from graph import GraphBuilder, GraphBuilderConfig
from ingestion.kafka_consumer import IncomingRecord


# Use the symbols defined in config.yaml so test data matches what the
# app will look for during graph building.
SYMBOLS: tuple[str, ...] = tuple(
    load_config()["data_sources"]["alpaca"]["symbols"]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rows(n_per_sym: int = 60, anomaly_p: float = 0.1) -> list[dict[str, Any]]:
    """Generate `n_per_sym` order book rows for each canonical symbol."""
    rng = np.random.default_rng(0)
    rows: list[dict[str, Any]] = []
    for sym_idx, sym in enumerate(SYMBOLS):
        price = 100.0 + sym_idx * 50
        for t in range(n_per_sym):
            price *= float(np.exp(rng.normal(0, 0.001)))
            row: dict[str, Any] = dict(
                timestamp=t * 100, symbol=sym, mid_price=price,
                spread_bps=float(rng.uniform(1, 4)),
                trade_imbalance=float(rng.uniform(-0.3, 0.3)),
                order_cancel_rate=float(rng.uniform(15, 35)),
                label=0,
                anomaly_severity=0.0,
                injection_id="",
            )
            for lvl in range(1, 11):
                row[f"bid_l{lvl}"] = price - lvl * 0.01
                row[f"ask_l{lvl}"] = price + lvl * 0.01
                row[f"bidsize_l{lvl}"] = float(rng.uniform(50, 500))
                row[f"asksize_l{lvl}"] = float(rng.uniform(50, 500))
            rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Health / stats
# ---------------------------------------------------------------------------

class TestHealthAndStats:
    def test_health_returns_ok(self):
        app = create_app(enable_streaming=False, checkpoint_path=Path("nope"))
        with TestClient(app) as client:
            r = client.get("/health")
            assert r.status_code == 200
            body = r.json()
            assert body["status"] == "ok"
            assert body["model_loaded"] is False  # no checkpoint
            assert "predictions_made" in body

    def test_stats_endpoint_starts_at_zero(self):
        app = create_app(enable_streaming=False, checkpoint_path=Path("nope"))
        with TestClient(app) as client:
            r = client.get("/stats")
            assert r.status_code == 200
            body = r.json()
            assert body["n_predictions"] == 0
            assert body["n_anomalies_detected"] == 0
            assert body["latency_p95_ms"] == 0.0

    def test_latest_returns_dict(self):
        app = create_app(enable_streaming=False, checkpoint_path=Path("nope"))
        with TestClient(app) as client:
            r = client.get("/latest")
            assert r.status_code == 200
            assert isinstance(r.json(), dict)


# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_predict_returns_per_symbol_predictions(self):
        app = create_app(enable_streaming=False, checkpoint_path=Path("nope"))
        rows = _make_rows(n_per_sym=60)
        with TestClient(app) as client:
            r = client.post("/predict", json={"rows": rows})
            assert r.status_code == 200, r.text
            body = r.json()
            assert "predictions" in body
            assert "inference_latency_ms" in body
            assert body["inference_latency_ms"] > 0
            assert len(body["predictions"]) == len(SYMBOLS)
            for p in body["predictions"]:
                assert {"symbol", "anomaly_score", "predicted_class",
                        "class_probabilities", "is_anomaly"} <= set(p.keys())
                assert 0.0 <= p["anomaly_score"] <= 1.0

    def test_predict_empty_rows_rejected(self):
        app = create_app(enable_streaming=False, checkpoint_path=Path("nope"))
        with TestClient(app) as client:
            r = client.post("/predict", json={"rows": []})
            # FastAPI/Pydantic catches min_length=1; some versions
            # return 422, others 400.
            assert r.status_code in (400, 422)

    def test_predict_missing_columns_rejected(self):
        app = create_app(enable_streaming=False, checkpoint_path=Path("nope"))
        with TestClient(app) as client:
            r = client.post("/predict", json={"rows": [{"foo": "bar"}]})
            assert r.status_code == 400

    def test_stats_increment_after_predict(self):
        app = create_app(enable_streaming=False, checkpoint_path=Path("nope"))
        rows = _make_rows(n_per_sym=60)
        with TestClient(app) as client:
            client.post("/predict", json={"rows": rows})
            r = client.get("/stats")
            body = r.json()
            assert body["n_predictions"] >= 1
            assert body["latency_p50_ms"] > 0.0


# ---------------------------------------------------------------------------
# StatsBuffer (unit)
# ---------------------------------------------------------------------------

class TestStatsBuffer:
    def test_initial_state(self):
        s = StatsBuffer()
        snap = s.snapshot()
        assert snap["n_predictions"] == 0
        assert snap["latency_p50_ms"] == 0.0

    def test_record_and_snapshot(self):
        s = StatsBuffer()
        s.record(latency_ms=10.0, n_anomalies=1)
        s.record(latency_ms=20.0, n_anomalies=0)
        snap = s.snapshot()
        assert snap["n_predictions"] == 2
        assert snap["n_anomalies_detected"] == 1
        assert 9.0 < snap["latency_p50_ms"] < 21.0

    def test_thread_safety_smoke(self):
        """Hammer the buffer from multiple threads — no exceptions."""
        import threading
        s = StatsBuffer()

        def writer():
            for _ in range(200):
                s.record(1.0, 0)

        ts = [threading.Thread(target=writer) for _ in range(8)]
        for t in ts: t.start()
        for t in ts: t.join()
        assert s.snapshot()["n_predictions"] == 1600


# ---------------------------------------------------------------------------
# Redis (graceful fallback when no Redis is available)
# ---------------------------------------------------------------------------

class TestRedisGracefulFallback:
    def test_unreachable_redis_does_not_crash(self):
        # Port 1 should be closed everywhere.
        client = RedisClient(host="127.0.0.1", port=1)
        assert client.connected is False
        client.set_latest("AAPL", {"x": 1})         # no exception
        assert client.get_latest("AAPL") is None
        assert client.get_all_latest() == {}
        assert client.ping() is False


# ---------------------------------------------------------------------------
# Streaming loop (uses mock consumer)
# ---------------------------------------------------------------------------

class FakeConsumer:
    """Minimal consumer: yields each IncomingRecord exactly once across all
    poll_forever() calls, then yields nothing on subsequent calls."""

    def __init__(self, records: list[IncomingRecord]) -> None:
        self._pending = list(records)
        self.committed: int = 0
        self.done: bool = False

    def poll_forever(self, poll_timeout_s: float = 0.1):
        while self._pending:
            yield self._pending.pop(0)
        self.done = True

    def commit(self, asynchronous: bool = True) -> None:
        self.committed += 1


class TestStreamingLoop:
    def test_loop_runs_inference_when_buffers_warm(self):
        """Push enough records for all symbols to trigger one prediction."""
        pipeline = build_pipeline_from_config()
        gb = GraphBuilder(GraphBuilderConfig(
            symbols=list(SYMBOLS), window_size=20,
            correlation_min_samples=10,
        ))
        stats = StatsBuffer()
        redis = RedisClient(host="127.0.0.1", port=1)  # offline -> no-ops

        rows = _make_rows(n_per_sym=25)
        records = [
            IncomingRecord(
                topic="orderbook.l2",
                partition=0, offset=i,
                key=row["symbol"].encode(),
                value=row, timestamp_ms=int(row["timestamp"]),
            )
            for i, row in enumerate(rows)
        ]
        consumer = FakeConsumer(records)
        loop = StreamingInferenceLoop(
            pipeline=pipeline,
            graph_builder=gb,
            config=StreamingLoopConfig(symbols=SYMBOLS, window_size=20),
            stats=stats,
            redis_client=redis,
            consumer=consumer,
            producer=None,
        )

        async def _drive():
            task = asyncio.create_task(loop.run_forever_async(
                poll_timeout_s=0.01
            ))
            # Wait a moment for it to process all records.
            for _ in range(20):
                await asyncio.sleep(0.05)
                if consumer.done:
                    break
            loop.request_shutdown()
            try:
                await asyncio.wait_for(task, timeout=2.0)
            except asyncio.TimeoutError:
                task.cancel()

        asyncio.run(_drive())

        # We expect at least one prediction to have been made.
        assert stats.snapshot()["n_predictions"] >= 1
        assert consumer.committed == len(records)

    def test_unknown_symbol_ignored(self):
        """Records for symbols not in config.symbols should be discarded."""
        pipeline = build_pipeline_from_config()
        gb = GraphBuilder(GraphBuilderConfig(
            symbols=list(SYMBOLS), window_size=20,
            correlation_min_samples=10,
        ))
        stats = StatsBuffer()
        redis = RedisClient(host="127.0.0.1", port=1)
        weird = {"symbol": "ZZZZ", "timestamp": 0, "mid_price": 1.0,
                 "spread_bps": 0.0, "trade_imbalance": 0.0,
                 "order_cancel_rate": 0.0}
        records = [IncomingRecord(
            topic="orderbook.l2", partition=0, offset=0,
            key=b"ZZZZ", value=weird, timestamp_ms=0,
        )]
        loop = StreamingInferenceLoop(
            pipeline=pipeline,
            graph_builder=gb,
            config=StreamingLoopConfig(symbols=SYMBOLS, window_size=20),
            stats=stats, redis_client=redis,
            consumer=FakeConsumer(records),
        )

        async def _drive():
            task = asyncio.create_task(loop.run_forever_async(0.01))
            await asyncio.sleep(0.2)
            loop.request_shutdown()
            try:
                await asyncio.wait_for(task, 2.0)
            except asyncio.TimeoutError:
                task.cancel()
        asyncio.run(_drive())

        assert stats.snapshot()["n_predictions"] == 0

    def test_news_record_routes_to_news_buffer(self):
        """Records on the news topic go to _news_buffers, not _buffers."""
        import time as _time
        pipeline = build_pipeline_from_config()
        gb = GraphBuilder(GraphBuilderConfig(
            symbols=list(SYMBOLS), window_size=20,
            correlation_min_samples=10,
        ))
        stats = StatsBuffer()
        redis = RedisClient(host="127.0.0.1", port=1)
        # Fresh timestamp so it doesn't get filtered as stale.
        ts_ms = int(_time.time() * 1000)
        news_value = {
            "timestamp": ts_ms,
            "headline": "Apple beats earnings expectations",
            "source_name": "Reuters",
            "symbols": [SYMBOLS[0]],
        }
        rec = IncomingRecord(
            topic="news.feed", partition=0, offset=0,
            key=b"Reuters", value=news_value, timestamp_ms=ts_ms,
        )
        loop = StreamingInferenceLoop(
            pipeline=pipeline,
            graph_builder=gb,
            config=StreamingLoopConfig(
                symbols=SYMBOLS, window_size=20, use_text=True,
            ),
            stats=stats, redis_client=redis,
            consumer=FakeConsumer([rec]),
        )

        async def _drive():
            task = asyncio.create_task(loop.run_forever_async(0.01))
            await asyncio.sleep(0.2)
            loop.request_shutdown()
            try:
                await asyncio.wait_for(task, 2.0)
            except asyncio.TimeoutError:
                task.cancel()
        asyncio.run(_drive())

        # Headline should have landed in the symbol's news buffer,
        # not the orderbook buffer.
        assert len(loop._news_buffers[SYMBOLS[0]]) == 1
        assert loop._news_buffers[SYMBOLS[0]][0][1].startswith("Apple")
        assert len(loop._buffers[SYMBOLS[0]]) == 0

    def test_gather_headlines_returns_none_when_use_text_off(self):
        """With use_text=False, _gather_headlines() always returns None."""
        pipeline = build_pipeline_from_config()
        gb = GraphBuilder(GraphBuilderConfig(
            symbols=list(SYMBOLS), window_size=20,
            correlation_min_samples=10,
        ))
        loop = StreamingInferenceLoop(
            pipeline=pipeline,
            graph_builder=gb,
            config=StreamingLoopConfig(
                symbols=SYMBOLS, window_size=20, use_text=False,
            ),
            stats=StatsBuffer(),
            redis_client=RedisClient(host="127.0.0.1", port=1),
            consumer=FakeConsumer([]),
        )
        # Even if there's a fresh headline in the buffer, use_text=False
        # forces None.
        loop._news_buffers[SYMBOLS[0]].append(
            (int(__import__("time").time() * 1000), "Some headline")
        )
        assert loop._gather_headlines() is None


# ---------------------------------------------------------------------------
# Pipeline construction
# ---------------------------------------------------------------------------

class TestPipelineConstruction:
    def test_builds_pipeline(self):
        pipeline = build_pipeline_from_config()
        # Smoke: forward pass works on a graph from the test helper.
        gb = GraphBuilder(GraphBuilderConfig(
            symbols=list(SYMBOLS), window_size=20,
            correlation_min_samples=10,
        ))
        df = pd.DataFrame(_make_rows(n_per_sym=25))
        graph = gb.build(df)
        with torch.no_grad():
            logits = pipeline(graph, headlines=None)
        assert logits.shape[0] == len(SYMBOLS)
