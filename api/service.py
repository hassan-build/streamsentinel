"""
api/service.py
==============
FastAPI service for StreamSentinel.

Endpoints
---------
  - GET  /health        liveness + model load info
  - POST /predict       synchronous one-shot prediction
  - GET  /stats         rolling latency + anomaly counters
  - GET  /latest        Redis cache snapshot (used by the dashboard)

Run
---
    python -m api.service                 # API + streaming loop
    python -m api.service --no-streaming  # API only
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_loader import load_config
from logger import get_logger
from graph import GraphBuilder, GraphBuilderConfig
from models.anomaly_scorer import ANOMALY_CLASSES, NORMAL_CLASS_IDX

from api.model_loader import build_pipeline_from_config, load_checkpoint
from api.state import RedisClient, StatsBuffer
from api.streaming_loop import StreamingInferenceLoop, StreamingLoopConfig

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pydantic request/response schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Input for `/predict`.

    `rows` is a list of order-book row dicts (one per symbol per
    timestamp). The handler reshapes them into a DataFrame, builds a
    graph, and runs the pipeline.
    """
    rows: list[dict[str, Any]] = Field(..., min_length=1)


class SymbolPrediction(BaseModel):
    symbol: str
    anomaly_score: float
    predicted_class: str
    class_probabilities: dict[str, float]
    is_anomaly: bool


class PredictResponse(BaseModel):
    predictions: list[SymbolPrediction]
    inference_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    checkpoint: str
    redis_connected: bool
    streaming_loop_active: bool
    predictions_made: int


class StatsResponse(BaseModel):
    n_predictions: int
    n_anomalies_detected: int
    anomaly_rate: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    uptime_seconds: int


# ---------------------------------------------------------------------------
# Application state holder
# ---------------------------------------------------------------------------

class AppState:
    """Container for shared mutable runtime state.

    Mounted onto `app.state` at startup; everything the endpoints need
    is reachable via `request.app.state.<attr>`.
    """
    pipeline: Any = None
    graph_builder: GraphBuilder | None = None
    stats: StatsBuffer | None = None
    redis: RedisClient | None = None
    streaming_loop: StreamingInferenceLoop | None = None
    streaming_task: asyncio.Task | None = None
    symbols: tuple[str, ...] = ()
    window_size: int = 60
    checkpoint_path: Path = Path("checkpoints/best_model.pt")
    model_loaded: bool = False
    enable_streaming: bool = True


# ---------------------------------------------------------------------------
# Lifespan: build + load on startup, flush on shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Construct shared resources on startup; clean up on shutdown."""
    st: AppState = app.state.app_state
    cfg_all = load_config()

    # Resolve symbols, window size from config.
    st.symbols = tuple(cfg_all["data_sources"]["alpaca"]["symbols"])
    st.window_size = cfg_all.get("api", {}).get(
        "streaming_window_size", 60
    )

    # Pipeline.
    st.pipeline = build_pipeline_from_config()
    st.model_loaded = load_checkpoint(st.pipeline, st.checkpoint_path)

    # GraphBuilder.
    st.graph_builder = GraphBuilder(GraphBuilderConfig(
        symbols=list(st.symbols),
        window_size=st.window_size,
        correlation_min_samples=min(st.window_size, 30),
    ))

    # Stats + Redis.
    st.stats = StatsBuffer()
    redis_cfg = cfg_all.get("redis", {})
    st.redis = RedisClient(
        host=redis_cfg.get("host", "localhost"),
        port=redis_cfg.get("port", 6379),
        db=redis_cfg.get("db", 0),
        ttl_seconds=redis_cfg.get("ttl_seconds", 300),
    )

    # Optional streaming loop.
    if st.enable_streaming:
        from ingestion.kafka_consumer import (
            _build_consumer_from_config,
        )
        kafka_cfg = cfg_all.get("kafka", {})
        topics_cfg = kafka_cfg.get("topics", {})
        orderbook_topic = topics_cfg.get("orderbook", "orderbook.l2")
        news_topic = topics_cfg.get("news", "news.feed")
        try:
            # Subscribe to BOTH orderbook AND news topics so FinBERT
            # gets real text input during the live demo.
            consumer = _build_consumer_from_config(
                extra_topics=[orderbook_topic, news_topic]
            )
            # Optional output producer
            output_producer = None
            try:
                from confluent_kafka import Producer  # type: ignore
                output_producer = Producer({
                    "bootstrap.servers": kafka_cfg.get(
                        "bootstrap_servers", "localhost:9050"
                    ),
                    "client.id": "streamsentinel-api-output",
                })
            except Exception as exc:  # noqa: BLE001
                log.warning(f"Output producer unavailable: {exc}")

            st.streaming_loop = StreamingInferenceLoop(
                pipeline=st.pipeline,
                graph_builder=st.graph_builder,
                config=StreamingLoopConfig(
                    symbols=st.symbols,
                    window_size=st.window_size,
                    output_topic=topics_cfg.get(
                        "anomalies", "anomaly.scores"
                    ),
                    # Turn on text fusion: news headlines on
                    # `news.feed` will flow through FinBERT.
                    use_text=True,
                ),
                stats=st.stats,
                redis_client=st.redis,
                consumer=consumer,
                producer=output_producer,
            )
            st.streaming_task = asyncio.create_task(
                st.streaming_loop.run_forever_async()
            )
            log.info("Background streaming loop scheduled.")
        except Exception as exc:  # noqa: BLE001
            log.warning(f"Streaming loop disabled: {exc}")
            st.streaming_loop = None

    log.info("API ready to serve requests.")

    yield

    # ---------------- shutdown ----------------
    log.info("Shutting down API.")
    if st.streaming_loop is not None:
        st.streaming_loop.request_shutdown()
    if st.streaming_task is not None:
        try:
            await asyncio.wait_for(st.streaming_task, timeout=5.0)
        except asyncio.TimeoutError:
            st.streaming_task.cancel()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(enable_streaming: bool = True,
               checkpoint_path: Path | None = None) -> FastAPI:
    """Build a FastAPI app instance. Used by both the CLI and tests."""
    st = AppState()
    st.enable_streaming = enable_streaming
    if checkpoint_path is not None:
        st.checkpoint_path = checkpoint_path
    app = FastAPI(title="StreamSentinel", lifespan=lifespan)
    app.state.app_state = st
    _register_routes(app)
    return app


# ---------------------------------------------------------------------------
# Endpoint handlers
# ---------------------------------------------------------------------------

def _register_routes(app: FastAPI) -> None:
    @app.get("/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        st: AppState = app.state.app_state
        return HealthResponse(
            status="ok",
            model_loaded=bool(st.model_loaded),
            checkpoint=str(st.checkpoint_path),
            redis_connected=bool(st.redis and st.redis.connected),
            streaming_loop_active=(st.streaming_loop is not None
                                   and st.streaming_task is not None
                                   and not st.streaming_task.done()),
            predictions_made=int(st.stats.snapshot()["n_predictions"])
                if st.stats else 0,
        )

    @app.get("/stats", response_model=StatsResponse)
    async def stats() -> StatsResponse:
        st: AppState = app.state.app_state
        snap = st.stats.snapshot() if st.stats else {
            "n_predictions": 0, "n_anomalies_detected": 0,
            "anomaly_rate": 0.0, "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0, "latency_p99_ms": 0.0,
            "uptime_seconds": 0,
        }
        return StatsResponse(**snap)

    @app.get("/latest")
    async def latest() -> dict[str, Any]:
        """Return Redis cache snapshot — used by the dashboard."""
        st: AppState = app.state.app_state
        if st.redis is None:
            return {}
        return st.redis.get_all_latest()

    @app.post("/predict", response_model=PredictResponse)
    async def predict(req: PredictRequest) -> PredictResponse:
        st: AppState = app.state.app_state
        if st.pipeline is None or st.graph_builder is None:
            raise HTTPException(503, "Model not ready")

        # Build dataframe + sanity check.
        df = pd.DataFrame(req.rows)
        if df.empty:
            raise HTTPException(400, "rows is empty")
        if "symbol" not in df.columns or "timestamp" not in df.columns:
            raise HTTPException(
                400,
                "Each row must contain 'symbol' and 'timestamp'."
            )
        # Ensure synthetic-injector columns exist (GraphBuilder doesn't
        # need them but downstream code might key off them).
        if "label" not in df.columns:
            df["label"] = 0
        if "anomaly_severity" not in df.columns:
            df["anomaly_severity"] = 0.0
        if "injection_id" not in df.columns:
            df["injection_id"] = ""

        try:
            t0 = time.perf_counter()
            graph = st.graph_builder.build(df)
            st.pipeline.eval()
            with torch.no_grad():
                logits = st.pipeline(graph, headlines=None)
                probs = torch.softmax(logits, dim=-1).cpu().numpy()
            latency_ms = (time.perf_counter() - t0) * 1000.0
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"Inference failed: {exc}")

        # One prediction per node in the graph (one per symbol).
        out: list[SymbolPrediction] = []
        n_anom = 0
        symbols = list(st.symbols) if st.symbols else (
            sorted(df["symbol"].unique().tolist())
        )
        for i, sym in enumerate(symbols[: probs.shape[0]]):
            class_probs = {
                ANOMALY_CLASSES[c]: float(probs[i, c])
                for c in range(probs.shape[1])
            }
            predicted_class = int(np.argmax(probs[i]))
            score = float(1.0 - probs[i, NORMAL_CLASS_IDX])
            is_anom = predicted_class != NORMAL_CLASS_IDX
            n_anom += int(is_anom)
            out.append(SymbolPrediction(
                symbol=sym,
                anomaly_score=score,
                predicted_class=ANOMALY_CLASSES[predicted_class],
                class_probabilities=class_probs,
                is_anomaly=is_anom,
            ))

        if st.stats:
            st.stats.record(latency_ms, n_anom)
        return PredictResponse(
            predictions=out,
            inference_latency_ms=latency_ms,
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(prog="api.service")
    parser.add_argument("--no-streaming", action="store_true",
                        help="Disable the Kafka streaming loop")
    parser.add_argument("--checkpoint", type=Path,
                        default=Path("checkpoints/best_model.pt"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    app = create_app(
        enable_streaming=not args.no_streaming,
        checkpoint_path=args.checkpoint,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
