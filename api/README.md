# API Module

FastAPI inference service for StreamSentinel. Serves predictions over
HTTP **and** runs the streaming-inference loop in the same process.

## What this replaces

The original architecture plan called for `processing/` (Apache Spark
Structured Streaming) to do the streaming layer. For the data scale
in this dissertation (≤100k rows in evaluation, ≤10 events/sec in
live demo), Spark would add JVM startup overhead and a separate
process to monitor without any throughput benefit. Pure-Python
`for record in consumer.poll_forever()` does the same work in ~150
lines.

This is an **honest dissertation finding**: we evaluated Spark and
found it added complexity without throughput benefit at our data
scale. The architecture remains production-ready — Spark could be
swapped in later by replacing `streaming_loop.py`.

## Files

| File | Purpose |
|---|---|
| `service.py` | FastAPI app + `/health`, `/predict`, `/stats` endpoints |
| `streaming_loop.py` | Async background task: consume Kafka → predict → cache to Redis + Kafka |
| `model_loader.py` | Load checkpoint into a `FullPipeline` |
| `state.py` | In-process stats counter + Redis client |

## Endpoints

### `GET /health`

Liveness + model load info. Streamlit polls this.

```json
{
  "status": "ok",
  "model_loaded": true,
  "checkpoint": "checkpoints/best_model.pt",
  "kafka_connected": true,
  "redis_connected": true,
  "streaming_loop_active": true,
  "predictions_made": 1234
}
```

### `POST /predict`

Synchronous one-shot prediction. Body = list of order book row dicts
(one row per symbol, same schema as the Kafka topic).

```json
{
  "rows": [
    {"timestamp": 1700000000000, "symbol": "AAPL", "mid_price": 195.0, ...},
    {"timestamp": 1700000000000, "symbol": "MSFT", "mid_price": 400.0, ...}
  ]
}
```

Returns per-symbol anomaly scores + latency:

```json
{
  "predictions": [
    {"symbol": "AAPL", "anomaly_score": 0.12, "predicted_class": "normal",
     "class_probabilities": {"normal": 0.88, "spoofing": 0.04, ...}},
    {"symbol": "MSFT", "anomaly_score": 0.81, "predicted_class": "spoofing",
     "class_probabilities": {"normal": 0.19, "spoofing": 0.72, ...}}
  ],
  "inference_latency_ms": 8.4
}
```

### `GET /stats`

Rolling latency + anomaly counts. Streamlit polls every 1s.

```json
{
  "n_predictions": 5421,
  "n_anomalies_detected": 244,
  "anomaly_rate": 0.045,
  "latency_p50_ms": 7.1,
  "latency_p95_ms": 18.4,
  "latency_p99_ms": 31.2,
  "uptime_seconds": 600
}
```

## Streaming loop

Runs as an `asyncio` background task started by FastAPI's `@app.on_event("startup")`.

1. Consumer subscribes to `orderbook.l2` (and optionally `news.feed`)
2. Maintains a rolling per-symbol window (default 60 ticks)
3. Once all symbols have ≥window_size ticks, builds a graph + runs the
   pipeline + emits a prediction
4. Prediction is written to Kafka topic `anomaly.scores` AND cached in
   Redis at key `latest:<symbol>` with 5-min TTL

To start the API + streaming loop:

```bash
python -m api.service
```

To start ONLY the API without streaming (e.g. you're hitting `/predict`
manually):

```bash
python -m api.service --no-streaming-loop
```

## Tests

```bash
pytest tests/test_api.py -v
```

Uses FastAPI's `TestClient` with mocked Kafka + Redis. No real broker
required for tests.
