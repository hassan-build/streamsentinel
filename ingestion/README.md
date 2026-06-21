# Ingestion Module

This module is the **bridge between data sources and the rest of
StreamSentinel**. It wires Kafka topics to:

- **Replayed synthetic data** (`data/synthetic/test.parquet`) — used for
  reproducible demos and load testing
- **Alpaca live ticks** — real-time market data from Alpaca's WebSocket
- **NewsAPI / GDELT** — financial news headlines polled at intervals

Every record passes through the **same Avro schema** regardless of
source, so the downstream pipeline (processing → graph → model) is
agnostic to where the data originated.

---

## Files

| File | Purpose |
|---|---|
| `schemas.py` | Avro schemas + Python dataclasses for ticks / orderbook / news |
| `kafka_producer.py` | CLI: produce to Kafka from any source |
| `kafka_consumer.py` | Pull-loop consumer for the processing layer |

---

## Topics

Defined in `config.yaml > kafka.topics`:

| Topic | Schema | Producer source |
|---|---|---|
| `market.ticks` | `TickSchema` | `replay` or `alpaca` |
| `orderbook.l2` | `OrderBookSchema` | `replay` only (Alpaca free tier has no L2) |
| `news.feed` | `NewsSchema` | `newsapi`, `gdelt` |
| `anomaly.scores` | written by the AI layer (downstream) | — |

---

## Producer modes

### Replay synthetic Parquet (most common for dissertation demos)

```bash
# Real-time replay (1 second of test data per second)
python -m ingestion.kafka_producer --source replay --speed 1x

# 10x speed (compressed timeline)
python -m ingestion.kafka_producer --source replay --speed 10x

# Instant — for stress testing
python -m ingestion.kafka_producer --source replay --speed instant
```

### Alpaca live ticks

```bash
python -m ingestion.kafka_producer --source alpaca
```

Requires `ALPACA_API_KEY` + `ALPACA_API_SECRET` in `.env`. Free tier
returns IEX feed only (no L2 order book — for that we use replay).

### NewsAPI

```bash
python -m ingestion.kafka_producer --source newsapi --poll-seconds 60
```

Polls every minute (free tier: 100 requests/day, so 60-sec polling
gives ~24 hours of headroom).

### Stopping

Send SIGINT (Ctrl+C). The producer flushes its in-flight buffer before
exiting so no messages are lost.

---

## Consumer

```python
from ingestion.kafka_consumer import StreamSentinelConsumer

consumer = StreamSentinelConsumer(topics=["market.ticks", "news.feed"])
for record in consumer.poll_forever():
    process(record)
```

The consumer uses **manual commits** (auto-commit is off) so a crash
mid-batch doesn't lose offsets. Commits happen after each successful
processing step in your downstream loop — call `consumer.commit()`.

---

## Schema enforcement

We use **Avro** instead of JSON. Two reasons:

1. **Smaller on the wire.** Avro is binary; a tick record is ~80 bytes
   vs ~280 bytes for JSON.
2. **Schema enforcement at produce time.** Typos in field names are
   caught when the message is written, not days later in the consumer.
   This has saved teams I've seen from days of debugging silent data
   corruption.

If you change a schema field, **bump the schema version** in `schemas.py`
so old consumers can detect they're talking to a newer producer.

---

## Tests

```bash
pytest tests/test_ingestion.py -v
```

The tests verify:
- Schemas round-trip (encode → decode = identity)
- Schema mismatches fail at produce time (not silently)
- Replay producer emits at the configured speed within tolerance
- Consumer reads what the producer wrote (end-to-end via mock-Kafka)
- Graceful shutdown works (no messages lost on Ctrl+C)

The end-to-end tests use a **mock Kafka client** rather than a real
broker — running against the Docker Kafka would add 30+ seconds per
test and tie CI to Docker. The mock satisfies the same interface
(confluent_kafka.Producer / Consumer).
