"""
ingestion/kafka_producer.py
===========================
Kafka producer for StreamSentinel. Supports three sources:

  - replay   : reads a Parquet file (synthetic test set) and replays it
               into Kafka at controllable speed (1x / 10x / instant)
  - alpaca   : connects to Alpaca's WebSocket and produces live ticks
  - newsapi  : polls NewsAPI every N seconds and produces headlines

The producer is **graceful** on SIGINT — it flushes the in-flight buffer
before exiting so no messages are lost mid-batch.

CLI
---
    # Replay synthetic Parquet (most common for demos)
    python -m ingestion.kafka_producer --source replay --speed 1x

    # 10x compressed replay
    python -m ingestion.kafka_producer --source replay --speed 10x

    # Stress test
    python -m ingestion.kafka_producer --source replay --speed instant

    # Live Alpaca (requires API keys in .env)
    python -m ingestion.kafka_producer --source alpaca

    # News polling
    python -m ingestion.kafka_producer --source newsapi --poll-seconds 60

Design notes
------------
- Uses `confluent_kafka.Producer` (NOT kafka-python). confluent-kafka is
  C-based, ~10× faster, and supported officially by Confluent.
- Avro serialisation via the helpers in `schemas.py`.
- Delivery callbacks log failures but don't crash the producer — we
  prefer "drop one message and continue" over "kill the producer."
- The replay loop emits batched flushes every `flush_interval_ms` to
  bound memory under instant-speed mode.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

# Make `python -m ingestion.kafka_producer` work from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_loader import load_config
from logger import get_logger

from ingestion.schemas import (
    NewsRecord,
    OrderBookRecord,
    TickRecord,
    encode_news,
    encode_orderbook,
    encode_tick,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ProducerConfig:
    """Producer configuration.

    Attributes
    ----------
    bootstrap_servers : str
        Comma-separated host:port pairs (e.g. "localhost:9092").
    topic_ticks : str
        Topic name for individual ticks.
    topic_orderbook : str
        Topic name for L2 order-book snapshots.
    topic_news : str
        Topic name for news headlines.
    client_id : str
        Producer identifier (logged by the broker).
    linger_ms : int
        Wait this many ms to batch messages before sending.
    batch_size : int
        Producer batch size in bytes (confluent_kafka default: 16384).
    compression : str
        Wire compression. "lz4" gives ~3x compression with negligible CPU.
    acks : str
        "1" = leader ack only (fast); "all" = full replication (safe).
        Default "1" is fine for non-financial workloads; for production
        market data go to "all".
    """
    bootstrap_servers: str = "localhost:9092"
    topic_ticks: str = "market.ticks"
    topic_orderbook: str = "orderbook.l2"
    topic_news: str = "news.feed"
    client_id: str = "streamsentinel-producer"
    linger_ms: int = 10
    batch_size: int = 32_768
    compression: str = "lz4"
    acks: str = "1"


# ---------------------------------------------------------------------------
# Producer wrapper
# ---------------------------------------------------------------------------

class StreamSentinelProducer:
    """Thin wrapper around confluent_kafka.Producer with graceful shutdown.

    Designed to be subclass-free: source-specific code lives in module-level
    `run_<source>()` functions that drive this wrapper.
    """

    def __init__(self, config: ProducerConfig) -> None:
        self.config = config
        self._producer = None  # lazy import to keep tests fast
        self._n_produced: int = 0
        self._n_failed: int = 0
        self._shutdown = False

    def _ensure_producer(self) -> None:
        if self._producer is not None:
            return
        from confluent_kafka import Producer
        self._producer = Producer({
            "bootstrap.servers": self.config.bootstrap_servers,
            "client.id": self.config.client_id,
            "linger.ms": self.config.linger_ms,
            "batch.size": self.config.batch_size,
            "compression.type": self.config.compression,
            "acks": self.config.acks,
        })

    def _on_delivery(self, err: Any, msg: Any) -> None:
        """Per-message delivery callback."""
        if err is not None:
            self._n_failed += 1
            log.warning(f"Delivery failed (topic={msg.topic()}): {err}")
        else:
            self._n_produced += 1

    # ------------------------------------------------------------------
    # Produce methods
    # ------------------------------------------------------------------
    def produce_tick(self, record: TickRecord) -> None:
        """Serialise & enqueue one tick."""
        self._ensure_producer()
        self._producer.poll(0)  # drain delivery reports without blocking
        try:
            self._producer.produce(
                topic=self.config.topic_ticks,
                value=encode_tick(record),
                key=record.symbol.encode("utf-8"),
                on_delivery=self._on_delivery,
            )
        except BufferError:
            # Local queue full — block briefly to drain, then retry.
            self._producer.flush(timeout=1.0)
            self._producer.produce(
                topic=self.config.topic_ticks,
                value=encode_tick(record),
                key=record.symbol.encode("utf-8"),
                on_delivery=self._on_delivery,
            )

    def produce_orderbook(self, record: OrderBookRecord) -> None:
        """Serialise & enqueue one L2 snapshot."""
        self._ensure_producer()
        self._producer.poll(0)
        try:
            self._producer.produce(
                topic=self.config.topic_orderbook,
                value=encode_orderbook(record),
                key=record.symbol.encode("utf-8"),
                on_delivery=self._on_delivery,
            )
        except BufferError:
            self._producer.flush(timeout=1.0)
            self._producer.produce(
                topic=self.config.topic_orderbook,
                value=encode_orderbook(record),
                key=record.symbol.encode("utf-8"),
                on_delivery=self._on_delivery,
            )

    def produce_news(self, record: NewsRecord) -> None:
        """Serialise & enqueue one news item."""
        self._ensure_producer()
        self._producer.poll(0)
        try:
            self._producer.produce(
                topic=self.config.topic_news,
                value=encode_news(record),
                key=(record.source_name or "unknown").encode("utf-8"),
                on_delivery=self._on_delivery,
            )
        except BufferError:
            self._producer.flush(timeout=1.0)
            self._producer.produce(
                topic=self.config.topic_news,
                value=encode_news(record),
                key=(record.source_name or "unknown").encode("utf-8"),
                on_delivery=self._on_delivery,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def request_shutdown(self) -> None:
        """Signal the producer to stop after the current message."""
        self._shutdown = True

    def is_shutting_down(self) -> bool:
        return self._shutdown

    def flush(self, timeout: float = 10.0) -> int:
        """Block until all in-flight messages are acked. Returns leftover count."""
        if self._producer is None:
            return 0
        return int(self._producer.flush(timeout=timeout))

    @property
    def stats(self) -> dict[str, int]:
        return {"produced": self._n_produced, "failed": self._n_failed}


# ---------------------------------------------------------------------------
# Source-specific runners
# ---------------------------------------------------------------------------

def _parse_speed(speed: str) -> float | None:
    """Parse `--speed` into a multiplier.

    Returns
    -------
    None for 'instant' (no sleep between messages); else a multiplier
    such that real-time-elapsed = file-time-elapsed / multiplier.
    """
    if speed == "instant":
        return None
    if speed.endswith("x"):
        return float(speed[:-1])
    return float(speed)


def run_replay(
    producer: StreamSentinelProducer,
    parquet_path: Path,
    speed: str = "1x",
    flush_interval_ms: int = 1000,
    max_records: int | None = None,
) -> None:
    """
    Replay a Parquet file into Kafka, preserving inter-event timing.

    Parameters
    ----------
    producer : StreamSentinelProducer
        Already-configured producer.
    parquet_path : Path
        File to read. Must have a `timestamp` column (ms epoch).
    speed : str
        "1x" (real time), "10x" (compressed), "instant" (as fast as possible).
    flush_interval_ms : int
        Flush in-flight messages every this many wall-clock ms.
    max_records : int, optional
        Stop after producing this many records (for testing).

    Notes
    -----
    We sleep between records to preserve timing. For accurate sleep on
    Windows, we use `time.sleep` (~15ms granularity) which is fine for
    market-data replay at <100k events/sec. For instant mode we don't
    sleep at all — Kafka's own batching paces throughput.
    """
    import pandas as pd

    multiplier = _parse_speed(speed)
    log.info(f"Replaying {parquet_path} at speed={speed}")
    df = pd.read_parquet(parquet_path)
    if len(df) == 0:
        log.warning("Parquet file is empty.")
        return

    df = df.sort_values("timestamp")
    # Are these orderbook records (40 L2 columns) or just ticks?
    is_orderbook = "bid_l1" in df.columns

    last_ts_ms: int | None = None
    last_wall_clock: float = time.time()
    last_flush_wall: float = time.time()
    n_produced = 0

    for _, row in df.iterrows():
        if producer.is_shutting_down():
            log.info("Shutdown requested; stopping replay.")
            break
        if max_records is not None and n_produced >= max_records:
            break

        ts_ms = int(row["timestamp"])

        # Pace inter-event timing.
        if multiplier is not None and last_ts_ms is not None:
            file_delta_ms = ts_ms - last_ts_ms
            target_sleep_s = (file_delta_ms / 1000.0) / multiplier
            elapsed = time.time() - last_wall_clock
            sleep_s = max(0.0, target_sleep_s - elapsed)
            if sleep_s > 0:
                time.sleep(sleep_s)

        # Build the record.
        if is_orderbook:
            row_dict = row.to_dict()
            row_dict["source"] = "replay"
            # Cast numpy types -> plain Python (Avro doesn't like np.float64
            # in some setups).
            for k, v in list(row_dict.items()):
                if hasattr(v, "item"):
                    row_dict[k] = v.item()
            # Keep only the fields the schema accepts.
            record = OrderBookRecord.from_dict(row_dict)
            producer.produce_orderbook(record)
        else:
            record = TickRecord(
                timestamp=ts_ms,
                symbol=str(row["symbol"]),
                mid_price=float(row["mid_price"]),
                spread_bps=float(row.get("spread_bps", 0.0)),
                trade_imbalance=float(row.get("trade_imbalance", 0.0)),
                order_cancel_rate=float(row.get("order_cancel_rate", 0.0)),
                source="replay",
            )
            producer.produce_tick(record)

        n_produced += 1
        last_ts_ms = ts_ms
        last_wall_clock = time.time()

        # Periodic flush so memory stays bounded.
        if (time.time() - last_flush_wall) * 1000 > flush_interval_ms:
            producer.flush(timeout=2.0)
            last_flush_wall = time.time()
            if n_produced % 1000 == 0:
                log.info(f"replay: produced={n_produced} "
                         f"failed={producer.stats['failed']}")

    leftover = producer.flush(timeout=15.0)
    log.info(f"Replay finished. produced={producer.stats['produced']} "
             f"failed={producer.stats['failed']} flush_leftover={leftover}")


def run_alpaca(
    producer: StreamSentinelProducer,
    symbols: list[str],
    api_key: str,
    api_secret: str,
    feed: str = "iex",
) -> None:
    """
    Stream live ticks from Alpaca's WebSocket into Kafka.

    Requires `alpaca-py` (which we don't import at module top-level so
    that the rest of the producer works in test envs without Alpaca
    installed).

    Notes
    -----
    Free Alpaca tier uses the IEX feed (no L2). Paid SIP feed has full
    market data. We emit TickRecord (not OrderBookRecord) here because
    the feed only gives quote/trade events, not depth.
    """
    try:
        from alpaca.data.live import StockDataStream  # type: ignore
    except ImportError:
        log.error(
            "alpaca-py not installed. Add `alpaca-py` to requirements.txt "
            "and `pip install alpaca-py`."
        )
        return

    log.info(f"Connecting to Alpaca ({feed}) for symbols: {symbols}")
    stream = StockDataStream(api_key, api_secret, feed=feed)

    async def _on_quote(quote: Any) -> None:
        if producer.is_shutting_down():
            await stream.stop_ws()
            return
        try:
            bid = float(quote.bid_price)
            ask = float(quote.ask_price)
            mid = (bid + ask) / 2.0
            spread_bps = (ask - bid) / mid * 10_000.0 if mid > 0 else 0.0
            record = TickRecord(
                timestamp=int(quote.timestamp.timestamp() * 1000),
                symbol=str(quote.symbol),
                mid_price=mid,
                spread_bps=spread_bps,
                trade_imbalance=0.0,    # Alpaca doesn't give us this
                order_cancel_rate=0.0,  # nor this
                source="alpaca",
            )
            producer.produce_tick(record)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"Alpaca handler error: {exc}")

    stream.subscribe_quotes(_on_quote, *symbols)

    try:
        stream.run()
    finally:
        producer.flush(timeout=15.0)


def run_newsapi(
    producer: StreamSentinelProducer,
    api_key: str,
    queries: list[str],
    poll_seconds: int = 60,
    max_per_poll: int = 20,
) -> None:
    """
    Poll NewsAPI every `poll_seconds` and produce items to `news.feed`.

    Parameters
    ----------
    producer : StreamSentinelProducer
    api_key : str
        From `https://newsapi.org/`. Pass via .env / NEWSAPI_KEY.
    queries : list of str
        Search queries (e.g. ["AAPL", "MSFT stock"]).
    poll_seconds : int
        Sleep between polls. Free tier: 100 requests/day, so 864s/poll
        is the floor.
    max_per_poll : int
        NewsAPI's `pageSize` cap.

    Notes
    -----
    We deduplicate by URL across polls — NewsAPI often returns the same
    headline for several hours.
    """
    try:
        import requests
    except ImportError:
        log.error("requests not installed.")
        return

    seen_urls: set[str] = set()
    base_url = "https://newsapi.org/v2/everything"

    while not producer.is_shutting_down():
        for query in queries:
            try:
                resp = requests.get(
                    base_url,
                    params={
                        "q": query,
                        "pageSize": max_per_poll,
                        "language": "en",
                        "sortBy": "publishedAt",
                        "apiKey": api_key,
                    },
                    timeout=10,
                )
                data = resp.json()
            except Exception as exc:  # noqa: BLE001
                log.warning(f"NewsAPI fetch failed for '{query}': {exc}")
                continue

            for art in data.get("articles", []):
                url = art.get("url", "")
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                # Parse publishedAt (ISO 8601) to ms epoch.
                from datetime import datetime
                try:
                    ts = int(datetime.fromisoformat(
                        art["publishedAt"].replace("Z", "+00:00")
                    ).timestamp() * 1000)
                except Exception:
                    ts = int(time.time() * 1000)

                record = NewsRecord(
                    timestamp=ts,
                    headline=art.get("title") or "",
                    url=url,
                    source_name=(art.get("source") or {}).get("name", "?"),
                    symbols=[query.upper()],
                )
                producer.produce_news(record)

        producer.flush(timeout=5.0)
        log.info(f"NewsAPI poll done. seen={len(seen_urls)} produced="
                 f"{producer.stats['produced']}")
        # Sleep but interrupt cleanly on shutdown.
        slept = 0
        while slept < poll_seconds and not producer.is_shutting_down():
            time.sleep(1)
            slept += 1


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def _install_signal_handlers(producer: StreamSentinelProducer) -> None:
    """Convert SIGINT/SIGTERM into a graceful shutdown flag."""
    def _handler(signum: int, frame: Any) -> None:
        log.info(f"Signal {signum} received; flushing and exiting.")
        producer.request_shutdown()
    signal.signal(signal.SIGINT, _handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handler)


def _build_producer_from_config() -> StreamSentinelProducer:
    """Construct producer from config.yaml + .env."""
    cfg = load_config()
    kafka_cfg = cfg.get("kafka", {})
    topics = kafka_cfg.get("topics", {})

    pc = ProducerConfig(
        bootstrap_servers=kafka_cfg.get("bootstrap_servers",
                                        "localhost:9092"),
        topic_ticks=topics.get("market_ticks", "market.ticks"),
        topic_orderbook=topics.get("orderbook_l2", "orderbook.l2"),
        topic_news=topics.get("news_feed", "news.feed"),
        client_id=kafka_cfg.get("producer", {}).get(
            "client_id", "streamsentinel-producer"
        ),
        linger_ms=kafka_cfg.get("producer", {}).get("linger_ms", 10),
        compression=kafka_cfg.get("producer", {}).get("compression", "lz4"),
        acks=kafka_cfg.get("producer", {}).get("acks", "1"),
    )
    return StreamSentinelProducer(pc)


def main() -> int:
    parser = argparse.ArgumentParser(prog="ingestion.kafka_producer")
    parser.add_argument(
        "--source",
        choices=["replay", "alpaca", "newsapi"],
        required=True,
        help="Where to read from."
    )
    parser.add_argument("--parquet", type=Path,
                        default=Path("data/synthetic/test.parquet"),
                        help="Replay source file (replay mode)")
    parser.add_argument("--speed", default="1x",
                        help="Replay speed: 1x, 10x, instant")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--symbols", default=None,
                        help="Comma-separated (alpaca mode)")
    parser.add_argument("--queries", default=None,
                        help="Comma-separated NewsAPI queries")
    parser.add_argument("--poll-seconds", type=int, default=60,
                        help="NewsAPI poll interval (s)")
    args = parser.parse_args()

    producer = _build_producer_from_config()
    _install_signal_handlers(producer)

    try:
        if args.source == "replay":
            if not args.parquet.exists():
                log.error(f"Replay file not found: {args.parquet}")
                return 1
            run_replay(producer, args.parquet, speed=args.speed,
                       max_records=args.max_records)

        elif args.source == "alpaca":
            api_key = os.environ.get("ALPACA_API_KEY", "")
            api_secret = os.environ.get("ALPACA_API_SECRET", "")
            if not api_key or not api_secret:
                log.error(
                    "ALPACA_API_KEY / ALPACA_API_SECRET not set in .env."
                )
                return 1
            symbols = (args.symbols or "AAPL,MSFT,TSLA").split(",")
            run_alpaca(producer, symbols, api_key, api_secret)

        elif args.source == "newsapi":
            api_key = os.environ.get("NEWSAPI_KEY", "")
            if not api_key:
                log.error("NEWSAPI_KEY not set in .env.")
                return 1
            queries = (args.queries or "AAPL,MSFT,TSLA").split(",")
            run_newsapi(producer, api_key, queries,
                        poll_seconds=args.poll_seconds)
    finally:
        leftover = producer.flush(timeout=15.0)
        log.info(
            f"Producer exiting. produced={producer.stats['produced']} "
            f"failed={producer.stats['failed']} flush_leftover={leftover}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
