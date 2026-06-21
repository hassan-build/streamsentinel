"""
ingestion/kafka_consumer.py
===========================
Kafka consumer for StreamSentinel. Used by the processing layer and by
the FastAPI service to stream records out of Kafka topics.

Design choices
--------------
- **Manual commits.** Auto-commit (the default) commits offsets every N
  seconds regardless of whether the consumer actually processed the
  messages. A crash mid-processing then loses messages silently. We
  disable auto-commit and require the caller to invoke `commit()` after
  successful processing.

- **Pull loop, not callbacks.** Confluent-Kafka supports both. The pull
  loop is simpler to integrate with FastAPI/Streamlit (synchronous
  generators) and matches how the rest of StreamSentinel processes data.

- **Schema-aware decoding.** The consumer dispatches by topic: each
  topic has a known schema, and the decoded dict is yielded with topic
  metadata so the caller can branch correctly.

Usage
-----
    consumer = StreamSentinelConsumer(topics=["market.ticks", "news.feed"])
    for record in consumer.poll_forever():
        process(record)            # do something with record.value
        consumer.commit()          # commit AFTER successful processing
"""

from __future__ import annotations

import signal
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_loader import load_config
from logger import get_logger
from ingestion.schemas import (
    decode_news,
    decode_orderbook,
    decode_tick,
)

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class IncomingRecord:
    """One decoded record yielded by `poll_forever()`.

    Attributes
    ----------
    topic : str
        Source Kafka topic.
    partition : int
    offset : int
    key : bytes | None
        Routing key (usually symbol).
    value : dict[str, Any]
        Avro-decoded payload as a Python dict.
    timestamp_ms : int
        Broker-side message timestamp.
    """
    topic: str
    partition: int
    offset: int
    key: bytes | None
    value: dict[str, Any]
    timestamp_ms: int


# ---------------------------------------------------------------------------
# Consumer config
# ---------------------------------------------------------------------------

@dataclass
class ConsumerConfig:
    """Consumer configuration.

    Attributes
    ----------
    bootstrap_servers : str
    group_id : str
        Consumer-group ID. Two consumers with the same group ID share
        the partitions of the topic between them; with different groups
        they each get a full copy.
    topics : list of str
    auto_offset_reset : str
        Where to start if there's no committed offset: "earliest" or
        "latest". For replay/demo we use "earliest"; for production
        live processing we'd use "latest".
    max_poll_interval_ms : int
        How long a single poll iteration can take before the broker
        rebalances. Default 5 min — enough for slow downstream processing.
    """
    bootstrap_servers: str = "localhost:9092"
    group_id: str = "streamsentinel"
    topics: list[str] = field(default_factory=list)
    auto_offset_reset: str = "earliest"
    max_poll_interval_ms: int = 300_000
    session_timeout_ms: int = 30_000


# ---------------------------------------------------------------------------
# Topic -> decoder dispatch
# ---------------------------------------------------------------------------

# Map of topic name -> decode function. The CLI uses config.yaml to fill
# this in dynamically; tests use the defaults below.
DEFAULT_TOPIC_DECODERS: dict[str, Any] = {
    "market.ticks": decode_tick,
    "orderbook.l2": decode_orderbook,
    "news.feed": decode_news,
}


# ---------------------------------------------------------------------------
# Consumer
# ---------------------------------------------------------------------------

class StreamSentinelConsumer:
    """Pull-loop Kafka consumer with manual commits.

    Designed for synchronous integration: call `poll_forever()` and
    iterate. Call `commit()` after each successful downstream step.
    """

    def __init__(
        self,
        config: ConsumerConfig,
        topic_decoders: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.topic_decoders = topic_decoders or dict(DEFAULT_TOPIC_DECODERS)
        self._consumer = None  # lazy import
        self._shutdown = False
        self._last_msg = None
        self._n_consumed: int = 0
        self._n_decode_errors: int = 0

    def _ensure_consumer(self) -> None:
        if self._consumer is not None:
            return
        from confluent_kafka import Consumer
        self._consumer = Consumer({
            "bootstrap.servers": self.config.bootstrap_servers,
            "group.id": self.config.group_id,
            "auto.offset.reset": self.config.auto_offset_reset,
            "enable.auto.commit": False,
            "max.poll.interval.ms": self.config.max_poll_interval_ms,
            "session.timeout.ms": self.config.session_timeout_ms,
        })
        self._consumer.subscribe(self.config.topics)
        log.info(f"Consumer subscribed to topics: {self.config.topics}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def poll_forever(
        self, poll_timeout_s: float = 1.0
    ) -> Iterator[IncomingRecord]:
        """Yield records until shutdown is requested or the process exits.

        Parameters
        ----------
        poll_timeout_s : float
            How long each poll() call blocks waiting for a message.
            Smaller = more responsive shutdown; larger = lower CPU.
        """
        self._ensure_consumer()
        while not self._shutdown:
            msg = self._consumer.poll(timeout=poll_timeout_s)
            if msg is None:
                continue
            if msg.error() is not None:
                log.warning(f"Consumer error: {msg.error()}")
                continue

            topic = msg.topic()
            decoder = self.topic_decoders.get(topic)
            if decoder is None:
                log.warning(f"No decoder for topic '{topic}'; skipping.")
                continue

            try:
                value = decoder(msg.value())
            except Exception as exc:  # noqa: BLE001
                self._n_decode_errors += 1
                log.warning(f"Decode error on topic={topic}: {exc}")
                continue

            self._last_msg = msg
            self._n_consumed += 1
            yield IncomingRecord(
                topic=topic,
                partition=msg.partition(),
                offset=msg.offset(),
                key=msg.key(),
                value=value,
                timestamp_ms=msg.timestamp()[1] if msg.timestamp() else 0,
            )

    def commit(self, asynchronous: bool = True) -> None:
        """Commit offsets for the most recently yielded message.

        If `asynchronous=True` (default), the commit happens in the
        background and the call returns immediately. Use False if you
        need synchronous confirmation.
        """
        if self._consumer is None or self._last_msg is None:
            return
        self._consumer.commit(message=self._last_msg,
                              asynchronous=asynchronous)

    def request_shutdown(self) -> None:
        """Stop the poll loop after the current message."""
        self._shutdown = True

    def close(self) -> None:
        """Final cleanup — closes the underlying consumer."""
        if self._consumer is not None:
            self._consumer.close()
            self._consumer = None

    @property
    def stats(self) -> dict[str, int]:
        return {
            "consumed": self._n_consumed,
            "decode_errors": self._n_decode_errors,
        }


# ---------------------------------------------------------------------------
# CLI: smoke-tests the consumer by printing records as they arrive
# ---------------------------------------------------------------------------

def _build_consumer_from_config(
    extra_topics: list[str] | None = None,
) -> StreamSentinelConsumer:
    """Construct consumer from config.yaml."""
    cfg = load_config()
    kafka_cfg = cfg.get("kafka", {})
    topics_cfg = kafka_cfg.get("topics", {})

    # Resolve canonical names so the dispatch table matches.
    topic_ticks = topics_cfg.get("market_ticks", "market.ticks")
    topic_orderbook = topics_cfg.get("orderbook_l2", "orderbook.l2")
    topic_news = topics_cfg.get("news_feed", "news.feed")

    all_topics = extra_topics or [topic_ticks, topic_orderbook, topic_news]

    cc = ConsumerConfig(
        bootstrap_servers=kafka_cfg.get("bootstrap_servers",
                                        "localhost:9092"),
        group_id=kafka_cfg.get("consumer", {}).get(
            "group_id", "streamsentinel"
        ),
        topics=all_topics,
        auto_offset_reset=kafka_cfg.get("consumer", {}).get(
            "auto_offset_reset", "earliest"
        ),
    )

    decoders = {
        topic_ticks: decode_tick,
        topic_orderbook: decode_orderbook,
        topic_news: decode_news,
    }
    return StreamSentinelConsumer(cc, topic_decoders=decoders)


def main() -> int:
    """CLI: dump a few records from the configured topics.

    Useful for verifying end-to-end:
        # Terminal 1: start the producer
        python -m ingestion.kafka_producer --source replay --speed instant

        # Terminal 2: see records arrive
        python -m ingestion.kafka_consumer --max 20
    """
    import argparse
    parser = argparse.ArgumentParser(prog="ingestion.kafka_consumer")
    parser.add_argument("--max", type=int, default=10,
                        help="Stop after this many records.")
    parser.add_argument("--topics", default=None,
                        help="Override config topics (comma-separated).")
    args = parser.parse_args()

    extra = args.topics.split(",") if args.topics else None
    consumer = _build_consumer_from_config(extra_topics=extra)

    def _handler(signum: int, frame: Any) -> None:
        log.info(f"Signal {signum} received; shutting down consumer.")
        consumer.request_shutdown()
    signal.signal(signal.SIGINT, _handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _handler)

    n = 0
    try:
        for rec in consumer.poll_forever():
            print(f"[{rec.topic}] offset={rec.offset} key={rec.key} "
                  f"value={rec.value}")
            consumer.commit()
            n += 1
            if n >= args.max:
                break
    finally:
        consumer.close()
        log.info(f"Consumer exiting. consumed={consumer.stats['consumed']} "
                 f"decode_errors={consumer.stats['decode_errors']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
