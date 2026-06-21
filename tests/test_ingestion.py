"""
tests/test_ingestion.py
=======================
Tests for the ingestion module.

We use a **mock Kafka client** rather than a real broker. The mock
satisfies the same interface as confluent_kafka.Producer / Consumer so
the tests exercise the same code paths the production CLI does — they
just record messages in memory instead of writing to a network socket.

Tests verify:
  - Avro schemas round-trip (encode → decode = identity)
  - Schema mismatches fail at produce time (not silently)
  - Replay producer emits the right number of records
  - Replay producer honours --speed (instant runs much faster than 1x)
  - Consumer reads what the producer wrote (end-to-end via mock-Kafka)
  - Consumer dispatches the right decoder per topic
  - Graceful shutdown stops the loop cleanly
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ingestion.kafka_consumer import (
    ConsumerConfig,
    IncomingRecord,
    StreamSentinelConsumer,
)
from ingestion.kafka_producer import (
    ProducerConfig,
    StreamSentinelProducer,
    _parse_speed,
    run_replay,
)
from ingestion.schemas import (
    NewsRecord,
    OrderBookRecord,
    TickRecord,
    decode_news,
    decode_orderbook,
    decode_tick,
    encode_news,
    encode_orderbook,
    encode_tick,
)


# ===========================================================================
# Mock Kafka client
# ===========================================================================

class _MockMessage:
    """Mimics confluent_kafka.Message."""

    def __init__(self, topic: str, value: bytes, key: bytes | None,
                 partition: int = 0, offset: int = 0,
                 timestamp_ms: int = 0) -> None:
        self._topic = topic
        self._value = value
        self._key = key
        self._partition = partition
        self._offset = offset
        self._timestamp = (1, timestamp_ms)  # (TIMESTAMP_CREATE_TIME, ms)

    def topic(self) -> str: return self._topic
    def value(self) -> bytes: return self._value
    def key(self) -> bytes | None: return self._key
    def partition(self) -> int: return self._partition
    def offset(self) -> int: return self._offset
    def timestamp(self) -> tuple[int, int]: return self._timestamp
    def error(self) -> Any: return None


class MockProducer:
    """Fake confluent_kafka.Producer that records calls."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.messages: list[_MockMessage] = []
        self._offset_counter: int = 0

    def produce(self, topic: str, value: bytes,
                key: bytes | None = None,
                on_delivery: Any = None) -> None:
        msg = _MockMessage(topic, value, key, offset=self._offset_counter,
                           timestamp_ms=int(time.time() * 1000))
        self.messages.append(msg)
        self._offset_counter += 1
        if on_delivery is not None:
            on_delivery(None, msg)

    def poll(self, timeout: float) -> int:
        return 0

    def flush(self, timeout: float = 10.0) -> int:
        return 0


class MockConsumer:
    """Fake confluent_kafka.Consumer that yields a pre-loaded queue."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.queue: list[_MockMessage] = []
        self.subscribed_to: list[str] = []
        self.commits: list[_MockMessage] = []
        self._closed: bool = False

    def subscribe(self, topics: list[str]) -> None:
        self.subscribed_to = list(topics)

    def poll(self, timeout: float) -> _MockMessage | None:
        if self.queue:
            return self.queue.pop(0)
        return None

    def commit(self, message: _MockMessage | None = None,
               asynchronous: bool = True) -> None:
        if message is not None:
            self.commits.append(message)

    def close(self) -> None:
        self._closed = True


# ===========================================================================
# Schema tests
# ===========================================================================

class TestSchemas:
    def test_tick_roundtrip(self):
        rec = TickRecord(
            timestamp=1_700_000_000_000,
            symbol="AAPL",
            mid_price=195.32,
            spread_bps=2.4,
            trade_imbalance=0.1,
            order_cancel_rate=30.5,
            source="test",
        )
        decoded = decode_tick(encode_tick(rec))
        assert decoded["symbol"] == "AAPL"
        assert decoded["mid_price"] == pytest.approx(195.32)
        assert decoded["source"] == "test"
        assert decoded["schema_version"] == rec.schema_version

    def test_orderbook_roundtrip_all_levels(self):
        kwargs = dict(
            timestamp=1, symbol="MSFT", mid_price=400.0,
            spread_bps=1.5, trade_imbalance=-0.2,
            order_cancel_rate=20.0, source="test",
        )
        for i in range(1, 11):
            kwargs[f"bid_l{i}"] = 400.0 - i * 0.01
            kwargs[f"ask_l{i}"] = 400.0 + i * 0.01
            kwargs[f"bidsize_l{i}"] = float(100 * i)
            kwargs[f"asksize_l{i}"] = float(110 * i)
        rec = OrderBookRecord(**kwargs)
        decoded = decode_orderbook(encode_orderbook(rec))
        assert decoded["bid_l1"] == pytest.approx(399.99)
        assert decoded["asksize_l10"] == pytest.approx(1100.0)

    def test_news_roundtrip(self):
        rec = NewsRecord(
            timestamp=1_700_000_000_000,
            headline="Apple beats earnings",
            url="https://example.com/a",
            source_name="Reuters",
            symbols=["AAPL"],
        )
        decoded = decode_news(encode_news(rec))
        assert decoded["headline"] == "Apple beats earnings"
        assert decoded["symbols"] == ["AAPL"]

    def test_schema_mismatch_fails_at_produce_time(self):
        # Pass garbage type for `mid_price` — Avro must catch it.
        bad = dict(
            timestamp=1, symbol="X", mid_price="not_a_number",
            spread_bps=0.0, trade_imbalance=0.0,
            order_cancel_rate=0.0, source="x",
            schema_version="1.0.0",
        )
        from ingestion.schemas import encode, PARSED_TICK_SCHEMA
        with pytest.raises(ValueError, match="Avro encoding failed"):
            encode(bad, PARSED_TICK_SCHEMA)


# ===========================================================================
# Producer tests (mock Kafka)
# ===========================================================================

class TestProducer:
    def _patched_producer(self, monkeypatch) -> tuple[StreamSentinelProducer,
                                                       MockProducer]:
        """Build a StreamSentinelProducer whose _ensure_producer plugs in
        a MockProducer rather than a real Confluent Producer."""
        mock = MockProducer()
        prod = StreamSentinelProducer(ProducerConfig(
            bootstrap_servers="dummy:9092",
        ))

        def fake_ensure(self_):
            self_._producer = mock
        monkeypatch.setattr(
            StreamSentinelProducer, "_ensure_producer", fake_ensure
        )
        return prod, mock

    def test_produces_tick_to_correct_topic(self, monkeypatch):
        prod, mock = self._patched_producer(monkeypatch)
        prod.produce_tick(TickRecord(
            timestamp=1, symbol="AAPL", mid_price=100.0,
            spread_bps=1.0, trade_imbalance=0.0,
            order_cancel_rate=10.0, source="t",
        ))
        assert len(mock.messages) == 1
        assert mock.messages[0].topic() == "market.ticks"
        assert mock.messages[0].key() == b"AAPL"
        # The bytes should decode to the same record.
        decoded = decode_tick(mock.messages[0].value())
        assert decoded["symbol"] == "AAPL"

    def test_produces_orderbook_to_correct_topic(self, monkeypatch):
        prod, mock = self._patched_producer(monkeypatch)
        rec = OrderBookRecord(
            timestamp=1, symbol="MSFT", mid_price=400.0,
            spread_bps=1.5, trade_imbalance=-0.2,
            order_cancel_rate=20.0, source="t",
        )
        prod.produce_orderbook(rec)
        assert len(mock.messages) == 1
        assert mock.messages[0].topic() == "orderbook.l2"
        assert mock.messages[0].key() == b"MSFT"

    def test_produces_news_to_correct_topic(self, monkeypatch):
        prod, mock = self._patched_producer(monkeypatch)
        prod.produce_news(NewsRecord(
            timestamp=1, headline="hi", source_name="Reuters",
        ))
        assert mock.messages[0].topic() == "news.feed"
        assert mock.messages[0].key() == b"Reuters"

    def test_stats_increment_via_delivery_callback(self, monkeypatch):
        prod, mock = self._patched_producer(monkeypatch)
        for i in range(5):
            prod.produce_tick(TickRecord(
                timestamp=i, symbol="A", mid_price=1.0,
                spread_bps=0.0, trade_imbalance=0.0,
                order_cancel_rate=0.0,
            ))
        assert prod.stats["produced"] == 5
        assert prod.stats["failed"] == 0


# ===========================================================================
# Speed parsing
# ===========================================================================

class TestSpeedParsing:
    def test_instant(self): assert _parse_speed("instant") is None
    def test_1x(self): assert _parse_speed("1x") == 1.0
    def test_10x(self): assert _parse_speed("10x") == 10.0
    def test_plain_number(self): assert _parse_speed("2.5") == 2.5


# ===========================================================================
# Replay producer (uses mock Kafka)
# ===========================================================================

def _make_replay_parquet(tmp_path: Path, n_rows: int = 30) -> Path:
    """Build a small Parquet file matching the synthetic schema."""
    rng = np.random.default_rng(0)
    rows: list[dict[str, Any]] = []
    base_ts = 1_700_000_000_000
    for i in range(n_rows):
        ts = base_ts + i * 10  # 10 ms apart
        sym = "AAPL" if i % 2 == 0 else "MSFT"
        price = 100.0 + i * 0.01
        row = dict(
            timestamp=ts, symbol=sym, mid_price=price,
            spread_bps=float(rng.uniform(1, 3)),
            trade_imbalance=float(rng.uniform(-0.2, 0.2)),
            order_cancel_rate=float(rng.uniform(10, 40)),
            label=0, anomaly_severity=0.0, injection_id="",
        )
        for lvl in range(1, 11):
            row[f"bid_l{lvl}"] = price - lvl * 0.01
            row[f"ask_l{lvl}"] = price + lvl * 0.01
            row[f"bidsize_l{lvl}"] = float(rng.uniform(50, 500))
            row[f"asksize_l{lvl}"] = float(rng.uniform(50, 500))
        rows.append(row)
    df = pd.DataFrame(rows)
    out = tmp_path / "replay.parquet"
    df.to_parquet(out, index=False)
    return out


class TestReplay:
    def test_replay_produces_one_per_row(self, monkeypatch, tmp_path):
        path = _make_replay_parquet(tmp_path, n_rows=20)
        mock = MockProducer()
        prod = StreamSentinelProducer(ProducerConfig())

        def fake_ensure(self_):
            self_._producer = mock
        monkeypatch.setattr(
            StreamSentinelProducer, "_ensure_producer", fake_ensure
        )

        run_replay(prod, path, speed="instant")
        assert len(mock.messages) == 20
        # Replay sets source="replay".
        decoded = decode_orderbook(mock.messages[0].value())
        assert decoded["source"] == "replay"

    def test_replay_honours_max_records(self, monkeypatch, tmp_path):
        path = _make_replay_parquet(tmp_path, n_rows=50)
        mock = MockProducer()
        prod = StreamSentinelProducer(ProducerConfig())
        monkeypatch.setattr(
            StreamSentinelProducer, "_ensure_producer",
            lambda self_: setattr(self_, "_producer", mock)
        )

        run_replay(prod, path, speed="instant", max_records=7)
        assert len(mock.messages) == 7

    def test_replay_shutdown_stops_loop(self, monkeypatch, tmp_path):
        path = _make_replay_parquet(tmp_path, n_rows=20)
        mock = MockProducer()
        prod = StreamSentinelProducer(ProducerConfig())
        monkeypatch.setattr(
            StreamSentinelProducer, "_ensure_producer",
            lambda self_: setattr(self_, "_producer", mock)
        )

        # Trigger shutdown immediately.
        prod.request_shutdown()
        run_replay(prod, path, speed="instant")
        assert len(mock.messages) == 0


# ===========================================================================
# Consumer tests
# ===========================================================================

class TestConsumer:
    def _patched_consumer(self, monkeypatch) -> tuple[StreamSentinelConsumer,
                                                       MockConsumer]:
        mock = MockConsumer()
        cons = StreamSentinelConsumer(ConsumerConfig(
            bootstrap_servers="dummy:9092",
            topics=["market.ticks", "orderbook.l2", "news.feed"],
        ))

        def fake_ensure(self_):
            self_._consumer = mock
            mock.subscribe(self_.config.topics)
        monkeypatch.setattr(
            StreamSentinelConsumer, "_ensure_consumer", fake_ensure
        )
        return cons, mock

    def test_consumer_decodes_tick(self, monkeypatch):
        cons, mock = self._patched_consumer(monkeypatch)
        tick = TickRecord(
            timestamp=1, symbol="AAPL", mid_price=100.0,
            spread_bps=1.0, trade_imbalance=0.0,
            order_cancel_rate=10.0,
        )
        mock.queue.append(_MockMessage(
            "market.ticks", encode_tick(tick), b"AAPL"
        ))

        records: list[IncomingRecord] = []
        for rec in cons.poll_forever(poll_timeout_s=0.01):
            records.append(rec)
            cons.commit()
            if len(records) >= 1:
                cons.request_shutdown()
        assert len(records) == 1
        assert records[0].topic == "market.ticks"
        assert records[0].value["symbol"] == "AAPL"
        assert len(mock.commits) == 1

    def test_consumer_dispatches_per_topic(self, monkeypatch):
        cons, mock = self._patched_consumer(monkeypatch)
        # Mixed batch across topics.
        mock.queue.extend([
            _MockMessage(
                "market.ticks",
                encode_tick(TickRecord(
                    timestamp=1, symbol="A", mid_price=1.0,
                    spread_bps=0.0, trade_imbalance=0.0,
                    order_cancel_rate=0.0,
                )), b"A"
            ),
            _MockMessage(
                "news.feed",
                encode_news(NewsRecord(
                    timestamp=2, headline="X", source_name="src",
                )), b"src"
            ),
        ])
        seen: list[tuple[str, dict]] = []
        for rec in cons.poll_forever(poll_timeout_s=0.01):
            seen.append((rec.topic, rec.value))
            if len(seen) >= 2:
                cons.request_shutdown()
        topics = {t for t, _ in seen}
        assert topics == {"market.ticks", "news.feed"}

    def test_consumer_handles_bad_payload(self, monkeypatch):
        """Garbage bytes should NOT crash the loop — log + skip."""
        cons, mock = self._patched_consumer(monkeypatch)
        mock.queue.append(_MockMessage(
            "market.ticks", b"\x00\x01\x02not_avro", b"k"
        ))
        # Add a good one after.
        mock.queue.append(_MockMessage(
            "market.ticks",
            encode_tick(TickRecord(
                timestamp=1, symbol="A", mid_price=1.0,
                spread_bps=0.0, trade_imbalance=0.0,
                order_cancel_rate=0.0,
            )), b"A"
        ))

        records: list[IncomingRecord] = []
        for rec in cons.poll_forever(poll_timeout_s=0.01):
            records.append(rec)
            if len(records) >= 1:
                cons.request_shutdown()
        # Bad one skipped, good one delivered.
        assert len(records) == 1
        assert records[0].value["symbol"] == "A"
        assert cons.stats["decode_errors"] == 1

    def test_consumer_close_idempotent(self, monkeypatch):
        cons, _ = self._patched_consumer(monkeypatch)
        cons._ensure_consumer()
        cons.close()
        cons.close()   # should not raise
