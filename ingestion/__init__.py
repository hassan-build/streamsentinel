"""StreamSentinel — ingestion package.

Public API:
  - Schema dataclasses: TickRecord, OrderBookRecord, NewsRecord
  - Producer: StreamSentinelProducer, ProducerConfig
  - Consumer: StreamSentinelConsumer, ConsumerConfig, IncomingRecord
"""

from ingestion.kafka_consumer import (
    ConsumerConfig,
    IncomingRecord,
    StreamSentinelConsumer,
)
from ingestion.kafka_producer import (
    ProducerConfig,
    StreamSentinelProducer,
    run_alpaca,
    run_newsapi,
    run_replay,
)
from ingestion.schemas import (
    NEWS_SCHEMA,
    ORDERBOOK_SCHEMA,
    SCHEMA_VERSION,
    TICK_SCHEMA,
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

__all__ = [
    "TickRecord", "OrderBookRecord", "NewsRecord",
    "encode_tick", "encode_orderbook", "encode_news",
    "decode_tick", "decode_orderbook", "decode_news",
    "TICK_SCHEMA", "ORDERBOOK_SCHEMA", "NEWS_SCHEMA", "SCHEMA_VERSION",
    "StreamSentinelProducer", "ProducerConfig",
    "StreamSentinelConsumer", "ConsumerConfig", "IncomingRecord",
    "run_replay", "run_alpaca", "run_newsapi",
]
