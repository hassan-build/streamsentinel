"""
ingestion/schemas.py
====================
Avro schemas and matching Python dataclasses for the three Kafka topics:
  - market.ticks    (TickRecord)
  - orderbook.l2    (OrderBookRecord)
  - news.feed       (NewsRecord)

Why both Avro AND dataclasses?
- Avro: enforces schema at produce/consume time, compact binary wire format
- Dataclasses: ergonomic Python construction with type hints and IDE support

The producer takes a dataclass, serialises to Avro bytes, and writes to
Kafka. The consumer reads bytes, decodes via Avro, and reconstructs the
dataclass. Schema version is embedded in the topic name + checked at
deserialisation time.
"""

from __future__ import annotations

import io
from dataclasses import asdict, dataclass, field
from typing import Any

import fastavro

# Schema version. Increment when changing any field name/type below.
SCHEMA_VERSION: str = "1.0.0"


# ---------------------------------------------------------------------------
# Avro schema definitions
# ---------------------------------------------------------------------------

TICK_SCHEMA: dict[str, Any] = {
    "type": "record",
    "namespace": "streamsentinel",
    "name": "Tick",
    "doc": "Single market tick (best bid/ask snapshot).",
    "fields": [
        {"name": "timestamp", "type": "long",
         "doc": "Millisecond epoch (UTC)"},
        {"name": "symbol", "type": "string"},
        {"name": "mid_price", "type": "double"},
        {"name": "spread_bps", "type": "double"},
        {"name": "trade_imbalance", "type": "double"},
        {"name": "order_cancel_rate", "type": "double"},
        {"name": "source", "type": "string",
         "doc": "Where this tick came from (replay, alpaca, synthetic)"},
        {"name": "schema_version", "type": "string",
         "default": SCHEMA_VERSION},
    ],
}


# Each side of the book gets 10 paired (price, size) levels.
_LEVELS = []
for i in range(1, 11):
    _LEVELS.append({"name": f"bid_l{i}", "type": "double"})
    _LEVELS.append({"name": f"ask_l{i}", "type": "double"})
    _LEVELS.append({"name": f"bidsize_l{i}", "type": "double"})
    _LEVELS.append({"name": f"asksize_l{i}", "type": "double"})

ORDERBOOK_SCHEMA: dict[str, Any] = {
    "type": "record",
    "namespace": "streamsentinel",
    "name": "OrderBookSnapshot",
    "doc": "Level-2 order book snapshot with 10-level depth on each side.",
    "fields": [
        {"name": "timestamp", "type": "long"},
        {"name": "symbol", "type": "string"},
        {"name": "mid_price", "type": "double"},
        {"name": "spread_bps", "type": "double"},
        {"name": "trade_imbalance", "type": "double"},
        {"name": "order_cancel_rate", "type": "double"},
        *_LEVELS,
        {"name": "source", "type": "string"},
        {"name": "schema_version", "type": "string",
         "default": SCHEMA_VERSION},
        # Optional fields that flow through from synthetic data so
        # evaluation can verify against ground truth.
        {"name": "label", "type": ["null", "int"], "default": None,
         "doc": "Ground-truth class label (synthetic data only)."},
        {"name": "anomaly_severity", "type": ["null", "double"],
         "default": None},
        {"name": "injection_id", "type": ["null", "string"], "default": None},
    ],
}


NEWS_SCHEMA: dict[str, Any] = {
    "type": "record",
    "namespace": "streamsentinel",
    "name": "NewsItem",
    "doc": "Single financial news headline with metadata.",
    "fields": [
        {"name": "timestamp", "type": "long"},
        {"name": "headline", "type": "string"},
        {"name": "url", "type": ["null", "string"], "default": None},
        {"name": "source_name", "type": "string"},
        {"name": "symbols", "type": {"type": "array", "items": "string"},
         "default": [],
         "doc": "Tickers mentioned (NER output or naive match)."},
        {"name": "schema_version", "type": "string",
         "default": SCHEMA_VERSION},
    ],
}


# ---------------------------------------------------------------------------
# Parsed schema objects (fastavro caches these for speed)
# ---------------------------------------------------------------------------

PARSED_TICK_SCHEMA = fastavro.parse_schema(TICK_SCHEMA)
PARSED_ORDERBOOK_SCHEMA = fastavro.parse_schema(ORDERBOOK_SCHEMA)
PARSED_NEWS_SCHEMA = fastavro.parse_schema(NEWS_SCHEMA)


# ---------------------------------------------------------------------------
# Dataclasses (the public API for producer code)
# ---------------------------------------------------------------------------

@dataclass
class TickRecord:
    """A single market tick (best bid/ask only)."""
    timestamp: int
    symbol: str
    mid_price: float
    spread_bps: float
    trade_imbalance: float
    order_cancel_rate: float
    source: str = "unknown"
    schema_version: str = SCHEMA_VERSION


@dataclass
class OrderBookRecord:
    """A 10-level L2 order book snapshot.

    All `bid_l1..l10`, `ask_l1..l10`, `bidsize_l1..l10`, `asksize_l1..l10`
    are required. Set unused deeper levels to 0.0 if you only have e.g.
    top-5 depth.
    """
    timestamp: int
    symbol: str
    mid_price: float
    spread_bps: float
    trade_imbalance: float
    order_cancel_rate: float
    # Levels — populated via `from_dict` for ergonomics.
    bid_l1: float = 0.0
    ask_l1: float = 0.0
    bidsize_l1: float = 0.0
    asksize_l1: float = 0.0
    bid_l2: float = 0.0
    ask_l2: float = 0.0
    bidsize_l2: float = 0.0
    asksize_l2: float = 0.0
    bid_l3: float = 0.0
    ask_l3: float = 0.0
    bidsize_l3: float = 0.0
    asksize_l3: float = 0.0
    bid_l4: float = 0.0
    ask_l4: float = 0.0
    bidsize_l4: float = 0.0
    asksize_l4: float = 0.0
    bid_l5: float = 0.0
    ask_l5: float = 0.0
    bidsize_l5: float = 0.0
    asksize_l5: float = 0.0
    bid_l6: float = 0.0
    ask_l6: float = 0.0
    bidsize_l6: float = 0.0
    asksize_l6: float = 0.0
    bid_l7: float = 0.0
    ask_l7: float = 0.0
    bidsize_l7: float = 0.0
    asksize_l7: float = 0.0
    bid_l8: float = 0.0
    ask_l8: float = 0.0
    bidsize_l8: float = 0.0
    asksize_l8: float = 0.0
    bid_l9: float = 0.0
    ask_l9: float = 0.0
    bidsize_l9: float = 0.0
    asksize_l9: float = 0.0
    bid_l10: float = 0.0
    ask_l10: float = 0.0
    bidsize_l10: float = 0.0
    asksize_l10: float = 0.0
    source: str = "unknown"
    schema_version: str = SCHEMA_VERSION
    label: int | None = None
    anomaly_severity: float | None = None
    injection_id: str | None = None

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "OrderBookRecord":
        """Construct from a row dict produced by synthetic/anomaly_injector.py."""
        # Filter out any keys we don't model.
        known = {f for f in cls.__dataclass_fields__.keys()}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)


@dataclass
class NewsRecord:
    """A single news headline."""
    timestamp: int
    headline: str
    source_name: str
    url: str | None = None
    symbols: list[str] = field(default_factory=list)
    schema_version: str = SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Encode / decode helpers
# ---------------------------------------------------------------------------

def encode(record: Any, parsed_schema: dict[str, Any]) -> bytes:
    """Serialise a dataclass record to Avro-encoded bytes.

    Raises
    ------
    ValueError
        If the record's fields don't satisfy the schema. fastavro's
        exception is wrapped in a clearer message.
    """
    if hasattr(record, "__dataclass_fields__"):
        payload = asdict(record)
    elif isinstance(record, dict):
        payload = record
    else:
        raise TypeError(
            f"encode() expects a dataclass or dict, got {type(record)}"
        )

    buf = io.BytesIO()
    try:
        fastavro.schemaless_writer(buf, parsed_schema, payload)
    except Exception as exc:
        raise ValueError(
            f"Avro encoding failed (schema={parsed_schema.get('name', '?')}): "
            f"{exc}"
        ) from exc
    return buf.getvalue()


def decode(data: bytes, parsed_schema: dict[str, Any]) -> dict[str, Any]:
    """Deserialise Avro bytes back to a dict."""
    buf = io.BytesIO(data)
    return fastavro.schemaless_reader(buf, parsed_schema)


# Convenience wrappers — these are what producer/consumer code uses.
def encode_tick(rec: TickRecord) -> bytes:
    return encode(rec, PARSED_TICK_SCHEMA)


def encode_orderbook(rec: OrderBookRecord) -> bytes:
    return encode(rec, PARSED_ORDERBOOK_SCHEMA)


def encode_news(rec: NewsRecord) -> bytes:
    return encode(rec, PARSED_NEWS_SCHEMA)


def decode_tick(data: bytes) -> dict[str, Any]:
    return decode(data, PARSED_TICK_SCHEMA)


def decode_orderbook(data: bytes) -> dict[str, Any]:
    return decode(data, PARSED_ORDERBOOK_SCHEMA)


def decode_news(data: bytes) -> dict[str, Any]:
    return decode(data, PARSED_NEWS_SCHEMA)
