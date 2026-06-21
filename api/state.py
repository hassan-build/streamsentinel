"""
api/state.py
============
In-process stats counter + Redis client used across the API.

Stats are computed from a fixed-size ring buffer of recent latencies
and prediction events. We avoid pulling stats from MLflow / Prometheus
because the dissertation demo doesn't need a metrics backend — the
buffer is sufficient and the data fits in RAM trivially.
"""

from __future__ import annotations

import json
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from logger import get_logger

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Stats — in-process ring buffer
# ---------------------------------------------------------------------------

@dataclass
class StatsBuffer:
    """Thread-safe rolling-window stats for the API.

    Stores the last `maxlen` latencies and per-call anomaly flags so
    the `/stats` endpoint can serve quick aggregates without scanning
    the full history.
    """
    maxlen: int = 1000
    _latencies: deque = field(default_factory=lambda: deque(maxlen=1000))
    _anomalies: deque = field(default_factory=lambda: deque(maxlen=1000))
    _start_time: float = field(default_factory=time.time)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _n_total: int = 0
    _n_anomalies_total: int = 0

    def record(self, latency_ms: float, n_anomalies: int) -> None:
        """Record one prediction event."""
        with self._lock:
            self._latencies.append(float(latency_ms))
            self._anomalies.append(int(n_anomalies))
            self._n_total += 1
            self._n_anomalies_total += int(n_anomalies)

    def snapshot(self) -> dict[str, Any]:
        """Return a current view of stats."""
        with self._lock:
            lat = np.asarray(self._latencies, dtype=np.float64)
            return {
                "n_predictions": self._n_total,
                "n_anomalies_detected": self._n_anomalies_total,
                "anomaly_rate": (
                    self._n_anomalies_total / max(1, self._n_total)
                ),
                "latency_p50_ms": float(np.percentile(lat, 50)) if lat.size else 0.0,
                "latency_p95_ms": float(np.percentile(lat, 95)) if lat.size else 0.0,
                "latency_p99_ms": float(np.percentile(lat, 99)) if lat.size else 0.0,
                "uptime_seconds": int(time.time() - self._start_time),
            }


# ---------------------------------------------------------------------------
# Redis — thin wrapper with graceful no-op fallback
# ---------------------------------------------------------------------------

class RedisClient:
    """Thin Redis wrapper that degrades gracefully if Redis is unavailable.

    If Redis is unreachable we log a warning ONCE and turn every method
    into a no-op. This means the API + streaming loop continue running
    even if Redis is down — predictions just don't get cached.
    """

    def __init__(self, host: str = "localhost", port: int = 6379,
                 db: int = 0, ttl_seconds: int = 300,
                 prefix: str = "latest:") -> None:
        self.host = host
        self.port = port
        self.db = db
        self.ttl_seconds = ttl_seconds
        self.prefix = prefix
        self._client = None
        self._connected: bool = False
        self._warned: bool = False
        self._connect()

    def _connect(self) -> None:
        try:
            import redis  # type: ignore
            self._client = redis.Redis(
                host=self.host, port=self.port, db=self.db,
                socket_connect_timeout=2.0,
                socket_timeout=2.0,
                decode_responses=True,
            )
            self._client.ping()
            self._connected = True
            log.info(f"Connected to Redis at {self.host}:{self.port}")
        except Exception as exc:  # noqa: BLE001
            if not self._warned:
                log.warning(
                    f"Redis at {self.host}:{self.port} unreachable: {exc}. "
                    "Cache writes will be no-ops."
                )
                self._warned = True
            self._client = None
            self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    def set_latest(self, symbol: str, payload: dict[str, Any]) -> None:
        """Store latest prediction for a symbol."""
        if not self._connected or self._client is None:
            return
        try:
            self._client.setex(
                f"{self.prefix}{symbol}",
                self.ttl_seconds,
                json.dumps(payload),
            )
        except Exception as exc:  # noqa: BLE001
            if not self._warned:
                log.warning(f"Redis set failed: {exc}")
                self._warned = True

    def get_latest(self, symbol: str) -> dict[str, Any] | None:
        """Retrieve latest prediction for a symbol; None if absent."""
        if not self._connected or self._client is None:
            return None
        try:
            raw = self._client.get(f"{self.prefix}{symbol}")
            return json.loads(raw) if raw else None
        except Exception as exc:  # noqa: BLE001
            log.warning(f"Redis get failed: {exc}")
            return None

    def get_all_latest(self) -> dict[str, dict[str, Any]]:
        """Retrieve every cached latest record."""
        if not self._connected or self._client is None:
            return {}
        out: dict[str, dict[str, Any]] = {}
        try:
            for k in self._client.scan_iter(match=f"{self.prefix}*"):
                raw = self._client.get(k)
                if raw:
                    sym = k[len(self.prefix):]
                    out[sym] = json.loads(raw)
        except Exception as exc:  # noqa: BLE001
            log.warning(f"Redis scan failed: {exc}")
        return out

    def ping(self) -> bool:
        if not self._connected or self._client is None:
            return False
        try:
            return bool(self._client.ping())
        except Exception:  # noqa: BLE001
            return False
