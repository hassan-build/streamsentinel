"""
tests/test_dashboard.py
=======================
Unit tests for the dashboard's data layer.

Streamlit's UI rendering itself isn't unit-tested (no good story for
that), but `dashboard/data.py` is pure and we cover it thoroughly.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
import requests

from dashboard.data import (
    fetch_health,
    fetch_latest,
    fetch_stats,
    latest_to_dataframe,
)


# ---------------------------------------------------------------------------
# Fetchers
# ---------------------------------------------------------------------------

class _FakeResponse:
    def __init__(self, status_code: int, body: dict | None = None) -> None:
        self.status_code = status_code
        self._body = body or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

    def json(self) -> dict:
        return self._body


class TestFetchHealth:
    def test_returns_body_on_success(self):
        with patch("dashboard.data.requests.get",
                   return_value=_FakeResponse(200, {"status": "ok",
                                                    "model_loaded": True})):
            out = fetch_health()
            assert out["status"] == "ok"

    def test_returns_sentinel_on_connection_error(self):
        with patch("dashboard.data.requests.get",
                   side_effect=requests.ConnectionError("nope")):
            out = fetch_health()
            assert out["status"] == "unreachable"
            assert out["model_loaded"] is False

    def test_returns_sentinel_on_http_error(self):
        with patch("dashboard.data.requests.get",
                   return_value=_FakeResponse(500)):
            out = fetch_health()
            assert out["status"] == "unreachable"


class TestFetchStats:
    def test_returns_body_on_success(self):
        body = {
            "n_predictions": 100, "n_anomalies_detected": 10,
            "anomaly_rate": 0.1, "latency_p50_ms": 5.0,
            "latency_p95_ms": 12.0, "latency_p99_ms": 20.0,
            "uptime_seconds": 60,
        }
        with patch("dashboard.data.requests.get",
                   return_value=_FakeResponse(200, body)):
            out = fetch_stats()
            assert out == body

    def test_returns_zeros_on_failure(self):
        with patch("dashboard.data.requests.get",
                   side_effect=requests.Timeout("slow")):
            out = fetch_stats()
            assert out["n_predictions"] == 0
            assert out["latency_p95_ms"] == 0.0


class TestFetchLatest:
    def test_returns_dict_on_success(self):
        body = {"AAPL": {"symbol": "AAPL", "anomaly_score": 0.42}}
        with patch("dashboard.data.requests.get",
                   return_value=_FakeResponse(200, body)):
            out = fetch_latest()
            assert "AAPL" in out

    def test_returns_empty_on_failure(self):
        with patch("dashboard.data.requests.get",
                   side_effect=requests.ConnectionError("nope")):
            assert fetch_latest() == {}


# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------

class TestLatestToDataFrame:
    def test_empty_input(self):
        df = latest_to_dataframe({})
        assert df.empty

    def test_populated_input_sorted_by_score(self):
        payload = {
            "AAPL": dict(symbol="AAPL", predicted_class="normal",
                         anomaly_score=0.1, is_anomaly=False,
                         inference_latency_ms=5.0, timestamp_ms=1),
            "MSFT": dict(symbol="MSFT", predicted_class="spoofing",
                         anomaly_score=0.8, is_anomaly=True,
                         inference_latency_ms=6.0, timestamp_ms=2),
            "TSLA": dict(symbol="TSLA", predicted_class="layering",
                         anomaly_score=0.5, is_anomaly=True,
                         inference_latency_ms=7.0, timestamp_ms=3),
        }
        df = latest_to_dataframe(payload)
        assert len(df) == 3
        # Sorted descending by anomaly_score.
        assert df.iloc[0]["symbol"] == "MSFT"
        assert df.iloc[1]["symbol"] == "TSLA"
        assert df.iloc[2]["symbol"] == "AAPL"
        assert {"symbol", "predicted_class", "anomaly_score",
                "is_anomaly", "inference_latency_ms",
                "timestamp_ms"} == set(df.columns)
