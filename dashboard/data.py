"""
dashboard/data.py
=================
HTTP fetchers used by the Streamlit dashboard.

These are pure functions taking an `api_base` URL and returning typed
dicts. Factored out of the UI so they can be unit-tested without
spinning up Streamlit.
"""

from __future__ import annotations

from typing import Any

import requests


DEFAULT_API_BASE = "http://localhost:8000"
DEFAULT_TIMEOUT = 2.0


def fetch_health(api_base: str = DEFAULT_API_BASE,
                 timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Return `/health` response or a sentinel dict on failure.

    Never raises — UI must keep rendering even when the API is down.
    """
    try:
        r = requests.get(f"{api_base}/health", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as exc:  # noqa: BLE001
        return {
            "status": "unreachable",
            "error": str(exc),
            "model_loaded": False,
            "redis_connected": False,
            "streaming_loop_active": False,
            "predictions_made": 0,
        }


def fetch_stats(api_base: str = DEFAULT_API_BASE,
                timeout: float = DEFAULT_TIMEOUT) -> dict[str, Any]:
    """Return `/stats` response or zeros on failure."""
    try:
        r = requests.get(f"{api_base}/stats", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {
            "n_predictions": 0,
            "n_anomalies_detected": 0,
            "anomaly_rate": 0.0,
            "latency_p50_ms": 0.0,
            "latency_p95_ms": 0.0,
            "latency_p99_ms": 0.0,
            "uptime_seconds": 0,
        }


def fetch_latest(api_base: str = DEFAULT_API_BASE,
                 timeout: float = DEFAULT_TIMEOUT
                 ) -> dict[str, dict[str, Any]]:
    """Return `/latest` response — Redis snapshot. Empty dict on failure."""
    try:
        r = requests.get(f"{api_base}/latest", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {}


def latest_to_dataframe(latest: dict[str, dict[str, Any]]):
    """Flatten the `/latest` dict into a DataFrame for st.dataframe.

    Returns a DataFrame with columns: symbol, predicted_class,
    anomaly_score, is_anomaly, timestamp_ms, inference_latency_ms.
    """
    import pandas as pd

    rows: list[dict[str, Any]] = []
    for sym, payload in latest.items():
        rows.append({
            "symbol": payload.get("symbol", sym),
            "predicted_class": payload.get("predicted_class", "?"),
            "anomaly_score": payload.get("anomaly_score", 0.0),
            "is_anomaly": payload.get("is_anomaly", False),
            "inference_latency_ms": payload.get(
                "inference_latency_ms", 0.0
            ),
            "timestamp_ms": payload.get("timestamp_ms", 0),
        })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values("anomaly_score", ascending=False)
    return df
