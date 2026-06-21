"""
dashboard/streamlit_app.py
==========================
Streamlit demo UI for StreamSentinel.

Run with:
    streamlit run dashboard/streamlit_app.py

Three panels (top → bottom):
  1. System health badges + control panel sidebar
  2. Live anomaly score timeline (per symbol, Plotly)
  3. Latest prediction per symbol (DataFrame)
"""

from __future__ import annotations

import sys
import time
from collections import deque
from pathlib import Path

# Make `dashboard.data` importable when launched via `streamlit run ...`
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import plotly.graph_objects as go
import streamlit as st

from dashboard.data import (
    DEFAULT_API_BASE,
    fetch_health,
    fetch_latest,
    fetch_stats,
    latest_to_dataframe,
)


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="StreamSentinel — Live",
    page_icon="📡",
    layout="wide",
)
st.title("StreamSentinel — Live Anomaly Detection")
st.caption(
    "Real-time order-book monitoring with GNN + LLM fusion. "
    "Predictions stream from the FastAPI service."
)


# ---------------------------------------------------------------------------
# Sidebar — control panel
# ---------------------------------------------------------------------------

st.sidebar.header("Settings")
api_base = st.sidebar.text_input("API URL", value=DEFAULT_API_BASE)
refresh_seconds = st.sidebar.slider(
    "Refresh every (seconds)", min_value=1, max_value=10, value=2,
)
threshold = st.sidebar.slider(
    "Anomaly threshold", min_value=0.0, max_value=1.0,
    value=0.5, step=0.05,
    help="Display threshold only — does not change inference."
)
auto_refresh = st.sidebar.checkbox("Auto-refresh", value=True)
if st.sidebar.button("Refresh now"):
    st.rerun()


# ---------------------------------------------------------------------------
# History buffer in session state
# ---------------------------------------------------------------------------

if "history" not in st.session_state:
    # symbol -> deque of (timestamp_ms, anomaly_score) tuples
    st.session_state.history = {}
HISTORY_MAX = 120  # ~2 minutes at 1s refresh


def _append_history(latest_records: dict[str, dict]) -> None:
    """Add the latest snapshot to the history rolling window."""
    for sym, payload in latest_records.items():
        h = st.session_state.history.setdefault(sym, deque(maxlen=HISTORY_MAX))
        ts = payload.get("timestamp_ms", int(time.time() * 1000))
        score = payload.get("anomaly_score", 0.0)
        # Avoid appending the same timestamp twice.
        if not h or h[-1][0] != ts:
            h.append((ts, score))


# ---------------------------------------------------------------------------
# Fetch data this render
# ---------------------------------------------------------------------------

health = fetch_health(api_base)
stats = fetch_stats(api_base)
latest = fetch_latest(api_base)
_append_history(latest)


# ---------------------------------------------------------------------------
# Panel 1 — system health
# ---------------------------------------------------------------------------

c1, c2, c3, c4, c5, c6 = st.columns(6)


def _badge(col, label: str, ok: bool, value: str | None = None) -> None:
    sym = "🟢" if ok else "🔴"
    col.metric(label, value if value is not None else (sym + " up" if ok else sym + " down"))


_badge(c1, "API", health.get("status") == "ok")
_badge(c2, "Model", health.get("model_loaded", False))
_badge(c3, "Redis", health.get("redis_connected", False))
_badge(c4, "Streaming", health.get("streaming_loop_active", False))
c5.metric("p95 latency (ms)", f"{stats['latency_p95_ms']:.1f}")
c6.metric("Anomaly rate", f"{stats['anomaly_rate'] * 100:.1f}%")

st.divider()


# ---------------------------------------------------------------------------
# Panel 2 — anomaly score timeline
# ---------------------------------------------------------------------------

st.subheader("Live anomaly scores")
if st.session_state.history:
    fig = go.Figure()
    for sym, h in st.session_state.history.items():
        if not h:
            continue
        xs = [t for t, _ in h]
        ys = [s for _, s in h]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines+markers", name=sym,
        ))
    fig.add_hline(
        y=threshold, line_dash="dash", line_color="red",
        annotation_text=f"threshold = {threshold:.2f}",
        annotation_position="bottom right",
    )
    fig.update_layout(
        height=380, margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="timestamp (ms)",
        yaxis_title="P(anomaly)",
        yaxis_range=[0, 1],
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info(
        "No predictions yet. Start the producer "
        "(`python -m ingestion.kafka_producer --source replay --speed 10x`) "
        "and the chart will populate as the streaming loop processes "
        "messages."
    )

st.divider()


# ---------------------------------------------------------------------------
# Panel 3 — latest prediction per symbol
# ---------------------------------------------------------------------------

st.subheader("Latest prediction per symbol")
df = latest_to_dataframe(latest)
if not df.empty:
    df_display = df.copy()
    df_display["anomaly_score"] = df_display["anomaly_score"].round(3)
    df_display["inference_latency_ms"] = df_display[
        "inference_latency_ms"
    ].round(2)
    df_display["above_threshold"] = (
        df_display["anomaly_score"] >= threshold
    )
    st.dataframe(df_display, hide_index=True, use_container_width=True)
else:
    st.info("Waiting for first prediction…")


# ---------------------------------------------------------------------------
# Footer / counters
# ---------------------------------------------------------------------------

st.caption(
    f"Predictions made: {stats['n_predictions']:,} | "
    f"Anomalies detected: {stats['n_anomalies_detected']:,} | "
    f"Uptime: {stats['uptime_seconds']}s"
)


# ---------------------------------------------------------------------------
# Auto-refresh
# ---------------------------------------------------------------------------

if auto_refresh:
    time.sleep(refresh_seconds)
    st.rerun()
