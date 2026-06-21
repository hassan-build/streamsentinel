# Grafana Operational Dashboard

Auto-provisioned Grafana dashboard for StreamSentinel.

## What this provides

A second monitoring surface (the Streamlit dashboard at port 8501 is
the primary one). Grafana is geared towards ops-style metrics: long
time-range queries, alerting, multi-tenant access. The Streamlit
dashboard is for live demos and interactive exploration.

## Files

| File | Purpose |
|---|---|
| `provisioning/datasources/timescaledb.yml` | Auto-configures the PostgreSQL datasource pointing at the TimescaleDB container |
| `provisioning/dashboards/dashboards.yml` | Tells Grafana to load every JSON in the dashboards folder |
| `provisioning/dashboards/streamsentinel_ops.json` | The operational dashboard: anomaly rate, latest scores, latency, class distribution |

## Panels

1. **Anomaly rate (rolling 5 min)** — line chart of fraction of predictions flagged as anomalous per minute
2. **Latest anomaly score by symbol** — bar chart with red/yellow/green thresholds at 0.5/0.75
3. **Inference latency (p50/p95/p99)** — three-line time-series of latency percentiles
4. **Predicted class distribution** — donut chart of normal vs spoofing vs layering etc.

## Current state

The dashboard panels query a `predictions` table in TimescaleDB. The
table **does not exist yet** — the streaming loop currently writes
predictions to Redis (for the Streamlit dashboard) and Kafka topic
`anomaly.scores` (for downstream consumers), but not TimescaleDB.

A future revision will add a TimescaleDB writer to the streaming loop;
this is documented as future work in the report. Until then the
Grafana panels show "No data".

The provisioning still has value: it demonstrates the operational
monitoring story end-to-end (datasource configured, dashboard layout
designed, queries written), and shows where production telemetry
would flow.

## Access

After `docker compose up -d`, open http://localhost:3000

Default login: `admin` / `admin`

Grafana will prompt you to change the password on first login. For a
local dissertation demo, "admin/admin" again is fine.
