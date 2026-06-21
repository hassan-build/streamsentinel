-- =============================================================================
-- StreamSentinel — TimescaleDB Initialisation Schema
-- =============================================================================
-- Run automatically by Docker Compose on first container start.
-- Creates the hypertables used to store labelled anomaly events and
-- raw trade data with time-series indexing.
-- =============================================================================

-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ---------------------------------------------------------------------------
-- anomaly_events — stores each anomaly prediction from the AI layer
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS anomaly_events (
    time            TIMESTAMPTZ     NOT NULL,
    symbol          TEXT            NOT NULL,
    anomaly_type    TEXT            NOT NULL,   -- e.g. spoofing, flash_crash
    anomaly_score   DOUBLE PRECISION NOT NULL,  -- probability output [0, 1]
    is_anomaly      BOOLEAN         NOT NULL,
    threshold_used  DOUBLE PRECISION,
    pipeline_stage  TEXT,                       -- which stage produced this
    latency_ms      DOUBLE PRECISION,           -- end-to-end pipeline latency
    model_version   TEXT,
    metadata        JSONB                        -- arbitrary extra fields
);

-- Convert to TimescaleDB hypertable (chunks by day)
SELECT create_hypertable('anomaly_events', 'time', if_not_exists => TRUE);

-- Index for fast symbol lookups
CREATE INDEX IF NOT EXISTS idx_anomaly_events_symbol
    ON anomaly_events (symbol, time DESC);

-- ---------------------------------------------------------------------------
-- trade_labels — ground truth labels (real or synthetic) for evaluation
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS trade_labels (
    time            TIMESTAMPTZ     NOT NULL,
    symbol          TEXT            NOT NULL,
    label           TEXT            NOT NULL,   -- normal | spoofing | etc.
    label_source    TEXT            NOT NULL,   -- synthetic | manual | sec_filing
    confidence      DOUBLE PRECISION DEFAULT 1.0,
    metadata        JSONB
);

SELECT create_hypertable('trade_labels', 'time', if_not_exists => TRUE);

CREATE INDEX IF NOT EXISTS idx_trade_labels_symbol
    ON trade_labels (symbol, time DESC);

-- ---------------------------------------------------------------------------
-- pipeline_latency — per-stage timing records for dissertation evaluation
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pipeline_latency (
    time            TIMESTAMPTZ     NOT NULL DEFAULT NOW(),
    stage           TEXT            NOT NULL,   -- ingestion | processing | gnn | fusion | api
    latency_ms      DOUBLE PRECISION NOT NULL,
    batch_size      INTEGER,
    run_id          TEXT                        -- MLflow run ID for cross-reference
);

SELECT create_hypertable('pipeline_latency', 'time', if_not_exists => TRUE);

-- ---------------------------------------------------------------------------
-- Views for Grafana dashboards
-- ---------------------------------------------------------------------------

-- Rolling 5-minute anomaly rate per symbol
CREATE OR REPLACE VIEW v_anomaly_rate_5min AS
SELECT
    time_bucket('5 minutes', time) AS bucket,
    symbol,
    COUNT(*) FILTER (WHERE is_anomaly) AS n_anomalies,
    COUNT(*) AS n_total,
    ROUND(
        100.0 * COUNT(*) FILTER (WHERE is_anomaly) / NULLIF(COUNT(*), 0), 2
    ) AS anomaly_rate_pct
FROM anomaly_events
GROUP BY bucket, symbol
ORDER BY bucket DESC;

-- Average pipeline latency per stage (last 1 hour)
CREATE OR REPLACE VIEW v_latency_summary AS
SELECT
    stage,
    ROUND(AVG(latency_ms)::numeric, 2)  AS avg_ms,
    ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY latency_ms)::numeric, 2) AS p50_ms,
    ROUND(PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY latency_ms)::numeric, 2) AS p95_ms,
    COUNT(*) AS n_observations
FROM pipeline_latency
WHERE time > NOW() - INTERVAL '1 hour'
GROUP BY stage;
