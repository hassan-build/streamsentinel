# Dashboard Module

Streamlit demo UI for StreamSentinel. Reads live anomaly scores from
the FastAPI service + Redis cache, renders three panels:

1. **Live anomaly feed** — table of the most recent predictions per symbol
2. **Per-symbol score timeline** — Plotly line chart of anomaly probability over the last 60s
3. **System health** — green/red indicators for Kafka / Redis / model / API

Plus a control panel: pause/resume auto-refresh, set a probability threshold.

## How it gets data

```
Streamlit ──HTTP──> FastAPI /latest  ──── reads ──> Redis cache
Streamlit ──HTTP──> FastAPI /stats   ──── reads ──> StatsBuffer
Streamlit ──HTTP──> FastAPI /health  ──── reads ──> service state
```

This is deliberately simple. The dashboard doesn't talk to Kafka or
Redis directly — it goes through FastAPI so the network surface stays
small (one port to expose) and the dashboard works against any
StreamSentinel deployment that exposes the same endpoints.

## Run

```bash
# 1. Start the API (also starts the streaming loop)
python -m api.service

# 2. Start a data producer (so the streaming loop has input)
python -m ingestion.kafka_producer --source replay --speed 10x

# 3. Start the dashboard
streamlit run dashboard/streamlit_app.py
```

Open <http://localhost:8501>.

## Files

- `streamlit_app.py` — the UI page
- `data.py` — pure-function HTTP fetchers (unit-tested)

## Tests

```bash
pytest tests/test_dashboard.py -v
```

The UI rendering itself isn't unit-tested (Streamlit doesn't have a
good story for that), but the data-fetching layer is — that's where
the dashboard could realistically break.
