"""
scripts/validate_setup.py
==========================
Run this script immediately after cloning the repo and setting up .env.
It checks every dependency, credential, and service connection before
you write a single line of model code.

Usage
-----
    python scripts/validate_setup.py

Exit codes
----------
    0 — all checks passed
    1 — one or more checks failed (details printed to console)
"""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Callable

# Ensure repo root is on sys.path so config_loader is importable
sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Check registry — each tuple is (name, check_function)
# ---------------------------------------------------------------------------
CHECKS: list[tuple[str, Callable[[], tuple[bool, str]]]] = []


def check(name: str):
    """Decorator to register a check function."""
    def decorator(fn: Callable) -> Callable:
        CHECKS.append((name, fn))
        return fn
    return decorator


# ---------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------

@check("Python version ≥ 3.11")
def check_python_version() -> tuple[bool, str]:
    major, minor = sys.version_info[:2]
    ok = (major, minor) >= (3, 11)
    return ok, f"Found Python {major}.{minor}"


@check(".env file exists")
def check_env_file() -> tuple[bool, str]:
    path = Path(".env")
    if path.exists():
        return True, ".env found"
    return False, ".env not found — run: cp .env.example .env"


@check("config.yaml loads without errors")
def check_config_loads() -> tuple[bool, str]:
    try:
        from config_loader import load_config
        load_config.cache_clear()
        cfg = load_config()
        return True, f"Config loaded — project: {cfg['project']['name']}"
    except Exception as exc:
        return False, f"Config failed to load: {exc}"


@check("Required Python packages importable")
def check_packages() -> tuple[bool, str]:
    required = [
        "confluent_kafka",           # the package imports as confluent_kafka
        "pyspark",
        "torch",
        "torch_geometric",
        "transformers",
        "sklearn",         # scikit-learn
        "shap",
        "mlflow",
        "fastapi",
        "redis",
        "psycopg2",
        "yaml",
        "loguru",
        "streamlit",
        "plotly",
    ]
    missing = []
    for pkg in required:
        try:
            importlib.import_module(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        return False, f"Missing packages: {', '.join(missing)} — run: pip install -r requirements.txt"
    return True, f"All {len(required)} required packages importable"


@check("PyTorch version and device")
def check_torch() -> tuple[bool, str]:
    try:
        import torch
        device = "CUDA" if torch.cuda.is_available() else "CPU"
        return True, f"PyTorch {torch.__version__} | device: {device}"
    except Exception as exc:
        return False, str(exc)


@check("Redis connection (local Docker)")
def check_redis() -> tuple[bool, str]:
    try:
        import redis as redis_lib
        host = os.environ.get("REDIS_HOST", "localhost")
        password = os.environ.get("REDIS_PASSWORD") or None
        client = redis_lib.Redis(host=host, port=6379, password=password, socket_timeout=3)
        client.ping()
        return True, f"Redis reachable at {host}:6379"
    except Exception as exc:
        return False, f"Redis not reachable: {exc} — is Docker running? (docker compose up -d)"


@check("TimescaleDB connection (local Docker)")
def check_timescaledb() -> tuple[bool, str]:
    try:
        import psycopg2
        conn = psycopg2.connect(
            host=os.environ.get("TIMESCALE_HOST", "localhost"),
            port=5432,
            dbname="streamsentinel",
            user=os.environ.get("TIMESCALE_USER", "streamsentinel"),
            password=os.environ.get("TIMESCALE_PASSWORD", "streamsentinel_dev_password"),
            connect_timeout=5,
        )
        conn.close()
        return True, "TimescaleDB reachable"
    except Exception as exc:
        return False, f"TimescaleDB not reachable: {exc} — is Docker running?"


@check("Kafka broker reachable")
def check_kafka() -> tuple[bool, str]:
    try:
        from confluent_kafka.admin import AdminClient
        bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
        # For local (no auth) we skip SASL params
        conf: dict = {"bootstrap.servers": bootstrap, "socket.timeout.ms": 5000}
        api_key = os.environ.get("KAFKA_API_KEY", "")
        if api_key and api_key != "test":
            conf.update({
                "security.protocol": "SASL_SSL",
                "sasl.mechanism": "PLAIN",
                "sasl.username": api_key,
                "sasl.password": os.environ.get("KAFKA_API_SECRET", ""),
            })
        client = AdminClient(conf)
        metadata = client.list_topics(timeout=5)
        topics = list(metadata.topics.keys())
        return True, f"Kafka reachable at {bootstrap} | topics: {topics}"
    except Exception as exc:
        return False, f"Kafka not reachable: {exc}"


@check("Alpaca API key format")
def check_alpaca_key() -> tuple[bool, str]:
    key = os.environ.get("ALPACA_API_KEY", "")
    secret = os.environ.get("ALPACA_API_SECRET", "")
    if not key or key in ("your_alpaca_api_key", "test"):
        return False, "ALPACA_API_KEY not set — add to .env (get from app.alpaca.markets)"
    if len(key) < 10 or len(secret) < 10:
        return False, "ALPACA credentials look too short — double-check .env"
    return True, f"Alpaca key set (length={len(key)})"


@check("NewsAPI key format")
def check_newsapi_key() -> tuple[bool, str]:
    key = os.environ.get("NEWSAPI_KEY", "")
    if not key or key in ("your_newsapi_key", "test"):
        return False, "NEWSAPI_KEY not set — add to .env (get from newsapi.org)"
    return True, f"NewsAPI key set (length={len(key)})"


@check("MLflow tracking directory writeable")
def check_mlflow() -> tuple[bool, str]:
    try:
        import mlflow
        uri = os.environ.get("MLFLOW_TRACKING_URI", "./mlruns")
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("setup_validation")
        with mlflow.start_run(run_name="setup_check") as run:
            mlflow.log_param("check", "ok")
        return True, f"MLflow tracking works at {uri} | run_id: {run.info.run_id[:8]}..."
    except Exception as exc:
        return False, f"MLflow error: {exc}"


@check("data/ directory writeable")
def check_data_dir() -> tuple[bool, str]:
    try:
        test_dir = Path("data/synthetic")
        test_dir.mkdir(parents=True, exist_ok=True)
        test_file = test_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
        return True, "data/ directory is writeable"
    except Exception as exc:
        return False, f"Cannot write to data/: {exc}"


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def main() -> int:
    print("\n" + "=" * 70)
    print("  StreamSentinel — Setup Validation")
    print("=" * 70 + "\n")

    results = []
    for name, fn in CHECKS:
        try:
            passed, message = fn()
        except Exception as exc:
            passed, message = False, f"Unexpected error: {exc}"
        status = "✓ PASS" if passed else "✗ FAIL"
        color = "\033[92m" if passed else "\033[91m"
        reset = "\033[0m"
        print(f"  {color}{status}{reset}  {name}")
        print(f"         {message}\n")
        results.append(passed)

    n_passed = sum(results)
    n_total = len(results)
    n_failed = n_total - n_passed

    print("=" * 70)
    if n_failed == 0:
        print(f"\033[92m  All {n_total} checks passed. You are ready to build.\033[0m\n")
        return 0
    else:
        print(
            f"\033[91m  {n_failed}/{n_total} checks failed. "
            "Fix the issues above before continuing.\033[0m\n"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
