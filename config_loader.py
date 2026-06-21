"""
streamsentinel/config_loader.py
================================
Central configuration loader for the StreamSentinel project.

Every module imports its settings from here rather than reading config.yaml
directly. This ensures:
  1. A single source of truth — change one file, all modules pick it up.
  2. Environment variable overrides work consistently everywhere.
  3. Secret values (API keys) are never hardcoded in module files.

Usage
-----
    from config_loader import load_config, get_kafka_config, get_model_config

    cfg = load_config()                # full config dict
    kafka_cfg = get_kafka_config()     # kafka-specific sub-dict
"""

from __future__ import annotations

import os
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from loguru import logger

# Locate the repo root (one level above this file) and load .env once.
_REPO_ROOT = Path(__file__).parent
_ENV_PATH = _REPO_ROOT / ".env"

if _ENV_PATH.exists():
    load_dotenv(_ENV_PATH)
    logger.debug(f"Loaded environment from {_ENV_PATH}")
else:
    logger.debug(".env not found — relying on OS environment variables")


def _interpolate_env_vars(obj: Any) -> Any:
    """
    Recursively walk a parsed YAML structure and substitute ${VAR} and
    ${VAR:-default} placeholders with environment variable values.

    Parameters
    ----------
    obj : Any
        A dict, list, or scalar value from yaml.safe_load.

    Returns
    -------
    Any
        The same structure with all placeholders resolved.

    Raises
    ------
    ValueError
        If a required placeholder (no default) is missing from the environment.
    """
    if isinstance(obj, dict):
        return {k: _interpolate_env_vars(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_interpolate_env_vars(item) for item in obj]
    if isinstance(obj, str):
        # Match ${VAR} and ${VAR:-default}
        pattern = re.compile(r"\$\{([^}:]+)(?::-(.*?))?\}")
        def replacer(match: re.Match) -> str:
            var_name = match.group(1)
            default = match.group(2)          # None if no :- default
            value = os.environ.get(var_name)
            if value is not None:
                return value
            if default is not None:
                return default
            raise ValueError(
                f"Environment variable '{var_name}' is required in config.yaml "
                f"but is not set. Add it to your .env file."
            )
        return pattern.sub(replacer, obj)
    return obj


@lru_cache(maxsize=1)
def load_config(config_path: str | None = None) -> dict[str, Any]:
    """
    Load, validate, and cache the project configuration.

    The config is loaded once and cached for the lifetime of the process.
    Call ``load_config.cache_clear()`` in tests that need a fresh config.

    Parameters
    ----------
    config_path : str | None
        Path to config.yaml. Defaults to <repo_root>/config.yaml.

    Returns
    -------
    dict[str, Any]
        Fully resolved configuration dictionary.
    """
    path = Path(config_path) if config_path else _REPO_ROOT / "config.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"config.yaml not found at {path}. "
            "Make sure you are running from the repository root."
        )
    with open(path, "r") as fh:
        raw = yaml.safe_load(fh)
    resolved = _interpolate_env_vars(raw)
    logger.info(f"Configuration loaded from {path}")
    return resolved


# ---------------------------------------------------------------------------
# Convenience accessors — each returns the relevant sub-dict so callers
# don't have to drill into the full config every time.
# ---------------------------------------------------------------------------

def get_kafka_config() -> dict[str, Any]:
    """Return the kafka section of the config."""
    return load_config()["kafka"]


def get_spark_config() -> dict[str, Any]:
    """Return the spark section of the config."""
    return load_config()["spark"]


def get_model_config() -> dict[str, Any]:
    """Return the models section of the config."""
    return load_config()["models"]


def get_training_config() -> dict[str, Any]:
    """Return the training section of the config."""
    return load_config()["training"]


def get_redis_config() -> dict[str, Any]:
    """Return the redis section of the config."""
    return load_config()["redis"]


def get_timescaledb_config() -> dict[str, Any]:
    """Return the timescaledb section of the config."""
    return load_config()["timescaledb"]


def get_mlflow_config() -> dict[str, Any]:
    """Return the mlflow section of the config."""
    return load_config()["mlflow"]


def get_evaluation_config() -> dict[str, Any]:
    """Return the evaluation section of the config."""
    return load_config()["evaluation"]


def get_synthetic_config() -> dict[str, Any]:
    """Return the synthetic anomaly injection section."""
    return load_config()["synthetic"]


def get_data_sources_config() -> dict[str, Any]:
    """Return the data_sources section (Alpaca, Polygon, NewsAPI, GDELT)."""
    return load_config()["data_sources"]


def get_graph_config() -> dict[str, Any]:
    """Return the graph construction section."""
    return load_config()["graph"]


def get_api_config() -> dict[str, Any]:
    """Return the FastAPI inference service config."""
    return load_config()["api"]


def get_explainability_config() -> dict[str, Any]:
    """Return the explainability (SHAP + attention) config."""
    return load_config()["explainability"]
