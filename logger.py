"""
streamsentinel/logger.py
========================
Centralised logging configuration using Loguru.

Every module in StreamSentinel calls ``get_logger(__name__)`` instead of
``logging.getLogger(__name__)``. This ensures all log output is formatted
consistently, written to a rotating file, and uses the log level defined in
config.yaml.

Why Loguru over stdlib logging?
--------------------------------
  - Zero-boilerplate: no Handler/Formatter setup per module
  - Structured JSON output option with one flag
  - Built-in log rotation and retention
  - Better stack traces with variable values shown

Usage
-----
    from logger import get_logger
    log = get_logger(__name__)
    log.info("Module started")
    log.debug(f"Processing {n_records} records")
"""

from __future__ import annotations

import sys
from pathlib import Path

from loguru import logger


def setup_logging(
    level: str = "INFO",
    log_file: str = "./logs/streamsentinel.log",
    rotation: str = "100 MB",
    retention: str = "7 days",
    fmt: str = "{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{line} | {message}",
) -> None:
    """
    Configure global logging for the StreamSentinel process.

    Call this once at process startup (e.g. in main.py or the CLI entrypoint).
    Individual modules should not call this — they only call ``get_logger``.

    Parameters
    ----------
    level : str
        Minimum log level to capture. One of DEBUG, INFO, WARNING, ERROR.
    log_file : str
        Path to the rotating log file.
    rotation : str
        Loguru rotation rule, e.g. "100 MB" or "1 day".
    retention : str
        How long to keep rotated files, e.g. "7 days".
    fmt : str
        Loguru format string for all sinks.
    """
    # Remove the default handler so we can add our own with the right level.
    logger.remove()

    # Console sink — colourised for human readability.
    logger.add(
        sys.stderr,
        format=fmt,
        level=level,
        colorize=True,
        backtrace=True,       # show full stack trace on exceptions
        diagnose=False,       # set True in dev to show variable values
    )

    # File sink — plain text, auto-rotated.
    log_path = Path(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger.add(
        log_path,
        format=fmt,
        level=level,
        rotation=rotation,
        retention=retention,
        compression="zip",
        backtrace=True,
        diagnose=False,
    )

    logger.info(
        f"Logging initialised | level={level} | file={log_file}"
    )


def get_logger(name: str):
    """
    Return a Loguru logger bound to the given module name.

    Using ``logger.bind(module=name)`` lets us filter logs by module name
    in the file sink, which is useful for debugging specific pipeline stages.

    Parameters
    ----------
    name : str
        Typically ``__name__`` from the calling module.

    Returns
    -------
    loguru.Logger
        A logger instance with the module name bound as context.

    Example
    -------
        from logger import get_logger
        log = get_logger(__name__)
        log.info("Starting Kafka producer")
    """
    return logger.bind(module=name)


# ---------------------------------------------------------------------------
# Auto-initialise with sensible defaults when the module is first imported.
# Config-aware code (main.py) can call setup_logging() again to override.
# ---------------------------------------------------------------------------
try:
    from config_loader import load_config
    _cfg = load_config().get("logging", {})
    setup_logging(
        level=_cfg.get("level", "INFO"),
        log_file=_cfg.get("file", "./logs/streamsentinel.log"),
        rotation=_cfg.get("rotation", "100 MB"),
        retention=_cfg.get("retention", "7 days"),
    )
except Exception:
    # Config may not be available yet (e.g. during tests) — use defaults.
    setup_logging()
