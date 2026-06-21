"""StreamSentinel — dashboard package."""

from dashboard.data import (
    DEFAULT_API_BASE,
    fetch_health,
    fetch_latest,
    fetch_stats,
    latest_to_dataframe,
)

__all__ = [
    "DEFAULT_API_BASE",
    "fetch_health", "fetch_stats", "fetch_latest",
    "latest_to_dataframe",
]
