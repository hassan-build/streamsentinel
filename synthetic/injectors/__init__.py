"""
synthetic/injectors/__init__.py
===============================
Package of anomaly injectors used by the synthetic dataset assembler.

Exposes a unified registry that maps human-readable scenario names
(as they appear in config.yaml) to their corresponding injector classes.
"""

from synthetic.injectors.base import (
    LABEL_COORDINATED,
    LABEL_FLASH_CRASH,
    LABEL_LAYERING,
    LABEL_LIQUIDITY_SHOCK,
    LABEL_NORMAL,
    LABEL_SPOOFING,
    AnomalyInjector,
    InjectionResult,
)
from synthetic.injectors.coordinated_trading import CoordinatedTradingInjector
from synthetic.injectors.flash_crash import FlashCrashInjector
from synthetic.injectors.layering import LayeringInjector
from synthetic.injectors.liquidity_shock import LiquidityShockInjector
from synthetic.injectors.spoofing import SpoofingInjector

# Registry: config.yaml scenario name -> injector class.
# Add new scenarios here when extending the system; the dataset
# assembler picks up new entries automatically.
INJECTOR_REGISTRY: dict[str, type[AnomalyInjector]] = {
    "spoofing": SpoofingInjector,
    "layering": LayeringInjector,
    "flash_crash": FlashCrashInjector,
    "coordinated_trading": CoordinatedTradingInjector,
    "liquidity_shock": LiquidityShockInjector,
}

__all__ = [
    "AnomalyInjector",
    "InjectionResult",
    "INJECTOR_REGISTRY",
    "SpoofingInjector",
    "LayeringInjector",
    "FlashCrashInjector",
    "CoordinatedTradingInjector",
    "LiquidityShockInjector",
    "LABEL_NORMAL",
    "LABEL_SPOOFING",
    "LABEL_LAYERING",
    "LABEL_FLASH_CRASH",
    "LABEL_COORDINATED",
    "LABEL_LIQUIDITY_SHOCK",
]
