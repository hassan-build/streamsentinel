"""
StreamSentinel — synthetic data generation package.

See synthetic/README.md for the full description.
"""

from synthetic.base_market import (
    BaseMarketConfig,
    BaseMarketSimulator,
    OrderBookSnapshot,
)
from synthetic.injectors import (
    INJECTOR_REGISTRY,
    AnomalyInjector,
    CoordinatedTradingInjector,
    FlashCrashInjector,
    InjectionResult,
    LayeringInjector,
    LiquidityShockInjector,
    SpoofingInjector,
)

__all__ = [
    "BaseMarketConfig",
    "BaseMarketSimulator",
    "OrderBookSnapshot",
    "AnomalyInjector",
    "InjectionResult",
    "INJECTOR_REGISTRY",
    "SpoofingInjector",
    "LayeringInjector",
    "FlashCrashInjector",
    "CoordinatedTradingInjector",
    "LiquidityShockInjector",
]
