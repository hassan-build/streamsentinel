"""
StreamSentinel — graph construction package.

Public API:
  - GraphBuilder, GraphBuilderConfig (stateless builder)
  - DynamicGraphUpdater, DynamicGraphUpdaterConfig (streaming wrapper)
  - FEATURE_NAMES, NODE_FEATURE_DIM, EDGE_FEATURE_DIM (schema constants)
"""

from graph.dynamic_graph_updater import (
    DynamicGraphUpdater,
    DynamicGraphUpdaterConfig,
)
from graph.graph_builder import (
    EDGE_FEATURE_DIM,
    FEATURE_NAMES,
    NODE_FEATURE_DIM,
    GraphBuilder,
    GraphBuilderConfig,
)

__all__ = [
    "GraphBuilder",
    "GraphBuilderConfig",
    "DynamicGraphUpdater",
    "DynamicGraphUpdaterConfig",
    "FEATURE_NAMES",
    "NODE_FEATURE_DIM",
    "EDGE_FEATURE_DIM",
]
