"""StreamSentinel — explainability package.

Public API:
  - SHAPExplainer, SHAPExplainerConfig, SHAPResult
  - AttentionVisualiser, AttentionResult
"""

from explainability.attention_visualiser import (
    AttentionResult,
    AttentionVisualiser,
)
from explainability.shap_explainer import (
    SHAPExplainer,
    SHAPExplainerConfig,
    SHAPResult,
)

__all__ = [
    "SHAPExplainer", "SHAPExplainerConfig", "SHAPResult",
    "AttentionVisualiser", "AttentionResult",
]
