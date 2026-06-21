"""StreamSentinel — api package.

Public API:
  - create_app                    FastAPI app factory
  - StreamingInferenceLoop        Background loop
  - build_pipeline_from_config    Pipeline construction
  - load_checkpoint               Load weights into pipeline
  - StatsBuffer, RedisClient      State helpers
"""

from api.model_loader import build_pipeline_from_config, load_checkpoint
from api.service import create_app
from api.state import RedisClient, StatsBuffer
from api.streaming_loop import StreamingInferenceLoop, StreamingLoopConfig

__all__ = [
    "create_app",
    "build_pipeline_from_config", "load_checkpoint",
    "StatsBuffer", "RedisClient",
    "StreamingInferenceLoop", "StreamingLoopConfig",
]
