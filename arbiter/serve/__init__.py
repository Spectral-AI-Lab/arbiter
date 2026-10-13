"""ARBITER serving package for high-throughput online inference.

Provides:
- vLLM Structured Decisions API (/v1/systemone)
- Native flat decision API (/decide)
- Dynamic asynchronous micro-batch scheduling
"""

try:
    from .app import create_app
except ImportError:  # pragma: no cover
    create_app = None  # type: ignore[assignment]
from .schemas import (
    DecideRequest,
    DecideResponse,
    HealthResponse,
    MetricsResponse,
    SystemOneQuestion,
    SystemOneRequest,
    SystemOneResponse,
)
from .scheduler import DynamicBatchScheduler, QueueFullError

__all__ = [
    "create_app",
    "DynamicBatchScheduler",
    "QueueFullError",
    "SystemOneRequest",
    "SystemOneResponse",
    "SystemOneQuestion",
    "DecideRequest",
    "DecideResponse",
    "HealthResponse",
    "MetricsResponse",
]
