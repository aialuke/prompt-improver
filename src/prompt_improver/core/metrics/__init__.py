"""Core Metrics Module - Unified Metrics Interface for 2025 Best Practices.

This module provides a unified interface for metrics collection that follows
2025 observability standards with OpenTelemetry as the primary backend.

Features:
- Protocol-based dependency injection interface
- OpenTelemetry-first with graceful fallbacks
- Unified API across different metrics backends
- Clean separation of concerns for DI containers
"""

from prompt_improver.core.metrics.unified_metrics_adapter import UnifiedMetricsAdapter

__all__ = ["UnifiedMetricsAdapter"]
