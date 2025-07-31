"""
Unified Metrics Adapter - 2025 Best Practices Implementation

This adapter implements the MetricsRegistryProtocol interface and provides
a unified API for metrics collection across different backends, with
OpenTelemetry as the primary choice following 2025 observability standards.

Features:
- Implements MetricsRegistryProtocol for clean DI integration
- OpenTelemetry-first with automatic fallbacks
- Graceful degradation when telemetry unavailable
- Thread-safe operations with proper error handling
- Support for labels/tags across all backends
"""

import logging
from typing import Any, Dict, Optional
from threading import Lock

from ..protocols.retry_protocols import MetricsRegistryProtocol

logger = logging.getLogger(__name__)


class UnifiedMetricsAdapter(MetricsRegistryProtocol):
    """
    Unified metrics adapter implementing MetricsRegistryProtocol.
    
    Provides a clean interface for dependency injection while supporting
    multiple metrics backends with OpenTelemetry as the preferred choice.
    """
    
    def __init__(self, backend: str = "opentelemetry"):
        """Initialize the unified metrics adapter.
        
        Args:
            backend: Preferred metrics backend ("opentelemetry", "prometheus", "memory")
        """
        self.backend = backend
        self._lock = Lock()
        self._initialized = False
        self._otel_available = False
        self._prometheus_available = False
        self._fallback_metrics: Dict[str, Any] = {}
        
        # Initialize the preferred backend
        self._initialize_backend()
        
        logger.info(f"UnifiedMetricsAdapter initialized with backend: {backend}")
    
    def _initialize_backend(self) -> None:
        """Initialize the metrics backend with fallback strategy."""
        with self._lock:
            if self._initialized:
                return
            
            # Try OpenTelemetry first (2025 standard)
            if self.backend == "opentelemetry" or self.backend == "auto":
                try:
                    from ...monitoring.opentelemetry.metrics import record_counter, record_histogram, record_gauge
                    self._otel_record_counter = record_counter
                    self._otel_record_histogram = record_histogram
                    self._otel_record_gauge = record_gauge
                    self._otel_available = True
                    logger.info("OpenTelemetry metrics backend initialized")
                except ImportError as e:
                    logger.warning(f"OpenTelemetry not available: {e}")
            
            # Try Prometheus fallback
            if not self._otel_available and (self.backend == "prometheus" or self.backend == "auto"):
                try:
                    from ...performance.monitoring.metrics_registry import get_metrics_registry
                    self._prometheus_registry = get_metrics_registry()
                    self._prometheus_available = True
                    logger.info("Prometheus metrics backend initialized")
                except ImportError as e:
                    logger.warning(f"Prometheus not available: {e}")
            
            # Always have in-memory fallback
            if not self._otel_available and not self._prometheus_available:
                logger.info("Using in-memory metrics fallback")
            
            self._initialized = True
    
    def increment_counter(self, name: str, labels: Optional[Dict] = None) -> None:
        """Increment a counter metric.
        
        Args:
            name: Counter name
            labels: Optional labels/tags for the metric
        """
        try:
            if self._otel_available:
                # Use OpenTelemetry
                self._otel_record_counter(name, 1, labels)
            elif self._prometheus_available:
                # Use Prometheus via registry
                counter = self._prometheus_registry.get_or_create_counter(
                    name, f"Counter metric: {name}", list(labels.keys()) if labels else None
                )
                if labels and hasattr(counter, 'labels'):
                    counter.labels(**labels).inc()
                else:
                    counter.inc()
            else:
                # In-memory fallback
                key = f"{name}:{labels}" if labels else name
                self._fallback_metrics[key] = self._fallback_metrics.get(key, 0) + 1
                
        except Exception as e:
            logger.error(f"Failed to increment counter {name}: {e}")
    
    def record_histogram(self, name: str, value: float, labels: Optional[Dict] = None) -> None:
        """Record a histogram value.
        
        Args:
            name: Histogram name
            value: Value to record
            labels: Optional labels/tags for the metric
        """
        try:
            if self._otel_available:
                # Use OpenTelemetry
                self._otel_record_histogram(name, value, labels)
            elif self._prometheus_available:
                # Use Prometheus via registry
                histogram = self._prometheus_registry.get_or_create_histogram(
                    name, f"Histogram metric: {name}", list(labels.keys()) if labels else None
                )
                if labels and hasattr(histogram, 'labels'):
                    histogram.labels(**labels).observe(value)
                else:
                    histogram.observe(value)
            else:
                # In-memory fallback - store latest value
                key = f"{name}:{labels}" if labels else name
                self._fallback_metrics[key] = value
                
        except Exception as e:
            logger.error(f"Failed to record histogram {name}: {e}")
    
    def set_gauge(self, name: str, value: float, labels: Optional[Dict] = None) -> None:
        """Set a gauge value.
        
        Args:
            name: Gauge name
            value: Value to set
            labels: Optional labels/tags for the metric
        """
        try:
            if self._otel_available:
                # Use OpenTelemetry
                self._otel_record_gauge(name, value, labels)
            elif self._prometheus_available:
                # Use Prometheus via registry
                gauge = self._prometheus_registry.get_or_create_gauge(
                    name, f"Gauge metric: {name}", list(labels.keys()) if labels else None
                )
                if labels and hasattr(gauge, 'labels'):
                    gauge.labels(**labels).set(value)
                else:
                    gauge.set(value)
            else:
                # In-memory fallback
                key = f"{name}:{labels}" if labels else name
                self._fallback_metrics[key] = value
                
        except Exception as e:
            logger.error(f"Failed to set gauge {name}: {e}")
    
    def get_metric(self, name: str) -> Optional[Any]:
        """Get a metric by name.
        
        Args:
            name: Metric name
            
        Returns:
            Metric object or None if not found
        """
        try:
            if self._prometheus_available:
                # Try to get from Prometheus registry
                return getattr(self._prometheus_registry, name, None)
            else:
                # Return from fallback storage
                return self._fallback_metrics.get(name)
        except Exception as e:
            logger.error(f"Failed to get metric {name}: {e}")
            return None
    
    def get_backend_info(self) -> Dict[str, Any]:
        """Get information about the active backend.
        
        Returns:
            Dictionary with backend information
        """
        return {
            "backend": self.backend,
            "otel_available": self._otel_available,
            "prometheus_available": self._prometheus_available,
            "fallback_metrics_count": len(self._fallback_metrics),
            "initialized": self._initialized
        }
