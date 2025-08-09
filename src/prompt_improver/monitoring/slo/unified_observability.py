"""Unified SLO Observability Integration
====================================

Enhanced observability integration between SLO monitoring components and
UnifiedConnectionManager, providing comprehensive tracing, metrics, and
cache correlation for SLO operations.
"""
import asyncio
import logging
import time
from collections.abc import Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime, timezone
from typing import Any, Dict, List, Optional
from prompt_improver.database.unified_connection_manager import ManagerMode, get_unified_manager
from prompt_improver.monitoring.slo.framework import SLOTarget, SLOTimeWindow
try:
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Status, StatusCode
    OPENTELEMETRY_AVAILABLE = True
    slo_tracer = trace.get_tracer(__name__ + '.slo')
    slo_meter = metrics.get_meter(__name__ + '.slo')
    slo_operations_counter = slo_meter.create_counter('slo_operations_total', description='Total SLO operations by type, component, and status', unit='1')
    slo_cache_performance_histogram = slo_meter.create_histogram('slo_cache_operation_duration_seconds', description='SLO cache operation duration by type and level', unit='s')
    slo_compliance_ratio_gauge = slo_meter.create_gauge('slo_compliance_ratio', description='SLO compliance ratio by service and target', unit='ratio')
    slo_error_budget_gauge = slo_meter.create_gauge('slo_error_budget_remaining', description='SLO error budget remaining percentage by service and target', unit='percent')
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    slo_tracer = None
    slo_meter = None
    slo_operations_counter = None
    slo_cache_performance_histogram = None
    slo_compliance_ratio_gauge = None
    slo_error_budget_gauge = None
logger = logging.getLogger(__name__)

@dataclass
class SLOObservabilityContext:
    """Context for SLO operation observability tracking."""
    operation: str
    service_name: str
    target_name: str | None = None
    component: str = 'slo_monitoring'
    start_time: float = field(default_factory=time.time)
    trace_id: str | None = None
    span_id: str | None = None
    cache_operations: list[dict[str, Any]] = field(default_factory=list)
    custom_attributes: dict[str, Any] = field(default_factory=dict)

class UnifiedSLOObservability:
    """Unified observability integration for SLO monitoring operations.

    Provides comprehensive tracing, metrics, and correlation between
    SLO monitoring components and UnifiedConnectionManager cache operations.
    """

    def __init__(self, unified_manager=None):
        """Initialize unified SLO observability.

        Args:
            unified_manager: Optional UnifiedConnectionManager instance
        """
        self._unified_manager = unified_manager or get_unified_manager(ManagerMode.ASYNC_MODERN)
        self._active_contexts: dict[str, SLOObservabilityContext] = {}
        self._correlation_callbacks: list[Callable] = []
        self._operation_stats = {'sli_calculation': {'count': 0, 'total_duration': 0.0, 'cache_hits': 0, 'cache_misses': 0}, 'error_budget_update': {'count': 0, 'total_duration': 0.0, 'cache_hits': 0, 'cache_misses': 0}, 'metrics_storage': {'count': 0, 'total_duration': 0.0, 'cache_hits': 0, 'cache_misses': 0}, 'alert_evaluation': {'count': 0, 'total_duration': 0.0, 'cache_hits': 0, 'cache_misses': 0}}
        self._alert_contexts: dict[str, dict[str, Any]] = {}
        logger.info('UnifiedSLOObservability initialized with enhanced cache correlation')

    @asynccontextmanager
    async def observe_slo_operation(self, operation: str, service_name: str, target_name: str | None=None, component: str='slo_monitoring', **custom_attributes):
        """Context manager for observing SLO operations with full tracing.

        Args:
            operation: Operation name (e.g., "sli_calculation", "error_budget_update")
            service_name: Service name for correlation
            target_name: Optional SLO target name
            component: Component name for classification
            **custom_attributes: Additional attributes for tracing
        """
        context_id = f"{operation}_{service_name}_{target_name or 'all'}_{time.time()}"
        context = SLOObservabilityContext(operation=operation, service_name=service_name, target_name=target_name, component=component, custom_attributes=custom_attributes)
        span = None
        if OPENTELEMETRY_AVAILABLE and slo_tracer:
            span = slo_tracer.start_span(name=f'slo.{operation}', attributes={'slo.operation': operation, 'slo.service_name': service_name, 'slo.target_name': target_name or 'all', 'slo.component': component, **custom_attributes})
            context.trace_id = f'{span.get_span_context().trace_id:032x}'
            context.span_id = f'{span.get_span_context().span_id:016x}'
        self._active_contexts[context_id] = context
        try:
            if not self._unified_manager._is_initialized:
                await self._unified_manager.initialize()
            yield context
            if span:
                span.set_status(Status(StatusCode.OK))
            if OPENTELEMETRY_AVAILABLE and slo_operations_counter:
                slo_operations_counter.add(1, {'operation': operation, 'component': component, 'service_name': service_name, 'target_name': target_name or 'all', 'status': 'success'})
        except Exception as e:
            if span:
                span.set_status(Status(StatusCode.ERROR, str(e)))
                span.record_exception(e)
            if OPENTELEMETRY_AVAILABLE and slo_operations_counter:
                slo_operations_counter.add(1, {'operation': operation, 'component': component, 'service_name': service_name, 'target_name': target_name or 'all', 'status': 'error'})
            logger.error('SLO operation {operation} failed for {service_name}: %s', e)
            raise
        finally:
            duration = time.time() - context.start_time
            if operation in self._operation_stats:
                stats = self._operation_stats[operation]
                stats['count'] += 1
                stats['total_duration'] += duration
                cache_hits = sum((1 for op in context.cache_operations if op.get('hit', False)))
                cache_misses = sum((1 for op in context.cache_operations if not op.get('hit', False)))
                stats['cache_hits'] += cache_hits
                stats['cache_misses'] += cache_misses
            if OPENTELEMETRY_AVAILABLE and slo_cache_performance_histogram:
                slo_cache_performance_histogram.record(duration, {'operation': operation, 'component': component, 'service_name': service_name})
            if span:
                span.end()
            if context_id in self._active_contexts:
                del self._active_contexts[context_id]

    async def record_cache_operation(self, context_id: str, operation_type: str, cache_key: str, hit: bool, duration_ms: float, cache_level: str='unified'):
        """Record cache operation for correlation with SLO operations.

        Args:
            context_id: Context ID to correlate with
            operation_type: Type of cache operation
            cache_key: Cache key used
            hit: Whether operation was a cache hit
            duration_ms: Operation duration in milliseconds
            cache_level: Cache level (l1, l2, unified)
        """
        cache_op = {'operation_type': operation_type, 'cache_key': cache_key, 'hit': hit, 'duration_ms': duration_ms, 'cache_level': cache_level, 'timestamp': time.time()}
        for context in self._active_contexts.values():
            if context_id in [context.trace_id, context.span_id]:
                context.cache_operations.append(cache_op)
                break
        if OPENTELEMETRY_AVAILABLE and slo_cache_performance_histogram:
            slo_cache_performance_histogram.record(duration_ms / 1000.0, {'operation_type': operation_type, 'cache_level': cache_level, 'hit': str(hit).lower()})

    async def record_slo_compliance(self, service_name: str, target_name: str, compliance_ratio: float, error_budget_remaining: float, time_window: SLOTimeWindow, additional_metrics: dict[str, float] | None=None):
        """Record SLO compliance metrics for observability.

        Args:
            service_name: Service name
            target_name: SLO target name
            compliance_ratio: Current compliance ratio (0.0 to 1.0)
            error_budget_remaining: Error budget remaining percentage
            time_window: Time window for the measurement
            additional_metrics: Optional additional metrics
        """
        labels = {'service_name': service_name, 'target_name': target_name, 'time_window': time_window.value}
        if OPENTELEMETRY_AVAILABLE and slo_compliance_ratio_gauge:
            slo_compliance_ratio_gauge.set(compliance_ratio, labels)
        if OPENTELEMETRY_AVAILABLE and slo_error_budget_gauge:
            slo_error_budget_gauge.set(error_budget_remaining, labels)
        if additional_metrics and OPENTELEMETRY_AVAILABLE:
            for metric_name, value in additional_metrics.items():
                try:
                    custom_gauge = slo_meter.create_gauge(f'slo_{metric_name}', description=f'SLO {metric_name} metric', unit='1')
                    custom_gauge.set(value, labels)
                except Exception as e:
                    logger.warning('Failed to record custom SLO metric %s: %s', metric_name, e)
        logger.debug('Recorded SLO compliance for %s.%s: %s compliance, %s%% budget remaining', service_name, target_name, format(compliance_ratio, '.3f'), format(error_budget_remaining, '.1f'))

    async def correlate_alert_with_cache(self, alert_id: str, service_name: str, target_name: str, alert_severity: str, cache_context: dict[str, Any] | None=None):
        """Correlate alert generation with cache performance context.

        Args:
            alert_id: Unique alert identifier
            service_name: Service name
            target_name: SLO target name
            alert_severity: Alert severity level
            cache_context: Optional cache performance context
        """
        cache_stats = self._unified_manager.get_cache_stats() if hasattr(self._unified_manager, 'get_cache_stats') else {}
        alert_context = {'alert_id': alert_id, 'service_name': service_name, 'target_name': target_name, 'alert_severity': alert_severity, 'timestamp': datetime.now(UTC).isoformat(), 'cache_stats': cache_stats, 'cache_context': cache_context or {}, 'unified_manager_initialized': self._unified_manager._is_initialized, 'unified_manager_mode': self._unified_manager.mode.value if hasattr(self._unified_manager, 'mode') else 'unknown'}
        self._alert_contexts[alert_id] = alert_context
        for callback in self._correlation_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_context)
                else:
                    callback(alert_context)
            except Exception as e:
                logger.error('Alert correlation callback failed: %s', e)
        logger.info('Correlated alert %s with cache context for %s.%s', alert_id, service_name, target_name)

    def register_correlation_callback(self, callback: Callable):
        """Register callback for alert correlation events.

        Args:
            callback: Callback function that receives alert context
        """
        self._correlation_callbacks.append(callback)

    def get_operation_statistics(self) -> dict[str, Any]:
        """Get comprehensive operation statistics.

        Returns:
            Dictionary with operation statistics including cache performance
        """
        stats = {}
        for operation, data in self._operation_stats.items():
            if data['count'] > 0:
                avg_duration = data['total_duration'] / data['count']
                cache_hit_rate = data['cache_hits'] / (data['cache_hits'] + data['cache_misses']) if data['cache_hits'] + data['cache_misses'] > 0 else 0.0
                stats[operation] = {'total_operations': data['count'], 'average_duration_seconds': avg_duration, 'total_duration_seconds': data['total_duration'], 'cache_hit_rate': cache_hit_rate, 'cache_hits': data['cache_hits'], 'cache_misses': data['cache_misses']}
        if hasattr(self._unified_manager, 'get_cache_stats'):
            stats['unified_manager_cache'] = self._unified_manager.get_cache_stats()
        stats['alert_correlation'] = {'total_alerts_correlated': len(self._alert_contexts), 'correlation_callbacks_registered': len(self._correlation_callbacks)}
        return stats

    def get_alert_contexts(self, limit: int=100) -> list[dict[str, Any]]:
        """Get recent alert contexts for analysis.

        Args:
            limit: Maximum number of contexts to return

        Returns:
            List of alert contexts with cache correlation data
        """
        sorted_contexts = sorted(self._alert_contexts.values(), key=lambda x: x['timestamp'], reverse=True)
        return sorted_contexts[:limit]

    async def generate_observability_report(self) -> dict[str, Any]:
        """Generate comprehensive observability report.

        Returns:
            Dictionary with observability metrics and recommendations
        """
        report = {'timestamp': datetime.now(UTC).isoformat(), 'operation_statistics': self.get_operation_statistics(), 'active_contexts': len(self._active_contexts), 'unified_manager_status': {'initialized': self._unified_manager._is_initialized, 'mode': self._unified_manager.mode.value if hasattr(self._unified_manager, 'mode') else 'unknown'}, 'opentelemetry_available': OPENTELEMETRY_AVAILABLE, 'recommendations': []}
        stats = report['operation_statistics']
        for operation, data in stats.items():
            if isinstance(data, dict) and 'cache_hit_rate' in data:
                hit_rate = data['cache_hit_rate']
                if hit_rate < 0.5:
                    report['recommendations'].append({'type': 'cache_performance', 'severity': 'warning', 'operation': operation, 'message': f'Low cache hit rate ({hit_rate:.1%}) for {operation} operations. Consider cache warming or TTL optimization.'})
                elif hit_rate > 0.9:
                    report['recommendations'].append({'type': 'cache_performance', 'severity': 'info', 'operation': operation, 'message': f'Excellent cache hit rate ({hit_rate:.1%}) for {operation} operations.'})
                if 'average_duration_seconds' in data:
                    avg_duration = data['average_duration_seconds']
                    if avg_duration > 1.0:
                        report['recommendations'].append({'type': 'performance', 'severity': 'warning', 'operation': operation, 'message': f'Slow {operation} operations averaging {avg_duration:.2f}s. Consider optimization.'})
        return report
_slo_observability: UnifiedSLOObservability | None = None

def get_slo_observability() -> UnifiedSLOObservability:
    """Get or create global SLO observability instance."""
    global _slo_observability
    if _slo_observability is None:
        _slo_observability = UnifiedSLOObservability()
    return _slo_observability
