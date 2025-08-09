"""Performance optimization utilities for achieving <200ms response times.

This module implements comprehensive performance optimization techniques based on
2025 best practices for high-performance Python applications.
"""
import asyncio
import json
import logging
import statistics
import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Dict, List, Optional
import aiofiles
logger = logging.getLogger(__name__)
try:
    from prompt_improver.monitoring.opentelemetry.metrics import get_http_metrics
    _http_metrics = get_http_metrics()
    _metrics_available = True
except ImportError:
    _http_metrics = None
    _metrics_available = False
    logger.warning('OpenTelemetry metrics not available, using in-memory fallback')

@dataclass
class PerformanceMetrics:
    """Container for performance measurement data."""
    operation_name: str
    start_time: float
    end_time: float | None = None
    duration_ms: float | None = None
    status: str = 'success'
    metadata: dict[str, Any] = field(default_factory=dict)

    def complete(self, status: str='success', **metadata) -> None:
        """Mark the operation as complete and calculate duration."""
        self.end_time = time.perf_counter()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = status
        self.metadata.update(metadata)
        if _metrics_available and _http_metrics:
            _http_metrics.record_request(method='INTERNAL', endpoint=self.operation_name, status_code=200 if self.status == 'success' else 500, duration_ms=self.duration_ms)
        if self.duration_ms > 200:
            logger.warning('Response time target violation: %s took %sms (target: <200ms)', self.operation_name, format(self.duration_ms, '.2f'))

@dataclass
class PerformanceBaseline:
    """Baseline performance measurements for comparison."""
    operation_name: str
    sample_count: int
    avg_duration_ms: float
    p50_duration_ms: float
    p95_duration_ms: float
    p99_duration_ms: float
    max_duration_ms: float
    min_duration_ms: float
    success_rate: float
    timestamp: datetime

    def meets_target(self, target_ms: float=200) -> bool:
        """Check if performance meets the target."""
        return self.p95_duration_ms <= target_ms

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {'operation_name': self.operation_name, 'sample_count': self.sample_count, 'avg_duration_ms': self.avg_duration_ms, 'p50_duration_ms': self.p50_duration_ms, 'p95_duration_ms': self.p95_duration_ms, 'p99_duration_ms': self.p99_duration_ms, 'max_duration_ms': self.max_duration_ms, 'min_duration_ms': self.min_duration_ms, 'success_rate': self.success_rate, 'timestamp': self.timestamp.isoformat(), 'meets_200ms_target': self.meets_target(200)}

class PerformanceOptimizer:
    """Comprehensive performance optimization and monitoring system."""

    def __init__(self):
        self._measurements: dict[str, list[PerformanceMetrics]] = {}
        self._baselines: dict[str, PerformanceBaseline] = {}
        self._optimization_enabled = True

    @asynccontextmanager
    async def measure_operation(self, operation_name: str, **metadata) -> AsyncGenerator[PerformanceMetrics, None]:
        """Context manager for measuring operation performance.

        Args:
            operation_name: Name of the operation being measured
            **metadata: Additional metadata to include

        Yields:
            PerformanceMetrics instance for the operation
        """
        if _metrics_available and _http_metrics:
            pass
        metrics = PerformanceMetrics(operation_name=operation_name, start_time=time.perf_counter(), metadata=metadata)
        try:
            yield metrics
            metrics.complete(status='success')
        except Exception as e:
            metrics.complete(status='error', error=str(e))
            raise
        finally:
            if operation_name not in self._measurements:
                self._measurements[operation_name] = []
            self._measurements[operation_name].append(metrics)
            if len(self._measurements[operation_name]) > 1000:
                self._measurements[operation_name] = self._measurements[operation_name][-1000:]

    async def calculate_baseline(self, operation_name: str, min_samples: int=10) -> PerformanceBaseline | None:
        """Calculate performance baseline for an operation.

        Args:
            operation_name: Name of the operation
            min_samples: Minimum number of samples required

        Returns:
            PerformanceBaseline if enough samples, None otherwise
        """
        measurements = self._measurements.get(operation_name, [])
        if len(measurements) < min_samples:
            logger.warning('Insufficient samples for %s: %s < %s', operation_name, len(measurements), min_samples)
            return None
        successful_measurements = [m for m in measurements if m.status == 'success' and m.duration_ms is not None]
        if not successful_measurements:
            logger.warning('No successful measurements for %s', operation_name)
            return None
        durations = [m.duration_ms for m in successful_measurements]
        baseline = PerformanceBaseline(operation_name=operation_name, sample_count=len(successful_measurements), avg_duration_ms=statistics.mean(durations), p50_duration_ms=statistics.median(durations), p95_duration_ms=self._percentile(durations, 95), p99_duration_ms=self._percentile(durations, 99), max_duration_ms=max(durations), min_duration_ms=min(durations), success_rate=len(successful_measurements) / len(measurements), timestamp=datetime.now(UTC))
        self._baselines[operation_name] = baseline
        return baseline

    @staticmethod
    def _percentile(data: list[float], percentile: float) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = percentile / 100 * (len(sorted_data) - 1)
        if index.is_integer():
            return sorted_data[int(index)]
        lower = sorted_data[int(index)]
        upper = sorted_data[int(index) + 1]
        return lower + (upper - lower) * (index - int(index))

    async def get_baseline(self, operation_name: str) -> PerformanceBaseline | None:
        """Get existing baseline for an operation."""
        return self._baselines.get(operation_name)

    async def get_all_baselines(self) -> dict[str, PerformanceBaseline]:
        """Get all calculated baselines."""
        return self._baselines.copy()

    async def save_baselines(self, filepath: str) -> None:
        """Save baselines to file for persistence."""
        baseline_data = {name: baseline.to_dict() for name, baseline in self._baselines.items()}
        async with aiofiles.open(filepath, 'w') as f:
            await f.write(json.dumps(baseline_data, indent=2))
        logger.info('Saved {len(baseline_data)} baselines to %s', filepath)

    async def load_baselines(self, filepath: str) -> None:
        """Load baselines from file."""
        try:
            async with aiofiles.open(filepath) as f:
                content = await f.read()
                baseline_data = json.loads(content)
            for name, data in baseline_data.items():
                baseline = PerformanceBaseline(operation_name=data['operation_name'], sample_count=data['sample_count'], avg_duration_ms=data['avg_duration_ms'], p50_duration_ms=data['p50_duration_ms'], p95_duration_ms=data['p95_duration_ms'], p99_duration_ms=data['p99_duration_ms'], max_duration_ms=data['max_duration_ms'], min_duration_ms=data['min_duration_ms'], success_rate=data['success_rate'], timestamp=datetime.fromisoformat(data['timestamp']))
                self._baselines[name] = baseline
            logger.info('Loaded {len(baseline_data)} baselines from %s', filepath)
        except FileNotFoundError:
            logger.info('Baseline file %s not found, starting fresh', filepath)
        except Exception as e:
            logger.error('Failed to load baselines from {filepath}: %s', e)

    async def run_performance_benchmark(self, operation_name: str, operation_func, sample_count: int=100, **operation_kwargs) -> PerformanceBaseline:
        """Run a performance benchmark for an operation.

        Args:
            operation_name: Name of the operation
            operation_func: Async function to benchmark
            sample_count: Number of samples to collect
            **operation_kwargs: Arguments to pass to operation_func

        Returns:
            PerformanceBaseline with benchmark results
        """
        logger.info('Running performance benchmark for %s (%s samples)', operation_name, sample_count)
        for i in range(sample_count):
            async with self.measure_operation(operation_name, sample_index=i):
                await operation_func(**operation_kwargs)
            if i % 10 == 0:
                await asyncio.sleep(0.001)
        baseline = await self.calculate_baseline(operation_name, min_samples=1)
        if baseline:
            logger.info('Benchmark complete for %s: avg=%sms, p95=%sms, meets_target=%s', operation_name, format(baseline.avg_duration_ms, '.2f'), format(baseline.p95_duration_ms, '.2f'), baseline.meets_target())
        return baseline

    def get_performance_summary(self) -> dict[str, Any]:
        """Get a summary of current performance metrics."""
        summary = {'total_operations': len(self._measurements), 'total_measurements': sum((len(measurements) for measurements in self._measurements.values())), 'operations_with_baselines': len(self._baselines), 'operations_meeting_target': sum((1 for baseline in self._baselines.values() if baseline.meets_target())), 'timestamp': datetime.now(UTC).isoformat()}
        operation_summaries = {}
        for operation_name, baseline in self._baselines.items():
            operation_summaries[operation_name] = {'avg_duration_ms': baseline.avg_duration_ms, 'p95_duration_ms': baseline.p95_duration_ms, 'success_rate': baseline.success_rate, 'meets_target': baseline.meets_target(), 'sample_count': baseline.sample_count}
        summary['operations'] = operation_summaries
        return summary
_global_optimizer: PerformanceOptimizer | None = None

def get_performance_optimizer() -> PerformanceOptimizer:
    """Get the global performance optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = PerformanceOptimizer()
    return _global_optimizer

async def measure_mcp_operation(operation_name: str, **metadata):
    """Convenience function for measuring MCP operations."""
    optimizer = get_performance_optimizer()
    return optimizer.measure_operation(f'mcp_{operation_name}', **metadata)

async def measure_database_operation(operation_name: str, **metadata):
    """Convenience function for measuring database operations."""
    optimizer = get_performance_optimizer()
    return optimizer.measure_operation(f'db_{operation_name}', **metadata)

async def measure_cache_operation(operation_name: str, **metadata):
    """Convenience function for measuring cache operations."""
    optimizer = get_performance_optimizer()
    return optimizer.measure_operation(f'cache_{operation_name}', **metadata)
