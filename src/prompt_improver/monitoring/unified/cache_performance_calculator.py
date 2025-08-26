"""Cache Performance Calculation Helper.

Extracted from cache_monitoring_service.py to keep services under 500 lines.
Contains cache statistics and performance calculation utilities.
"""

import time
from collections import deque
from typing import Any

from prompt_improver.core.types import CacheLevel


class CachePerformanceCalculator:
    """Helper class for cache performance calculations."""

    @staticmethod
    def calculate_overall_stats(operation_history: dict[str, deque]) -> dict[str, Any]:
        """Calculate overall cache statistics."""
        all_operations = []
        for history in operation_history.values():
            all_operations.extend(history)

        if not all_operations:
            return {"message": "No cache operations recorded"}

        # Filter recent operations (last hour)
        cutoff_time = time.time() - 3600
        recent_ops = [op for op in all_operations if op["timestamp"] >= cutoff_time]

        if not recent_ops:
            return {"message": "No recent cache operations"}

        total_ops = len(recent_ops)
        hits = sum(1 for op in recent_ops if op["hit"])
        hit_rate = hits / total_ops if total_ops > 0 else 0

        durations = [op["duration_ms"] for op in recent_ops]
        avg_duration = sum(durations) / len(durations) if durations else 0

        return {
            "total_operations": total_ops,
            "hit_rate": hit_rate,
            "avg_response_time_ms": avg_duration,
            "total_hits": hits,
            "total_misses": total_ops - hits,
        }

    @staticmethod
    def calculate_level_stats(operation_history: dict[str, deque], level: CacheLevel) -> dict[str, Any]:
        """Calculate statistics for specific cache level."""
        level_operations = []

        for metric_key, history in operation_history.items():
            if level.value in metric_key:
                level_operations.extend(history)

        if not level_operations:
            return {"message": f"No operations recorded for {level.value}"}

        # Filter recent operations
        cutoff_time = time.time() - 3600
        recent_ops = [op for op in level_operations if op["timestamp"] >= cutoff_time]

        if not recent_ops:
            return {"message": f"No recent operations for {level.value}"}

        total_ops = len(recent_ops)
        hits = sum(1 for op in recent_ops if op["hit"])
        hit_rate = hits / total_ops if total_ops > 0 else 0

        return {
            "total_operations": total_ops,
            "hit_rate": hit_rate,
            "total_hits": hits,
            "total_misses": total_ops - hits,
        }

    @staticmethod
    def calculate_performance_metrics(operation_history: dict[str, deque]) -> dict[str, Any]:
        """Calculate comprehensive performance metrics."""
        import statistics

        metrics = {}
        for metric_key, history in operation_history.items():
            if not history:
                continue
            recent_data = list(history)[-100:]
            hit_rates = [d["hit"] for d in recent_data]
            durations = [d["duration_ms"] for d in recent_data]
            metrics[metric_key] = {
                "operation_count": len(recent_data),
                "hit_rate": sum(hit_rates) / len(hit_rates) if hit_rates else 0,
                "avg_duration_ms": statistics.mean(durations) if durations else 0,
                "p95_duration_ms": statistics.quantiles(durations, n=20)[18]
                if len(durations) >= 20
                else 0,
                "p99_duration_ms": statistics.quantiles(durations, n=100)[98]
                if len(durations) >= 100
                else 0,
            }
        return metrics

    @staticmethod
    def calculate_invalidation_stats(invalidation_history: deque) -> dict[str, Any]:
        """Calculate invalidation statistics."""
        from collections import defaultdict
        from datetime import UTC, datetime

        if not invalidation_history:
            return {"total_events": 0}

        recent_events = [
            event
            for event in invalidation_history
            if (datetime.now(UTC) - event.timestamp).total_seconds() < 3600
        ]

        type_counts = defaultdict(int)
        for event in recent_events:
            type_counts[event.invalidation_type.value] += 1

        return {
            "total_events": len(invalidation_history),
            "recent_events_1h": len(recent_events),
            "events_by_type": dict(type_counts),
        }

    @staticmethod
    def calculate_warming_stats(warming_patterns: dict[str, dict[str, Any]]) -> dict[str, Any]:
        """Calculate cache warming statistics."""
        import statistics

        if not warming_patterns:
            return {"active_patterns": 0}

        patterns = list(warming_patterns.values())
        avg_frequency = statistics.mean([p["frequency"] for p in patterns])
        avg_success_rate = statistics.mean([p["success_rate"] for p in patterns])

        return {
            "active_patterns": len(warming_patterns),
            "avg_access_frequency": avg_frequency,
            "avg_success_rate": avg_success_rate,
            "warming_effectiveness": avg_success_rate,
        }
