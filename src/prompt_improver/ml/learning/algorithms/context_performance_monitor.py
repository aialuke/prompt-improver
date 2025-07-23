"""Performance Monitoring for Context Learning.

Extracted from ContextSpecificLearner to improve maintainability.
Handles performance metrics, monitoring, and optimization tracking.
"""

import logging
from typing import Any, Dict

from ....security import MemoryGuard


class ContextPerformanceMonitor:
    """Monitors performance metrics for context learning components."""
    
    def __init__(self, memory_guard: MemoryGuard, logger: logging.Logger = None):
        """Initialize performance monitor."""
        self.memory_guard = memory_guard
        self.logger = logger or logging.getLogger(__name__)
        
        # Performance tracking
        self.operation_counts = {}
        self.operation_times = {}
        self.privacy_budget_used = 0.0
    
    def get_performance_metrics(
        self,
        cache_stats: Dict[str, Any],
        context_clusters_count: int = 0
    ) -> Dict[str, Any]:
        """Get comprehensive performance metrics."""
        memory_stats = self.memory_guard.check_memory_usage()
        
        metrics = {
            "memory_usage_mb": memory_stats.get("current_mb", 0),
            "memory_peak_mb": memory_stats.get("peak_mb", 0),
            "memory_usage_percent": memory_stats.get("usage_percent", 0),
            "context_clusters": context_clusters_count,
            "privacy_budget_used": self.privacy_budget_used,
            "operation_counts": self.operation_counts.copy(),
        }
        
        # Add cache statistics
        metrics.update(cache_stats)
        
        # Add operation timing statistics
        if self.operation_times:
            metrics["avg_operation_times"] = {
                op: sum(times) / len(times) 
                for op, times in self.operation_times.items()
            }
        
        return metrics
    
    def track_operation(self, operation: str, execution_time: float):
        """Track operation execution for performance analysis."""
        if operation not in self.operation_counts:
            self.operation_counts[operation] = 0
            self.operation_times[operation] = []
        
        self.operation_counts[operation] += 1
        self.operation_times[operation].append(execution_time)
        
        # Keep only recent timing data (last 100 operations per type)
        if len(self.operation_times[operation]) > 100:
            self.operation_times[operation] = self.operation_times[operation][-100:]
    
    def update_privacy_budget(self, used_budget: float):
        """Update privacy budget tracking."""
        self.privacy_budget_used += used_budget
        
        # Log warning if approaching privacy budget limits
        if self.privacy_budget_used > 8.0:  # High privacy budget usage
            self.logger.warning(
                f"High privacy budget usage: {self.privacy_budget_used:.2f}. "
                "Consider reducing epsilon or clearing state."
            )
    
    def reset_privacy_budget(self):
        """Reset privacy budget tracking."""
        old_budget = self.privacy_budget_used
        self.privacy_budget_used = 0.0
        self.logger.info(f"Reset privacy budget from {old_budget:.2f} to 0.0")
    
    def get_privacy_budget_status(self) -> Dict[str, float]:
        """Get current privacy budget status."""
        return {
            "budget_used": self.privacy_budget_used,
            "budget_remaining": max(0, 10.0 - self.privacy_budget_used),  # Assume 10.0 total budget
            "budget_utilization": min(1.0, self.privacy_budget_used / 10.0)
        }
    
    def log_performance_summary(self):
        """Log performance summary for debugging."""
        metrics = self.get_performance_metrics({}, 0)
        
        self.logger.info(
            f"Performance Summary - Memory: {metrics['memory_usage_mb']:.1f}MB "
            f"({metrics['memory_usage_percent']:.1f}%), "
            f"Privacy Budget: {metrics['privacy_budget_used']:.2f}, "
            f"Operations: {sum(self.operation_counts.values())}"
        )
    
    def clear_performance_history(self):
        """Clear performance tracking history."""
        self.operation_counts.clear()
        self.operation_times.clear()
        self.logger.info("Cleared performance tracking history")