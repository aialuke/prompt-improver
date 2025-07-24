"""Dynamic Batch Size Optimization System (2025 Best Practices)

Implements intelligent batch size optimization with:
- Memory-aware adaptation
- Performance-based optimization
- Efficiency metrics tracking
- Real-time adjustment algorithms
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import psutil

logger = logging.getLogger(__name__)

@dataclass
class BatchPerformanceMetrics:
    """Performance metrics for batch processing"""
    batch_size: int
    processing_time: float
    memory_usage_mb: float
    memory_peak_mb: float
    throughput_samples_per_sec: float
    efficiency_score: float
    success_rate: float
    timestamp: datetime
    error_count: int = 0
    
@dataclass
class BatchOptimizationConfig:
    """Configuration for dynamic batch optimization"""
    min_batch_size: int = 10
    max_batch_size: int = 1000
    initial_batch_size: int = 100
    memory_limit_mb: float = 1000.0
    memory_safety_margin: float = 0.8  # Use 80% of memory limit
    performance_window: int = 10  # Number of recent batches to consider
    adaptation_rate: float = 0.1  # How aggressively to adapt batch size
    efficiency_threshold: float = 0.7  # Minimum efficiency to maintain
    
class DynamicBatchOptimizer:
    """Dynamic batch size optimizer with 2025 best practices"""
    
    def __init__(self, config: BatchOptimizationConfig = None):
        self.config = config or BatchOptimizationConfig()
        self.current_batch_size = self.config.initial_batch_size
        self.performance_history: List[BatchPerformanceMetrics] = []
        self.memory_monitor = psutil.Process()
        
        # Optimization state
        self.consecutive_failures = 0
        self.last_optimization_time = time.time()
        self.optimization_interval = 30.0  # Optimize every 30 seconds
        
    async def get_optimal_batch_size(
        self,
        target_samples: int,
        current_memory_usage: Optional[float] = None
    ) -> int:
        """Get optimal batch size based on current conditions"""
        
        # Get current memory usage
        if current_memory_usage is None:
            current_memory_usage = self._get_memory_usage()
            
        # Check if we need to optimize
        if self._should_optimize():
            await self._optimize_batch_size()
            
        # Apply memory constraints
        memory_constrained_size = self._calculate_memory_constrained_batch_size(
            current_memory_usage
        )
        
        # Apply performance constraints
        performance_optimized_size = self._calculate_performance_optimized_batch_size()
        
        # Take the minimum to ensure safety
        optimal_size = min(
            memory_constrained_size,
            performance_optimized_size,
            target_samples,  # Don't exceed target
            self.config.max_batch_size
        )
        
        # Ensure minimum batch size
        optimal_size = max(optimal_size, self.config.min_batch_size)
        
        logger.debug(f"Optimal batch size: {optimal_size} (memory: {memory_constrained_size}, "
                    f"performance: {performance_optimized_size}, target: {target_samples})")
        
        return optimal_size
        
    async def record_batch_performance(
        self,
        batch_size: int,
        processing_time: float,
        success_count: int,
        error_count: int = 0
    ) -> None:
        """Record performance metrics for a completed batch"""
        
        # Get memory metrics
        memory_usage = self._get_memory_usage()
        memory_peak = self._get_peak_memory_usage()
        
        # Calculate performance metrics
        throughput = success_count / processing_time if processing_time > 0 else 0.0
        success_rate = success_count / (success_count + error_count) if (success_count + error_count) > 0 else 0.0
        efficiency_score = self._calculate_efficiency_score(
            batch_size, processing_time, memory_usage, throughput
        )
        
        # Create metrics record
        metrics = BatchPerformanceMetrics(
            batch_size=batch_size,
            processing_time=processing_time,
            memory_usage_mb=memory_usage,
            memory_peak_mb=memory_peak,
            throughput_samples_per_sec=throughput,
            efficiency_score=efficiency_score,
            success_rate=success_rate,
            timestamp=datetime.now(),
            error_count=error_count
        )
        
        # Add to history
        self.performance_history.append(metrics)
        
        # Keep only recent history
        if len(self.performance_history) > self.config.performance_window * 2:
            self.performance_history = self.performance_history[-self.config.performance_window:]
            
        # Update consecutive failures
        if success_rate < 0.5:  # More than 50% failures
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0
            
        logger.debug(f"Recorded batch performance: size={batch_size}, "
                    f"efficiency={efficiency_score:.3f}, throughput={throughput:.1f}")
        
    def _should_optimize(self) -> bool:
        """Check if batch size optimization should be performed"""
        current_time = time.time()
        
        # Time-based optimization
        if current_time - self.last_optimization_time > self.optimization_interval:
            return True
            
        # Performance-based optimization
        if len(self.performance_history) >= 3:
            recent_metrics = self.performance_history[-3:]
            avg_efficiency = np.mean([m.efficiency_score for m in recent_metrics])
            if avg_efficiency < self.config.efficiency_threshold:
                return True
                
        # Failure-based optimization
        if self.consecutive_failures >= 2:
            return True
            
        return False
        
    async def _optimize_batch_size(self) -> None:
        """Optimize batch size based on performance history"""
        if len(self.performance_history) < 2:
            return
            
        self.last_optimization_time = time.time()
        
        # Analyze recent performance
        recent_metrics = self.performance_history[-self.config.performance_window:]
        
        # Calculate performance trends
        efficiency_trend = self._calculate_trend([m.efficiency_score for m in recent_metrics])
        memory_trend = self._calculate_trend([m.memory_usage_mb for m in recent_metrics])
        throughput_trend = self._calculate_trend([m.throughput_samples_per_sec for m in recent_metrics])
        
        # Determine optimization direction
        optimization_factor = 1.0
        
        if efficiency_trend < -0.1:  # Efficiency declining
            optimization_factor *= 0.9  # Reduce batch size
        elif efficiency_trend > 0.1:  # Efficiency improving
            optimization_factor *= 1.1  # Increase batch size
            
        if memory_trend > 0.1:  # Memory usage increasing
            optimization_factor *= 0.95  # Reduce batch size
            
        if throughput_trend < -0.1:  # Throughput declining
            optimization_factor *= 0.9  # Reduce batch size
            
        # Apply consecutive failure penalty
        if self.consecutive_failures > 0:
            optimization_factor *= (0.8 ** self.consecutive_failures)
            
        # Update batch size
        new_batch_size = int(self.current_batch_size * optimization_factor)
        new_batch_size = max(self.config.min_batch_size, 
                           min(new_batch_size, self.config.max_batch_size))
        
        if new_batch_size != self.current_batch_size:
            logger.info(f"Optimizing batch size: {self.current_batch_size} -> {new_batch_size} "
                       f"(efficiency_trend={efficiency_trend:.3f}, factor={optimization_factor:.3f})")
            self.current_batch_size = new_batch_size
            
    def _calculate_memory_constrained_batch_size(self, current_memory_mb: float) -> int:
        """Calculate maximum batch size based on memory constraints"""
        available_memory = (self.config.memory_limit_mb * self.config.memory_safety_margin) - current_memory_mb
        
        if available_memory <= 0:
            return self.config.min_batch_size
            
        # Estimate memory per sample (rough heuristic)
        if self.performance_history:
            recent_metrics = self.performance_history[-5:]
            memory_per_sample = np.mean([
                m.memory_usage_mb / m.batch_size for m in recent_metrics
                if m.batch_size > 0
            ])
        else:
            memory_per_sample = 1.0  # Default estimate: 1MB per sample
            
        max_samples = int(available_memory / memory_per_sample)
        return max(self.config.min_batch_size, min(max_samples, self.config.max_batch_size))
        
    def _calculate_performance_optimized_batch_size(self) -> int:
        """Calculate optimal batch size based on performance history"""
        if len(self.performance_history) < 3:
            return self.current_batch_size
            
        # Find batch size with best efficiency
        recent_metrics = self.performance_history[-self.config.performance_window:]
        
        # Group by batch size and calculate average efficiency
        batch_size_performance = {}
        for metrics in recent_metrics:
            if metrics.batch_size not in batch_size_performance:
                batch_size_performance[metrics.batch_size] = []
            batch_size_performance[metrics.batch_size].append(metrics.efficiency_score)
            
        # Find best performing batch size
        best_batch_size = self.current_batch_size
        best_efficiency = 0.0
        
        for batch_size, efficiencies in batch_size_performance.items():
            avg_efficiency = np.mean(efficiencies)
            if avg_efficiency > best_efficiency:
                best_efficiency = avg_efficiency
                best_batch_size = batch_size
                
        return best_batch_size
        
    def _calculate_efficiency_score(
        self,
        batch_size: int,
        processing_time: float,
        memory_usage: float,
        throughput: float
    ) -> float:
        """Calculate efficiency score for batch processing"""
        if processing_time <= 0 or batch_size <= 0:
            return 0.0
            
        # Normalize metrics
        time_efficiency = min(1.0, batch_size / (processing_time * 10))  # Prefer faster processing
        memory_efficiency = max(0.0, 1.0 - (memory_usage / self.config.memory_limit_mb))
        throughput_efficiency = min(1.0, throughput / 100.0)  # Normalize to reasonable range
        
        # Combined efficiency score
        efficiency = (
            0.4 * time_efficiency +
            0.3 * memory_efficiency +
            0.3 * throughput_efficiency
        )
        
        return min(1.0, max(0.0, efficiency))
        
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in a series of values (-1 to 1)"""
        if len(values) < 2:
            return 0.0
            
        # Simple linear trend calculation
        x = np.arange(len(values))
        y = np.array(values)
        
        if np.std(y) == 0:
            return 0.0
            
        correlation = np.corrcoef(x, y)[0, 1]
        return correlation if not np.isnan(correlation) else 0.0
        
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        try:
            memory_info = self.memory_monitor.memory_info()
            return memory_info.rss / (1024 * 1024)
        except Exception:
            return 0.0
            
    def _get_peak_memory_usage(self) -> float:
        """Get peak memory usage in MB"""
        try:
            memory_info = self.memory_monitor.memory_info()
            return memory_info.vms / (1024 * 1024)
        except Exception:
            return 0.0
            
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        if not self.performance_history:
            return {"status": "no_data"}
            
        recent_metrics = self.performance_history[-self.config.performance_window:]
        
        return {
            "current_batch_size": self.current_batch_size,
            "total_batches_processed": len(self.performance_history),
            "recent_avg_efficiency": np.mean([m.efficiency_score for m in recent_metrics]),
            "recent_avg_throughput": np.mean([m.throughput_samples_per_sec for m in recent_metrics]),
            "recent_avg_memory_usage": np.mean([m.memory_usage_mb for m in recent_metrics]),
            "consecutive_failures": self.consecutive_failures,
            "optimization_interval": self.optimization_interval,
            "last_optimization": self.last_optimization_time
        }
