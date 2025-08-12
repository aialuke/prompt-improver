"""Dynamic connection pool scaler with intelligent optimization algorithms.

Extracted from database.unified_connection_manager.py to provide:
- Load-based scaling decisions using multiple metrics
- Time-based scaling patterns and history analysis
- Performance-aware scaling to avoid thrashing
- Resource constraint awareness and safety limits
- Multi-pool coordination for complex topologies

This centralizes all pool scaling logic from the monolithic manager.
"""

import logging
import statistics
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class ScalingAction(Enum):
    """Pool scaling action types."""

    NO_ACTION = "no_action"
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    EMERGENCY_SCALE = "emergency_scale"


class ScalingReason(Enum):
    """Reasons for scaling decisions."""

    HIGH_UTILIZATION = "high_utilization"
    LOW_UTILIZATION = "low_utilization"
    HIGH_LATENCY = "high_latency"
    QUEUE_BUILDUP = "queue_buildup"
    ERROR_RATE = "error_rate"
    TIME_PATTERN = "time_pattern"
    RESOURCE_CONSTRAINT = "resource_constraint"
    COOLDOWN_PERIOD = "cooldown_period"


@dataclass
class ScalingMetrics:
    """Current metrics for scaling decisions."""

    current_pool_size: int
    active_connections: int
    waiting_requests: int = 0
    avg_response_time_ms: float = 0.0
    error_rate: float = 0.0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def utilization_ratio(self) -> float:
        """Calculate connection utilization ratio."""
        if self.current_pool_size <= 0:
            return 0.0
        return self.active_connections / self.current_pool_size

    @property
    def is_overloaded(self) -> bool:
        """Check if pool is currently overloaded."""
        return (
            self.utilization_ratio > 0.9
            or self.waiting_requests > 0
            or self.avg_response_time_ms > 1000
        )

    @property
    def is_underutilized(self) -> bool:
        """Check if pool is underutilized."""
        return (
            self.utilization_ratio < 0.3
            and self.waiting_requests == 0
            and self.avg_response_time_ms < 100
        )


@dataclass
class ScalingConfiguration:
    """Configuration for pool scaling behavior."""

    # Size constraints
    min_pool_size: int = 1
    max_pool_size: int = 100
    scale_up_increment: int = 5
    scale_down_increment: int = 3

    # Utilization thresholds
    scale_up_threshold: float = 0.8
    scale_down_threshold: float = 0.3
    emergency_threshold: float = 0.95

    # Timing constraints
    cooldown_seconds: int = 60
    min_stable_duration: int = 30

    # Performance thresholds
    max_response_time_ms: float = 500.0
    max_error_rate: float = 0.05

    # Resource constraints
    max_cpu_utilization: float = 0.8
    max_memory_utilization: float = 0.9

    # Advanced features
    enable_predictive_scaling: bool = True
    enable_time_based_patterns: bool = True
    enable_burst_detection: bool = True

    @classmethod
    def for_environment(cls, env: str) -> "ScalingConfiguration":
        """Create scaling configuration optimized for environment."""
        configs = {
            "development": cls(
                min_pool_size=2,
                max_pool_size=20,
                scale_up_increment=3,
                scale_down_increment=2,
                cooldown_seconds=30,
                enable_predictive_scaling=False,
            ),
            "testing": cls(
                min_pool_size=1,
                max_pool_size=10,
                scale_up_increment=2,
                scale_down_increment=1,
                cooldown_seconds=10,
                enable_predictive_scaling=False,
                enable_time_based_patterns=False,
            ),
            "production": cls(
                min_pool_size=10,
                max_pool_size=200,
                scale_up_increment=10,
                scale_down_increment=5,
                cooldown_seconds=120,
                enable_predictive_scaling=True,
                enable_time_based_patterns=True,
                enable_burst_detection=True,
            ),
        }
        return configs.get(env, configs["development"])


@dataclass
class ScalingDecision:
    """A scaling decision with reasoning and recommendations."""

    action: ScalingAction
    target_size: int
    current_size: int
    reason: ScalingReason
    confidence: float  # 0.0 to 1.0
    estimated_benefit: str
    metrics_snapshot: ScalingMetrics
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def size_change(self) -> int:
        """Calculate the size change."""
        return self.target_size - self.current_size

    @property
    def is_scaling_action(self) -> bool:
        """Check if this is an actual scaling action."""
        return self.action in [
            ScalingAction.SCALE_UP,
            ScalingAction.SCALE_DOWN,
            ScalingAction.EMERGENCY_SCALE,
        ]


class PoolScalerProtocol(Protocol):
    """Protocol for pool implementations that can be scaled."""

    async def get_current_metrics(self) -> ScalingMetrics:
        """Get current pool metrics for scaling decisions."""
        ...

    async def scale_to_size(self, new_size: int) -> bool:
        """Scale pool to specified size. Returns success status."""
        ...

    def get_current_size(self) -> int:
        """Get current pool size."""
        ...


class PoolScaler:
    """Advanced pool scaler with intelligent optimization algorithms.

    Provides comprehensive pool scaling with:
    - Multi-metric analysis for scaling decisions
    - Predictive scaling based on historical patterns
    - Time-based scaling pattern recognition
    - Resource constraint awareness
    - Burst traffic detection and handling
    - Anti-thrashing safeguards with cooldown periods
    """

    def __init__(self, config: ScalingConfiguration, service_name: str = "pool_scaler"):
        self.config = config
        self.service_name = service_name

        # Scaling state
        self.last_scaling_time = 0.0
        self.last_scaling_action = ScalingAction.NO_ACTION
        self.scaling_history = deque(maxlen=1000)

        # Performance tracking
        self.metrics_history = deque(maxlen=500)
        self.performance_window = deque(maxlen=100)

        # Pattern detection
        self.time_patterns: Dict[int, float] = {}  # hour -> typical utilization
        self.burst_detector = BurstDetector(window_minutes=5)

        # Decision tracking
        self.decision_history = deque(maxlen=200)
        self._statistics = {
            "total_decisions": 0,
            "scale_up_count": 0,
            "scale_down_count": 0,
            "no_action_count": 0,
            "emergency_scale_count": 0,
        }

        logger.info(f"PoolScaler initialized: {service_name}")
        logger.info(
            f"Config: min={config.min_pool_size}, max={config.max_pool_size}, cooldown={config.cooldown_seconds}s"
        )

    async def evaluate_scaling(
        self, pool: PoolScalerProtocol, force_evaluation: bool = False
    ) -> ScalingDecision:
        """Evaluate if pool scaling is needed and return decision.

        Args:
            pool: Pool implementation to evaluate
            force_evaluation: Skip cooldown checks

        Returns:
            ScalingDecision with action and reasoning
        """
        current_metrics = await pool.get_current_metrics()
        self._record_metrics(current_metrics)

        # Check cooldown period
        if not force_evaluation and self._is_in_cooldown():
            decision = ScalingDecision(
                action=ScalingAction.NO_ACTION,
                target_size=current_metrics.current_pool_size,
                current_size=current_metrics.current_pool_size,
                reason=ScalingReason.COOLDOWN_PERIOD,
                confidence=1.0,
                estimated_benefit="Respecting cooldown period to prevent thrashing",
                metrics_snapshot=current_metrics,
            )
            self._record_decision(decision)
            return decision

        # Emergency scaling check
        if current_metrics.utilization_ratio >= self.config.emergency_threshold:
            decision = await self._emergency_scaling_decision(current_metrics)
            self._record_decision(decision)
            return decision

        # Normal scaling evaluation
        decision = await self._evaluate_normal_scaling(current_metrics, pool)
        self._record_decision(decision)
        return decision

    async def execute_scaling_decision(
        self, decision: ScalingDecision, pool: PoolScalerProtocol
    ) -> Dict[str, Any]:
        """Execute a scaling decision on the pool.

        Args:
            decision: Scaling decision to execute
            pool: Pool to scale

        Returns:
            Execution result with status and metrics
        """
        if not decision.is_scaling_action:
            # Still need to update statistics for no-action decisions
            self._update_statistics(decision.action)
            return {
                "status": "no_action",
                "reason": decision.reason.value,
                "current_size": decision.current_size,
            }

        start_time = time.time()

        try:
            logger.info(
                f"Executing scaling: {decision.current_size} → {decision.target_size} "
                f"({decision.reason.value}, confidence={decision.confidence:.2f})"
            )

            success = await pool.scale_to_size(decision.target_size)

            if success:
                self.last_scaling_time = time.time()
                self.last_scaling_action = decision.action

                # Record scaling event
                scaling_event = {
                    "timestamp": time.time(),
                    "action": decision.action.value,
                    "from_size": decision.current_size,
                    "to_size": decision.target_size,
                    "reason": decision.reason.value,
                    "confidence": decision.confidence,
                    "duration_ms": (time.time() - start_time) * 1000,
                }
                self.scaling_history.append(scaling_event)

                # Update statistics
                self._update_statistics(decision.action)

                logger.info(
                    f"Scaling executed successfully in {scaling_event['duration_ms']:.1f}ms"
                )

                return {
                    "status": "success",
                    "action": decision.action.value,
                    "previous_size": decision.current_size,
                    "new_size": decision.target_size,
                    "reason": decision.reason.value,
                    "confidence": decision.confidence,
                    "duration_ms": scaling_event["duration_ms"],
                    "estimated_benefit": decision.estimated_benefit,
                }
            else:
                logger.error("Pool scaling failed - pool implementation returned false")
                return {
                    "status": "failed",
                    "reason": "pool_implementation_error",
                    "current_size": decision.current_size,
                }

        except Exception as e:
            logger.error(f"Failed to execute scaling decision: {e}")
            return {
                "status": "error",
                "error": str(e),
                "current_size": decision.current_size,
            }

    async def _evaluate_normal_scaling(
        self, metrics: ScalingMetrics, pool: PoolScalerProtocol
    ) -> ScalingDecision:
        """Evaluate normal scaling needs based on multiple factors."""

        # Collect scaling factors
        factors = []

        # Factor 1: Utilization-based scaling
        utilization_factor = self._evaluate_utilization_factor(metrics)
        if utilization_factor:
            factors.append(utilization_factor)

        # Factor 2: Performance-based scaling
        performance_factor = self._evaluate_performance_factor(metrics)
        if performance_factor:
            factors.append(performance_factor)

        # Factor 3: Queue-based scaling
        queue_factor = self._evaluate_queue_factor(metrics)
        if queue_factor:
            factors.append(queue_factor)

        # Factor 4: Resource constraint factor
        resource_factor = self._evaluate_resource_constraints(metrics)
        if resource_factor:
            factors.append(resource_factor)

        # Factor 5: Predictive scaling (if enabled)
        if self.config.enable_predictive_scaling:
            predictive_factor = self._evaluate_predictive_factor(metrics)
            if predictive_factor:
                factors.append(predictive_factor)

        # Factor 6: Time-based patterns (if enabled)
        if self.config.enable_time_based_patterns:
            time_factor = self._evaluate_time_factor(metrics)
            if time_factor:
                factors.append(time_factor)

        # Combine factors to make final decision
        return self._combine_scaling_factors(factors, metrics)

    def _evaluate_utilization_factor(
        self, metrics: ScalingMetrics
    ) -> Optional[Dict[str, Any]]:
        """Evaluate scaling based on connection utilization."""
        utilization = metrics.utilization_ratio

        if utilization >= self.config.scale_up_threshold:
            return {
                "action": ScalingAction.SCALE_UP,
                "reason": ScalingReason.HIGH_UTILIZATION,
                "weight": min(utilization / self.config.scale_up_threshold, 2.0),
                "confidence": 0.8,
                "target_increment": self.config.scale_up_increment,
            }
        elif utilization <= self.config.scale_down_threshold:
            return {
                "action": ScalingAction.SCALE_DOWN,
                "reason": ScalingReason.LOW_UTILIZATION,
                "weight": (self.config.scale_down_threshold - utilization)
                / self.config.scale_down_threshold,
                "confidence": 0.7,
                "target_increment": -self.config.scale_down_increment,
            }

        return None

    def _evaluate_performance_factor(
        self, metrics: ScalingMetrics
    ) -> Optional[Dict[str, Any]]:
        """Evaluate scaling based on performance metrics."""
        if metrics.avg_response_time_ms > self.config.max_response_time_ms:
            severity = metrics.avg_response_time_ms / self.config.max_response_time_ms
            return {
                "action": ScalingAction.SCALE_UP,
                "reason": ScalingReason.HIGH_LATENCY,
                "weight": min(severity, 3.0),
                "confidence": 0.85,
                "target_increment": int(self.config.scale_up_increment * severity),
            }

        return None

    def _evaluate_queue_factor(
        self, metrics: ScalingMetrics
    ) -> Optional[Dict[str, Any]]:
        """Evaluate scaling based on waiting requests."""
        if metrics.waiting_requests > 0:
            queue_ratio = metrics.waiting_requests / max(metrics.current_pool_size, 1)
            return {
                "action": ScalingAction.SCALE_UP,
                "reason": ScalingReason.QUEUE_BUILDUP,
                "weight": min(queue_ratio * 2, 3.0),
                "confidence": 0.9,
                "target_increment": max(
                    self.config.scale_up_increment, metrics.waiting_requests
                ),
            }

        return None

    def _evaluate_resource_constraints(
        self, metrics: ScalingMetrics
    ) -> Optional[Dict[str, Any]]:
        """Evaluate scaling based on resource constraints."""

        # Check if we're hitting resource limits that prevent scaling up
        if (
            metrics.cpu_utilization > self.config.max_cpu_utilization
            or metrics.memory_utilization > self.config.max_memory_utilization
        ):
            return {
                "action": ScalingAction.NO_ACTION,
                "reason": ScalingReason.RESOURCE_CONSTRAINT,
                "weight": 5.0,  # High weight to prevent scaling
                "confidence": 0.95,
                "target_increment": 0,
            }

        return None

    def _evaluate_predictive_factor(
        self, metrics: ScalingMetrics
    ) -> Optional[Dict[str, Any]]:
        """Evaluate scaling based on predictive patterns."""
        if len(self.metrics_history) < 10:
            return None

        # Analyze recent trend
        recent_utilizations = [
            m.utilization_ratio for m in list(self.metrics_history)[-10:]
        ]
        trend = self._calculate_trend(recent_utilizations)

        if trend > 0.05 and metrics.utilization_ratio > 0.6:  # Growing trend
            return {
                "action": ScalingAction.SCALE_UP,
                "reason": ScalingReason.HIGH_UTILIZATION,
                "weight": trend * 2,
                "confidence": 0.6,
                "target_increment": self.config.scale_up_increment,
            }
        elif trend < -0.05 and metrics.utilization_ratio < 0.4:  # Declining trend
            return {
                "action": ScalingAction.SCALE_DOWN,
                "reason": ScalingReason.LOW_UTILIZATION,
                "weight": abs(trend) * 2,
                "confidence": 0.5,
                "target_increment": -self.config.scale_down_increment,
            }

        return None

    def _evaluate_time_factor(
        self, metrics: ScalingMetrics
    ) -> Optional[Dict[str, Any]]:
        """Evaluate scaling based on time-based patterns."""
        current_hour = datetime.now(UTC).hour

        if current_hour in self.time_patterns:
            expected_utilization = self.time_patterns[current_hour]
            actual_utilization = metrics.utilization_ratio

            if actual_utilization > expected_utilization + 0.2:
                return {
                    "action": ScalingAction.SCALE_UP,
                    "reason": ScalingReason.TIME_PATTERN,
                    "weight": 1.5,
                    "confidence": 0.7,
                    "target_increment": self.config.scale_up_increment,
                }

        return None

    def _combine_scaling_factors(
        self, factors: List[Dict[str, Any]], metrics: ScalingMetrics
    ) -> ScalingDecision:
        """Combine multiple scaling factors into a final decision."""

        if not factors:
            return ScalingDecision(
                action=ScalingAction.NO_ACTION,
                target_size=metrics.current_pool_size,
                current_size=metrics.current_pool_size,
                reason=ScalingReason.LOW_UTILIZATION,
                confidence=0.8,
                estimated_benefit="Pool operating within normal parameters",
                metrics_snapshot=metrics,
            )

        # Separate scale up and scale down factors
        scale_up_factors = [f for f in factors if f["action"] == ScalingAction.SCALE_UP]
        scale_down_factors = [
            f for f in factors if f["action"] == ScalingAction.SCALE_DOWN
        ]
        constraint_factors = [
            f for f in factors if f["action"] == ScalingAction.NO_ACTION
        ]

        # Resource constraints override everything
        if constraint_factors:
            strongest_constraint = max(constraint_factors, key=lambda x: x["weight"])
            return ScalingDecision(
                action=ScalingAction.NO_ACTION,
                target_size=metrics.current_pool_size,
                current_size=metrics.current_pool_size,
                reason=strongest_constraint["reason"],
                confidence=strongest_constraint["confidence"],
                estimated_benefit="Prevented scaling due to resource constraints",
                metrics_snapshot=metrics,
            )

        # Calculate weighted scores
        scale_up_score = sum(f["weight"] * f["confidence"] for f in scale_up_factors)
        scale_down_score = sum(
            f["weight"] * f["confidence"] for f in scale_down_factors
        )

        # Make decision based on stronger signal
        if scale_up_score > scale_down_score and scale_up_score > 0.5:
            # Scale up
            increment = int(
                statistics.mean([f["target_increment"] for f in scale_up_factors])
            )
            new_size = min(
                metrics.current_pool_size + increment, self.config.max_pool_size
            )
            confidence = min(scale_up_score / len(scale_up_factors), 1.0)
            primary_reason = max(scale_up_factors, key=lambda x: x["weight"])["reason"]

            return ScalingDecision(
                action=ScalingAction.SCALE_UP,
                target_size=new_size,
                current_size=metrics.current_pool_size,
                reason=primary_reason,
                confidence=confidence,
                estimated_benefit=f"Improve performance by adding {increment} connections",
                metrics_snapshot=metrics,
            )

        elif scale_down_score > 0.5:
            # Scale down
            increment = int(
                statistics.mean([
                    abs(f["target_increment"]) for f in scale_down_factors
                ])
            )
            new_size = max(
                metrics.current_pool_size - increment, self.config.min_pool_size
            )
            confidence = min(scale_down_score / len(scale_down_factors), 1.0)
            primary_reason = max(scale_down_factors, key=lambda x: x["weight"])[
                "reason"
            ]

            return ScalingDecision(
                action=ScalingAction.SCALE_DOWN,
                target_size=new_size,
                current_size=metrics.current_pool_size,
                reason=primary_reason,
                confidence=confidence,
                estimated_benefit=f"Reduce resource usage by removing {increment} connections",
                metrics_snapshot=metrics,
            )

        # No clear signal - no action
        return ScalingDecision(
            action=ScalingAction.NO_ACTION,
            target_size=metrics.current_pool_size,
            current_size=metrics.current_pool_size,
            reason=ScalingReason.LOW_UTILIZATION,
            confidence=0.6,
            estimated_benefit="Conflicting signals - maintaining current size",
            metrics_snapshot=metrics,
        )

    async def _emergency_scaling_decision(
        self, metrics: ScalingMetrics
    ) -> ScalingDecision:
        """Create emergency scaling decision for critical load."""
        emergency_increment = max(
            self.config.scale_up_increment * 2,
            metrics.waiting_requests,
            int(metrics.current_pool_size * 0.5),  # Scale up by 50%
        )

        new_size = min(
            metrics.current_pool_size + emergency_increment, self.config.max_pool_size
        )

        return ScalingDecision(
            action=ScalingAction.EMERGENCY_SCALE,
            target_size=new_size,
            current_size=metrics.current_pool_size,
            reason=ScalingReason.HIGH_UTILIZATION,
            confidence=1.0,
            estimated_benefit=f"Emergency response to critical load - adding {emergency_increment} connections",
            metrics_snapshot=metrics,
        )

    def _is_in_cooldown(self) -> bool:
        """Check if we're in cooldown period after last scaling."""
        return (time.time() - self.last_scaling_time) < self.config.cooldown_seconds

    def _record_metrics(self, metrics: ScalingMetrics) -> None:
        """Record metrics for historical analysis."""
        self.metrics_history.append(metrics)

        # Update time patterns
        if self.config.enable_time_based_patterns:
            current_hour = metrics.timestamp.hour
            if current_hour in self.time_patterns:
                # Exponential moving average
                alpha = 0.1
                self.time_patterns[current_hour] = (
                    alpha * metrics.utilization_ratio
                    + (1 - alpha) * self.time_patterns[current_hour]
                )
            else:
                self.time_patterns[current_hour] = metrics.utilization_ratio

    def _record_decision(self, decision: ScalingDecision) -> None:
        """Record scaling decision for analysis."""
        self.decision_history.append(decision)
        self._statistics["total_decisions"] += 1

    def _update_statistics(self, action: ScalingAction) -> None:
        """Update scaling action statistics."""
        if action == ScalingAction.SCALE_UP:
            self._statistics["scale_up_count"] += 1
        elif action == ScalingAction.SCALE_DOWN:
            self._statistics["scale_down_count"] += 1
        elif action == ScalingAction.EMERGENCY_SCALE:
            self._statistics["emergency_scale_count"] += 1
        else:
            self._statistics["no_action_count"] += 1

    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in a series of values using linear regression."""
        if len(values) < 2:
            return 0.0

        n = len(values)
        x_values = list(range(n))

        # Simple linear regression
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n

        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get scaling statistics and analysis."""
        recent_decisions = (
            list(self.decision_history)[-50:] if self.decision_history else []
        )
        recent_history = (
            list(self.scaling_history)[-20:] if self.scaling_history else []
        )

        return {
            "service": self.service_name,
            "configuration": {
                "min_pool_size": self.config.min_pool_size,
                "max_pool_size": self.config.max_pool_size,
                "scale_up_threshold": self.config.scale_up_threshold,
                "scale_down_threshold": self.config.scale_down_threshold,
                "cooldown_seconds": self.config.cooldown_seconds,
            },
            "statistics": self._statistics.copy(),
            "last_scaling": {
                "time_ago_seconds": time.time() - self.last_scaling_time
                if self.last_scaling_time > 0
                else None,
                "action": self.last_scaling_action.value,
                "in_cooldown": self._is_in_cooldown(),
            },
            "recent_decisions": [
                {
                    "action": d.action.value,
                    "reason": d.reason.value,
                    "confidence": d.confidence,
                    "size_change": d.size_change,
                }
                for d in recent_decisions
            ],
            "recent_scaling_events": recent_history,
            "time_patterns": self.time_patterns.copy(),
            "metrics_history_size": len(self.metrics_history),
            "decision_history_size": len(self.decision_history),
        }


class BurstDetector:
    """Detects burst traffic patterns for intelligent scaling."""

    def __init__(self, window_minutes: int = 5):
        self.window_size = window_minutes * 60  # Convert to seconds
        self.events = deque(maxlen=1000)

    def record_event(self, timestamp: float, value: float) -> None:
        """Record a metric event."""
        self.events.append((timestamp, value))

    def detect_burst(self, current_timestamp: float) -> bool:
        """Detect if we're currently in a burst period."""
        # Remove old events outside window
        cutoff_time = current_timestamp - self.window_size
        recent_events = [(t, v) for t, v in self.events if t >= cutoff_time]

        if len(recent_events) < 5:
            return False

        values = [v for _, v in recent_events]

        # Simple burst detection: recent average significantly higher than historical
        if len(self.events) < 20:
            return False

        recent_avg = statistics.mean(values[-5:])
        historical_avg = statistics.mean([v for _, v in list(self.events)[:-5]])

        return recent_avg > historical_avg * 1.5
