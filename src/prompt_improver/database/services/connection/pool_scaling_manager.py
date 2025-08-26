"""Dynamic pool scaling management for PostgreSQL connections.

Handles automatic pool size optimization based on utilization patterns.
Extracted from PostgreSQLPoolManager following single responsibility principle.
"""

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

from prompt_improver.database.services.connection.pool_shared_context import (
    PoolSharedContext,
    PoolState,
)

logger = logging.getLogger(__name__)


class PoolScalingManager:
    """Dynamic pool scaling management.

    Responsible for:
    - Pool size optimization based on utilization patterns
    - Scaling decision algorithms and execution
    - Performance metrics collection for scaling decisions
    - Scaling threshold management and cooldown enforcement
    """

    def __init__(self, shared_context: PoolSharedContext) -> None:
        self.context = shared_context

        logger.info(
            f"PoolScalingManager initialized for service: {self.context.service_name}"
        )

    async def optimize_pool_size(self) -> dict[str, Any]:
        """Dynamically optimize pool size based on load patterns."""
        current_metrics = await self.collect_pool_metrics()

        # Check cooldown period
        if (datetime.now(UTC) - timedelta(minutes=5)) < datetime.fromtimestamp(
            self.context.last_scale_time, UTC
        ):
            return {"status": "skipped", "reason": "optimization cooldown"}

        utilization = current_metrics.get("utilization", 0) / 100.0
        waiting_requests = current_metrics.get("waiting_requests", 0)
        recommendations = []
        new_pool_size = self.context.current_pool_size

        # Scale up logic
        if utilization > 0.9 and waiting_requests > 0:
            increase = min(5, self.context.max_pool_size - self.context.current_pool_size)
            if increase > 0:
                new_pool_size += increase
                recommendations.append(
                    f"Increase pool size by {increase} (high utilization: {utilization:.1%})"
                )
                self.context.pool_state = PoolState.STRESSED

        # Scale down logic
        elif utilization < 0.3 and self.context.current_pool_size > self.context.min_pool_size:
            decrease = min(3, self.context.current_pool_size - self.context.min_pool_size)
            if decrease > 0:
                new_pool_size -= decrease
                recommendations.append(
                    f"Decrease pool size by {decrease} (low utilization: {utilization:.1%})"
                )

        # Apply scaling
        if new_pool_size != self.context.current_pool_size:
            try:
                await self.scale_pool(new_pool_size)
                return {
                    "status": "optimized",
                    "previous_size": self.context.current_pool_size,
                    "new_size": new_pool_size,
                    "utilization": utilization,
                    "recommendations": recommendations,
                }
            except Exception as e:
                logger.exception(f"Failed to optimize pool size: {e}")
                return {"status": "error", "error": str(e)}

        return {
            "status": "no_change_needed",
            "current_size": self.context.current_pool_size,
            "utilization": utilization,
            "state": self.context.pool_state.value,
        }

    async def evaluate_scaling_need(self) -> dict[str, Any]:
        """Evaluate if pool scaling is needed based on current metrics."""
        if not self._can_scale():
            return {
                "scaling_needed": False,
                "reason": "scaling_cooldown_active",
                "cooldown_remaining": self._get_cooldown_remaining(),
            }

        utilization = self.context.metrics.pool_utilization / 100.0

        # Evaluate scale up need
        if self.context.should_scale_up():
            return {
                "scaling_needed": True,
                "direction": "up",
                "current_size": self.context.current_pool_size,
                "recommended_size": min(
                    self.context.current_pool_size + 5,
                    self.context.max_pool_size
                ),
                "utilization": utilization,
                "reason": f"high_utilization_{utilization:.1%}",
            }

        # Evaluate scale down need
        if self.context.should_scale_down():
            return {
                "scaling_needed": True,
                "direction": "down",
                "current_size": self.context.current_pool_size,
                "recommended_size": max(
                    self.context.current_pool_size - 3,
                    self.context.min_pool_size
                ),
                "utilization": utilization,
                "reason": f"low_utilization_{utilization:.1%}",
            }

        return {
            "scaling_needed": False,
            "reason": "utilization_within_bounds",
            "utilization": utilization,
            "current_size": self.context.current_pool_size,
        }

    async def scale_pool(self, new_size: int) -> None:
        """Scale the connection pool to new size."""
        if not self.context.async_engine:
            logger.warning("Cannot scale pool - engine not initialized")
            return

        if new_size < self.context.min_pool_size or new_size > self.context.max_pool_size:
            raise ValueError(
                f"New pool size {new_size} outside allowed range "
                f"[{self.context.min_pool_size}, {self.context.max_pool_size}]"
            )

        old_size = self.context.current_pool_size

        try:
            logger.info(f"Pool scaling: {old_size} → {new_size} connections")

            # Record scaling event
            self.context.record_scale_event(new_size)

            # Update pool state based on scaling direction
            if new_size > old_size:
                self.context.pool_state = PoolState.SCALING_UP
            elif new_size < old_size:
                self.context.pool_state = PoolState.SCALING_DOWN
            else:
                self.context.pool_state = PoolState.HEALTHY

            logger.info(f"Pool successfully scaled to {new_size} connections")

        except Exception as e:
            logger.exception(f"Failed to scale pool from {old_size} to {new_size}: {e}")
            raise

    async def collect_pool_metrics(self) -> dict[str, Any]:
        """Collect current pool metrics from SQLAlchemy engine."""
        if not self.context.async_engine:
            return {}

        pool = self.context.async_engine.pool
        return {
            "pool_size": pool.size(),
            "available": pool.checkedin(),
            "active": pool.checkedout(),
            "utilization": pool.checkedout() / pool.size() * 100
            if pool.size() > 0
            else 0,
            "waiting_requests": 0,  # SQLAlchemy doesn't expose this directly
            "overflow": pool.overflow(),
            "invalid": pool.invalid(),
        }

    def set_scaling_thresholds(
        self,
        scale_up_threshold: float,
        scale_down_threshold: float
    ) -> None:
        """Set scaling thresholds for automatic scaling."""
        if not (0.0 <= scale_up_threshold <= 1.0):
            raise ValueError("scale_up_threshold must be between 0.0 and 1.0")
        if not (0.0 <= scale_down_threshold <= 1.0):
            raise ValueError("scale_down_threshold must be between 0.0 and 1.0")
        if scale_down_threshold >= scale_up_threshold:
            raise ValueError("scale_down_threshold must be less than scale_up_threshold")

        old_up = self.context.scale_up_threshold
        old_down = self.context.scale_down_threshold

        self.context.scale_up_threshold = scale_up_threshold
        self.context.scale_down_threshold = scale_down_threshold

        logger.info(
            f"Scaling thresholds updated: "
            f"up={old_up:.1%}→{scale_up_threshold:.1%}, "
            f"down={old_down:.1%}→{scale_down_threshold:.1%}"
        )

    def get_scaling_metrics(self) -> dict[str, Any]:
        """Get scaling-specific metrics and status."""
        return {
            "service": self.context.service_name,
            "current_pool_size": self.context.current_pool_size,
            "min_pool_size": self.context.min_pool_size,
            "max_pool_size": self.context.max_pool_size,
            "thresholds": {
                "scale_up": self.context.scale_up_threshold,
                "scale_down": self.context.scale_down_threshold,
            },
            "cooldown": {
                "seconds": self.context.scale_cooldown_seconds,
                "last_scale_time": self.context.last_scale_time,
                "remaining": self._get_cooldown_remaining(),
                "can_scale": self._can_scale(),
            },
            "utilization": self.context.metrics.pool_utilization,
            "scaling_state": self.context.pool_state.value,
            "last_scale_event": self.context.metrics.last_scale_event.isoformat()
            if self.context.metrics.last_scale_event
            else None,
        }

    async def perform_background_scaling_evaluation(self) -> None:
        """Perform background scaling evaluation and execute if needed.

        This method is called by the monitoring service's background loop.
        """
        try:
            evaluation = await self.evaluate_scaling_need()

            if evaluation["scaling_needed"]:
                recommended_size = evaluation["recommended_size"]
                direction = evaluation["direction"]

                logger.info(
                    f"Background scaling triggered: {direction} to {recommended_size} "
                    f"(reason: {evaluation['reason']})"
                )

                await self.scale_pool(recommended_size)

                # Update pool state after scaling
                if direction == "up":
                    self.context.pool_state = PoolState.SCALING_UP
                else:
                    self.context.pool_state = PoolState.SCALING_DOWN
            # Reset to healthy if no scaling needed
            elif self.context.pool_state in {PoolState.SCALING_UP, PoolState.SCALING_DOWN}:
                self.context.pool_state = PoolState.HEALTHY

        except Exception as e:
            logger.exception(f"Error during background scaling evaluation: {e}")

    def _can_scale(self) -> bool:
        """Check if scaling is allowed based on cooldown."""
        return time.time() - self.context.last_scale_time >= self.context.scale_cooldown_seconds

    def _get_cooldown_remaining(self) -> float:
        """Get remaining cooldown time in seconds."""
        elapsed = time.time() - self.context.last_scale_time
        return max(0, self.context.scale_cooldown_seconds - elapsed)

    @property
    def current_pool_size(self) -> int:
        """Get current pool size."""
        return self.context.current_pool_size
