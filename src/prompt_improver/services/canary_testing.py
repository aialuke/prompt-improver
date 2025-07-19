"""Canary A/B Testing Service for APES Pattern Cache Rollout
"""

import asyncio
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import yaml
from rich.console import Console

from prompt_improver.database import get_sessionmanager
from prompt_improver.utils.redis_cache import redis_client

console = Console()


@dataclass
class CanaryMetrics:
    """Metrics for canary testing"""
    total_requests: int
    successful_requests: int
    failed_requests: int
    avg_response_time_ms: float
    cache_hit_ratio: float
    error_rate: float
    p95_response_time_ms: float
    p99_response_time_ms: float
    timestamp: datetime


@dataclass
class CanaryGroup:
    """Configuration for a canary group"""
    name: str
    percentage: float
    enabled: bool
    start_time: datetime
    end_time: datetime | None = None
    success_criteria: dict = None
    rollback_criteria: dict = None


class CanaryTestingService:
    """Service for managing canary A/B testing of pattern cache rollout"""

    def __init__(self):
        self.config = self._load_config()
        self.redis_client = redis_client
        self.metrics_store = {}

    def _load_config(self) -> dict:
        """Load canary testing configuration from Redis config"""
        config_file = "config/redis_config.yaml"
        try:
            with open(config_file) as f:
                config = yaml.safe_load(f)
                return config.get('feature_flags', {}).get('pattern_cache', {})
        except FileNotFoundError:
            console.print(f"❌ Config file not found: {config_file}", style="red")
            return {}
        except yaml.YAMLError as e:
            console.print(f"❌ YAML parsing error: {e}", style="red")
            return {}

    async def should_enable_cache(self, user_id: str, session_id: str) -> bool:
        """Determine if pattern cache should be enabled for this user/session
        based on canary testing configuration
        """
        if not self.config.get('enabled', False):
            return False

        # Check if feature is fully rolled out
        rollout_percentage = self.config.get('rollout_percentage', 0)
        if rollout_percentage >= 100:
            return True

        # Check canary configuration
        canary_config = self.config.get('ab_testing', {}).get('canary', {})
        if not canary_config.get('enabled', False):
            return False

        # Determine group assignment based on user/session hash
        group_assignment = self._get_group_assignment(user_id, session_id)
        current_percentage = canary_config.get('initial_percentage', 0)

        # Check if user falls within the current canary percentage
        return group_assignment < current_percentage

    def _get_group_assignment(self, user_id: str, session_id: str) -> int:
        """Get consistent group assignment (0-100) for a user/session"""
        # Create a consistent hash for the user/session
        hash_input = f"{user_id}:{session_id}"
        hash_value = hash(hash_input)
        # Convert to 0-100 range
        return abs(hash_value) % 100

    async def record_request_metrics(self, user_id: str, session_id: str,
                                   response_time_ms: float, success: bool,
                                   cache_hit: bool = False) -> None:
        """Record metrics for a request"""
        cache_enabled = await self.should_enable_cache(user_id, session_id)
        group = "treatment" if cache_enabled else "control"

        # Store metrics in Redis with expiration
        metric_key = f"canary_metrics:{group}:{datetime.now().strftime('%Y%m%d_%H')}"

        try:
            # Get existing metrics or create new
            existing_metrics = self.redis_client.get(metric_key)
            if existing_metrics:
                metrics = json.loads(existing_metrics)
            else:
                metrics = {
                    "total_requests": 0,
                    "successful_requests": 0,
                    "failed_requests": 0,
                    "response_times": [],
                    "cache_hits": 0,
                    "cache_misses": 0,
                    "start_time": datetime.now().isoformat()
                }

            # Update metrics
            metrics["total_requests"] += 1
            if success:
                metrics["successful_requests"] += 1
            else:
                metrics["failed_requests"] += 1

            metrics["response_times"].append(response_time_ms)

            if cache_enabled:
                if cache_hit:
                    metrics["cache_hits"] += 1
                else:
                    metrics["cache_misses"] += 1

            # Store updated metrics with 48-hour expiration
            self.redis_client.setex(
                metric_key,
                86400 * 2,  # 48 hours
                json.dumps(metrics)
            )

        except Exception as e:
            console.print(f"❌ Error recording canary metrics: {e}", style="red")

    async def get_canary_metrics(self, hours: int = 24) -> dict[str, CanaryMetrics]:
        """Get aggregated canary metrics for the specified time period"""
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=hours)

        control_metrics = await self._aggregate_metrics("control", start_time, end_time)
        treatment_metrics = await self._aggregate_metrics("treatment", start_time, end_time)

        return {
            "control": control_metrics,
            "treatment": treatment_metrics
        }

    async def _aggregate_metrics(self, group: str, start_time: datetime,
                               end_time: datetime) -> CanaryMetrics:
        """Aggregate metrics for a specific group over a time period"""
        total_requests = 0
        successful_requests = 0
        failed_requests = 0
        all_response_times = []
        cache_hits = 0
        cache_misses = 0

        # Get all relevant metric keys
        current_time = start_time
        while current_time <= end_time:
            metric_key = f"canary_metrics:{group}:{current_time.strftime('%Y%m%d_%H')}"

            try:
                metrics_data = self.redis_client.get(metric_key)
                if metrics_data:
                    metrics = json.loads(metrics_data)
                    total_requests += metrics.get("total_requests", 0)
                    successful_requests += metrics.get("successful_requests", 0)
                    failed_requests += metrics.get("failed_requests", 0)
                    all_response_times.extend(metrics.get("response_times", []))
                    cache_hits += metrics.get("cache_hits", 0)
                    cache_misses += metrics.get("cache_misses", 0)

            except Exception as e:
                console.print(f"❌ Error aggregating metrics for {metric_key}: {e}", style="red")

            current_time += timedelta(hours=1)

        # Calculate aggregated metrics
        avg_response_time = sum(all_response_times) / len(all_response_times) if all_response_times else 0
        error_rate = (failed_requests / total_requests) if total_requests > 0 else 0
        cache_hit_ratio = (cache_hits / (cache_hits + cache_misses)) if (cache_hits + cache_misses) > 0 else 0

        # Calculate percentiles
        sorted_times = sorted(all_response_times)
        p95_index = int(len(sorted_times) * 0.95)
        p99_index = int(len(sorted_times) * 0.99)

        p95_response_time = sorted_times[p95_index] if sorted_times else 0
        p99_response_time = sorted_times[p99_index] if sorted_times else 0

        return CanaryMetrics(
            total_requests=total_requests,
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_response_time_ms=avg_response_time,
            cache_hit_ratio=cache_hit_ratio,
            error_rate=error_rate,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            timestamp=datetime.now()
        )

    async def evaluate_canary_success(self) -> dict:
        """Evaluate if the canary test meets success criteria"""
        metrics = await self.get_canary_metrics(hours=24)
        control = metrics["control"]
        treatment = metrics["treatment"]

        canary_config = self.config.get('ab_testing', {}).get('canary', {})
        success_criteria = canary_config.get('success_criteria', {})
        rollback_criteria = canary_config.get('rollback_criteria', {})

        # Check success criteria
        max_error_rate = success_criteria.get('max_error_rate', 0.01)
        min_performance_improvement = success_criteria.get('min_performance_improvement', 0.1)
        min_sample_size = success_criteria.get('min_sample_size', 1000)

        # Check rollback criteria
        max_rollback_error_rate = rollback_criteria.get('max_error_rate', 0.05)
        max_latency_increase = rollback_criteria.get('max_latency_increase', 0.2)

        results = {
            "control_metrics": control,
            "treatment_metrics": treatment,
            "success_criteria_met": False,
            "rollback_triggered": False,
            "recommendations": []
        }

        # Check if we have enough data
        if treatment.total_requests < min_sample_size:
            results["recommendations"].append(
                f"Need more data: {treatment.total_requests} < {min_sample_size} requests"
            )
            return results

        # Check rollback criteria first
        if treatment.error_rate > max_rollback_error_rate:
            results["rollback_triggered"] = True
            results["recommendations"].append(
                f"ROLLBACK: Error rate too high: {treatment.error_rate:.3f} > {max_rollback_error_rate}"
            )
            return results

        # Check for performance regression
        if control.avg_response_time_ms > 0:
            latency_change = (treatment.avg_response_time_ms - control.avg_response_time_ms) / control.avg_response_time_ms
            if latency_change > max_latency_increase:
                results["rollback_triggered"] = True
                results["recommendations"].append(
                    f"ROLLBACK: Latency increase too high: {latency_change:.3f} > {max_latency_increase}"
                )
                return results

        # Check success criteria
        success_checks = []

        # Error rate check
        if treatment.error_rate <= max_error_rate:
            success_checks.append("Error rate within limits")
        else:
            results["recommendations"].append(
                f"Error rate too high: {treatment.error_rate:.3f} > {max_error_rate}"
            )

        # Performance improvement check
        if control.avg_response_time_ms > 0:
            performance_improvement = (control.avg_response_time_ms - treatment.avg_response_time_ms) / control.avg_response_time_ms
            if performance_improvement >= min_performance_improvement:
                success_checks.append("Performance improvement achieved")
                results["recommendations"].append(
                    f"Performance improved by {performance_improvement:.1%}"
                )
            else:
                results["recommendations"].append(
                    f"Performance improvement insufficient: {performance_improvement:.3f} < {min_performance_improvement}"
                )

        # Sample size check
        if treatment.total_requests >= min_sample_size:
            success_checks.append("Sufficient sample size")

        # Overall success determination
        results["success_criteria_met"] = len(success_checks) >= 2

        if results["success_criteria_met"]:
            results["recommendations"].append("SUCCESS: Ready to increase rollout percentage")

        return results

    async def auto_adjust_rollout(self) -> dict:
        """Automatically adjust rollout percentage based on canary results"""
        evaluation = await self.evaluate_canary_success()

        canary_config = self.config.get('ab_testing', {}).get('canary', {})
        current_percentage = canary_config.get('initial_percentage', 0)
        increment_percentage = canary_config.get('increment_percentage', 10)
        max_percentage = canary_config.get('max_percentage', 100)

        result = {
            "previous_percentage": current_percentage,
            "new_percentage": current_percentage,
            "action": "no_change",
            "reason": "Evaluation pending"
        }

        if evaluation["rollback_triggered"]:
            # Rollback to 0%
            result["new_percentage"] = 0
            result["action"] = "rollback"
            result["reason"] = "Rollback criteria triggered"

            # Update configuration
            await self._update_rollout_percentage(0)

        elif evaluation["success_criteria_met"]:
            # Increase rollout percentage
            new_percentage = min(current_percentage + increment_percentage, max_percentage)
            result["new_percentage"] = new_percentage
            result["action"] = "increase"
            result["reason"] = "Success criteria met"

            # Update configuration
            await self._update_rollout_percentage(new_percentage)

        return result

    async def _update_rollout_percentage(self, new_percentage: int) -> None:
        """Update the rollout percentage in configuration"""
        try:
            # Store in Redis for immediate use
            config_key = "canary_config:rollout_percentage"
            self.redis_client.set(config_key, str(new_percentage))

            # Update local config
            if 'ab_testing' not in self.config:
                self.config['ab_testing'] = {}
            if 'canary' not in self.config['ab_testing']:
                self.config['ab_testing']['canary'] = {}

            self.config['ab_testing']['canary']['initial_percentage'] = new_percentage

            console.print(f"✅ Updated rollout percentage to {new_percentage}%", style="green")

        except Exception as e:
            console.print(f"❌ Error updating rollout percentage: {e}", style="red")

    async def get_canary_status(self) -> dict:
        """Get current canary testing status"""
        canary_config = self.config.get('ab_testing', {}).get('canary', {})

        # Get current percentage from Redis (most up-to-date)
        try:
            stored_percentage = self.redis_client.get("canary_config:rollout_percentage")
            current_percentage = int(stored_percentage) if stored_percentage else canary_config.get('initial_percentage', 0)
        except Exception:
            current_percentage = canary_config.get('initial_percentage', 0)

        metrics = await self.get_canary_metrics(hours=24)
        evaluation = await self.evaluate_canary_success()

        return {
            "enabled": canary_config.get('enabled', False),
            "current_percentage": current_percentage,
            "max_percentage": canary_config.get('max_percentage', 100),
            "increment_percentage": canary_config.get('increment_percentage', 10),
            "control_metrics": metrics["control"],
            "treatment_metrics": metrics["treatment"],
            "evaluation": evaluation,
            "recommendations": evaluation.get("recommendations", [])
        }

    async def generate_canary_report(self, hours: int = 24) -> dict:
        """Generate a comprehensive canary testing report"""
        status = await self.get_canary_status()

        report = {
            "timestamp": datetime.now().isoformat(),
            "period_hours": hours,
            "canary_status": status,
            "summary": {
                "total_control_requests": status["control_metrics"].total_requests,
                "total_treatment_requests": status["treatment_metrics"].total_requests,
                "control_error_rate": status["control_metrics"].error_rate,
                "treatment_error_rate": status["treatment_metrics"].error_rate,
                "performance_delta_ms": status["treatment_metrics"].avg_response_time_ms - status["control_metrics"].avg_response_time_ms,
                "cache_hit_ratio": status["treatment_metrics"].cache_hit_ratio,
                "rollout_percentage": status["current_percentage"]
            },
            "recommendations": status["recommendations"]
        }

        return report


# Global instance
canary_service = CanaryTestingService()
