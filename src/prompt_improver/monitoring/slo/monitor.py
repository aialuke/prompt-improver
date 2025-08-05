"""
SLO Monitoring and Alerting Components
=====================================

Implements comprehensive SLO monitoring, error budget tracking, and burn rate alerting
following Google SRE practices with automated response capabilities.
"""

import asyncio
import time
import logging
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime, timedelta, UTC
from dataclasses import dataclass, field
from enum import Enum
from collections import deque

# Unified Redis operations through UnifiedConnectionManager
from ...database.unified_connection_manager import (
    get_unified_manager, ManagerMode, create_security_context
)

# Legacy coredis imports for backward compatibility
try:
    import coredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    coredis = None

from .framework import (
    SLODefinition, SLOTarget, SLOTimeWindow, SLOType, 

)
from .calculator import MultiWindowSLICalculator, SLIResult
from ...performance.monitoring.health.background_manager import get_background_task_manager, TaskPriority

logger = logging.getLogger(__name__)

class AlertSeverity(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class AlertState(Enum):
    """Alert state tracking"""
    OK = "ok"
    PENDING = "pending"
    FIRING = "firing"
    RESOLVED = "resolved"

@dataclass
class Alert:
    """SLO alert definition"""
    id: str
    slo_name: str
    service_name: str
    severity: AlertSeverity
    state: AlertState
    message: str

    # Alert timing
    started_at: datetime
    resolved_at: Optional[datetime] = None
    last_fired_at: Optional[datetime] = None

    # Alert context
    current_value: float = 0.0
    target_value: float = 0.0
    burn_rate: float = 0.0
    error_budget_remaining: float = 0.0

    # Metadata
    labels: Dict[str, str] = field(default_factory=dict)
    annotations: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary format"""
        return {
            "id": self.id,
            "slo_name": self.slo_name,
            "service_name": self.service_name,
            "severity": self.severity.value,
            "state": self.state.value,
            "message": self.message,
            "started_at": self.started_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "last_fired_at": self.last_fired_at.isoformat() if self.last_fired_at else None,
            "current_value": self.current_value,
            "target_value": self.target_value,
            "burn_rate": self.burn_rate,
            "error_budget_remaining": self.error_budget_remaining,
            "labels": self.labels,
            "annotations": self.annotations
        }

class BurnRateAlert:
    """Burn rate alerting with multiple thresholds following Google SRE practices"""

    def __init__(
        self,
        slo_target: SLOTarget,
        short_window: SLOTimeWindow = SLOTimeWindow.HOUR_1,
        long_window: SLOTimeWindow = SLOTimeWindow.DAY_1
    ):
        self.slo_target = slo_target
        self.short_window = short_window
        self.long_window = long_window

        # Google SRE burn rate thresholds
        self.burn_rate_thresholds = {
            # (burn_rate, short_window_hours, long_window_hours, severity)
            (36.0, 1, 6, AlertSeverity.EMERGENCY),   # 5% budget in 1 hour
            (6.0, 6, 24, AlertSeverity.CRITICAL),    # 20% budget in 6 hours  
            (1.0, 24, 72, AlertSeverity.WARNING),    # Normal burn rate over 3 days
            (0.5, 72, 168, AlertSeverity.INFO)       # Slow burn over a week
        }

        # State tracking
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        self.last_evaluation: Optional[datetime] = None

        # Alert cooldown to prevent spam
        self.alert_cooldown_seconds = 300  # 5 minutes
        self.last_alert_times: Dict[str, datetime] = {}

    def evaluate_burn_rate(
        self,
        short_result: SLIResult,
        long_result: SLIResult
    ) -> List[Alert]:
        """Evaluate burn rate and generate alerts if necessary"""
        self.last_evaluation = datetime.now(UTC)
        new_alerts = []

        # Calculate burn rates for both windows
        short_burn_rate = self._calculate_burn_rate(short_result)
        long_burn_rate = self._calculate_burn_rate(long_result)

        # Check each threshold
        for burn_threshold, short_hours, long_hours, severity in self.burn_rate_thresholds:
            alert_id = f"{self.slo_target.name}_{severity.value}_burn_rate"

            # Check if both windows exceed threshold
            short_exceeds = short_burn_rate >= burn_threshold
            long_exceeds = long_burn_rate >= burn_threshold

            # Alert fires when both windows exceed threshold
            should_alert = short_exceeds and long_exceeds

            # Check cooldown
            if should_alert and alert_id in self.last_alert_times:
                time_since_last = datetime.now(UTC) - self.last_alert_times[alert_id]
                if time_since_last.total_seconds() < self.alert_cooldown_seconds:
                    continue

            # Existing alert handling
            existing_alert = self.active_alerts.get(alert_id)

            if should_alert:
                if existing_alert is None:
                    # Create new alert
                    alert = Alert(
                        id=alert_id,
                        slo_name=self.slo_target.name,
                        service_name=self.slo_target.service_name,
                        severity=severity,
                        state=AlertState.FIRING,
                        message=self._generate_burn_rate_message(
                            burn_threshold, short_burn_rate, long_burn_rate, severity
                        ),
                        started_at=datetime.now(UTC),
                        current_value=short_result.current_value,
                        target_value=short_result.target_value,
                        burn_rate=short_burn_rate,
                        labels={
                            "slo_type": self.slo_target.slo_type.value,
                            "alert_type": "burn_rate",
                            "short_window": self.short_window.value,
                            "long_window": self.long_window.value
                        },
                        annotations={
                            "short_burn_rate": str(short_burn_rate),
                            "long_burn_rate": str(long_burn_rate),
                            "threshold": str(burn_threshold),
                            "short_window_hours": str(short_hours),
                            "long_window_hours": str(long_hours)
                        }
                    )

                    self.active_alerts[alert_id] = alert
                    self.last_alert_times[alert_id] = datetime.now(UTC)
                    new_alerts.append(alert)

                    logger.warning(f"Burn rate alert fired: {alert.message}")

                else:
                    # Update existing alert
                    existing_alert.last_fired_at = datetime.now(UTC)
                    existing_alert.current_value = short_result.current_value
                    existing_alert.burn_rate = short_burn_rate

            else:
                if existing_alert is not None and existing_alert.state == AlertState.FIRING:
                    # Resolve alert
                    existing_alert.state = AlertState.RESOLVED
                    existing_alert.resolved_at = datetime.now(UTC)

                    # Move to history
                    self.alert_history.append(existing_alert)
                    del self.active_alerts[alert_id]

                    logger.info(f"Burn rate alert resolved: {alert_id}")

        return new_alerts

    def _calculate_burn_rate(self, result: SLIResult) -> float:
        """Calculate burn rate from SLI result"""
        if result.measurement_count == 0:
            return 0.0

        # Burn rate = actual error rate / allowed error rate
        if self.slo_target.slo_type == SLOType.AVAILABILITY:
            # For availability: burn rate = (1 - availability) / (1 - target)
            availability = result.current_value / 100.0  # Convert percentage
            target_availability = self.slo_target.target_value / 100.0

            actual_error_rate = 1.0 - availability
            allowed_error_rate = 1.0 - target_availability

            if allowed_error_rate > 0:
                return actual_error_rate / allowed_error_rate
            else:
                return 0.0 if actual_error_rate == 0 else float('inf')

        elif self.slo_target.slo_type == SLOType.ERROR_RATE:
            # For error rate: burn rate = actual rate / target rate
            actual_rate = result.current_value / 100.0
            target_rate = self.slo_target.target_value / 100.0

            if target_rate > 0:
                return actual_rate / target_rate
            else:
                return 0.0 if actual_rate == 0 else float('inf')

        elif self.slo_target.slo_type == SLOType.LATENCY:
            # For latency: burn rate = actual latency / target latency
            if self.slo_target.target_value > 0:
                return result.current_value / self.slo_target.target_value
            else:
                return 0.0

        else:
            # Generic calculation
            if self.slo_target.target_value > 0:
                return result.current_value / self.slo_target.target_value
            else:
                return 0.0

    def _generate_burn_rate_message(
        self,
        threshold: float,
        short_rate: float,
        long_rate: float,
        severity: AlertSeverity
    ) -> str:
        """Generate human-readable burn rate alert message"""
        return (
            f"SLO burn rate alert ({severity.value}): "
            f"{self.slo_target.service_name}/{self.slo_target.name} "
            f"burning error budget at {short_rate:.1f}x rate "
            f"(threshold: {threshold:.1f}x, "
            f"short window: {short_rate:.1f}x, "
            f"long window: {long_rate:.1f}x)"
        )

    def get_active_alerts(self) -> List[Alert]:
        """Get all currently active alerts"""
        return list(self.active_alerts.values())

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alert status"""
        return {
            "active_alert_count": len(self.active_alerts),
            "active_alerts": [alert.to_dict() for alert in self.active_alerts.values()],
            "last_evaluation": self.last_evaluation.isoformat() if self.last_evaluation else None,
            "alert_history_count": len(self.alert_history)
        }

class ErrorBudgetMonitor:
    """Monitor error budget consumption and policy enforcement"""

    def __init__(
        self,
        slo_definition: SLODefinition,
        redis_url: Optional[str] = None,
        unified_manager = None
    ):
        self.slo_definition = slo_definition
        self.redis_url = redis_url  # Kept for backward compatibility
        self._redis_client = None  # Legacy field
        
        # Unified connection manager for Redis operations
        self._unified_manager = unified_manager or get_unified_manager(ManagerMode.ASYNC_MODERN)
        self._security_context = None

        # Error budget tracking
        self.error_budgets: Dict[str, ErrorBudget] = {}
        self.budget_history: List[Dict[str, Any]] = []

        # Policy enforcement
        self.policy_actions: Dict[str, Callable] = {}
        self.exhaustion_callbacks: List[Callable] = []

        # Monitoring configuration
        self.check_interval_seconds = 60  # Check every minute
        self.monitoring_task_id: Optional[str] = None
        self.is_monitoring = False

    async def get_redis_client(self) -> Optional[coredis.Redis]:
        """Legacy Redis client method - deprecated, uses UnifiedConnectionManager internally"""
        # For backward compatibility, but internally uses UnifiedConnectionManager
        logger.info("Using UnifiedConnectionManager for Redis operations instead of direct client")
        return None  # Always return None to force usage of unified cache methods
    
    async def _ensure_security_context(self):
        """Ensure security context exists for Redis operations"""
        if self._security_context is None:
            self._security_context = await create_security_context(
                agent_id=f"slo_error_budget_monitor_{self.slo_definition.service_name}",
                tier="professional",
                authenticated=True
            )
        return self._security_context

    def register_policy_action(self, action_name: str, callback: Callable) -> None:
        """Register a policy action callback"""
        self.policy_actions[action_name] = callback

    def register_exhaustion_callback(self, callback: Callable) -> None:
        """Register callback for error budget exhaustion"""
        self.exhaustion_callbacks.append(callback)

    async def update_error_budget(
        self,
        slo_target: SLOTarget,
        total_requests: int,
        failed_requests: int,
        time_window: SLOTimeWindow
    ) -> ErrorBudget:
        """Update error budget for SLO target"""
        budget_key = f"{slo_target.name}_{time_window.value}"

        # Get or create error budget
        if budget_key not in self.error_budgets:
            self.error_budgets[budget_key] = ErrorBudget(
                slo_target=slo_target,
                time_window=time_window
            )

        budget = self.error_budgets[budget_key]

        # Calculate error budget
        budget.calculate_budget(total_requests, failed_requests)

        # Store in unified cache system
        await self._store_budget_unified(budget_key, budget)

        # Check for exhaustion and policy enforcement
        await self._check_budget_exhaustion(budget)

        # Add to history
        self.budget_history.append({
            "timestamp": time.time(),
            "slo_target": slo_target.name,
            "time_window": time_window.value,
            "total_budget": budget.total_budget,
            "consumed_budget": budget.consumed_budget,
            "remaining_budget": budget.remaining_budget,
            "budget_percentage": budget.budget_percentage
        })

        # Keep history size manageable
        if len(self.budget_history) > 10000:
            self.budget_history = self.budget_history[-5000:]

        return budget

    async def _store_budget_unified(self, budget_key: str, budget: ErrorBudget) -> None:
        """Store error budget in unified cache system"""
        try:
            # Initialize unified manager if needed
            if not self._unified_manager._is_initialized:
                await self._unified_manager.initialize()
            
            # Get security context
            security_context = await self._ensure_security_context()
            
            key = f"error_budget:{self.slo_definition.service_name}:{budget_key}"
            data = {
                "total_budget": budget.total_budget,
                "consumed_budget": budget.consumed_budget,
                "remaining_budget": budget.remaining_budget,
                "budget_percentage": budget.budget_percentage,
                "current_burn_rate": budget.current_burn_rate,
                "last_updated": budget.last_updated.isoformat() if budget.last_updated else None,
                "service_name": self.slo_definition.service_name,
                "budget_key": budget_key,
                "time_window_seconds": budget.time_window.seconds
            }

            # Calculate TTL based on budget time window
            ttl_seconds = budget.time_window.seconds * 2
            
            # Store in unified cache (L1 + Redis L2)
            success = await self._unified_manager.set_cached(
                key=key,
                value=data,
                ttl_seconds=ttl_seconds,
                security_context=security_context
            )
            
            if success:
                logger.debug(f"Stored error budget for {self.slo_definition.service_name}:{budget_key} in unified cache")
            else:
                logger.warning(f"Failed to store error budget for {self.slo_definition.service_name}:{budget_key} in unified cache")

        except Exception as e:
            logger.warning(f"Failed to store error budget in unified cache: {e}")

    async def _check_budget_exhaustion(self, budget: ErrorBudget) -> None:
        """Check for error budget exhaustion and enforce policies"""
        if budget.is_budget_exhausted():
            logger.critical(
                f"Error budget exhausted for {budget.slo_target.name} "
                f"({budget.time_window.value} window)"
            )

            # Execute policy actions based on configuration
            policy = self.slo_definition.error_budget_policy

            if policy == "block_deploys":
                await self._execute_policy_action("block_deploys", budget)
            elif policy == "rollback_features":
                await self._execute_policy_action("rollback_features", budget)
            elif policy == "alerting_only":
                await self._execute_policy_action("send_alert", budget)

            # Call exhaustion callbacks
            for callback in self.exhaustion_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(budget)
                    else:
                        callback(budget)
                except Exception as e:
                    logger.error(f"Error budget exhaustion callback failed: {e}")

    async def _execute_policy_action(self, action_name: str, budget: ErrorBudget) -> None:
        """Execute registered policy action"""
        if action_name in self.policy_actions:
            try:
                callback = self.policy_actions[action_name]
                if asyncio.iscoroutinefunction(callback):
                    await callback(budget)
                else:
                    callback(budget)

                logger.info(f"Executed policy action: {action_name}")

            except Exception as e:
                logger.error(f"Policy action {action_name} failed: {e}")
        else:
            logger.warning(f"Policy action {action_name} not registered")

    def get_budget_status(self) -> Dict[str, Any]:
        """Get comprehensive error budget status"""
        status = {
            "service_name": self.slo_definition.service_name,
            "slo_name": self.slo_definition.name,
            "error_budget_policy": self.slo_definition.error_budget_policy,
            "budgets": {},
            "overall_status": "healthy"
        }

        exhausted_budgets = []

        for budget_key, budget in self.error_budgets.items():
            budget_status = {
                "total_budget": budget.total_budget,
                "consumed_budget": budget.consumed_budget,
                "remaining_budget": budget.remaining_budget,
                "budget_percentage": budget.budget_percentage,
                "current_burn_rate": budget.current_burn_rate,
                "is_exhausted": budget.is_budget_exhausted(),
                "time_to_exhaustion": None,
                "last_updated": budget.last_updated.isoformat() if budget.last_updated else None
            }

            # Calculate time to exhaustion
            tte = budget.time_to_exhaustion()
            if tte:
                budget_status["time_to_exhaustion"] = tte.total_seconds()

            status["budgets"][budget_key] = budget_status

            if budget.is_budget_exhausted():
                exhausted_budgets.append(budget_key)

        if exhausted_budgets:
            status["overall_status"] = "error_budget_exhausted"
            status["exhausted_budgets"] = exhausted_budgets
        elif any(b.budget_percentage > 80 for b in self.error_budgets.values()):
            status["overall_status"] = "budget_warning"

        return status

    async def start_monitoring(self) -> None:
        """Start continuous error budget monitoring"""
        if self.is_monitoring:
            return

        self.is_monitoring = True
        task_manager = get_background_task_manager()
        self.monitoring_task_id = await task_manager.submit_enhanced_task(
            task_id=f"error_budget_monitor_{self.slo_definition.service_name}_{id(self)}",
            coroutine=self._monitoring_loop,
            priority=TaskPriority.HIGH,
            tags={
                "type": "slo_monitoring",
                "service": self.slo_definition.service_name,
                "component": "error_budget"
            }
        )
        logger.info("Started error budget monitoring")

    async def stop_monitoring(self) -> None:
        """Stop error budget monitoring"""
        self.is_monitoring = False
        if self.monitoring_task_id:
            task_manager = get_background_task_manager()
            await task_manager.cancel_task(self.monitoring_task_id)
            self.monitoring_task_id = None
        logger.info("Stopped error budget monitoring")

    async def _monitoring_loop(self) -> None:
        """Continuous monitoring loop"""
        while self.is_monitoring:
            try:
                # Check all error budgets
                for budget in self.error_budgets.values():
                    await self._check_budget_exhaustion(budget)

                await asyncio.sleep(self.check_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in budget monitoring loop: {e}")
                await asyncio.sleep(self.check_interval_seconds)

class SLOMonitor:
    """Comprehensive SLO monitoring orchestrator"""

    def __init__(
        self,
        slo_definition: SLODefinition,
        redis_url: Optional[str] = None,
        alert_callbacks: Optional[List[Callable]] = None
    ):
        self.slo_definition = slo_definition
        self.redis_url = redis_url
        self.alert_callbacks = alert_callbacks or []

        # Get unified manager for Redis operations
        self._unified_manager = get_unified_manager(ManagerMode.ASYNC_MODERN)
        
        # Create calculators for each SLO target with unified manager
        self.calculators: Dict[str, MultiWindowSLICalculator] = {}
        for target in slo_definition.targets:
            self.calculators[target.name] = MultiWindowSLICalculator(
                slo_target=target,
                redis_url=redis_url,  # Legacy parameter
                unified_manager=self._unified_manager  # Use shared unified manager
            )

        # Create burn rate alerting for each target
        self.burn_rate_alerts: Dict[str, BurnRateAlert] = {}
        for target in slo_definition.targets:
            self.burn_rate_alerts[target.name] = BurnRateAlert(slo_target=target)

        # Error budget monitoring with unified manager
        self.error_budget_monitor = ErrorBudgetMonitor(
            slo_definition=slo_definition,
            redis_url=redis_url,  # Legacy parameter
            unified_manager=self._unified_manager  # Use shared unified manager
        )

        # Monitoring state
        self.is_monitoring = False
        self.monitoring_task_id: Optional[str] = None
        self.check_interval_seconds = 60

        # Alert aggregation
        self.alert_aggregator = AlertAggregator()

    def add_measurement(
        self,
        target_name: str,
        value: float,
        timestamp: Optional[float] = None,
        success: bool = True,
        labels: Optional[Dict[str, str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """Add measurement to specific SLO target"""
        if target_name in self.calculators:
            self.calculators[target_name].add_measurement(
                value=value,
                timestamp=timestamp,
                success=success,
                labels=labels,
                metadata=metadata
            )
        else:
            logger.warning(f"SLO target {target_name} not found")

    async def evaluate_slos(self) -> Dict[str, Any]:
        """Evaluate all SLOs and generate alerts"""
        evaluation_results = {
            "service_name": self.slo_definition.service_name,
            "evaluation_time": datetime.now(UTC).isoformat(),
            "slo_results": {},
            "alerts": [],
            "error_budget_status": {}
        }

        all_alerts = []

        # Evaluate each SLO target
        for target_name, calculator in self.calculators.items():
            try:
                # Calculate SLI for multiple windows
                window_results = calculator.calculate_all_windows()

                # Get trend analysis
                trends = calculator.analyze_trends()

                # Store results
                evaluation_results["slo_results"][target_name] = {
                    "window_results": {
                        window.value: {
                            "current_value": result.current_value,
                            "target_value": result.target_value,
                            "compliance_ratio": result.compliance_ratio,
                            "is_compliant": result.is_compliant,
                            "measurement_count": result.measurement_count
                        }
                        for window, result in window_results.items()
                    },
                    "trends": trends
                }

                # Check burn rate alerts
                if len(window_results) >= 2:
                    burn_rate_alerter = self.burn_rate_alerts[target_name]

                    # Use 1h and 24h windows for burn rate evaluation
                    short_result = window_results.get(SLOTimeWindow.HOUR_1)
                    long_result = window_results.get(SLOTimeWindow.DAY_1)

                    if short_result and long_result:
                        new_alerts = burn_rate_alerter.evaluate_burn_rate(
                            short_result, long_result
                        )
                        all_alerts.extend(new_alerts)

                # Update error budget
                target = next(t for t in self.slo_definition.targets if t.name == target_name)
                primary_result = window_results.get(SLOTimeWindow.DAY_1)

                if primary_result and primary_result.measurement_count > 0:
                    # Calculate failed requests for error budget
                    if target.slo_type == SLOType.AVAILABILITY:
                        total_requests = primary_result.measurement_count
                        success_rate = primary_result.current_value / 100.0
                        failed_requests = int(total_requests * (1.0 - success_rate))
                    elif target.slo_type == SLOType.ERROR_RATE:
                        total_requests = primary_result.measurement_count
                        error_rate = primary_result.current_value / 100.0
                        failed_requests = int(total_requests * error_rate)
                    else:
                        # For other types, approximate based on compliance
                        total_requests = primary_result.measurement_count
                        failed_requests = int(total_requests * (1.0 - primary_result.compliance_ratio))

                    await self.error_budget_monitor.update_error_budget(
                        target, total_requests, failed_requests, SLOTimeWindow.DAY_1
                    )

            except Exception as e:
                logger.error(f"Failed to evaluate SLO target {target_name}: {e}")
                evaluation_results["slo_results"][target_name] = {"error": str(e)}

        # Get error budget status
        evaluation_results["error_budget_status"] = self.error_budget_monitor.get_budget_status()

        # Aggregate and deduplicate alerts
        aggregated_alerts = self.alert_aggregator.aggregate_alerts(all_alerts)
        evaluation_results["alerts"] = [alert.to_dict() for alert in aggregated_alerts]

        # Send alerts through callbacks
        for alert in aggregated_alerts:
            await self._send_alert(alert)

        return evaluation_results

    async def _send_alert(self, alert: Alert) -> None:
        """Send alert through registered callbacks"""
        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Alert callback failed: {e}")

    def register_alert_callback(self, callback: Callable) -> None:
        """Register alert callback"""
        self.alert_callbacks.append(callback)

    async def start_monitoring(self) -> None:
        """Start continuous SLO monitoring"""
        if self.is_monitoring:
            return

        self.is_monitoring = True

        # Start error budget monitoring
        await self.error_budget_monitor.start_monitoring()

        # Start main monitoring loop
        task_manager = get_background_task_manager()
        self.monitoring_task_id = await task_manager.submit_enhanced_task(
            task_id=f"slo_monitor_{self.slo_definition.service_name}_{id(self)}",
            coroutine=self._monitoring_loop,
            priority=TaskPriority.HIGH,
            tags={
                "type": "slo_monitoring",
                "service": self.slo_definition.service_name,
                "component": "slo_monitor"
            }
        )

        logger.info(f"Started SLO monitoring for {self.slo_definition.service_name}")

    async def stop_monitoring(self) -> None:
        """Stop SLO monitoring"""
        self.is_monitoring = False

        # Stop error budget monitoring
        await self.error_budget_monitor.stop_monitoring()

        # Stop main monitoring loop
        if self.monitoring_task_id:
            task_manager = get_background_task_manager()
            await task_manager.cancel_task(self.monitoring_task_id)
            self.monitoring_task_id = None

        logger.info(f"Stopped SLO monitoring for {self.slo_definition.service_name}")

    async def _monitoring_loop(self) -> None:
        """Main monitoring loop"""
        while self.is_monitoring:
            try:
                await self.evaluate_slos()
                await asyncio.sleep(self.check_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in SLO monitoring loop: {e}")
                await asyncio.sleep(self.check_interval_seconds)

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive monitoring status"""
        return {
            "service_name": self.slo_definition.service_name,
            "slo_name": self.slo_definition.name,
            "is_monitoring": self.is_monitoring,
            "targets_count": len(self.slo_definition.targets),
            "active_alerts": sum(
                len(alerter.get_active_alerts()) 
                for alerter in self.burn_rate_alerts.values()
            ),
            "error_budget_status": self.error_budget_monitor.get_budget_status(),
            "last_evaluation": datetime.now(UTC).isoformat()
        }

class AlertAggregator:
    """Aggregate and deduplicate alerts to reduce noise"""

    def __init__(self, aggregation_window_seconds: int = 300):
        self.aggregation_window_seconds = aggregation_window_seconds
        self.recent_alerts: deque = deque()

    def aggregate_alerts(self, alerts: List[Alert]) -> List[Alert]:
        """Aggregate similar alerts within time window"""
        current_time = datetime.now(UTC)

        # Remove old alerts from recent list
        cutoff_time = current_time - timedelta(seconds=self.aggregation_window_seconds)
        self.recent_alerts = deque([
            alert for alert in self.recent_alerts 
            if alert.started_at > cutoff_time
        ])

        # Deduplicate new alerts
        unique_alerts = []

        for alert in alerts:
            is_duplicate = False

            for recent_alert in self.recent_alerts:
                if self._are_similar_alerts(alert, recent_alert):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_alerts.append(alert)
                self.recent_alerts.append(alert)

        return unique_alerts

    def _are_similar_alerts(self, alert1: Alert, alert2: Alert) -> bool:
        """Check if two alerts are similar enough to be considered duplicates"""
        return (
            alert1.slo_name == alert2.slo_name and
            alert1.service_name == alert2.service_name and
            alert1.severity == alert2.severity and
            alert1.labels.get("alert_type") == alert2.labels.get("alert_type")
        )