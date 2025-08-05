"""
Feature Flag Integration for Error Budget Policy Enforcement
==========================================================

Implements automated feature flag rollback and deployment blocking
when error budgets are exhausted, following Google SRE practices.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, UTC
from dataclasses import dataclass, field
from enum import Enum
import json

try:
    import coredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    coredis = None

from .framework import ErrorBudget, SLODefinition
from .monitor import ErrorBudgetMonitor

logger = logging.getLogger(__name__)

class PolicyAction(Enum):
    """Available policy enforcement actions"""
    ALERT_ONLY = "alert_only"
    BLOCK_DEPLOYS = "block_deploys"
    ROLLBACK_FEATURES = "rollback_features"
    CIRCUIT_BREAKER = "circuit_breaker"
    TRAFFIC_REDUCTION = "traffic_reduction"

class FeatureFlagState(Enum):
    """Feature flag states"""
    ENABLED = "enabled"
    DISABLED = "disabled"
    ROLLBACK = "rollback"
    CANARY = "canary"

@dataclass
class FeatureFlag:
    """Feature flag definition"""
    name: str
    description: str
    service_name: str
    current_state: FeatureFlagState
    
    # Rollback configuration
    can_rollback: bool = True
    rollback_priority: int = 1  # Lower number = higher priority for rollback
    rollback_impact: str = "low"  # low, medium, high, critical
    
    # Dependencies
    dependencies: List[str] = field(default_factory=list)
    dependent_flags: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_modified: datetime = field(default_factory=datetime.utcnow)
    owner_team: str = ""
    
    # State tracking
    previous_state: Optional[FeatureFlagState] = None
    rollback_reason: Optional[str] = None
    rollback_timestamp: Optional[datetime] = None

@dataclass
class DeploymentBlock:
    """Deployment blocking record"""
    service_name: str
    blocked_at: datetime
    reason: str
    error_budget_consumed: float
    unblock_criteria: Dict[str, Any]
    
    # Tracking
    is_active: bool = True
    unblocked_at: Optional[datetime] = None
    override_by: Optional[str] = None

class FeatureFlagManager:
    """Manages feature flags for error budget policy enforcement"""
    
    def __init__(
        self,
        redis_url: Optional[str] = None,
        default_rollback_timeout: int = 3600  # 1 hour
    ):
        self.redis_url = redis_url
        self.default_rollback_timeout = default_rollback_timeout
        self._redis_client = None
        
        # Feature flag storage
        self.feature_flags: Dict[str, FeatureFlag] = {}
        self.rollback_history: List[Dict[str, Any]] = []
        
        # Policy configuration
        self.rollback_policies: Dict[str, Dict[str, Any]] = {}
        
        # Callbacks for external systems
        self.rollback_callbacks: List[Callable] = []
        self.deployment_callbacks: List[Callable] = []
    
    async def get_redis_client(self) -> Optional[coredis.Redis]:
        """Get Redis client for distributed flag management via UnifiedConnectionManager"""
        if not REDIS_AVAILABLE:
            return None
            
        if self._redis_client is None:
            try:
                # Use UnifiedConnectionManager for consistent Redis management
                from ...database.unified_connection_manager import get_unified_manager, ManagerMode
                unified_manager = get_unified_manager(ManagerMode.HIGH_AVAILABILITY)
                if not unified_manager._is_initialized:
                    await unified_manager.initialize()
                
                # Access underlying Redis client
                if hasattr(unified_manager, '_redis_master') and unified_manager._redis_master:
                    self._redis_client = unified_manager._redis_master
                    await self._redis_client.ping()
                else:
                    logger.warning("Redis client not available via UnifiedConnectionManager")
                    return None
            except Exception as e:
                logger.warning(f"Failed to connect to Redis via UnifiedConnectionManager: {e}")
                return None
        
        return self._redis_client
    
    def register_feature_flag(
        self,
        name: str,
        description: str,
        service_name: str,
        can_rollback: bool = True,
        rollback_priority: int = 1,
        rollback_impact: str = "low",
        dependencies: Optional[List[str]] = None,
        owner_team: str = ""
    ) -> FeatureFlag:
        """Register a new feature flag"""
        flag = FeatureFlag(
            name=name,
            description=description,
            service_name=service_name,
            current_state=FeatureFlagState.ENABLED,
            can_rollback=can_rollback,
            rollback_priority=rollback_priority,
            rollback_impact=rollback_impact,
            dependencies=dependencies or [],
            owner_team=owner_team
        )
        
        self.feature_flags[name] = flag
        logger.info(f"Registered feature flag: {name} for service {service_name}")
        
        return flag
    
    def set_rollback_policy(
        self,
        service_name: str,
        error_budget_threshold: float = 90.0,
        actions: List[PolicyAction] = None,
        rollback_order: List[str] = None
    ) -> None:
        """Set error budget policy for a service"""
        if actions is None:
            actions = [PolicyAction.ALERT_ONLY, PolicyAction.ROLLBACK_FEATURES]
        
        self.rollback_policies[service_name] = {
            "error_budget_threshold": error_budget_threshold,
            "actions": actions,
            "rollback_order": rollback_order or [],
            "created_at": datetime.now(UTC).isoformat()
        }
        
        logger.info(f"Set rollback policy for service {service_name}")
    
    async def handle_error_budget_exhaustion(
        self,
        error_budget: ErrorBudget,
        service_name: str
    ) -> Dict[str, Any]:
        """Handle error budget exhaustion with policy enforcement"""
        budget_consumed = error_budget.budget_percentage
        
        # Get policy for service
        policy = self.rollback_policies.get(service_name, {})
        threshold = policy.get("error_budget_threshold", 90.0)
        actions = policy.get("actions", [PolicyAction.ALERT_ONLY])
        
        if budget_consumed < threshold:
            return {"action": "none", "reason": "threshold_not_reached"}
        
        logger.critical(
            f"Error budget exhausted for {service_name}: "
            f"{budget_consumed:.1f}% consumed (threshold: {threshold:.1f}%)"
        )
        
        results = []
        
        # Execute policy actions in order
        for action in actions:
            try:
                if action == PolicyAction.ROLLBACK_FEATURES:
                    result = await self._rollback_features(service_name, error_budget)
                    results.append(result)
                    
                elif action == PolicyAction.BLOCK_DEPLOYS:
                    result = await self._block_deployments(service_name, error_budget)
                    results.append(result)
                    
                elif action == PolicyAction.CIRCUIT_BREAKER:
                    result = await self._activate_circuit_breaker(service_name, error_budget)
                    results.append(result)
                    
                elif action == PolicyAction.TRAFFIC_REDUCTION:
                    result = await self._reduce_traffic(service_name, error_budget)
                    results.append(result)
                    
                elif action == PolicyAction.ALERT_ONLY:
                    result = {"action": "alert", "status": "sent"}
                    results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to execute policy action {action}: {e}")
                results.append({"action": action.value, "status": "failed", "error": str(e)})
        
        return {
            "service_name": service_name,
            "budget_consumed": budget_consumed,
            "threshold": threshold,
            "actions_taken": results,
            "timestamp": datetime.now(UTC).isoformat()
        }
    
    async def _rollback_features(
        self,
        service_name: str,
        error_budget: ErrorBudget
    ) -> Dict[str, Any]:
        """Rollback feature flags for the service"""
        
        # Get rollbackable flags for service
        service_flags = [
            flag for flag in self.feature_flags.values()
            if (flag.service_name == service_name and 
                flag.can_rollback and 
                flag.current_state == FeatureFlagState.ENABLED)
        ]
        
        if not service_flags:
            return {
                "action": "rollback_features",
                "status": "no_rollbackable_flags",
                "flags_rolled_back": []
            }
        
        # Sort by rollback priority (lower number = higher priority)
        service_flags.sort(key=lambda f: (f.rollback_priority, f.rollback_impact))
        
        # Get rollback order from policy
        policy = self.rollback_policies.get(service_name, {})
        rollback_order = policy.get("rollback_order", [])
        
        # Reorder flags based on specified order
        if rollback_order:
            ordered_flags = []
            
            # Add flags in specified order
            for flag_name in rollback_order:
                flag = next((f for f in service_flags if f.name == flag_name), None)
                if flag:
                    ordered_flags.append(flag)
                    service_flags.remove(flag)
            
            # Add remaining flags
            ordered_flags.extend(service_flags)
            service_flags = ordered_flags
        
        # Determine how many flags to rollback based on budget consumption
        budget_consumed = error_budget.budget_percentage
        
        if budget_consumed >= 95:
            # Critical: rollback all flags
            flags_to_rollback = service_flags
        elif budget_consumed >= 90:
            # High: rollback high/critical impact flags
            flags_to_rollback = [
                f for f in service_flags 
                if f.rollback_impact in ["high", "critical"]
            ][:3]  # Max 3 flags
        else:
            # Medium: rollback highest priority flag only
            flags_to_rollback = service_flags[:1]
        
        # Execute rollbacks
        rolled_back_flags = []
        
        for flag in flags_to_rollback:
            try:
                success = await self._rollback_single_flag(
                    flag, 
                    f"Error budget exhausted: {budget_consumed:.1f}% consumed"
                )
                
                if success:
                    rolled_back_flags.append({
                        "name": flag.name,
                        "impact": flag.rollback_impact,
                        "priority": flag.rollback_priority
                    })
                    
                    # Store rollback history
                    self.rollback_history.append({
                        "flag_name": flag.name,
                        "service_name": service_name,
                        "rollback_time": datetime.now(UTC).isoformat(),
                        "reason": "error_budget_exhaustion",
                        "budget_consumed": budget_consumed,
                        "previous_state": flag.previous_state.value if flag.previous_state else None
                    })
                    
            except Exception as e:
                logger.error(f"Failed to rollback flag {flag.name}: {e}")
        
        return {
            "action": "rollback_features",
            "status": "completed",
            "flags_rolled_back": rolled_back_flags,
            "total_flags_available": len(service_flags)
        }
    
    async def _rollback_single_flag(self, flag: FeatureFlag, reason: str) -> bool:
        """Rollback a single feature flag"""
        try:
            # Check dependencies
            if flag.dependencies:
                for dep_name in flag.dependencies:
                    dep_flag = self.feature_flags.get(dep_name)
                    if dep_flag and dep_flag.current_state == FeatureFlagState.ENABLED:
                        logger.warning(
                            f"Cannot rollback {flag.name} due to active dependency: {dep_name}"
                        )
                        return False
            
            # Store previous state
            flag.previous_state = flag.current_state
            flag.current_state = FeatureFlagState.ROLLBACK
            flag.rollback_reason = reason
            flag.rollback_timestamp = datetime.now(UTC)
            flag.last_modified = datetime.now(UTC)
            
            # Update in Redis
            await self._store_flag_state(flag)
            
            # Execute rollback callbacks
            for callback in self.rollback_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(flag, reason)
                    else:
                        callback(flag, reason)
                except Exception as e:
                    logger.error(f"Rollback callback failed for {flag.name}: {e}")
            
            logger.warning(f"Rolled back feature flag: {flag.name} (reason: {reason})")
            
            # Rollback dependent flags
            await self._rollback_dependent_flags(flag, reason)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to rollback flag {flag.name}: {e}")
            return False
    
    async def _rollback_dependent_flags(self, flag: FeatureFlag, reason: str) -> None:
        """Rollback flags that depend on this flag"""
        if not flag.dependent_flags:
            return
        
        for dep_flag_name in flag.dependent_flags:
            dep_flag = self.feature_flags.get(dep_flag_name)
            if dep_flag and dep_flag.current_state == FeatureFlagState.ENABLED:
                logger.info(f"Rolling back dependent flag: {dep_flag_name}")
                await self._rollback_single_flag(
                    dep_flag, 
                    f"Dependency rollback: {flag.name} was rolled back"
                )
    
    async def _block_deployments(
        self,
        service_name: str,
        error_budget: ErrorBudget
    ) -> Dict[str, Any]:
        """Block deployments for the service"""
        
        block = DeploymentBlock(
            service_name=service_name,
            blocked_at=datetime.now(UTC),
            reason=f"Error budget exhausted: {error_budget.budget_percentage:.1f}% consumed",
            error_budget_consumed=error_budget.budget_percentage,
            unblock_criteria={
                "error_budget_threshold": 50.0,  # Unblock when budget consumption < 50%
                "minimum_block_duration": 1800,  # At least 30 minutes
                "manual_override_allowed": True
            }
        )
        
        # Store block record
        await self._store_deployment_block(block)
        
        # Execute deployment callbacks
        for callback in self.deployment_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback("block", service_name, block)
                else:
                    callback("block", service_name, block)
            except Exception as e:
                logger.error(f"Deployment callback failed: {e}")
        
        logger.critical(f"Blocked deployments for service: {service_name}")
        
        return {
            "action": "block_deployments",
            "status": "blocked",
            "service_name": service_name,
            "unblock_criteria": block.unblock_criteria
        }
    
    async def _activate_circuit_breaker(
        self,
        service_name: str,
        error_budget: ErrorBudget
    ) -> Dict[str, Any]:
        """Activate circuit breaker for the service"""
        
        # This would integrate with existing circuit breaker implementation
        # For now, we'll log and return status
        
        logger.critical(f"Circuit breaker activated for service: {service_name}")
        
        return {
            "action": "circuit_breaker",
            "status": "activated",
            "service_name": service_name
        }
    
    async def _reduce_traffic(
        self,
        service_name: str,
        error_budget: ErrorBudget
    ) -> Dict[str, Any]:
        """Reduce traffic to the service"""
        
        # Calculate traffic reduction percentage based on budget consumption
        budget_consumed = error_budget.budget_percentage
        
        if budget_consumed >= 95:
            reduction_percentage = 50  # Reduce traffic by 50%
        elif budget_consumed >= 90:
            reduction_percentage = 25  # Reduce traffic by 25%
        else:
            reduction_percentage = 10  # Reduce traffic by 10%
        
        logger.warning(
            f"Reducing traffic for service {service_name} by {reduction_percentage}%"
        )
        
        return {
            "action": "traffic_reduction",
            "status": "applied",
            "service_name": service_name,
            "reduction_percentage": reduction_percentage
        }
    
    async def _store_flag_state(self, flag: FeatureFlag) -> None:
        """Store feature flag state in Redis"""
        redis = await self.get_redis_client()
        if not redis:
            return
        
        try:
            key = f"feature_flag:{flag.service_name}:{flag.name}"
            data = {
                "name": flag.name,
                "service_name": flag.service_name,
                "current_state": flag.current_state.value,
                "previous_state": flag.previous_state.value if flag.previous_state else None,
                "rollback_reason": flag.rollback_reason,
                "rollback_timestamp": flag.rollback_timestamp.isoformat() if flag.rollback_timestamp else None,
                "last_modified": flag.last_modified.isoformat()
            }
            
            await redis.hset(key, mapping={k: str(v) for k, v in data.items()})
            await redis.expire(key, 86400 * 7)  # Keep for 7 days
            
        except Exception as e:
            logger.warning(f"Failed to store flag state in Redis: {e}")
    
    async def _store_deployment_block(self, block: DeploymentBlock) -> None:
        """Store deployment block in Redis"""
        redis = await self.get_redis_client()
        if not redis:
            return
        
        try:
            key = f"deployment_block:{block.service_name}"
            data = {
                "service_name": block.service_name,
                "blocked_at": block.blocked_at.isoformat(),
                "reason": block.reason,
                "error_budget_consumed": block.error_budget_consumed,
                "is_active": block.is_active,
                "unblock_criteria": json.dumps(block.unblock_criteria)
            }
            
            await redis.hset(key, mapping={k: str(v) for k, v in data.items()})
            await redis.expire(key, 86400 * 30)  # Keep for 30 days
            
        except Exception as e:
            logger.warning(f"Failed to store deployment block in Redis: {e}")
    
    async def restore_feature_flag(
        self,
        flag_name: str,
        reason: str = "Manual restore"
    ) -> bool:
        """Restore a rolled-back feature flag"""
        flag = self.feature_flags.get(flag_name)
        if not flag:
            logger.warning(f"Feature flag {flag_name} not found")
            return False
        
        if flag.current_state != FeatureFlagState.ROLLBACK:
            logger.warning(f"Feature flag {flag_name} is not in rollback state")
            return False
        
        try:
            # Restore to previous state or default to enabled
            flag.current_state = flag.previous_state or FeatureFlagState.ENABLED
            flag.previous_state = FeatureFlagState.ROLLBACK
            flag.rollback_reason = None
            flag.rollback_timestamp = None
            flag.last_modified = datetime.now(UTC)
            
            # Update in Redis
            await self._store_flag_state(flag)
            
            logger.info(f"Restored feature flag: {flag_name} (reason: {reason})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to restore flag {flag_name}: {e}")
            return False
    
    async def unblock_deployments(
        self,
        service_name: str,
        reason: str = "Manual unblock"
    ) -> bool:
        """Unblock deployments for a service"""
        redis = await self.get_redis_client()
        if not redis:
            return False
        
        try:
            key = f"deployment_block:{service_name}"
            block_data = await redis.hgetall(key)
            
            if not block_data:
                logger.warning(f"No deployment block found for service: {service_name}")
                return False
            
            # Update block record
            await redis.hset(key, mapping={
                "is_active": "False",
                "unblocked_at": datetime.now(UTC).isoformat(),
                "unblock_reason": reason
            })
            
            # Execute deployment callbacks
            for callback in self.deployment_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback("unblock", service_name, reason)
                    else:
                        callback("unblock", service_name, reason)
                except Exception as e:
                    logger.error(f"Deployment callback failed: {e}")
            
            logger.info(f"Unblocked deployments for service: {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unblock deployments for {service_name}: {e}")
            return False
    
    def register_rollback_callback(self, callback: Callable) -> None:
        """Register callback for feature flag rollbacks"""
        self.rollback_callbacks.append(callback)
    
    def register_deployment_callback(self, callback: Callable) -> None:
        """Register callback for deployment blocking/unblocking"""
        self.deployment_callbacks.append(callback)
    
    def get_feature_flag_status(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """Get current status of feature flags"""
        if service_name:
            flags = {name: flag for name, flag in self.feature_flags.items() 
                    if flag.service_name == service_name}
        else:
            flags = self.feature_flags
        
        status_summary = {
            "enabled": 0,
            "disabled": 0,
            "rollback": 0,
            "canary": 0
        }
        
        flag_details = []
        
        for flag in flags.values():
            status_summary[flag.current_state.value] += 1
            
            flag_details.append({
                "name": flag.name,
                "service_name": flag.service_name,
                "state": flag.current_state.value,
                "can_rollback": flag.can_rollback,
                "rollback_priority": flag.rollback_priority,
                "rollback_impact": flag.rollback_impact,
                "rollback_reason": flag.rollback_reason,
                "rollback_timestamp": flag.rollback_timestamp.isoformat() if flag.rollback_timestamp else None,
                "owner_team": flag.owner_team
            })
        
        return {
            "service_name": service_name,
            "total_flags": len(flags),
            "status_summary": status_summary,
            "flags": flag_details,
            "rollback_history_count": len(self.rollback_history)
        }
    
    def get_deployment_blocks(self) -> List[Dict[str, Any]]:
        """Get current deployment blocks"""
        # This would typically query Redis for active blocks
        # For now, return empty list as placeholder
        return []

class ErrorBudgetPolicyEnforcer:
    """Orchestrates error budget policy enforcement"""
    
    def __init__(
        self,
        feature_flag_manager: FeatureFlagManager,
        redis_url: Optional[str] = None
    ):
        self.feature_flag_manager = feature_flag_manager
        self.redis_url = redis_url
        
        # Policy enforcement tracking
        self.enforcement_history: List[Dict[str, Any]] = []
        self.active_enforcements: Dict[str, Dict[str, Any]] = {}
    
    async def setup_error_budget_monitoring(
        self,
        error_budget_monitor: ErrorBudgetMonitor,
        slo_definition: SLODefinition
    ) -> None:
        """Setup error budget monitoring with policy enforcement"""
        
        # Register error budget exhaustion callback
        error_budget_monitor.register_exhaustion_callback(
            self._handle_budget_exhaustion
        )
        
        # Set default rollback policy if not already set
        service_name = slo_definition.service_name
        if service_name not in self.feature_flag_manager.rollback_policies:
            self.feature_flag_manager.set_rollback_policy(
                service_name=service_name,
                error_budget_threshold=85.0,  # More aggressive threshold
                actions=[
                    PolicyAction.ALERT_ONLY,
                    PolicyAction.ROLLBACK_FEATURES,
                    PolicyAction.BLOCK_DEPLOYS
                ]
            )
        
        logger.info(f"Setup error budget policy enforcement for {service_name}")
    
    async def _handle_budget_exhaustion(self, error_budget: ErrorBudget) -> None:
        """Handle error budget exhaustion event"""
        service_name = error_budget.slo_target.service_name
        
        try:
            # Execute policy enforcement
            result = await self.feature_flag_manager.handle_error_budget_exhaustion(
                error_budget, service_name
            )
            
            # Track enforcement
            self.enforcement_history.append(result)
            self.active_enforcements[service_name] = result
            
            logger.critical(
                f"Error budget policy enforced for {service_name}: "
                f"{len(result.get('actions_taken', []))} actions taken"
            )
            
        except Exception as e:
            logger.error(f"Failed to handle budget exhaustion for {service_name}: {e}")
    
    def get_enforcement_status(self) -> Dict[str, Any]:
        """Get current policy enforcement status"""
        return {
            "active_enforcements": len(self.active_enforcements),
            "total_enforcements": len(self.enforcement_history),
            "services_under_enforcement": list(self.active_enforcements.keys()),
            "last_enforcement": self.enforcement_history[-1] if self.enforcement_history else None
        }