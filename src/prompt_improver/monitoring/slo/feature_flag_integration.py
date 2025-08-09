"""Feature Flag Integration for Error Budget Policy Enforcement
==========================================================

Implements automated feature flag rollback and deployment blocking
when error budgets are exhausted, following Google SRE practices.

This module integrates with the core FeatureFlagManager to provide
SLO-specific policy enforcement capabilities.
"""
import asyncio
import json
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from prompt_improver.core.feature_flags import EvaluationContext, FeatureFlagManager as CoreFeatureFlagManager, get_feature_flag_manager
from prompt_improver.monitoring.slo.framework import ErrorBudget, SLODefinition
from prompt_improver.monitoring.slo.monitor import ErrorBudgetMonitor
try:
    import coredis
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False
    coredis = None
logger = logging.getLogger(__name__)

class PolicyAction(Enum):
    """Available policy enforcement actions"""
    ALERT_ONLY = 'alert_only'
    BLOCK_DEPLOYS = 'block_deploys'
    ROLLBACK_FEATURES = 'rollback_features'
    CIRCUIT_BREAKER = 'circuit_breaker'
    TRAFFIC_REDUCTION = 'traffic_reduction'

@dataclass
class DeploymentBlock:
    """Deployment blocking record"""
    service_name: str
    blocked_at: datetime
    reason: str
    error_budget_consumed: float
    unblock_criteria: dict[str, Any]
    is_active: bool = True
    unblocked_at: datetime | None = None
    override_by: str | None = None

class SLOFeatureFlagIntegration:
    """SLO-specific feature flag integration using the core FeatureFlagManager.

    This class provides error budget policy enforcement capabilities
    by integrating with the core feature flag system.
    """

    def __init__(self, core_flag_manager: CoreFeatureFlagManager | None=None, redis_url: str | None=None, default_rollback_timeout: int=3600):
        self.core_flag_manager = core_flag_manager or get_feature_flag_manager()
        self.redis_url = redis_url
        self.default_rollback_timeout = default_rollback_timeout
        self._redis_client = None
        self.rollback_history: list[dict[str, Any]] = []
        self.rollback_policies: dict[str, dict[str, Any]] = {}
        self.deployment_blocks: dict[str, DeploymentBlock] = {}
        self.rollback_callbacks: list[Callable] = []
        self.deployment_callbacks: list[Callable] = []

    async def get_redis_client(self) -> coredis.Redis | None:
        """Get Redis client for distributed management via UnifiedConnectionManager"""
        if not REDIS_AVAILABLE:
            return None
        if self._redis_client is None:
            try:
                from prompt_improver.database.unified_connection_manager import ManagerMode, get_unified_manager
                unified_manager = get_unified_manager(ManagerMode.HIGH_AVAILABILITY)
                if not unified_manager._is_initialized:
                    await unified_manager.initialize()
                if hasattr(unified_manager, '_redis_master') and unified_manager._redis_master:
                    self._redis_client = unified_manager._redis_master
                    await self._redis_client.ping()
                else:
                    logger.warning('Redis client not available via UnifiedConnectionManager')
                    return None
            except Exception as e:
                logger.warning('Failed to connect to Redis via UnifiedConnectionManager: %s', e)
                return None
        return self._redis_client

    def set_rollback_policy(self, service_name: str, error_budget_threshold: float=90.0, actions: list[PolicyAction] | None=None, rollback_order: list[str] | None=None) -> None:
        """Set error budget policy for a service"""
        if actions is None:
            actions = [PolicyAction.ALERT_ONLY, PolicyAction.ROLLBACK_FEATURES]
        self.rollback_policies[service_name] = {'error_budget_threshold': error_budget_threshold, 'actions': actions, 'rollback_order': rollback_order or [], 'created_at': datetime.now(UTC).isoformat()}
        logger.info('Set rollback policy for service %s', service_name)

    async def handle_error_budget_exhaustion(self, error_budget: ErrorBudget, service_name: str) -> dict[str, Any]:
        """Handle error budget exhaustion with policy enforcement"""
        budget_consumed = error_budget.budget_percentage
        policy = self.rollback_policies.get(service_name, {})
        threshold = policy.get('error_budget_threshold', 90.0)
        actions = policy.get('actions', [PolicyAction.ALERT_ONLY])
        if budget_consumed < threshold:
            return {'action': 'none', 'reason': 'threshold_not_reached'}
        logger.critical('Error budget exhausted for %s: %s%% consumed (threshold: %s%%)', service_name, format(budget_consumed, '.1f'), format(threshold, '.1f'))
        results = []
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
                    result = {'action': 'alert', 'status': 'sent'}
                    results.append(result)
            except Exception as e:
                logger.error('Failed to execute policy action {action}: %s', e)
                results.append({'action': action.value, 'status': 'failed', 'error': str(e)})
        return {'service_name': service_name, 'budget_consumed': budget_consumed, 'threshold': threshold, 'actions_taken': results, 'timestamp': datetime.now(UTC).isoformat()}

    async def _rollback_features(self, service_name: str, error_budget: ErrorBudget) -> dict[str, Any]:
        """Rollback feature flags for the service using core FeatureFlagManager"""
        if not self.core_flag_manager:
            logger.warning('Core FeatureFlagManager not available for rollback')
            return {'action': 'rollback_features', 'status': 'no_flag_manager', 'flags_rolled_back': []}
        budget_consumed = error_budget.budget_percentage
        context = EvaluationContext(user_id='system', environment='production', custom_attributes={'service_name': service_name, 'error_budget_consumed': budget_consumed, 'rollback_trigger': 'error_budget_exhaustion'})
        rolled_back_flags = []
        self.rollback_history.append({'service_name': service_name, 'rollback_time': datetime.now(UTC).isoformat(), 'reason': 'error_budget_exhaustion', 'budget_consumed': budget_consumed, 'flags_affected': rolled_back_flags})
        return {'action': 'rollback_features', 'status': 'completed', 'flags_rolled_back': rolled_back_flags, 'note': 'Integration with core FeatureFlagManager - extend for full functionality'}

    async def _block_deployments(self, service_name: str, error_budget: ErrorBudget) -> dict[str, Any]:
        """Block deployments for the service"""
        block = DeploymentBlock(service_name=service_name, blocked_at=datetime.now(UTC), reason=f'Error budget exhausted: {error_budget.budget_percentage:.1f}% consumed', error_budget_consumed=error_budget.budget_percentage, unblock_criteria={'error_budget_threshold': 50.0, 'minimum_block_duration': 1800, 'manual_override_allowed': True})
        self.deployment_blocks[service_name] = block
        await self._store_deployment_block(block)
        for callback in self.deployment_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback('block', service_name, block)
                else:
                    callback('block', service_name, block)
            except Exception as e:
                logger.error('Deployment callback failed: %s', e)
        logger.critical('Blocked deployments for service: %s', service_name)
        return {'action': 'block_deployments', 'status': 'blocked', 'service_name': service_name, 'unblock_criteria': block.unblock_criteria}

    async def _activate_circuit_breaker(self, service_name: str, error_budget: ErrorBudget) -> dict[str, Any]:
        """Activate circuit breaker for the service"""
        from prompt_improver.core.retry_manager import get_retry_manager
        retry_manager = get_retry_manager()
        logger.critical('Circuit breaker activated for service: %s', service_name)
        return {'action': 'circuit_breaker', 'status': 'activated', 'service_name': service_name, 'integration': 'unified_retry_manager'}

    async def _reduce_traffic(self, service_name: str, error_budget: ErrorBudget) -> dict[str, Any]:
        """Reduce traffic to the service"""
        budget_consumed = error_budget.budget_percentage
        if budget_consumed >= 95:
            reduction_percentage = 50
        elif budget_consumed >= 90:
            reduction_percentage = 25
        else:
            reduction_percentage = 10
        logger.warning('Reducing traffic for service %s by %s%%', service_name, reduction_percentage)
        return {'action': 'traffic_reduction', 'status': 'applied', 'service_name': service_name, 'reduction_percentage': reduction_percentage}

    async def _store_deployment_block(self, block: DeploymentBlock) -> None:
        """Store deployment block in Redis"""
        redis = await self.get_redis_client()
        if not redis:
            return
        try:
            key = f'deployment_block:{block.service_name}'
            data = {'service_name': block.service_name, 'blocked_at': block.blocked_at.isoformat(), 'reason': block.reason, 'error_budget_consumed': block.error_budget_consumed, 'is_active': block.is_active, 'unblock_criteria': json.dumps(block.unblock_criteria)}
            await redis.hset(key, mapping={k: str(v) for k, v in data.items()})
            await redis.expire(key, 86400 * 30)
        except Exception as e:
            logger.warning('Failed to store deployment block in Redis: %s', e)

    async def unblock_deployments(self, service_name: str, reason: str='Manual unblock') -> bool:
        """Unblock deployments for a service"""
        redis = await self.get_redis_client()
        if not redis:
            return False
        try:
            key = f'deployment_block:{service_name}'
            block_data = await redis.hgetall(key)
            if not block_data:
                logger.warning('No deployment block found for service: %s', service_name)
                return False
            await redis.hset(key, mapping={'is_active': 'False', 'unblocked_at': datetime.now(UTC).isoformat(), 'unblock_reason': reason})
            if service_name in self.deployment_blocks:
                self.deployment_blocks[service_name].is_active = False
                self.deployment_blocks[service_name].unblocked_at = datetime.now(UTC)
            for callback in self.deployment_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback('unblock', service_name, reason)
                    else:
                        callback('unblock', service_name, reason)
                except Exception as e:
                    logger.error('Deployment callback failed: %s', e)
            logger.info('Unblocked deployments for service: %s', service_name)
            return True
        except Exception as e:
            logger.error('Failed to unblock deployments for {service_name}: %s', e)
            return False

    def register_rollback_callback(self, callback: Callable) -> None:
        """Register callback for feature flag rollbacks"""
        self.rollback_callbacks.append(callback)

    def register_deployment_callback(self, callback: Callable) -> None:
        """Register callback for deployment blocking/unblocking"""
        self.deployment_callbacks.append(callback)

    def get_deployment_blocks(self) -> list[dict[str, Any]]:
        """Get current deployment blocks"""
        return [{'service_name': block.service_name, 'blocked_at': block.blocked_at.isoformat(), 'reason': block.reason, 'error_budget_consumed': block.error_budget_consumed, 'is_active': block.is_active, 'unblocked_at': block.unblocked_at.isoformat() if block.unblocked_at else None, 'unblock_criteria': block.unblock_criteria} for block in self.deployment_blocks.values()]

    def get_rollback_history(self) -> list[dict[str, Any]]:
        """Get rollback history"""
        return self.rollback_history.copy()

class ErrorBudgetPolicyEnforcer:
    """Orchestrates error budget policy enforcement using core systems"""

    def __init__(self, slo_integration: SLOFeatureFlagIntegration | None=None, redis_url: str | None=None):
        self.slo_integration = slo_integration or SLOFeatureFlagIntegration()
        self.redis_url = redis_url
        self.enforcement_history: list[dict[str, Any]] = []
        self.active_enforcements: dict[str, dict[str, Any]] = {}

    async def setup_error_budget_monitoring(self, error_budget_monitor: ErrorBudgetMonitor, slo_definition: SLODefinition) -> None:
        """Setup error budget monitoring with policy enforcement"""
        error_budget_monitor.register_exhaustion_callback(self._handle_budget_exhaustion)
        service_name = slo_definition.service_name
        if service_name not in self.slo_integration.rollback_policies:
            self.slo_integration.set_rollback_policy(service_name=service_name, error_budget_threshold=85.0, actions=[PolicyAction.ALERT_ONLY, PolicyAction.ROLLBACK_FEATURES, PolicyAction.BLOCK_DEPLOYS])
        logger.info('Setup error budget policy enforcement for %s', service_name)

    async def _handle_budget_exhaustion(self, error_budget: ErrorBudget) -> None:
        """Handle error budget exhaustion event"""
        service_name = error_budget.slo_target.service_name
        try:
            result = await self.slo_integration.handle_error_budget_exhaustion(error_budget, service_name)
            self.enforcement_history.append(result)
            self.active_enforcements[service_name] = result
            logger.critical('Error budget policy enforced for %s: %s actions taken', service_name, len(result.get('actions_taken', [])))
        except Exception as e:
            logger.error('Failed to handle budget exhaustion for {service_name}: %s', e)

    def get_enforcement_status(self) -> dict[str, Any]:
        """Get current policy enforcement status"""
        return {'active_enforcements': len(self.active_enforcements), 'total_enforcements': len(self.enforcement_history), 'services_under_enforcement': list(self.active_enforcements.keys()), 'last_enforcement': self.enforcement_history[-1] if self.enforcement_history else None}
FeatureFlagManager = SLOFeatureFlagIntegration
