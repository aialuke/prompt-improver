"""ML Circuit Breaker Service.

Provides circuit breaker resilience patterns specifically for ML operations.
Extracted from intelligence_processor.py god object to follow single responsibility principle.

Performance Target: <1ms circuit breaker state checks
Memory Target: <10MB for state management
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass

from prompt_improver.ml.services.intelligence.protocols.intelligence_service_protocols import (
    MLCircuitBreakerServiceProtocol,
    CircuitBreakerState,
)
from prompt_improver.performance.monitoring.health.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerOpen,
    CircuitState,
)
from prompt_improver.performance.monitoring.metrics_registry import (
    StandardMetrics,
    get_metrics_registry,
)

logger = logging.getLogger(__name__)


@dataclass
class MLComponentConfig:
    """Configuration for ML component circuit breakers."""
    failure_threshold: int = 5
    recovery_timeout: float = 30.0
    slow_call_duration_threshold: float = 5.0
    slow_call_rate_threshold: float = 0.8
    minimum_number_of_calls: int = 10


class MLCircuitBreakerService:
    """Circuit breaker service for ML operations.
    
    Provides resilience patterns specifically designed for ML components with
    their unique failure patterns and recovery requirements.
    """
    
    def __init__(self):
        """Initialize ML circuit breaker service."""
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}
        self._component_configs: Dict[str, MLComponentConfig] = {}
        self._metrics_registry = get_metrics_registry()
        self._lock = asyncio.Lock()
        
        # Default configurations for common ML components
        self._default_configs = {
            "pattern_discovery": MLComponentConfig(
                failure_threshold=3,
                recovery_timeout=15.0,
                slow_call_duration_threshold=10.0
            ),
            "rule_optimization": MLComponentConfig(
                failure_threshold=5,
                recovery_timeout=30.0,
                slow_call_duration_threshold=15.0
            ),
            "prediction_generation": MLComponentConfig(
                failure_threshold=4,
                recovery_timeout=20.0,
                slow_call_duration_threshold=8.0
            ),
            "batch_processing": MLComponentConfig(
                failure_threshold=2,
                recovery_timeout=45.0,
                slow_call_duration_threshold=30.0
            )
        }
        
        logger.info("MLCircuitBreakerService initialized with default configurations")
    
    async def setup_circuit_breakers(self, components: List[str]) -> None:
        """Initialize circuit breakers for ML components.
        
        Args:
            components: List of ML component names to protect
        """
        async with self._lock:
            for component in components:
                if component not in self._circuit_breakers:
                    config = self._get_component_config(component)
                    circuit_breaker_config = CircuitBreakerConfig(
                        failure_threshold=config.failure_threshold,
                        recovery_timeout=timedelta(seconds=config.recovery_timeout),
                        slow_call_duration_threshold=timedelta(seconds=config.slow_call_duration_threshold),
                        slow_call_rate_threshold=config.slow_call_rate_threshold,
                        minimum_number_of_calls=config.minimum_number_of_calls
                    )
                    
                    self._circuit_breakers[component] = CircuitBreaker(
                        name=f"ml_{component}",
                        config=circuit_breaker_config
                    )
                    self._component_configs[component] = config
                    
                    logger.info(f"Circuit breaker initialized for ML component: {component}")
    
    async def call_with_breaker(self, component: str, operation: Callable[..., Any], *args, **kwargs) -> Any:
        """Execute operation with circuit breaker protection.
        
        Args:
            component: ML component name
            operation: Operation to execute
            *args, **kwargs: Operation arguments
            
        Returns:
            Operation result
            
        Raises:
            CircuitBreakerOpen: When circuit breaker is open
        """
        # Ensure circuit breaker exists
        if component not in self._circuit_breakers:
            await self.setup_circuit_breakers([component])
        
        circuit_breaker = self._circuit_breakers[component]
        
        try:
            # Record operation attempt
            self._metrics_registry.increment(
                StandardMetrics.CIRCUIT_BREAKER_CALLS_TOTAL,
                tags={"component": component, "type": "ml"}
            )
            
            # Execute with circuit breaker
            start_time = datetime.now(timezone.utc)
            result = await circuit_breaker.call(operation, *args, **kwargs)
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()
            
            # Record success metrics
            self._metrics_registry.increment(
                StandardMetrics.CIRCUIT_BREAKER_CALLS_TOTAL,
                tags={"component": component, "type": "ml", "result": "success"}
            )
            self._metrics_registry.record_value(
                "ml_operation_duration_seconds",
                execution_time,
                tags={"component": component}
            )
            
            await self.handle_state_change(component, is_success=True)
            return result
            
        except CircuitBreakerOpen as e:
            # Circuit breaker is open
            self._metrics_registry.increment(
                StandardMetrics.CIRCUIT_BREAKER_CALLS_TOTAL,
                tags={"component": component, "type": "ml", "result": "circuit_open"}
            )
            logger.warning(f"Circuit breaker open for ML component {component}: {e}")
            raise
            
        except Exception as e:
            # Operation failed
            self._metrics_registry.increment(
                StandardMetrics.CIRCUIT_BREAKER_CALLS_TOTAL,
                tags={"component": component, "type": "ml", "result": "failure"}
            )
            logger.error(f"ML operation failed for component {component}: {e}")
            await self.handle_state_change(component, is_success=False)
            raise
    
    async def get_circuit_state(self, component: str) -> CircuitBreakerState:
        """Get current circuit breaker state for component.
        
        Args:
            component: ML component name
            
        Returns:
            Current circuit breaker state
        """
        if component not in self._circuit_breakers:
            return CircuitBreakerState(
                is_open=False,
                failure_count=0,
                last_failure_time=None,
                recovery_timeout=0.0,
                component_name=component
            )
        
        circuit_breaker = self._circuit_breakers[component]
        config = self._component_configs[component]
        
        return CircuitBreakerState(
            is_open=circuit_breaker.state == CircuitState.OPEN,
            failure_count=circuit_breaker.failure_count,
            last_failure_time=circuit_breaker.last_failure_time,
            recovery_timeout=config.recovery_timeout,
            component_name=component
        )
    
    async def handle_state_change(self, component: str, is_success: bool) -> None:
        """Handle circuit breaker state transitions.
        
        Args:
            component: ML component name
            is_success: Whether the operation was successful
        """
        if component not in self._circuit_breakers:
            return
        
        circuit_breaker = self._circuit_breakers[component]
        previous_state = circuit_breaker.state
        
        # State transition happens automatically in circuit breaker
        current_state = circuit_breaker.state
        
        # Record state change if it occurred
        if previous_state != current_state:
            self._metrics_registry.increment(
                StandardMetrics.CIRCUIT_BREAKER_STATE_TRANSITIONS,
                tags={
                    "component": component,
                    "type": "ml",
                    "from_state": previous_state.value,
                    "to_state": current_state.value
                }
            )
            
            logger.info(
                f"ML Circuit breaker state transition for {component}: "
                f"{previous_state.value} -> {current_state.value}"
            )
        
        # Update state metrics
        self._metrics_registry.set_gauge(
            StandardMetrics.CIRCUIT_BREAKER_STATE,
            1 if current_state == CircuitState.OPEN else 0,
            tags={"component": component, "type": "ml"}
        )
    
    def _get_component_config(self, component: str) -> MLComponentConfig:
        """Get configuration for ML component.
        
        Args:
            component: ML component name
            
        Returns:
            Component configuration
        """
        if component in self._default_configs:
            return self._default_configs[component]
        
        # Return default config for unknown components
        return MLComponentConfig()
    
    async def get_all_states(self) -> Dict[str, CircuitBreakerState]:
        """Get states of all circuit breakers.
        
        Returns:
            Dictionary of component states
        """
        states = {}
        for component in self._circuit_breakers:
            states[component] = await self.get_circuit_state(component)
        return states
    
    async def reset_circuit_breaker(self, component: str) -> bool:
        """Reset circuit breaker for component.
        
        Args:
            component: ML component name
            
        Returns:
            True if reset successful, False otherwise
        """
        if component not in self._circuit_breakers:
            return False
        
        try:
            circuit_breaker = self._circuit_breakers[component]
            await circuit_breaker.reset()
            
            self._metrics_registry.increment(
                "ml_circuit_breaker_manual_resets_total",
                tags={"component": component}
            )
            
            logger.info(f"Circuit breaker manually reset for ML component: {component}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to reset circuit breaker for {component}: {e}")
            return False