"""Error Injection Utility for Real Behavior Testing.

Provides controlled error injection capabilities for testing error handling services,
retry mechanisms, and circuit breakers with realistic failure scenarios.
"""

import asyncio
import logging
import random
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Union
import uuid

logger = logging.getLogger(__name__)


class InjectionTrigger(Enum):
    """When to trigger error injection."""
    
    IMMEDIATE = "immediate"          # Trigger immediately
    AFTER_DELAY = "after_delay"      # Trigger after specified delay
    ON_CALL_COUNT = "on_call_count"  # Trigger after N calls
    PROBABILITY = "probability"      # Trigger based on probability
    TIME_WINDOW = "time_window"      # Trigger within time window


class InjectionScope(Enum):
    """Scope of error injection."""
    
    SINGLE_CALL = "single_call"      # Affect only one call
    MULTIPLE_CALLS = "multiple_calls" # Affect multiple calls
    TIME_BASED = "time_based"        # Affect calls within time period
    PERMANENT = "permanent"          # Affect all future calls until cleared


@dataclass
class ErrorInjectionConfig:
    """Configuration for error injection scenario."""
    
    config_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    
    # Error details
    exception_type: type = RuntimeError
    error_message: str = "Injected error for testing"
    custom_exception: Optional[Exception] = None
    
    # Trigger configuration
    trigger: InjectionTrigger = InjectionTrigger.IMMEDIATE
    trigger_value: Optional[Union[int, float]] = None  # Delay, call count, probability
    
    # Scope configuration
    scope: InjectionScope = InjectionScope.SINGLE_CALL
    scope_duration_ms: Optional[int] = None
    max_injections: Optional[int] = None
    
    # State tracking
    call_count: int = 0
    injection_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_injection_at: Optional[float] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)


class ErrorInjector:
    """Error injection utility for controlled failure testing."""
    
    def __init__(self):
        """Initialize error injector."""
        self.injector_id = str(uuid.uuid4())[:8]
        self.configs: Dict[str, ErrorInjectionConfig] = {}
        self.global_enabled = True
        self.injection_history: List[Dict[str, Any]] = []
        
        logger.info(f"ErrorInjector initialized: {self.injector_id}")
    
    def add_injection_config(
        self,
        name: str,
        exception_type: type = RuntimeError,
        error_message: str = "Injected error for testing",
        trigger: InjectionTrigger = InjectionTrigger.IMMEDIATE,
        trigger_value: Optional[Union[int, float]] = None,
        scope: InjectionScope = InjectionScope.SINGLE_CALL,
        scope_duration_ms: Optional[int] = None,
        max_injections: Optional[int] = None,
        **metadata
    ) -> str:
        """Add error injection configuration.
        
        Args:
            name: Configuration name
            exception_type: Type of exception to raise
            error_message: Error message
            trigger: When to trigger injection
            trigger_value: Trigger threshold/value
            scope: Injection scope
            scope_duration_ms: Duration for time-based scope
            max_injections: Maximum number of injections
            **metadata: Additional metadata
            
        Returns:
            Configuration ID
        """
        config = ErrorInjectionConfig(
            name=name,
            description=f"Error injection: {exception_type.__name__}",
            exception_type=exception_type,
            error_message=error_message,
            trigger=trigger,
            trigger_value=trigger_value,
            scope=scope,
            scope_duration_ms=scope_duration_ms,
            max_injections=max_injections,
            metadata=metadata
        )
        
        self.configs[name] = config
        
        logger.info(f"Added error injection config: {name} ({config.config_id})")
        
        return config.config_id
    
    def remove_injection_config(self, name: str) -> bool:
        """Remove error injection configuration.
        
        Args:
            name: Configuration name to remove
            
        Returns:
            True if removed, False if not found
        """
        if name in self.configs:
            del self.configs[name]
            logger.info(f"Removed error injection config: {name}")
            return True
        return False
    
    def clear_all_configs(self):
        """Clear all error injection configurations."""
        config_count = len(self.configs)
        self.configs.clear()
        self.injection_history.clear()
        logger.info(f"Cleared {config_count} error injection configurations")
    
    def should_inject_error(self, config_name: str) -> tuple[bool, Optional[Exception]]:
        """Check if error should be injected for given configuration.
        
        Args:
            config_name: Configuration name to check
            
        Returns:
            Tuple of (should_inject, exception_to_raise)
        """
        if not self.global_enabled:
            return False, None
            
        config = self.configs.get(config_name)
        if not config:
            return False, None
        
        config.call_count += 1
        current_time = time.time()
        
        # Check if max injections reached
        if config.max_injections and config.injection_count >= config.max_injections:
            return False, None
        
        # Check scope constraints
        if config.scope == InjectionScope.TIME_BASED and config.scope_duration_ms:
            if config.last_injection_at:
                time_since_last = (current_time - config.last_injection_at) * 1000
                if time_since_last > config.scope_duration_ms:
                    # Time window expired, no more injections for this config
                    return False, None
        
        # Check trigger conditions
        should_inject = False
        
        if config.trigger == InjectionTrigger.IMMEDIATE:
            should_inject = config.injection_count == 0  # Only first call
            
        elif config.trigger == InjectionTrigger.AFTER_DELAY:
            if config.trigger_value and config.created_at:
                elapsed = (current_time - config.created_at) * 1000
                should_inject = elapsed >= config.trigger_value
            
        elif config.trigger == InjectionTrigger.ON_CALL_COUNT:
            if config.trigger_value:
                should_inject = config.call_count >= config.trigger_value
            
        elif config.trigger == InjectionTrigger.PROBABILITY:
            if config.trigger_value:
                should_inject = random.random() < config.trigger_value
            
        elif config.trigger == InjectionTrigger.TIME_WINDOW:
            if config.trigger_value and config.created_at:
                elapsed = (current_time - config.created_at) * 1000
                should_inject = elapsed <= config.trigger_value
        
        if not should_inject:
            return False, None
        
        # Check scope constraints for injection
        if config.scope == InjectionScope.SINGLE_CALL:
            if config.injection_count > 0:
                return False, None  # Already injected once
                
        # Create exception
        exception = None
        if config.custom_exception:
            exception = config.custom_exception
        else:
            exception = config.exception_type(config.error_message)
        
        # Update injection state
        config.injection_count += 1
        config.last_injection_at = current_time
        
        # Record injection history
        self.injection_history.append({
            "config_name": config_name,
            "config_id": config.config_id,
            "exception_type": type(exception).__name__,
            "error_message": str(exception),
            "injected_at": current_time,
            "call_count": config.call_count,
            "injection_count": config.injection_count,
        })
        
        logger.info(f"Error injected: {config_name} - {type(exception).__name__}: {exception}")
        
        return True, exception
    
    @asynccontextmanager
    async def inject_on_operation(self, config_name: str, operation: Callable, *args, **kwargs):
        """Context manager for injecting errors on specific operations.
        
        Args:
            config_name: Error injection configuration name
            operation: Operation to potentially inject error on
            *args, **kwargs: Arguments for the operation
            
        Yields:
            Operation result or raises injected exception
        """
        should_inject, exception = self.should_inject_error(config_name)
        
        if should_inject and exception:
            raise exception
        
        # Execute operation normally
        try:
            if asyncio.iscoroutinefunction(operation):
                result = await operation(*args, **kwargs)
            else:
                result = operation(*args, **kwargs)
            yield result
        except Exception as e:
            # Re-raise actual operation errors
            raise e
    
    def create_failing_operation(
        self,
        config_name: str,
        success_operation: Optional[Callable] = None,
        *success_args,
        **success_kwargs
    ) -> Callable:
        """Create an operation that may fail based on injection configuration.
        
        Args:
            config_name: Error injection configuration name
            success_operation: Operation to call on success
            *success_args, **success_kwargs: Arguments for success operation
            
        Returns:
            Callable that may raise injected errors
        """
        async def failing_operation(*args, **kwargs):
            should_inject, exception = self.should_inject_error(config_name)
            
            if should_inject and exception:
                raise exception
            
            # Execute success operation if provided
            if success_operation:
                if asyncio.iscoroutinefunction(success_operation):
                    return await success_operation(*success_args, **success_kwargs)
                else:
                    return success_operation(*success_args, **success_kwargs)
            
            return f"Success: {config_name}"
        
        return failing_operation
    
    def get_injection_statistics(self) -> Dict[str, Any]:
        """Get error injection statistics.
        
        Returns:
            Dictionary with injection statistics
        """
        stats = {
            "injector_id": self.injector_id,
            "global_enabled": self.global_enabled,
            "total_configs": len(self.configs),
            "total_injections": len(self.injection_history),
            "config_stats": {},
            "recent_injections": self.injection_history[-10:],  # Last 10
        }
        
        # Per-config statistics
        for name, config in self.configs.items():
            stats["config_stats"][name] = {
                "config_id": config.config_id,
                "call_count": config.call_count,
                "injection_count": config.injection_count,
                "exception_type": config.exception_type.__name__,
                "trigger": config.trigger.value,
                "scope": config.scope.value,
                "max_injections": config.max_injections,
                "last_injection_at": config.last_injection_at,
            }
        
        return stats
    
    def enable_injection(self):
        """Enable global error injection."""
        self.global_enabled = True
        logger.info("Error injection enabled globally")
    
    def disable_injection(self):
        """Disable global error injection."""
        self.global_enabled = False
        logger.info("Error injection disabled globally")
    
    async def create_database_error_scenarios(self) -> List[str]:
        """Create common database error injection scenarios.
        
        Returns:
            List of configuration names created
        """
        scenarios = []
        
        # Connection error
        scenarios.append(
            self.add_injection_config(
                name="database_connection_error",
                exception_type=ConnectionError,
                error_message="Connection to database failed: Connection refused",
                trigger=InjectionTrigger.PROBABILITY,
                trigger_value=0.3,  # 30% chance
                scope=InjectionScope.MULTIPLE_CALLS,
                max_injections=5,
            )
        )
        
        # Query timeout
        scenarios.append(
            self.add_injection_config(
                name="database_query_timeout",
                exception_type=TimeoutError,
                error_message="Query timeout: execution exceeded 30 seconds",
                trigger=InjectionTrigger.ON_CALL_COUNT,
                trigger_value=3,  # After 3 calls
                scope=InjectionScope.SINGLE_CALL,
            )
        )
        
        # Transaction error
        scenarios.append(
            self.add_injection_config(
                name="database_transaction_error",
                exception_type=RuntimeError,
                error_message="Transaction deadlock detected",
                trigger=InjectionTrigger.PROBABILITY,
                trigger_value=0.2,  # 20% chance
                scope=InjectionScope.TIME_BASED,
                scope_duration_ms=5000,  # 5 seconds
            )
        )
        
        logger.info(f"Created {len(scenarios)} database error scenarios")
        return scenarios
    
    async def create_network_error_scenarios(self) -> List[str]:
        """Create common network error injection scenarios.
        
        Returns:
            List of configuration names created
        """
        scenarios = []
        
        # Connection timeout
        scenarios.append(
            self.add_injection_config(
                name="network_connection_timeout",
                exception_type=asyncio.TimeoutError,
                error_message="HTTP request timeout after 30 seconds",
                trigger=InjectionTrigger.PROBABILITY,
                trigger_value=0.4,  # 40% chance
                scope=InjectionScope.MULTIPLE_CALLS,
                max_injections=3,
            )
        )
        
        # Service unavailable
        scenarios.append(
            self.add_injection_config(
                name="network_service_unavailable",
                exception_type=RuntimeError,
                error_message="HTTP 503: Service Unavailable",
                trigger=InjectionTrigger.AFTER_DELAY,
                trigger_value=1000,  # After 1 second
                scope=InjectionScope.TIME_BASED,
                scope_duration_ms=3000,  # 3 seconds
            )
        )
        
        # DNS resolution failure
        scenarios.append(
            self.add_injection_config(
                name="network_dns_failure",
                exception_type=OSError,
                error_message="DNS resolution failed: Name resolution failure",
                trigger=InjectionTrigger.ON_CALL_COUNT,
                trigger_value=2,
                scope=InjectionScope.SINGLE_CALL,
            )
        )
        
        logger.info(f"Created {len(scenarios)} network error scenarios")
        return scenarios
    
    async def create_validation_error_scenarios(self) -> List[str]:
        """Create common validation error injection scenarios.
        
        Returns:
            List of configuration names created
        """
        scenarios = []
        
        # Invalid input
        scenarios.append(
            self.add_injection_config(
                name="validation_invalid_input",
                exception_type=ValueError,
                error_message="Invalid input format: missing required field 'email'",
                trigger=InjectionTrigger.PROBABILITY,
                trigger_value=0.25,  # 25% chance
                scope=InjectionScope.MULTIPLE_CALLS,
                max_injections=10,
            )
        )
        
        # Security threat
        scenarios.append(
            self.add_injection_config(
                name="validation_security_threat",
                exception_type=SecurityError,
                error_message="Potential SQL injection detected in input",
                trigger=InjectionTrigger.ON_CALL_COUNT,
                trigger_value=5,  # After 5 calls
                scope=InjectionScope.PERMANENT,  # Security threats should be persistent
            )
        )
        
        # Business rule violation
        scenarios.append(
            self.add_injection_config(
                name="validation_business_rule_violation",
                exception_type=ValueError,
                error_message="Business rule violation: insufficient account balance",
                trigger=InjectionTrigger.TIME_WINDOW,
                trigger_value=2000,  # Within 2 seconds
                scope=InjectionScope.MULTIPLE_CALLS,
                max_injections=3,
            )
        )
        
        logger.info(f"Created {len(scenarios)} validation error scenarios")
        return scenarios
    
    async def cleanup(self):
        """Clean up error injector resources."""
        self.clear_all_configs()
        logger.info(f"ErrorInjector cleanup completed: {self.injector_id}")


# Define SecurityError if not available
class SecurityError(Exception):
    """Security-related error for injection testing."""
    pass