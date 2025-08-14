"""Retry Configuration Service - Centralized Configuration Management

This service eliminates redundant configuration classes and provides a unified interface
for retry configuration management with performance-optimized caching.

Key Features:
- Eliminates RetryConfigOverrides, StrategyOptions, RetryDecoratorOptions redundancy
- Domain-specific templates for database, ML, and API operations
- <1ms configuration lookups through intelligent caching
- Runtime configuration updates with validation
- Integration with existing retry patterns

Architecture:
- Protocol-based interface for dependency injection
- Template-based approach for common scenarios
- Composition over inheritance for configuration building
- Clean separation between configuration and execution
"""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Protocol, runtime_checkable

from prompt_improver.core.config.retry import RetryConfig, RetryStrategy
from prompt_improver.core.protocols.retry_protocols import RetryConfigProtocol

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConfigTemplate:
    """Lightweight template for retry configurations.
    
    Templates provide base configurations that can be customized
    for specific use cases while maintaining performance through caching.
    """
    
    name: str
    domain: str
    operation_type: str
    max_attempts: int = 3
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    base_delay: float = 1.0
    max_delay: float = 60.0
    jitter: bool = True
    jitter_factor: float = 0.1
    backoff_multiplier: float = 2.0
    retry_on_exceptions: list[type] = field(default_factory=lambda: [Exception])
    operation_timeout: float | None = None
    total_timeout: float | None = None
    enable_circuit_breaker: bool = False
    failure_threshold: int = 5
    recovery_timeout_seconds: float = 60.0
    description: str = ""
    
    def to_retry_config(self, **overrides: Any) -> RetryConfig:
        """Convert template to RetryConfig with optional overrides."""
        config_dict = {
            "max_attempts": self.max_attempts,
            "strategy": self.strategy,
            "base_delay": self.base_delay,
            "max_delay": self.max_delay,
            "jitter": self.jitter,
            "jitter_factor": self.jitter_factor,
            "backoff_multiplier": self.backoff_multiplier,
            "retry_on_exceptions": self.retry_on_exceptions.copy(),
            "operation_timeout": self.operation_timeout,
            "total_timeout": self.total_timeout,
            "enable_circuit_breaker": self.enable_circuit_breaker,
            "failure_threshold": self.failure_threshold,
            "recovery_timeout_seconds": self.recovery_timeout_seconds,
        }
        config_dict.update(overrides)
        return RetryConfig(**config_dict)


@runtime_checkable
class RetryConfigurationProtocol(Protocol):
    """Protocol for retry configuration service implementations."""
    
    def get_template(self, domain: str, operation: str) -> ConfigTemplate | None:
        """Get configuration template for domain and operation type."""
        ...
    
    def create_config(
        self, 
        template_name: str | None = None,
        domain: str = "general", 
        operation: str = "default",
        **overrides: Any
    ) -> RetryConfig:
        """Create retry configuration from template with overrides."""
        ...
    
    def get_cached_config(self, cache_key: str) -> RetryConfig | None:
        """Get cached configuration by key (<1ms lookup)."""
        ...
    
    def update_template(self, template: ConfigTemplate) -> None:
        """Update or add configuration template at runtime."""
        ...
    
    def list_templates(self, domain: str | None = None) -> list[ConfigTemplate]:
        """List available templates, optionally filtered by domain."""
        ...


class RetryConfigurationService:
    """Centralized retry configuration management service.
    
    Provides domain-specific templates, caching, and runtime updates
    for retry configurations across the system.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._templates: dict[str, ConfigTemplate] = {}
        self._config_cache: dict[str, tuple[RetryConfig, float]] = {}
        self._cache_ttl = 300.0  # 5 minutes TTL
        self._setup_default_templates()
        self.logger.info("RetryConfigurationService initialized with domain templates")
    
    def _setup_default_templates(self) -> None:
        """Initialize default domain-specific templates."""
        templates = [
            # Database Operation Templates
            ConfigTemplate(
                name="db_connection", domain="database", operation_type="connection",
                max_attempts=5, strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                base_delay=0.5, max_delay=30.0, jitter=True,
                retry_on_exceptions=[ConnectionError, TimeoutError, OSError],
                operation_timeout=10.0, total_timeout=120.0,
                enable_circuit_breaker=True, failure_threshold=3,
                description="Database connection with connection pool awareness"
            ),
            ConfigTemplate(
                name="db_transaction", domain="database", operation_type="transaction", 
                max_attempts=3, strategy=RetryStrategy.LINEAR_BACKOFF,
                base_delay=1.0, max_delay=10.0, jitter=False,
                retry_on_exceptions=[ConnectionError], operation_timeout=30.0,
                description="Database transaction with deadlock handling"
            ),
            ConfigTemplate(
                name="db_query", domain="database", operation_type="query",
                max_attempts=3, strategy=RetryStrategy.FIXED_DELAY,
                base_delay=2.0, max_delay=2.0, jitter=True,
                retry_on_exceptions=[TimeoutError], operation_timeout=60.0,
                description="Database query with timeout handling"
            ),
            
            # ML Operation Templates  
            ConfigTemplate(
                name="ml_training", domain="ml", operation_type="training",
                max_attempts=3, strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                base_delay=5.0, max_delay=300.0, jitter=True, backoff_multiplier=3.0,
                retry_on_exceptions=[RuntimeError, ConnectionError], 
                total_timeout=3600.0, enable_circuit_breaker=True, failure_threshold=2,
                description="ML model training with resource awareness"
            ),
            ConfigTemplate(
                name="ml_inference", domain="ml", operation_type="inference",
                max_attempts=5, strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                base_delay=0.1, max_delay=10.0, jitter=True,
                retry_on_exceptions=[RuntimeError, TimeoutError],
                operation_timeout=30.0, enable_circuit_breaker=True,
                description="ML inference with low-latency requirements"
            ),
            ConfigTemplate(
                name="ml_data_loading", domain="ml", operation_type="data_loading",
                max_attempts=4, strategy=RetryStrategy.LINEAR_BACKOFF,
                base_delay=2.0, max_delay=60.0, jitter=True,
                retry_on_exceptions=[IOError, ConnectionError], operation_timeout=120.0,
                description="ML data loading with I/O error handling"
            ),
            
            # API Operation Templates
            ConfigTemplate(
                name="api_external", domain="api", operation_type="external", 
                max_attempts=4, strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                base_delay=1.0, max_delay=60.0, jitter=True,
                retry_on_exceptions=[ConnectionError, TimeoutError],
                operation_timeout=30.0, enable_circuit_breaker=True,
                description="External API calls with rate limit awareness"
            ),
            ConfigTemplate(
                name="api_rate_limited", domain="api", operation_type="rate_limited",
                max_attempts=6, strategy=RetryStrategy.FIBONACCI_BACKOFF,
                base_delay=2.0, max_delay=120.0, jitter=True,
                retry_on_exceptions=[ConnectionError], operation_timeout=45.0,
                description="Rate-limited API with intelligent backoff"
            ),
            ConfigTemplate(
                name="api_auth", domain="api", operation_type="authentication",
                max_attempts=2, strategy=RetryStrategy.FIXED_DELAY,
                base_delay=5.0, max_delay=5.0, jitter=False,
                retry_on_exceptions=[ConnectionError], operation_timeout=15.0,
                description="API authentication with minimal retries"
            ),
            
            # Critical Operation Templates
            ConfigTemplate(
                name="critical_high_availability", domain="critical", operation_type="high_availability",
                max_attempts=10, strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                base_delay=0.5, max_delay=60.0, jitter=True, backoff_multiplier=1.5,
                retry_on_exceptions=[Exception], operation_timeout=30.0,
                enable_circuit_breaker=True, failure_threshold=3,
                description="High availability operations with extensive retries"
            ),
            
            # Background Task Templates
            ConfigTemplate(
                name="background_async", domain="background", operation_type="async_processing",
                max_attempts=5, strategy=RetryStrategy.LINEAR_BACKOFF,
                base_delay=10.0, max_delay=300.0, jitter=True,
                retry_on_exceptions=[Exception], total_timeout=1800.0,
                description="Background async processing with long delays"
            ),
        ]
        
        for template in templates:
            self._templates[f"{template.domain}_{template.operation_type}"] = template
            self.logger.debug(f"Registered template: {template.name} ({template.domain}.{template.operation_type})")
    
    @lru_cache(maxsize=100)
    def get_template(self, domain: str, operation: str) -> ConfigTemplate | None:
        """Get configuration template for domain and operation type."""
        key = f"{domain}_{operation}"
        template = self._templates.get(key)
        if template:
            self.logger.debug(f"Retrieved template: {key}")
        return template
    
    def create_config(
        self, 
        template_name: str | None = None,
        domain: str = "general", 
        operation: str = "default",
        **overrides: Any
    ) -> RetryConfig:
        """Create retry configuration from template with overrides."""
        start_time = time.perf_counter()
        
        # Get template or use default
        if template_name and "_" in template_name:
            template = self._templates.get(template_name)
        else:
            template = self.get_template(domain, operation)
            
        if not template:
            # Create default template if none found
            template = ConfigTemplate(
                name=f"{domain}_{operation}", domain=domain, operation_type=operation,
                description=f"Default template for {domain}.{operation}"
            )
        
        # Create configuration with overrides
        config = template.to_retry_config(**overrides)
        
        # Cache the result
        cache_key = f"{template.name}_{hash(frozenset(overrides.items()))}"
        self._config_cache[cache_key] = (config, time.time())
        
        duration_ms = (time.perf_counter() - start_time) * 1000
        self.logger.debug(f"Created config {cache_key} in {duration_ms:.3f}ms")
        
        return config
    
    def get_cached_config(self, cache_key: str) -> RetryConfig | None:
        """Get cached configuration by key (<1ms lookup)."""
        if cache_key not in self._config_cache:
            return None
            
        config, timestamp = self._config_cache[cache_key]
        
        # Check TTL
        if time.time() - timestamp > self._cache_ttl:
            del self._config_cache[cache_key]
            return None
            
        return config
    
    def update_template(self, template: ConfigTemplate) -> None:
        """Update or add configuration template at runtime."""
        key = f"{template.domain}_{template.operation_type}"
        self._templates[key] = template
        
        # Clear related cache entries
        self.get_template.cache_clear()
        keys_to_remove = [k for k in self._config_cache.keys() if template.name in k]
        for key in keys_to_remove:
            del self._config_cache[key]
            
        self.logger.info(f"Updated template: {template.name}")
    
    def list_templates(self, domain: str | None = None) -> list[ConfigTemplate]:
        """List available templates, optionally filtered by domain."""
        templates = list(self._templates.values())
        if domain:
            templates = [t for t in templates if t.domain == domain]
        return sorted(templates, key=lambda t: (t.domain, t.operation_type))
    
    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache statistics for monitoring."""
        cache_info = self.get_template.cache_info()
        return {
            "template_cache_hits": cache_info.hits,
            "template_cache_misses": cache_info.misses,
            "template_cache_size": cache_info.currsize,
            "config_cache_size": len(self._config_cache),
            "registered_templates": len(self._templates),
        }
    
    def clear_expired_cache(self) -> None:
        """Clear expired cache entries."""
        current_time = time.time()
        expired_keys = [
            key for key, (_, timestamp) in self._config_cache.items()
            if current_time - timestamp > self._cache_ttl
        ]
        for key in expired_keys:
            del self._config_cache[key]
        
        if expired_keys:
            self.logger.debug(f"Cleared {len(expired_keys)} expired cache entries")


# Global service instance
_global_service: RetryConfigurationService | None = None


def get_retry_configuration_service() -> RetryConfigurationService:
    """Get the global retry configuration service instance."""
    global _global_service
    if _global_service is None:
        _global_service = RetryConfigurationService()
    return _global_service