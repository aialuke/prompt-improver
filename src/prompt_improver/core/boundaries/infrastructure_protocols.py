"""Infrastructure Layer Boundary Protocols.

These protocols define the contracts between domain/application layers and
infrastructure concerns like databases, external services, and caching.

Clean Architecture Rule: Infrastructure layer implements domain protocols.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from datetime import datetime

from prompt_improver.core.domain.types import (
    SessionId,
    UserId,
    ModelId,
    AnalysisId,
    ImprovementSessionData,
    PromptSessionData,
    TrainingSessionData,
    UserFeedbackData,
    HealthCheckResultData,
)
from prompt_improver.core.domain.enums import (
    CacheLevel,
    HealthStatus,
    SessionStatus,
)


@runtime_checkable
class RepositoryProtocol(Protocol):
    """Base protocol for repository implementations."""
    
    async def initialize(self) -> None:
        """Initialize repository connection and resources."""
        ...
    
    async def cleanup(self) -> None:
        """Clean up repository resources."""
        ...
    
    async def health_check(self) -> HealthCheckResultData:
        """Check repository health and connectivity.
        
        Returns:
            Health check result
        """
        ...
    
    async def begin_transaction(self) -> str:
        """Begin a new transaction.
        
        Returns:
            Transaction identifier
        """
        ...
    
    async def commit_transaction(
        self,
        transaction_id: str
    ) -> bool:
        """Commit a transaction.
        
        Args:
            transaction_id: Transaction to commit
            
        Returns:
            Whether commit was successful
        """
        ...
    
    async def rollback_transaction(
        self,
        transaction_id: str
    ) -> bool:
        """Rollback a transaction.
        
        Args:
            transaction_id: Transaction to rollback
            
        Returns:
            Whether rollback was successful
        """
        ...


@runtime_checkable
class ExternalServiceProtocol(Protocol):
    """Protocol for external service integrations."""
    
    async def call_service(
        self,
        service_endpoint: str,
        request_data: Dict[str, Any],
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Make a call to an external service.
        
        Args:
            service_endpoint: Endpoint to call
            request_data: Request payload
            timeout: Optional timeout in seconds
            
        Returns:
            Service response data
        """
        ...
    
    async def check_service_availability(
        self,
        service_name: str
    ) -> HealthCheckResultData:
        """Check if external service is available.
        
        Args:
            service_name: Name of service to check
            
        Returns:
            Health check result for the service
        """
        ...
    
    async def get_service_metrics(
        self,
        service_name: str,
        time_range: Optional[tuple[datetime, datetime]] = None,
    ) -> Dict[str, Any]:
        """Get metrics for an external service.
        
        Args:
            service_name: Service to get metrics for
            time_range: Optional time range filter
            
        Returns:
            Service metrics and performance data
        """
        ...
    
    def get_supported_services(self) -> List[str]:
        """Get list of supported external services.
        
        Returns:
            List of service names
        """
        ...


@runtime_checkable
class CacheServiceProtocol(Protocol):
    """Protocol for caching service implementations."""
    
    async def get(
        self,
        key: str,
        cache_level: CacheLevel = CacheLevel.L1,
    ) -> Optional[Any]:
        """Get value from cache.
        
        Args:
            key: Cache key
            cache_level: Which cache level to use
            
        Returns:
            Cached value or None if not found
        """
        ...
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl_seconds: Optional[int] = None,
        cache_level: CacheLevel = CacheLevel.L1,
    ) -> bool:
        """Set value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
            cache_level: Which cache level to use
            
        Returns:
            Whether value was cached successfully
        """
        ...
    
    async def delete(
        self,
        key: str,
        cache_level: Optional[CacheLevel] = None,
    ) -> bool:
        """Delete value from cache.
        
        Args:
            key: Cache key to delete
            cache_level: Cache level (None = all levels)
            
        Returns:
            Whether deletion was successful
        """
        ...
    
    async def clear(
        self,
        pattern: Optional[str] = None,
        cache_level: Optional[CacheLevel] = None,
    ) -> int:
        """Clear cache entries.
        
        Args:
            pattern: Optional pattern to match keys
            cache_level: Cache level (None = all levels)
            
        Returns:
            Number of entries cleared
        """
        ...
    
    async def get_cache_stats(
        self,
        cache_level: Optional[CacheLevel] = None,
    ) -> Dict[str, Any]:
        """Get cache statistics.
        
        Args:
            cache_level: Cache level (None = all levels)
            
        Returns:
            Cache statistics and metrics
        """
        ...


@runtime_checkable
class MonitoringServiceProtocol(Protocol):
    """Protocol for monitoring and observability services."""
    
    async def record_metric(
        self,
        metric_name: str,
        value: float,
        tags: Optional[Dict[str, str]] = None,
        timestamp: Optional[datetime] = None,
    ) -> None:
        """Record a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Metric value
            tags: Optional metric tags
            timestamp: Optional timestamp (default: now)
        """
        ...
    
    async def record_event(
        self,
        event_type: str,
        event_data: Dict[str, Any],
        severity: str = "info",
    ) -> None:
        """Record a system event.
        
        Args:
            event_type: Type of event
            event_data: Event details
            severity: Event severity level
        """
        ...
    
    async def start_trace(
        self,
        trace_name: str,
        parent_trace_id: Optional[str] = None,
    ) -> str:
        """Start a new trace span.
        
        Args:
            trace_name: Name of the trace
            parent_trace_id: Optional parent trace ID
            
        Returns:
            Trace identifier
        """
        ...
    
    async def end_trace(
        self,
        trace_id: str,
        status: str = "success",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """End a trace span.
        
        Args:
            trace_id: Trace to end
            status: Trace completion status
            metadata: Optional trace metadata
        """
        ...
    
    async def get_metrics(
        self,
        metric_names: List[str],
        time_range: tuple[datetime, datetime],
        aggregation: str = "avg",
    ) -> Dict[str, List[float]]:
        """Get metric values over time.
        
        Args:
            metric_names: Metrics to retrieve
            time_range: Time range to query
            aggregation: Aggregation method
            
        Returns:
            Metric values by name
        """
        ...


@runtime_checkable
class DatabaseConnectionProtocol(Protocol):
    """Protocol for database connection management."""
    
    async def get_connection(
        self,
        readonly: bool = False
    ) -> Any:  # Database connection object
        """Get a database connection.
        
        Args:
            readonly: Whether connection is for read-only operations
            
        Returns:
            Database connection object
        """
        ...
    
    async def release_connection(
        self,
        connection: Any
    ) -> None:
        """Release a database connection back to pool.
        
        Args:
            connection: Connection to release
        """
        ...
    
    async def execute_query(
        self,
        query: str,
        parameters: Optional[Dict[str, Any]] = None,
        readonly: bool = False,
    ) -> List[Dict[str, Any]]:
        """Execute a database query.
        
        Args:
            query: SQL query to execute
            parameters: Optional query parameters
            readonly: Whether query is read-only
            
        Returns:
            Query results
        """
        ...
    
    async def execute_transaction(
        self,
        operations: List[Dict[str, Any]]
    ) -> bool:
        """Execute multiple operations in a transaction.
        
        Args:
            operations: List of operations to execute
            
        Returns:
            Whether transaction was successful
        """
        ...
    
    async def get_connection_pool_status(self) -> Dict[str, Any]:
        """Get connection pool status and metrics.
        
        Returns:
            Connection pool status information
        """
        ...


@runtime_checkable
class ConfigurationServiceProtocol(Protocol):
    """Protocol for configuration management."""
    
    def get_config_value(
        self,
        key: str,
        default: Optional[Any] = None,
    ) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        ...
    
    def set_config_value(
        self,
        key: str,
        value: Any,
        persistent: bool = True,
    ) -> bool:
        """Set configuration value.
        
        Args:
            key: Configuration key
            value: Value to set
            persistent: Whether to persist the change
            
        Returns:
            Whether value was set successfully
        """
        ...
    
    def get_config_section(
        self,
        section: str
    ) -> Dict[str, Any]:
        """Get all configuration values in a section.
        
        Args:
            section: Configuration section name
            
        Returns:
            Configuration values in the section
        """
        ...
    
    def reload_configuration(self) -> bool:
        """Reload configuration from source.
        
        Returns:
            Whether reload was successful
        """
        ...
    
    def validate_configuration(self) -> Dict[str, Any]:
        """Validate current configuration.
        
        Returns:
            Validation results
        """
        ...


@runtime_checkable
class SecurityServiceProtocol(Protocol):
    """Protocol for security-related infrastructure services."""
    
    async def encrypt_data(
        self,
        data: str,
        key_id: Optional[str] = None,
    ) -> str:
        """Encrypt sensitive data.
        
        Args:
            data: Data to encrypt
            key_id: Optional specific key ID to use
            
        Returns:
            Encrypted data
        """
        ...
    
    async def decrypt_data(
        self,
        encrypted_data: str,
        key_id: Optional[str] = None,
    ) -> str:
        """Decrypt sensitive data.
        
        Args:
            encrypted_data: Data to decrypt
            key_id: Optional specific key ID to use
            
        Returns:
            Decrypted data
        """
        ...
    
    async def hash_password(
        self,
        password: str
    ) -> str:
        """Hash a password securely.
        
        Args:
            password: Password to hash
            
        Returns:
            Hashed password
        """
        ...
    
    async def verify_password(
        self,
        password: str,
        hashed_password: str,
    ) -> bool:
        """Verify a password against its hash.
        
        Args:
            password: Plain text password
            hashed_password: Hashed password to verify against
            
        Returns:
            Whether password is valid
        """
        ...
    
    async def generate_token(
        self,
        payload: Dict[str, Any],
        expires_in: Optional[int] = None,
    ) -> str:
        """Generate a secure token.
        
        Args:
            payload: Token payload
            expires_in: Optional expiration time in seconds
            
        Returns:
            Generated token
        """
        ...
    
    async def validate_token(
        self,
        token: str
    ) -> Optional[Dict[str, Any]]:
        """Validate and decode a token.
        
        Args:
            token: Token to validate
            
        Returns:
            Token payload if valid, None otherwise
        """
        ...