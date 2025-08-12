"""SecurityServiceFacade - Unified Entry Point for All Security Operations

The main facade that consolidates all security services into a single, clean interface.
This facade delegates to specialized security components while maintaining backward
compatibility with existing security service interfaces.

Key Features:
- Single entry point for all security operations
- Component-based architecture with clean separation of concerns
- Backward compatibility with existing factory functions
- Comprehensive health monitoring and metrics collection
- Fail-secure policies with comprehensive error handling
- Integration with OpenTelemetry for observability

Architecture:
- SecurityServiceFacade: Main entry point and coordinator
- Security Components: Specialized handlers for each security domain
- Protocol Compliance: All components implement defined protocols
- Legacy Bridges: Factory functions delegate to facade components

Security Standards:
- OWASP 2025 compliance across all operations  
- Zero-trust architecture with mandatory authentication
- Fail-secure by default (no fail-open vulnerabilities)
- Comprehensive audit logging and monitoring
- Real-time threat detection and incident response
"""

import asyncio
import logging
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from prompt_improver.database import (
    ManagerMode,
    SecurityContext,
    SecurityPerformanceMetrics,
    SecurityThreatScore,
    SecurityValidationResult,
    create_security_context,
    get_database_services
)
from prompt_improver.security.unified.protocols import (
    AuthenticationProtocol,
    AuthorizationProtocol,
    CryptographyProtocol,
    RateLimitingProtocol,
    SecurityComponent,
    SecurityComponentStatus,
    SecurityOperationResult,
    SecurityServiceFacadeProtocol,
    ValidationProtocol,
)
from prompt_improver.utils.datetime_utils import aware_utc_now

try:
    from opentelemetry import metrics, trace
    from opentelemetry.trace import Status, StatusCode
    
    OPENTELEMETRY_AVAILABLE = True
    facade_tracer = trace.get_tracer(__name__ + ".security_facade")
    facade_meter = metrics.get_meter(__name__ + ".security_facade")
    security_facade_operations_counter = facade_meter.create_counter(
        "security_facade_operations_total",
        description="Total security facade operations by component and result",
        unit="1"
    )
    security_facade_latency_histogram = facade_meter.create_histogram(
        "security_facade_operation_duration_seconds", 
        description="Security facade operation duration by component",
        unit="s"
    )
except ImportError:
    OPENTELEMETRY_AVAILABLE = False
    facade_tracer = None
    facade_meter = None
    security_facade_operations_counter = None
    security_facade_latency_histogram = None

logger = logging.getLogger(__name__)


class SecurityServiceFacade:
    """Unified security service facade providing single entry point for all security operations.
    
    This facade consolidates functionality from:
    - UnifiedSecurityManager: Security context and threat detection
    - UnifiedAuthenticationManager: Authentication and session management
    - UnifiedValidationManager: Input/output validation and sanitization
    - UnifiedCryptoManager: Cryptographic operations
    - UnifiedRateLimiter: Rate limiting and traffic control
    - AuthorizationService: Role-based access control
    - UnifiedSecurityStack: Middleware integration
    
    The facade delegates to specialized components while providing a unified interface
    and maintaining backward compatibility with existing security service APIs.
    """
    
    def __init__(self):
        """Initialize security service facade."""
        self._initialized = False
        self._components_initialized = False
        self._initialization_lock = asyncio.Lock()
        
        # Security components (will be initialized lazily)
        self._authentication_component: Optional[AuthenticationProtocol] = None
        self._authorization_component: Optional[AuthorizationProtocol] = None
        self._validation_component: Optional[ValidationProtocol] = None
        self._cryptography_component: Optional[CryptographyProtocol] = None
        self._rate_limiting_component: Optional[RateLimitingProtocol] = None
        
        # Performance metrics
        self._operation_counts: Dict[str, int] = {}
        self._component_health: Dict[str, SecurityComponentStatus] = {}
        self._last_health_check = aware_utc_now()
        self._metrics_cache: Optional[SecurityPerformanceMetrics] = None
        self._metrics_cache_expiry = aware_utc_now()
        
        # Configuration
        self._security_mode = ManagerMode.PRODUCTION
        self._health_check_interval = 300  # 5 minutes
        self._metrics_cache_ttl = 60  # 1 minute
        
        logger.info("SecurityServiceFacade initialized - consolidating all security services")
    
    @property
    async def authentication(self) -> AuthenticationProtocol:
        """Get authentication component with lazy initialization."""
        if not self._authentication_component:
            await self._ensure_components_initialized()
        return self._authentication_component
    
    @property  
    async def authorization(self) -> AuthorizationProtocol:
        """Get authorization component with lazy initialization."""
        if not self._authorization_component:
            await self._ensure_components_initialized()
        return self._authorization_component
    
    @property
    async def validation(self) -> ValidationProtocol:
        """Get validation component with lazy initialization.""" 
        if not self._validation_component:
            await self._ensure_components_initialized()
        return self._validation_component
    
    @property
    async def cryptography(self) -> CryptographyProtocol:
        """Get cryptography component with lazy initialization."""
        if not self._cryptography_component:
            await self._ensure_components_initialized()
        return self._cryptography_component
    
    @property
    async def rate_limiting(self) -> RateLimitingProtocol:
        """Get rate limiting component with lazy initialization."""
        if not self._rate_limiting_component:
            await self._ensure_components_initialized()
        return self._rate_limiting_component
    
    async def _ensure_components_initialized(self) -> None:
        """Ensure all security components are initialized."""
        if self._components_initialized:
            return
            
        async with self._initialization_lock:
            if self._components_initialized:
                return
                
            try:
                # Import and initialize components here to avoid circular imports
                from prompt_improver.security.unified.authentication_component import AuthenticationComponent
                from prompt_improver.security.unified.authorization_component import AuthorizationComponent  
                from prompt_improver.security.unified.validation_component import ValidationComponent
                from prompt_improver.security.unified.cryptography_component import CryptographyComponent
                from prompt_improver.security.unified.rate_limiting_component import RateLimitingComponent
                
                # Initialize all components
                self._authentication_component = AuthenticationComponent()
                self._authorization_component = AuthorizationComponent()
                self._validation_component = ValidationComponent()
                self._cryptography_component = CryptographyComponent()
                self._rate_limiting_component = RateLimitingComponent()
                
                # Initialize each component
                initialization_results = await asyncio.gather(
                    self._authentication_component.initialize(),
                    self._authorization_component.initialize(),
                    self._validation_component.initialize(), 
                    self._cryptography_component.initialize(),
                    self._rate_limiting_component.initialize(),
                    return_exceptions=True
                )
                
                # Check for initialization failures
                component_names = ["authentication", "authorization", "validation", "cryptography", "rate_limiting"]
                failed_components = []
                
                for i, result in enumerate(initialization_results):
                    if isinstance(result, Exception):
                        logger.error(f"Failed to initialize {component_names[i]} component: {result}")
                        failed_components.append(component_names[i])
                    elif not result:
                        logger.error(f"Failed to initialize {component_names[i]} component: returned False")
                        failed_components.append(component_names[i])
                
                if failed_components:
                    raise RuntimeError(f"Failed to initialize security components: {failed_components}")
                
                self._components_initialized = True
                logger.info("All security components initialized successfully")
                
            except Exception as e:
                logger.error(f"Failed to initialize security components: {e}")
                # Clean up partially initialized components
                self._authentication_component = None
                self._authorization_component = None
                self._validation_component = None
                self._cryptography_component = None
                self._rate_limiting_component = None
                raise
    
    async def initialize_all_components(self) -> bool:
        """Initialize all security components.
        
        Returns:
            True if all components initialized successfully
        """
        try:
            await self._ensure_components_initialized()
            self._initialized = True
            logger.info("SecurityServiceFacade fully initialized")
            return True
        except Exception as e:
            logger.error(f"Failed to initialize SecurityServiceFacade: {e}")
            return False
    
    async def health_check_all_components(self) -> Dict[str, Tuple[SecurityComponentStatus, Dict[str, Any]]]:
        """Check health of all security components.
        
        Returns:
            Dictionary mapping component names to their health status
        """
        if not self._components_initialized:
            return {"facade": (SecurityComponentStatus.FAILED, {"error": "Components not initialized"})}
        
        try:
            # Run health checks in parallel
            health_checks = await asyncio.gather(
                self._authentication_component.health_check(),
                self._authorization_component.health_check(),
                self._validation_component.health_check(),
                self._cryptography_component.health_check(),
                self._rate_limiting_component.health_check(),
                return_exceptions=True
            )
            
            component_names = ["authentication", "authorization", "validation", "cryptography", "rate_limiting"]
            health_status = {}
            
            for i, result in enumerate(health_checks):
                component_name = component_names[i]
                if isinstance(result, Exception):
                    health_status[component_name] = (
                        SecurityComponentStatus.FAILED,
                        {"error": str(result)}
                    )
                    logger.error(f"Health check failed for {component_name}: {result}")
                else:
                    health_status[component_name] = result
                    self._component_health[component_name] = result[0]
            
            self._last_health_check = aware_utc_now()
            
            # Add facade-level health status
            failed_components = [name for name, (status, _) in health_status.items() 
                               if status == SecurityComponentStatus.FAILED]
            
            if failed_components:
                health_status["facade"] = (
                    SecurityComponentStatus.DEGRADED,
                    {"failed_components": failed_components}
                )
            else:
                health_status["facade"] = (
                    SecurityComponentStatus.HEALTHY,
                    {"last_check": self._last_health_check.isoformat()}
                )
            
            return health_status
            
        except Exception as e:
            logger.error(f"Failed to perform health checks: {e}")
            return {"facade": (SecurityComponentStatus.FAILED, {"error": str(e)})}
    
    async def get_overall_metrics(self) -> SecurityPerformanceMetrics:
        """Get overall security performance metrics.
        
        Returns:
            Aggregated security performance metrics from all components
        """
        # Check cache first
        now = aware_utc_now()
        if (self._metrics_cache and 
            now < self._metrics_cache_expiry):
            return self._metrics_cache
        
        if not self._components_initialized:
            # Return empty metrics if not initialized
            return SecurityPerformanceMetrics(
                operation_count=0,
                average_latency_ms=0.0,
                error_rate=0.0,
                threat_detection_count=0,
                last_updated=now
            )
        
        try:
            # Collect metrics from all components in parallel
            metrics_collection = await asyncio.gather(
                self._authentication_component.get_metrics(),
                self._authorization_component.get_metrics(), 
                self._validation_component.get_metrics(),
                self._cryptography_component.get_metrics(),
                self._rate_limiting_component.get_metrics(),
                return_exceptions=True
            )
            
            # Aggregate metrics
            total_operations = sum(self._operation_counts.values())
            total_latency_ms = 0.0
            total_errors = 0
            total_threats = 0
            
            valid_metrics = []
            for metric_result in metrics_collection:
                if isinstance(metric_result, SecurityPerformanceMetrics):
                    valid_metrics.append(metric_result)
                    total_latency_ms += metric_result.average_latency_ms * metric_result.operation_count
                    total_operations += metric_result.operation_count
                    total_errors += int(metric_result.error_rate * metric_result.operation_count)
                    total_threats += metric_result.threat_detection_count
                elif isinstance(metric_result, Exception):
                    logger.error(f"Failed to get component metrics: {metric_result}")
            
            # Calculate aggregated metrics
            average_latency_ms = total_latency_ms / total_operations if total_operations > 0 else 0.0
            error_rate = total_errors / total_operations if total_operations > 0 else 0.0
            
            aggregated_metrics = SecurityPerformanceMetrics(
                operation_count=total_operations,
                average_latency_ms=average_latency_ms,
                error_rate=error_rate,
                threat_detection_count=total_threats,
                last_updated=now
            )
            
            # Cache the result
            self._metrics_cache = aggregated_metrics
            self._metrics_cache_expiry = now.replace(microsecond=0).replace(
                second=(now.second + self._metrics_cache_ttl) % 60
            )
            
            return aggregated_metrics
            
        except Exception as e:
            logger.error(f"Failed to get overall metrics: {e}")
            # Return basic metrics with error indication
            return SecurityPerformanceMetrics(
                operation_count=sum(self._operation_counts.values()),
                average_latency_ms=0.0,
                error_rate=1.0,  # Indicate error state
                threat_detection_count=0,
                last_updated=now
            )
    
    async def _record_operation(self, component: str, operation: str, duration_ms: float, success: bool = True):
        """Record operation metrics for performance monitoring."""
        operation_key = f"{component}.{operation}"
        self._operation_counts[operation_key] = self._operation_counts.get(operation_key, 0) + 1
        
        # Record OpenTelemetry metrics if available
        if OPENTELEMETRY_AVAILABLE and security_facade_operations_counter:
            security_facade_operations_counter.add(
                1,
                {"component": component, "operation": operation, "success": str(success).lower()}
            )
            
        if OPENTELEMETRY_AVAILABLE and security_facade_latency_histogram:
            security_facade_latency_histogram.record(
                duration_ms / 1000.0,  # Convert to seconds
                {"component": component, "operation": operation}
            )
    
    async def cleanup(self) -> bool:
        """Cleanup all security components and facade resources.
        
        Returns:
            True if cleanup successful, False otherwise
        """
        if not self._components_initialized:
            return True
        
        try:
            # Cleanup all components in parallel
            cleanup_results = await asyncio.gather(
                self._authentication_component.cleanup(),
                self._authorization_component.cleanup(),
                self._validation_component.cleanup(),
                self._cryptography_component.cleanup(),
                self._rate_limiting_component.cleanup(),
                return_exceptions=True
            )
            
            # Check for cleanup failures
            component_names = ["authentication", "authorization", "validation", "cryptography", "rate_limiting"]
            cleanup_failures = []
            
            for i, result in enumerate(cleanup_results):
                if isinstance(result, Exception):
                    logger.error(f"Failed to cleanup {component_names[i]} component: {result}")
                    cleanup_failures.append(component_names[i])
                elif not result:
                    logger.warning(f"Cleanup returned False for {component_names[i]} component")
                    cleanup_failures.append(component_names[i])
            
            # Clear component references
            self._authentication_component = None
            self._authorization_component = None
            self._validation_component = None
            self._cryptography_component = None
            self._rate_limiting_component = None
            
            # Clear caches
            self._operation_counts.clear()
            self._component_health.clear()
            self._metrics_cache = None
            
            # Update state
            self._components_initialized = False
            self._initialized = False
            
            if cleanup_failures:
                logger.warning(f"Some components failed cleanup: {cleanup_failures}")
                return False
            
            logger.info("SecurityServiceFacade cleanup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup SecurityServiceFacade: {e}")
            return False


# Global facade instance
_security_facade: Optional[SecurityServiceFacade] = None
_facade_lock = asyncio.Lock()


async def get_security_service_facade() -> SecurityServiceFacade:
    """Get the global SecurityServiceFacade instance.
    
    This function provides the main entry point for accessing the unified security services.
    It implements the singleton pattern to ensure a single facade instance across the application.
    
    Returns:
        SecurityServiceFacade instance
    """
    global _security_facade
    
    if _security_facade is not None:
        return _security_facade
    
    async with _facade_lock:
        if _security_facade is not None:
            return _security_facade
        
        try:
            _security_facade = SecurityServiceFacade()
            await _security_facade.initialize_all_components()
            logger.info("Global SecurityServiceFacade created and initialized")
            return _security_facade
            
        except Exception as e:
            logger.error(f"Failed to create SecurityServiceFacade: {e}")
            _security_facade = None
            raise


async def cleanup_security_service_facade() -> bool:
    """Cleanup the global SecurityServiceFacade instance.
    
    This should be called during application shutdown to properly cleanup security resources.
    
    Returns:
        True if cleanup successful, False otherwise
    """
    global _security_facade
    
    if _security_facade is None:
        return True
    
    async with _facade_lock:
        if _security_facade is None:
            return True
        
        try:
            cleanup_result = await _security_facade.cleanup()
            _security_facade = None
            logger.info("Global SecurityServiceFacade cleaned up")
            return cleanup_result
            
        except Exception as e:
            logger.error(f"Failed to cleanup SecurityServiceFacade: {e}")
            _security_facade = None
            return False