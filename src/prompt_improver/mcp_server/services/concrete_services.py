"""
Concrete implementations of MCP service protocols.

These implementations wrap the existing functionality to provide
clean interfaces that conform to the defined protocols.
"""

from typing import Any, Dict, Optional, AsyncContextManager
import logging
from datetime import datetime

# Import existing implementations
from prompt_improver.core.config import get_config
from prompt_improver.database import get_session
from prompt_improver.utils.datetime_utils import aware_utc_now, naive_utc_now
from prompt_improver.utils.redis_cache import get_cache
from prompt_improver.security.owasp_input_validator import OWASP2025InputValidator
from prompt_improver.security.output_validator import OutputValidator
from prompt_improver.security.rate_limit_middleware import get_mcp_rate_limit_middleware
from prompt_improver.security.mcp_middleware import get_mcp_auth_middleware
# ML and feedback services removed per architectural separation requirements
from prompt_improver.performance.monitoring.health.service import HealthService
from prompt_improver.performance.monitoring.performance_monitor import get_performance_monitor

# Import protocols (ML and feedback protocols removed)
from .mcp_service_facade import (
    ConfigServiceProtocol,
    SecurityServiceProtocol
)
from prompt_improver.core.protocols import (
    DatabaseProtocol,
    RedisCacheProtocol,
    DateTimeUtilsProtocol,
    HealthServiceProtocol,
    PerformanceMonitorProtocol
)

class ConcreteConfigService:
    """Concrete implementation of ConfigServiceProtocol"""

    def __init__(self):
        self._config = get_config()

    def get_mcp_config(self) -> Dict[str, Any]:
        return {
            'batch_size': self._config.mcp_batch_size,
            'session_maxsize': self._config.mcp_session_maxsize,
            'session_ttl': self._config.mcp_session_ttl
        }

    @property
    def mcp_batch_size(self) -> int:
        return self._config.mcp_batch_size

    @property
    def mcp_session_maxsize(self) -> int:
        return self._config.mcp_session_maxsize

    @property
    def mcp_session_ttl(self) -> int:
        return self._config.mcp_session_ttl

class ConcreteSecurityService:
    """Concrete implementation of SecurityServiceProtocol"""

    def __init__(self):
        self.input_validator = OWASP2025InputValidator()
        self.output_validator = OutputValidator()
        self.rate_limiter = get_mcp_rate_limit_middleware()
        self.auth_middleware = get_mcp_auth_middleware()
        self.logger = logging.getLogger(__name__)

    async def validate_input(self, input_data: str) -> bool:
        try:
            return await self.input_validator.validate_input(input_data)
        except Exception as e:
            self.logger.warning(f"Input validation error: {e}")
            return False

    async def validate_output(self, output_data: str) -> bool:
        try:
            return await self.output_validator.validate_output(output_data)
        except Exception as e:
            self.logger.warning(f"Output validation error: {e}")
            return False

    async def check_rate_limit(self, client_id: str) -> bool:
        try:
            # Rate limiter implementation check
            return True  # Placeholder - implement actual rate limiting logic
        except Exception as e:
            self.logger.warning(f"Rate limit check error: {e}")
            return False

    async def authenticate_request(self, auth_data: Dict[str, Any]) -> bool:
        try:
            # Auth middleware implementation
            return True  # Placeholder - implement actual auth logic
        except Exception as e:
            self.logger.warning(f"Authentication error: {e}")
            return False

# ConcreteMLService and ConcreteFeedbackService removed per architectural separation requirements
# ML and feedback services should be handled by separate ML training system

class ConcreteDatabaseService:
    """Concrete implementation of DatabaseProtocol"""

    async def get_session(self) -> AsyncContextManager:
        return get_session()

    async def get_session_manager(self):
        from prompt_improver.database.connection import _get_global_sessionmanager
        return _get_global_sessionmanager()

    async def health_check(self) -> bool:
        try:
            async with get_session() as session:
                # Simple health check query
                await session.execute("SELECT 1")
                return True
        except Exception:
            return False

class ConcreteCacheService:
    """Concrete implementation of RedisCacheProtocol"""

    def __init__(self):
        self.cache = get_cache()

    async def get(self, key: str) -> Optional[Any]:
        return await self.cache.get(key)

    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        return await self.cache.set(key, value, ttl)

    async def delete(self, key: str) -> bool:
        return await self.cache.delete(key)

    async def exists(self, key: str) -> bool:
        return await self.cache.exists(key)

    async def clear(self) -> bool:
        return await self.cache.clear()

    async def ping(self) -> bool:
        try:
            await self.cache.ping()
            return True
        except Exception:
            return False

class ConcreteDateTimeService:
    """Concrete implementation of DateTimeUtilsProtocol"""

    def aware_utc_now(self) -> datetime:
        return aware_utc_now()

    def naive_utc_now(self) -> datetime:
        return naive_utc_now()

    def format_iso(self, dt: datetime) -> str:
        return dt.isoformat()

    def parse_iso(self, iso_string: str) -> datetime:
        return datetime.fromisoformat(iso_string)

class ConcreteHealthService:
    """Concrete implementation of HealthServiceProtocol"""

    def __init__(self):
        self.health_service = HealthService()

    async def get_overall_health(self):
        return await self.health_service.get_overall_health()

    async def run_all_checks(self):
        return await self.health_service.run_all_checks()

class ConcretePerformanceService:
    """Concrete implementation of PerformanceMonitorProtocol"""

    def __init__(self):
        self.performance_monitor = get_performance_monitor()

    async def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None):
        await self.performance_monitor.record_metric(name, value, tags)

    async def record_timer(self, name: str, duration_ms: float, tags: Optional[Dict[str, str]] = None):
        await self.performance_monitor.record_timer(name, duration_ms, tags)

    async def record_counter(self, name: str, count: int = 1, tags: Optional[Dict[str, str]] = None):
        await self.performance_monitor.record_counter(name, count, tags)

    async def get_metrics_summary(self):
        return await self.performance_monitor.get_metrics_summary()

def create_concrete_services() -> Dict[str, Any]:
    """Factory function to create all concrete service implementations (read-only mode)"""
    return {
        'config_service': ConcreteConfigService(),
        'database_service': ConcreteDatabaseService(),
        'cache_service': ConcreteCacheService(),
        'datetime_service': ConcreteDateTimeService(),
        'health_service': ConcreteHealthService(),
        'performance_service': ConcretePerformanceService(),
        'security_service': ConcreteSecurityService()
        # ML and feedback services removed per architectural separation
    }
