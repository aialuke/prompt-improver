"""Cache services package for high-performance direct caching architecture.

This package provides focused cache services implementing direct cache-aside pattern:
- L1CacheService: In-memory cache (<1ms response time)
- L2RedisService: Redis cache (1-10ms response time)
- CacheFacade: Direct L1+L2 operations (eliminates coordination overhead)
- CacheMonitoringService: Health monitoring and metrics

Designed to achieve <2ms response times through elimination of coordination anti-patterns.
Architecture: App → CacheFacade → Direct L1 → Direct L2 → Storage
"""

from prompt_improver.services.cache.cache_facade import CacheFacade
from prompt_improver.services.cache.cache_monitoring_service import (
    CacheMonitoringService,
)
from prompt_improver.services.cache.l1_cache_service import L1CacheService
from prompt_improver.services.cache.l2_redis_service import L2RedisService
from prompt_improver.shared.interfaces.protocols.cache import (
    CacheHealthProtocol as CacheMonitoringProtocol,
    CacheServiceProtocol,
)

__all__ = [
    "CacheFacade",
    "CacheMonitoringProtocol",
    "CacheMonitoringService",
    "CacheServiceProtocol",
    "L1CacheService",
    "L2RedisService",
]
