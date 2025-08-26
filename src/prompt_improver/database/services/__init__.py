"""Database services module for modular connection management.

This package provides comprehensive database services extracted from the monolithic
unified_connection_manager.py with clear separation of concerns:

Connection Management (connection/):
    - PostgreSQLPoolManager: Advanced PostgreSQL connection pooling
    - Redis operations: Available through CacheFacade (services.cache)
    - SentinelManager: Redis Sentinel high availability
    - PoolScaler: Dynamic connection pool scaling
    - ConnectionMetrics: Performance monitoring

Cache Management (cache/):
    - MemoryCache: L1 in-memory LRU cache
    - RedisCache: L2 distributed Redis cache
    - DatabaseCache: Database fallback cache (eliminated in L1/L2 architecture)
    - CacheManager: Multi-level cache orchestration
    - CacheWarmer: Predictive cache warming

Each service is designed for:
- High performance and scalability
- Real behavior testing (no mocks for core functionality)
- Comprehensive observability and metrics
- Clean interfaces and protocols
- Zero backwards compatibility constraints
"""
