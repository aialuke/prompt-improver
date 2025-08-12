# Multi-Level Cache Architecture

This package implements a sophisticated multi-level caching system extracted from the monolithic `unified_connection_manager.py`.

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   L1 Memory     │    │   L2 Redis      │    │ L3 Database     │
│   Cache         │◄──►│   Cache         │◄──►│   Fallback      │
│                 │    │                 │    │                 │
│ • LRU eviction  │    │ • Distributed   │    │ • Query cache   │
│ • Sub-ms access │    │ • Persistent    │    │ • Long-term     │
│ • Hot data      │    │ • Multi-tenant  │    │ • Compressed    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │  Cache Manager  │
                    │                 │
                    │ • Orchestration │
                    │ • Warming       │
                    │ • Monitoring    │
                    │ • Security      │
                    └─────────────────┘
```

## Components

### L1 Memory Cache (`memory_cache.py`)
- **Purpose**: Ultra-fast in-memory caching for frequently accessed data
- **Features**: LRU eviction, TTL support, access pattern tracking
- **Performance**: Sub-millisecond response times
- **Capacity**: Configurable size limits with automatic eviction

### L2 Redis Cache (`redis_cache.py`) 
- **Purpose**: Distributed caching across multiple instances
- **Features**: Persistence, cluster/sentinel support, security validation
- **Performance**: Single-digit millisecond response times
- **Scalability**: Horizontal scaling with Redis cluster

### L3 Database Cache (`database_cache.py`)
- **Purpose**: Ultimate fallback with query result caching
- **Features**: Transaction safety, compression, long-term storage
- **Performance**: Database query performance with caching optimization
- **Reliability**: Always available as final fallback

### Cache Manager (`cache_manager.py`)
- **Purpose**: Multi-level cache orchestration and coordination
- **Features**: Intelligent level selection, automatic fallback, monitoring
- **Integration**: OpenTelemetry metrics, security context validation
- **Interface**: Unified API for all caching operations

### Cache Warmer (`cache_warmer.py`)
- **Purpose**: Predictive cache population based on access patterns
- **Features**: Background warming, priority algorithms, resource awareness
- **Intelligence**: Machine learning-based prediction of cache needs
- **Efficiency**: Minimizes cache misses through proactive warming

## Design Principles

### 1. **No Backwards Compatibility**
- Clean slate implementation without legacy constraints
- Modern async/await patterns throughout
- Type hints and validation on all interfaces

### 2. **Real Behavior Testing**
- No mocks for core functionality testing
- Actual Redis and PostgreSQL instances required
- Performance benchmarks under real load

### 3. **Observability First**
- Comprehensive metrics at every level
- OpenTelemetry integration for distributed tracing  
- Performance monitoring and alerting

### 4. **Security Integration**
- Security context validation across all levels
- Multi-tenant isolation and access control
- Audit logging for cache operations

### 5. **Performance Optimization**
- Sub-millisecond L1 cache access
- Intelligent cache level selection
- Resource-aware warming strategies
- Memory-bounded collections

## Implementation Phases

- ✅ **Phase 3.1**: Directory structure creation
- 🔄 **Phase 3.2**: L1 Memory cache implementation
- 🔄 **Phase 3.3**: L2 Redis cache implementation  
- 🔄 **Phase 3.4**: L3 Database fallback implementation
- 🔄 **Phase 3.5**: Cache Manager orchestration implementation

## Performance Targets

| Level | Response Time | Throughput | Availability |
|-------|--------------|------------|--------------|
| L1    | < 1ms        | 100K+ ops/sec | 99.99% |
| L2    | < 5ms        | 50K+ ops/sec  | 99.9%  |
| L3    | < 50ms       | 1K+ ops/sec   | 99.95% |

## Usage Example

```python
from prompt_improver.database.services.cache import CacheManager

# Initialize multi-level cache
cache_manager = CacheManager(
    l1_size=1000,
    redis_config=redis_config,
    db_config=db_config
)

# Cache operations with automatic level management
await cache_manager.set("user:123", user_data, ttl=300)
user_data = await cache_manager.get("user:123")

# Cache warming based on patterns
await cache_manager.warm_cache(patterns=["user:*", "session:*"])
```

This architecture provides high performance, reliability, and scalability while maintaining clean separation of concerns and comprehensive observability.