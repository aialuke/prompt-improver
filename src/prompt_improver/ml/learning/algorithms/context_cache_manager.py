"""Modern Cache Management for Context Learning.

Modernized cache implementation using cachetools with TTL support and observability.
Replaces manual LRU implementation with production-ready caching library.
"""

import time
from typing import Any, Dict, Optional
from cachetools import TTLCache
from opentelemetry import trace

tracer = trace.get_tracer(__name__)


class ContextCacheManager:
    """Modern cache manager with TTL support and observability."""
    
    def __init__(
        self, 
        linguistic_cache_size: int = 1000,
        domain_cache_size: int = 500,
        ttl_seconds: int = 3600,  # 1 hour default TTL
        linguistic_ttl_seconds: Optional[int] = None,
        domain_ttl_seconds: Optional[int] = None
    ):
        """Initialize modern cache manager with TTL support.
        
        Args:
            linguistic_cache_size: Max size of linguistic cache
            domain_cache_size: Max size of domain cache  
            ttl_seconds: Default TTL in seconds
            linguistic_ttl_seconds: TTL for linguistic cache (overrides default)
            domain_ttl_seconds: TTL for domain cache (overrides default)
        """
        self.linguistic_cache = TTLCache(
            maxsize=linguistic_cache_size,
            ttl=linguistic_ttl_seconds or ttl_seconds
        )
        self.domain_cache = TTLCache(
            maxsize=domain_cache_size, 
            ttl=domain_ttl_seconds or ttl_seconds
        )
        self.linguistic_cache_max_size = linguistic_cache_size
        self.domain_cache_max_size = domain_cache_size
        self._hit_counts = {"linguistic": 0, "domain": 0}
        self._miss_counts = {"linguistic": 0, "domain": 0}
    
    def update_linguistic_cache(self, key: str, value: Any):
        """Add/update linguistic cache with automatic TTL management."""
        with tracer.start_as_current_span("context_cache_update", attributes={
            "cache.type": "linguistic",
            "cache.key": key
        }):
            self.linguistic_cache[key] = value
    
    def update_domain_cache(self, key: str, value: Any):
        """Add/update domain cache with automatic TTL management."""
        with tracer.start_as_current_span("context_cache_update", attributes={
            "cache.type": "domain", 
            "cache.key": key
        }):
            self.domain_cache[key] = value
    
    def get_linguistic_cache(self, key: str) -> Any:
        """Get item from linguistic cache with hit/miss tracking."""
        with tracer.start_as_current_span("context_cache_get", attributes={
            "cache.type": "linguistic",
            "cache.key": key
        }) as span:
            try:
                value = self.linguistic_cache[key]
                self._hit_counts["linguistic"] += 1
                span.set_attribute("cache.hit", True)
                return value
            except KeyError:
                self._miss_counts["linguistic"] += 1
                span.set_attribute("cache.hit", False)
                return None
    
    def get_domain_cache(self, key: str) -> Any:
        """Get item from domain cache with hit/miss tracking."""
        with tracer.start_as_current_span("context_cache_get", attributes={
            "cache.type": "domain",
            "cache.key": key
        }) as span:
            try:
                value = self.domain_cache[key]
                self._hit_counts["domain"] += 1
                span.set_attribute("cache.hit", True)
                return value
            except KeyError:
                self._miss_counts["domain"] += 1
                span.set_attribute("cache.hit", False)
                return None
    
    def clear_all_caches(self) -> Dict[str, int]:
        """Clear all caches and return counts of cleared items."""
        with tracer.start_as_current_span("context_cache_clear"):
            linguistic_count = len(self.linguistic_cache)
            domain_count = len(self.domain_cache)
            
            self.linguistic_cache.clear()
            self.domain_cache.clear()
            
            # Reset hit/miss counters
            self._hit_counts = {"linguistic": 0, "domain": 0}
            self._miss_counts = {"linguistic": 0, "domain": 0}
            
            return {
                "linguistic_cleared": linguistic_count,
                "domain_cleared": domain_count
            }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics about cache utilization and performance."""
        linguistic_total = self._hit_counts["linguistic"] + self._miss_counts["linguistic"]
        domain_total = self._hit_counts["domain"] + self._miss_counts["domain"]
        
        return {
            "linguistic_cache_size": len(self.linguistic_cache),
            "linguistic_cache_max": self.linguistic_cache_max_size,
            "linguistic_cache_utilization": len(self.linguistic_cache) / self.linguistic_cache_max_size,
            "linguistic_hit_rate": self._hit_counts["linguistic"] / max(linguistic_total, 1),
            "linguistic_hits": self._hit_counts["linguistic"],
            "linguistic_misses": self._miss_counts["linguistic"],
            "domain_cache_size": len(self.domain_cache),
            "domain_cache_max": self.domain_cache_max_size,
            "domain_cache_utilization": len(self.domain_cache) / self.domain_cache_max_size,
            "domain_hit_rate": self._hit_counts["domain"] / max(domain_total, 1),
            "domain_hits": self._hit_counts["domain"],
            "domain_misses": self._miss_counts["domain"],
            "ttl_info": {
                "linguistic_ttl": self.linguistic_cache.ttl,
                "domain_ttl": self.domain_cache.ttl
            }
        }
    
    def expire_cache_entries(self) -> Dict[str, int]:
        """Manually trigger expiration of TTL entries.""" 
        with tracer.start_as_current_span("context_cache_expire"):
            # Access internal methods to trigger expiration
            linguistic_before = len(self.linguistic_cache)
            domain_before = len(self.domain_cache)
            
            # Force expiration by calling expire on caches
            self.linguistic_cache.expire()
            self.domain_cache.expire()
            
            return {
                "linguistic_expired": linguistic_before - len(self.linguistic_cache),
                "domain_expired": domain_before - len(self.domain_cache)
            }