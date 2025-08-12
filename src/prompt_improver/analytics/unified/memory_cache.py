"""Simple Memory Cache Implementation

This is a simple in-memory cache for testing and fallback scenarios
when Redis is not available.
"""

import asyncio
import time
from typing import Any, Dict, Optional


class MemoryCache:
    """Simple in-memory cache with TTL support"""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._access_times: Dict[str, float] = {}
    
    async def get(self, key: str) -> Optional[Any]:
        """Get cached value"""
        if key not in self._cache:
            return None
        
        entry = self._cache[key]
        
        # Check TTL
        if entry.get("expires_at") and time.time() > entry["expires_at"]:
            await self.delete(key)
            return None
        
        # Update access time
        self._access_times[key] = time.time()
        return entry["value"]
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set cached value with optional TTL"""
        try:
            # Cleanup if at max size
            if len(self._cache) >= self.max_size:
                await self._cleanup_lru()
            
            expires_at = time.time() + ttl if ttl else None
            
            self._cache[key] = {
                "value": value,
                "expires_at": expires_at,
                "created_at": time.time()
            }
            self._access_times[key] = time.time()
            
            return True
            
        except Exception:
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete cached value"""
        try:
            if key in self._cache:
                del self._cache[key]
            if key in self._access_times:
                del self._access_times[key]
            return True
        except Exception:
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear keys matching pattern (simple implementation)"""
        deleted_count = 0
        keys_to_delete = []
        
        for key in self._cache.keys():
            if pattern in key:  # Simple contains check
                keys_to_delete.append(key)
        
        for key in keys_to_delete:
            if await self.delete(key):
                deleted_count += 1
        
        return deleted_count
    
    async def _cleanup_lru(self) -> None:
        """Clean up least recently used entries"""
        # Remove 25% of entries (LRU)
        entries_to_remove = max(1, self.max_size // 4)
        
        # Sort by access time
        sorted_keys = sorted(self._access_times.keys(), key=self._access_times.get)
        
        for key in sorted_keys[:entries_to_remove]:
            await self.delete(key)