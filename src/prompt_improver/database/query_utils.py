"""Simplified query utilities using unified cache system.

Replaces the complex query_optimizer.py with simple functions that use
the unified cache architecture directly.
"""

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from prompt_improver.services.cache.cache_facade import CacheFacade

logger = logging.getLogger(__name__)

# Global query cache instance
_query_cache: Optional[CacheFacade] = None


def get_query_cache() -> CacheFacade:
    """Get the global query cache instance."""
    global _query_cache
    if _query_cache is None:
        _query_cache = CacheFacade(
            l1_max_size=500,      # Smaller for query caching
            l2_default_ttl=600,   # 10 minute default TTL
            enable_l2=True,       # Use Redis for query cache persistence
            enable_l3=False,      # No database persistence needed
            enable_warming=False  # No warming needed for query cache
        )
    return _query_cache


@asynccontextmanager
async def execute_optimized_query(
    session: AsyncSession,
    query: str,
    params: Optional[dict[str, Any]] = None,
    cache_ttl: int = 300,
    enable_cache: bool = True,
):
    """Execute a query with unified cache support.
    
    Args:
        session: Database session
        query: SQL query string
        params: Query parameters
        cache_ttl: Cache time-to-live in seconds
        enable_cache: Whether to use query result caching
        
    Yields:
        Dictionary with result, cache_hit status, and execution_time_ms
    """
    params = params or {}
    start_time = time.perf_counter()
    
    # Generate cache key if caching enabled
    cache_key = None
    if enable_cache:
        import hashlib
        import json
        
        cache_data = {
            "query": query.strip(),
            "params": sorted(params.items()) if params else [],
        }
        cache_string = json.dumps(cache_data, sort_keys=True)
        cache_key = f"query:{hashlib.md5(cache_string.encode(), usedforsecurity=False).hexdigest()}"
    
    # Try cache first
    if cache_key and enable_cache:
        try:
            cache = get_query_cache()
            cached_result = await cache.get(cache_key)
            if cached_result is not None:
                cache_time = (time.perf_counter() - start_time) * 1000
                logger.debug(f"Cache hit for query (saved time: {cache_time:.2f}ms)")
                yield {
                    "result": cached_result,
                    "cache_hit": True,
                    "execution_time_ms": cache_time,
                }
                return
        except Exception as e:
            logger.warning(f"Cache lookup failed: {e}, proceeding without cache")
    
    # Execute query
    try:
        result = await session.execute(text(query), params)
        rows = result.fetchall()
        execution_time = (time.perf_counter() - start_time) * 1000
        
        # Cache result if enabled and query was fast enough
        if cache_key and enable_cache and execution_time < 1000:  # Don't cache slow queries
            try:
                # Convert rows to serializable format
                serializable_rows = [
                    {k: str(v) if not isinstance(v, (str, int, float, bool, type(None))) else v 
                     for k, v in row._asdict().items() if hasattr(row, '_asdict')} 
                    if hasattr(row, '_asdict') else dict(row._mapping)
                    for row in rows
                ]
                
                cache = get_query_cache()
                await cache.set(cache_key, serializable_rows, l2_ttl=cache_ttl)
                logger.debug(f"Cached query result: {len(rows)} rows, TTL: {cache_ttl}s")
            except Exception as e:
                logger.warning(f"Failed to cache query result: {e}")
        
        if execution_time > 50:
            logger.warning(f"Slow query detected: {execution_time:.2f}ms - {query[:100]}...")
        
        yield {
            "result": rows,
            "cache_hit": False,
            "execution_time_ms": execution_time,
        }
        
    except Exception as e:
        logger.error(f"Query execution failed: {e}")
        raise


async def execute_simple_query(
    session: AsyncSession,
    query: str,
    params: Optional[dict[str, Any]] = None,
    enable_cache: bool = True,
    cache_ttl: int = 300,
):
    """Execute a simple query and return results directly.
    
    Convenience function that unwraps the context manager result.
    """
    async with execute_optimized_query(session, query, params, cache_ttl, enable_cache) as result:
        return result["result"]
