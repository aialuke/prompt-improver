"""Distributed locking services for database connections.

This package provides distributed locking functionality extracted from
unified_connection_manager.py, implementing:

- DistributedLockManager: Redis-based distributed locking with token security
- LockConfig: Configurable timeouts, retry policies, and monitoring settings
- LockInfo: Active lock tracking with expiration monitoring
- Context manager support for automatic lock cleanup
- OpenTelemetry metrics integration for production monitoring

Designed for production distributed systems with sub-5ms lock operations.
"""

from prompt_improver.database.services.locking.lock_manager import (
    DistributedLockManager,
    LockConfig,
    LockInfo,
    create_lock_manager,
    create_lock_manager_legacy,
)

__all__ = [
    "DistributedLockManager",
    "LockConfig",
    "LockInfo",
    "create_lock_manager",
    "create_lock_manager_legacy",
]
