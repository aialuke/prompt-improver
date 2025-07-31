#!/usr/bin/env python3
"""
Validation Script for UnifiedConnectionManager

This script validates that the unified connection manager works correctly
and provides backward compatibility through adapter interfaces.

The feature flag migration is complete - UnifiedConnectionManager is now the default.

Usage:
    python scripts/validate_unified_manager_migration.py
"""

import os
import sys
import argparse
import asyncio
import importlib
import time
from pathlib import Path
from typing import List, Dict, Any
from contextlib import contextmanager, asynccontextmanager

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Track validation results
validation_results = []

class ValidationResult:
    def __init__(self, test_name: str, success: bool, error: str = None, duration_ms: float = 0):
        self.test_name = test_name
        self.success = success
        self.error = error
        self.duration_ms = duration_ms
    
    def __str__(self):
        status = "✅ PASS" if self.success else "❌ FAIL"
        duration = f"({self.duration_ms:.1f}ms)" if self.duration_ms > 0 else ""
        error_msg = f" - {self.error}" if self.error else ""
        return f"{status} {self.test_name} {duration}{error_msg}"

def log_result(test_name: str, success: bool, error: str = None, duration_ms: float = 0):
    """Log a validation result"""
    result = ValidationResult(test_name, success, error, duration_ms)
    validation_results.append(result)
    print(result)

@contextmanager
def timer():
    """Context manager to time operations"""
    start = time.time()
    yield
    duration = (time.time() - start) * 1000
    return duration

def test_import_compatibility():
    """Test that imports work correctly with unified manager"""
    test_name = "Import Compatibility (unified manager)"
    
    try:
        start_time = time.time()
        
        # Force reload of database module to pick up environment change
        if 'prompt_improver.database' in sys.modules:
            importlib.reload(sys.modules['prompt_improver.database'])
        
        # Test basic imports
        from prompt_improver.database import DatabaseManager, DatabaseSessionManager
        from prompt_improver.database.connection import get_session, get_session_context
        
        # Test that classes are importable
        assert DatabaseManager is not None
        assert DatabaseSessionManager is not None
        assert get_session is not None
        assert get_session_context is not None
        
        duration = (time.time() - start_time) * 1000
        log_result(test_name, True, duration_ms=duration)
        
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        log_result(test_name, False, str(e), duration_ms=duration)

def test_database_manager_sync_pattern():
    """Test the synchronous DatabaseManager pattern used in 16 locations"""
    test_name = "DatabaseManager Sync Pattern"
    
    try:
        start_time = time.time()
        
        from prompt_improver.database import DatabaseManager
        from prompt_improver.database.config import DatabaseConfig
        
        # Create database URL (using test database if available)
        db_config = DatabaseConfig()
        database_url = f"postgresql+asyncpg://{db_config.postgres_username}:{db_config.postgres_password}@{db_config.postgres_host}:{db_config.postgres_port}/{db_config.postgres_database}"
        
        # Test DatabaseManager instantiation pattern (used in AprioriAnalyzer)
        db_manager = DatabaseManager(database_url, echo=False)
        
        # Test session factory access
        session_factory = db_manager.session_factory
        assert session_factory is not None
        
        # Test context manager pattern
        with db_manager.get_session() as session:
            # This is the pattern used across the codebase
            assert session is not None
        
        # Test connection pattern
        with db_manager.get_connection() as connection:
            assert connection is not None
        
        # Cleanup
        db_manager.close()
        
        duration = (time.time() - start_time) * 1000
        log_result(test_name, True, duration_ms=duration)
        
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        log_result(test_name, False, str(e), duration_ms=duration)

async def test_database_session_manager_async_pattern():
    """Test the async DatabaseSessionManager pattern"""
    test_name = "DatabaseSessionManager Async Pattern"
    
    try:
        start_time = time.time()
        
        from prompt_improver.database import get_session_context
        
        # Test async session context pattern (used in ML integration)
        async with get_session_context() as session:
            assert session is not None
            # Test basic query capability
            from sqlalchemy import text
            result = await session.execute(text("SELECT 1"))
            assert result.scalar() == 1
        
        duration = (time.time() - start_time) * 1000
        log_result(test_name, True, duration_ms=duration)
        
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        log_result(test_name, False, str(e), duration_ms=duration)

def test_apriori_analyzer_pattern():
    """Test the specific pattern used in AprioriAnalyzer"""
    test_name = "AprioriAnalyzer Usage Pattern"
    
    try:
        start_time = time.time()
        
        # Simulate the exact pattern from apriori_analyzer.py
        from prompt_improver.database.connection import DatabaseManager, get_database_url
        
        # This is how AprioriAnalyzer creates its database manager
        database_url = get_database_url(async_driver=False)
        db_manager = DatabaseManager(database_url, echo=False)
        
        # Test session access pattern
        with db_manager.get_session() as session:
            # This pattern is used extensively in AprioriAnalyzer
            from sqlalchemy import text
            result = session.execute(text("SELECT COUNT(*) FROM information_schema.tables"))
            count = result.scalar()
            assert count > 0  # Should have some system tables
        
        db_manager.close()
        
        duration = (time.time() - start_time) * 1000
        log_result(test_name, True, duration_ms=duration)
        
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        log_result(test_name, False, str(e), duration_ms=duration)

async def test_ml_integration_pattern():
    """Test the pattern used in ML integration"""
    test_name = "ML Integration Usage Pattern"
    
    try:
        start_time = time.time()
        
        # Simulate pattern from ml_integration.py
        from prompt_improver.database.connection import DatabaseSessionManager, get_database_url
        
        database_url = get_database_url(async_driver=True)
        session_manager = DatabaseSessionManager(database_url, echo=False)
        
        # Test async session pattern
        async with session_manager.session() as session:
            from sqlalchemy import text
            result = await session.execute(text("SELECT 1"))
            assert result.scalar() == 1
        
        await session_manager.close()
        
        duration = (time.time() - start_time) * 1000
        log_result(test_name, True, duration_ms=duration)
        
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        log_result(test_name, False, str(e), duration_ms=duration)

def test_api_endpoint_pattern():
    """Test the pattern used in API endpoints"""
    test_name = "API Endpoint Usage Pattern"
    
    try:
        start_time = time.time()
        
        # Simulate pattern from apriori_endpoints.py
        from prompt_improver.database.connection import DatabaseManager, get_database_url
        
        def create_db_manager():
            """Factory function like used in endpoints"""
            database_url = get_database_url(async_driver=False)
            return DatabaseManager(database_url, echo=False)
        
        db_manager = create_db_manager()
        
        # Test the usage pattern
        with db_manager.get_session() as session:
            # Pattern used for database queries in endpoints
            from sqlalchemy import text
            result = session.execute(text("SELECT version()"))
            version = result.scalar()
            assert version is not None
        
        db_manager.close()
        
        duration = (time.time() - start_time) * 1000
        log_result(test_name, True, duration_ms=duration)
        
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        log_result(test_name, False, str(e), duration_ms=duration)

async def test_performance_characteristics():
    """Test that performance characteristics are maintained"""
    test_name = "Performance Characteristics"
    
    try:
        from prompt_improver.database import get_session_context
        
        # Test connection acquisition time
        start_time = time.time()
        async with get_session_context() as session:
            from sqlalchemy import text
            await session.execute(text("SELECT 1"))
        acquisition_time = (time.time() - start_time) * 1000
        
        if acquisition_time > 1000:  # Should be under 1 second
            log_result(test_name, False, f"Connection acquisition too slow: {acquisition_time:.1f}ms")
            return
        
        # Test concurrent connections
        async def concurrent_task():
            async with get_session_context() as session:
                from sqlalchemy import text
                await session.execute(text("SELECT 1"))
        
        start_time = time.time()
        await asyncio.gather(*[concurrent_task() for _ in range(5)])
        concurrent_time = (time.time() - start_time) * 1000
        
        log_result(test_name, True, f"Acquisition: {acquisition_time:.1f}ms, Concurrent: {concurrent_time:.1f}ms")
        
    except Exception as e:
        log_result(test_name, False, str(e))

def test_registry_management():
    """Test registry management functionality"""
    test_name = "Registry Management"
    
    try:
        start_time = time.time()
        
        from prompt_improver.database.registry import get_registry_manager, clear_registry
        
        # Test registry manager access
        registry_manager = get_registry_manager()
        assert registry_manager is not None
        
        # Test registry operations
        registered_classes = registry_manager.get_registered_classes()
        assert isinstance(registered_classes, dict)
        
        # Test clear registry (important for test isolation)
        clear_registry()
        
        duration = (time.time() - start_time) * 1000
        log_result(test_name, True, duration_ms=duration)
        
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        log_result(test_name, False, str(e), duration_ms=duration)

async def test_unified_manager_specific_features():
    """Test features specific to the unified manager"""
    test_name = "Unified Manager Specific Features"
    
    # Unified manager is now the default - no feature flag needed
    
    try:
        start_time = time.time()
        
        from prompt_improver.database.unified_connection_manager import (
            get_unified_manager, ManagerMode
        )
        
        # Test different manager modes
        mcp_manager = get_unified_manager(ManagerMode.MCP_SERVER)
        ml_manager = get_unified_manager(ManagerMode.ML_TRAINING)
        ha_manager = get_unified_manager(ManagerMode.HIGH_AVAILABILITY)
        
        assert mcp_manager is not None
        assert ml_manager is not None
        assert ha_manager is not None
        
        # Test health check
        health = await mcp_manager.health_check()
        assert isinstance(health, dict)
        assert "status" in health
        
        # Test connection info
        info = await mcp_manager.get_connection_info()
        assert isinstance(info, dict)
        assert "mode" in info
        
        duration = (time.time() - start_time) * 1000
        log_result(test_name, True, duration_ms=duration)
        
    except Exception as e:
        duration = (time.time() - start_time) * 1000
        log_result(test_name, False, str(e), duration_ms=duration)

async def run_all_tests():
    """Run all validation tests"""
    print(f"\n🔍 Starting UnifiedConnectionManager Validation")
    print(f"{'='*60}")
    
    print("Testing unified manager (migration complete)...")
    test_import_compatibility()
    
    # Core compatibility tests
    test_database_manager_sync_pattern()
    await test_database_session_manager_async_pattern()
    
    # Specific usage pattern tests
    test_apriori_analyzer_pattern()
    await test_ml_integration_pattern()
    test_api_endpoint_pattern()
    
    # Performance and feature tests
    await test_performance_characteristics()
    test_registry_management()
    await test_unified_manager_specific_features()
    
    # Report results
    print(f"\n📊 Validation Results")
    print(f"{'='*60}")
    
    passed = sum(1 for r in validation_results if r.success)
    total = len(validation_results)
    
    for result in validation_results:
        print(result)
    
    print(f"\n{'✅ ALL TESTS PASSED' if passed == total else '❌ SOME TESTS FAILED'}")
    print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed != total:
        print("\n🚨 Migration validation failed. Do not proceed with deployment.")
        return False
    else:
        print("\n🎉 Migration validation successful. Safe to proceed with deployment.")
        return True

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Validate UnifiedConnectionManager")
    
    try:
        success = asyncio.run(run_all_tests())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Validation failed with unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()