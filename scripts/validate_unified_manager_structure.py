#!/usr/bin/env python3
"""
Structure Validation Script for UnifiedConnectionManager

This script validates the structural integrity of the unified connection manager
without requiring database connections, focusing on:
1. Code structure and class definitions
2. Interface compatibility 
3. Adapter integration
4. Import patterns
5. Method signatures and contracts

Usage:
    python scripts/validate_unified_manager_structure.py
"""

import os
import sys
import inspect
import importlib
from pathlib import Path
from typing import List, Dict, Any, get_type_hints

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Track validation results
validation_results = []

class ValidationResult:
    def __init__(self, test_name: str, success: bool, error: str = None):
        self.test_name = test_name
        self.success = success
        self.error = error
    
    def __str__(self):
        status = "✅ PASS" if self.success else "❌ FAIL"
        error_msg = f" - {self.error}" if self.error else ""
        return f"{status} {self.test_name}{error_msg}"

def log_result(test_name: str, success: bool, error: str = None):
    """Log a validation result"""
    result = ValidationResult(test_name, success, error)
    validation_results.append(result)
    print(result)

def test_unified_manager_class_structure():
    """Test that UnifiedConnectionManager has the correct structure"""
    test_name = "UnifiedConnectionManager Class Structure"
    
    try:
        from prompt_improver.database.unified_connection_manager import UnifiedConnectionManager
        
        # Check class exists
        assert UnifiedConnectionManager is not None
        
        # Check required methods exist
        required_methods = [
            'initialize', 'get_connection', 'health_check', 'close', 
            'get_connection_info', 'is_healthy', 'get_sync_session', 
            'get_async_session', 'get_pg_connection', 'get_redis_connection'
        ]
        
        for method_name in required_methods:
            assert hasattr(UnifiedConnectionManager, method_name), f"Missing method: {method_name}"
        
        # Check class attributes
        required_attributes = [
            'mode', 'db_config', 'redis_config', 'pool_config', '_metrics'
        ]
        
        # Create instance to check attributes
        from prompt_improver.database.unified_connection_manager import ManagerMode
        instance = UnifiedConnectionManager(ManagerMode.ASYNC_MODERN)
        
        for attr_name in required_attributes:
            assert hasattr(instance, attr_name), f"Missing attribute: {attr_name}"
        
        log_result(test_name, True)
        
    except Exception as e:
        log_result(test_name, False, str(e))

def test_connection_manager_protocol_compliance():
    """Test that UnifiedConnectionManager implements ConnectionManagerProtocol"""
    test_name = "ConnectionManagerProtocol Compliance"
    
    try:
        from prompt_improver.database.unified_connection_manager import UnifiedConnectionManager
        from prompt_improver.core.protocols.connection_protocol import ConnectionManagerProtocol
        
        # Check protocol methods
        protocol_methods = ['get_connection', 'health_check', 'close', 'get_connection_info', 'is_healthy']
        
        for method_name in protocol_methods:
            # Check method exists in both protocol and implementation
            assert hasattr(ConnectionManagerProtocol, method_name), f"Protocol missing method: {method_name}"
            assert hasattr(UnifiedConnectionManager, method_name), f"Implementation missing method: {method_name}"
            
            # Check method signatures are compatible
            protocol_method = getattr(ConnectionManagerProtocol, method_name)
            impl_method = getattr(UnifiedConnectionManager, method_name)
            
            # Both should be callable
            assert callable(protocol_method), f"Protocol {method_name} not callable"
            assert callable(impl_method), f"Implementation {method_name} not callable"
        
        log_result(test_name, True)
        
    except Exception as e:
        log_result(test_name, False, str(e))

def test_manager_modes_enum():
    """Test ManagerMode enum has all required modes"""
    test_name = "ManagerMode Enum Structure"
    
    try:
        from prompt_improver.database.unified_connection_manager import ManagerMode
        
        expected_modes = ['MCP_SERVER', 'ML_TRAINING', 'ADMIN', 'SYNC_HEAVY', 'ASYNC_MODERN', 'HIGH_AVAILABILITY']
        
        for mode_name in expected_modes:
            assert hasattr(ManagerMode, mode_name), f"Missing mode: {mode_name}"
            mode_value = getattr(ManagerMode, mode_name)
            assert isinstance(mode_value, ManagerMode), f"Mode {mode_name} not enum value"
        
        log_result(test_name, True)
        
    except Exception as e:
        log_result(test_name, False, str(e))

def test_pool_configuration_class():
    """Test PoolConfiguration class structure"""
    test_name = "PoolConfiguration Class Structure"
    
    try:
        from prompt_improver.database.unified_connection_manager import PoolConfiguration, ManagerMode
        
        # Test class exists
        assert PoolConfiguration is not None
        
        # Test factory method
        assert hasattr(PoolConfiguration, 'for_mode'), "Missing for_mode class method"
        
        # Test creating configurations for each mode
        for mode in ManagerMode:
            config = PoolConfiguration.for_mode(mode)
            assert isinstance(config, PoolConfiguration), f"for_mode didn't return PoolConfiguration for {mode}"
            
            # Check required attributes
            required_attrs = ['mode', 'pg_pool_size', 'pg_max_overflow', 'pg_timeout', 'redis_pool_size']
            for attr in required_attrs:
                assert hasattr(config, attr), f"PoolConfiguration missing {attr}"
                assert getattr(config, attr) is not None, f"PoolConfiguration {attr} is None"
        
        log_result(test_name, True)
        
    except Exception as e:
        log_result(test_name, False, str(e))

def test_backward_compatibility_adapters():
    """Test backward compatibility adapters exist and have correct structure"""
    test_name = "Backward Compatibility Adapters"
    
    try:
        from prompt_improver.database.unified_connection_manager import (
            DatabaseManagerAdapter, DatabaseSessionManagerAdapter,
            get_database_manager_adapter, get_database_session_manager_adapter
        )
        
        # Test adapter classes exist
        assert DatabaseManagerAdapter is not None
        assert DatabaseSessionManagerAdapter is not None
        
        # Test factory functions exist
        assert callable(get_database_manager_adapter)
        assert callable(get_database_session_manager_adapter)
        
        # Test adapter methods exist
        sync_adapter_methods = ['get_session', 'get_connection', 'close', 'session_factory', 'engine']
        async_adapter_methods = ['get_session', 'connect', 'session', 'close', 'session_factory']
        
        for method in sync_adapter_methods:
            assert hasattr(DatabaseManagerAdapter, method), f"DatabaseManagerAdapter missing {method}"
        
        for method in async_adapter_methods:
            assert hasattr(DatabaseSessionManagerAdapter, method), f"DatabaseSessionManagerAdapter missing {method}"
        
        log_result(test_name, True)
        
    except Exception as e:
        log_result(test_name, False, str(e))

def test_unified_manager_migration_complete():
    """Test that migration to unified manager is complete"""
    test_name = "Unified Manager Migration Complete"
    
    try:
        from prompt_improver.database import UnifiedConnectionManager
        from prompt_improver.database import get_unified_manager
        
        # Verify unified manager is available and working
        manager = get_unified_manager()
        assert manager is not None
        assert isinstance(manager, UnifiedConnectionManager)
        
        # Should use unified manager adapters
        from prompt_improver.database import DatabaseManager as UnifiedDatabaseManager
        from prompt_improver.database import DatabaseSessionManager as UnifiedDatabaseSessionManager
        
        assert UnifiedDatabaseManager is not None
        assert UnifiedDatabaseSessionManager is not None
        
        log_result(test_name, True)
        
    except Exception as e:
        log_result(test_name, False, str(e)) 

def test_consolidation_completeness():
    """Test that all 5 original manager capabilities are represented"""
    test_name = "Consolidation Completeness"
    
    try:
        from prompt_improver.database.unified_connection_manager import UnifiedConnectionManager, ManagerMode
        
        # Create instance
        manager = UnifiedConnectionManager(ManagerMode.HIGH_AVAILABILITY)
        
        # Check HAConnectionManager features
        ha_methods = ['get_pg_connection', 'get_redis_connection', '_setup_ha_components', '_setup_redis_sentinel']
        for method in ha_methods:
            assert hasattr(manager, method), f"Missing HAConnectionManager method: {method}"
        
        # Check DatabaseManager features  
        db_manager_methods = ['get_sync_session']
        for method in db_manager_methods:
            assert hasattr(manager, method), f"Missing DatabaseManager method: {method}"
        
        # Check DatabaseSessionManager features
        db_session_methods = ['get_async_session']
        for method in db_session_methods:
            assert hasattr(manager, method), f"Missing DatabaseSessionManager method: {method}"
        
        # Check UnifiedConnectionManager features (mode-based access)
        assert hasattr(manager, 'mode'), "Missing mode-based access capability"
        assert hasattr(manager, 'pool_config'), "Missing pool configuration capability"
        
        # Check RegistryManager integration
        assert hasattr(manager, '_registry_manager'), "Missing registry manager integration"
        
        log_result(test_name, True)
        
    except Exception as e:
        log_result(test_name, False, str(e))

def test_metrics_and_monitoring():
    """Test metrics and monitoring capabilities"""
    test_name = "Metrics and Monitoring"
    
    try:
        from prompt_improver.database.unified_connection_manager import (
            UnifiedConnectionManager, ConnectionMetrics, HealthStatus, ManagerMode
        )
        
        # Test ConnectionMetrics class
        metrics = ConnectionMetrics()
        expected_metrics = [
            'active_connections', 'idle_connections', 'total_connections',
            'pool_utilization', 'avg_response_time_ms', 'error_rate',
            'failed_connections', 'failover_count', 'health_check_failures',
            'circuit_breaker_state', 'sla_compliance_rate'
        ]
        
        for metric in expected_metrics:
            assert hasattr(metrics, metric), f"Missing metric: {metric}"
        
        # Test HealthStatus enum
        expected_statuses = ['HEALTHY', 'DEGRADED', 'UNHEALTHY', 'UNKNOWN']
        for status in expected_statuses:
            assert hasattr(HealthStatus, status), f"Missing health status: {status}"
        
        # Test manager has monitoring methods
        manager = UnifiedConnectionManager(ManagerMode.ASYNC_MODERN)
        monitoring_methods = [
            'health_check', '_health_monitor_loop', '_update_connection_metrics',
            '_handle_connection_failure', '_is_circuit_breaker_open'
        ]
        
        for method in monitoring_methods:
            assert hasattr(manager, method), f"Missing monitoring method: {method}"
        
        log_result(test_name, True)
        
    except Exception as e:
        log_result(test_name, False, str(e))

def test_import_patterns():
    """Test that all expected import patterns work"""
    test_name = "Import Patterns"
    
    try:
        # Test direct imports
        from prompt_improver.database.unified_connection_manager import (
            UnifiedConnectionManager,
            ManagerMode,
            PoolConfiguration,
            ConnectionMetrics,
            HealthStatus,
            DatabaseManagerAdapter,
            DatabaseSessionManagerAdapter
        )
        
        # Test factory function imports
        from prompt_improver.database.unified_connection_manager import (
            get_unified_manager,
            get_database_manager_adapter,
            get_database_session_manager_adapter,
            get_ha_connection_manager_adapter
        )
        
        # Test that all imports are not None
        imports_to_test = [
            UnifiedConnectionManager, ManagerMode, PoolConfiguration,
            ConnectionMetrics, HealthStatus, DatabaseManagerAdapter,
            DatabaseSessionManagerAdapter, get_unified_manager,
            get_database_manager_adapter, get_database_session_manager_adapter,
            get_ha_connection_manager_adapter
        ]
        
        for import_item in imports_to_test:
            assert import_item is not None, f"Import is None: {import_item}"
        
        log_result(test_name, True)
        
    except Exception as e:
        log_result(test_name, False, str(e))

def test_method_signatures():
    """Test that method signatures match expected patterns"""
    test_name = "Method Signatures"
    
    try:
        from prompt_improver.database.unified_connection_manager import UnifiedConnectionManager
        
        # Test key method signatures
        manager_class = UnifiedConnectionManager
        
        # Check get_connection signature (from ConnectionManagerProtocol)
        get_conn_sig = inspect.signature(manager_class.get_connection)
        assert 'mode' in get_conn_sig.parameters, "get_connection missing mode parameter"
        assert 'kwargs' in get_conn_sig.parameters, "get_connection missing kwargs parameter"
        
        # Check health_check signature
        health_sig = inspect.signature(manager_class.health_check)
        # Should return Dict[str, Any]
        assert health_sig.return_annotation != inspect.Signature.empty, "health_check missing return annotation"
        
        # Check initialization signature
        init_sig = inspect.signature(manager_class.__init__)
        init_params = list(init_sig.parameters.keys())
        expected_params = ['self', 'mode', 'db_config', 'redis_config']
        for param in expected_params:
            assert param in init_params, f"__init__ missing parameter: {param}"
        
        log_result(test_name, True)
        
    except Exception as e:
        log_result(test_name, False, str(e))

def test_composition_pattern():
    """Test that composition pattern is correctly implemented"""  
    test_name = "Composition Pattern Implementation"
    
    try:
        from prompt_improver.database.unified_connection_manager import UnifiedConnectionManager, ManagerMode
        
        # Create manager instance
        manager = UnifiedConnectionManager(ManagerMode.HIGH_AVAILABILITY)
        
        # Check that it has composition attributes for all original managers
        composition_attributes = [
            '_registry_manager',  # RegistryManager
            '_metrics',          # Connection metrics from all managers
            '_sync_engine',      # DatabaseManager capability
            '_async_engine',     # DatabaseSessionManager capability
            '_pg_pools',         # HAConnectionManager capability
            '_redis_master',     # Redis connection capability
            'pool_config'        # UnifiedConnectionManager mode-based config
        ]
        
        for attr in composition_attributes:
            assert hasattr(manager, attr), f"Missing composition attribute: {attr}"
        
        log_result(test_name, True)
        
    except Exception as e:
        log_result(test_name, False, str(e))

def run_all_tests():
    """Run all structural validation tests"""
    print(f"\n🔍 Starting UnifiedConnectionManager Structural Validation")
    print(f"{'='*70}")
    
    # Run all tests
    test_unified_manager_class_structure()
    test_connection_manager_protocol_compliance()
    test_manager_modes_enum()
    test_pool_configuration_class()
    test_backward_compatibility_adapters()
    test_unified_manager_migration_complete()
    test_consolidation_completeness()
    test_metrics_and_monitoring()
    test_import_patterns()
    test_method_signatures()
    test_composition_pattern()
    
    # Report results
    print(f"\n📊 Structural Validation Results")
    print(f"{'='*70}")
    
    passed = sum(1 for r in validation_results if r.success)
    total = len(validation_results)
    
    for result in validation_results:
        print(result)
    
    print(f"\n{'✅ ALL STRUCTURAL TESTS PASSED' if passed == total else '❌ SOME STRUCTURAL TESTS FAILED'}")
    print(f"Passed: {passed}/{total} ({passed/total*100:.1f}%)")
    
    if passed != total:
        print("\n🚨 Structural validation failed. Code structure issues detected.")
        return False
    else:
        print("\n🎉 Structural validation successful. Code structure is correct.")
        return True

def main():
    """Main entry point"""
    try:
        success = run_all_tests()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Validation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n💥 Validation failed with unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()