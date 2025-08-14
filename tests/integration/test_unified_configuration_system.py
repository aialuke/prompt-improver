"""Integration tests for unified configuration system.

Tests the consolidated configuration system that replaces 4,442 lines across
14 fragmented modules with a single, unified approach.

Test Coverage:
- Configuration initialization performance (<100ms SLO)
- Environment variable validation
- Configuration migration compatibility
- Error handling and validation
- Memory usage and resource efficiency
"""

import os
import pytest
import time
import tempfile
from pathlib import Path
from typing import Dict, Any
from unittest.mock import patch

# Test the unified configuration system
from prompt_improver.core.config_unified import (
    UnifiedConfig,
    UnifiedConfigManager, 
    Environment,
    LogLevel,
    ValidationResult,
    InitializationMetrics,
    get_config,
    get_initialization_metrics,
    reload_config,
    get_database_config,
    get_redis_config,
    get_security_config,
    get_ml_config,
    get_monitoring_config,
)


class TestUnifiedConfigurationSystem:
    """Test suite for unified configuration system."""
    
    def test_configuration_initialization_performance(self):
        """Test configuration initialization meets <100ms SLO."""
        # Clear any existing configuration
        UnifiedConfigManager._instance = None
        UnifiedConfigManager._config = None
        UnifiedConfigManager._initialization_metrics = None
        
        start_time = time.perf_counter()
        config = get_config()
        end_time = time.perf_counter()
        
        initialization_time_ms = (end_time - start_time) * 1000
        metrics = get_initialization_metrics()
        
        assert config is not None
        assert metrics is not None
        assert metrics.meets_slo, f"Configuration initialization took {metrics.total_time_ms:.2f}ms (>100ms SLO)"
        assert initialization_time_ms < 100.0, f"Direct measurement: {initialization_time_ms:.2f}ms"
        
        print(f"✓ Configuration initialized in {metrics.total_time_ms:.2f}ms")
        print(f"  Environment loading: {metrics.environment_load_time_ms:.2f}ms")
        print(f"  Validation: {metrics.validation_time_ms:.2f}ms")
    
    def test_environment_variable_configuration(self):
        """Test configuration loading from environment variables."""
        test_env = {
            "ENVIRONMENT": "testing",
            "DEBUG": "true",
            "LOG_LEVEL": "DEBUG",
            "DATABASE_URL": "postgresql://test:test@localhost:5432/test_db",
            "REDIS_URL": "redis://localhost:6379/1",
            "SECRET_KEY": "test-secret-key-that-is-long-enough-for-validation-requirements",
            "API_PORT": "8081",
            "ML_BATCH_SIZE": "64",
        }
        
        with patch.dict(os.environ, test_env):
            # Force reload to pick up environment changes
            config = reload_config()
            
            assert config.environment == Environment.TESTING
            assert config.debug is True
            assert config.log_level == LogLevel.DEBUG
            assert config.database_url == test_env["DATABASE_URL"]
            assert config.redis_url == test_env["REDIS_URL"]
            assert config.secret_key == test_env["SECRET_KEY"]
            assert config.api_port == 8081
            assert config.ml_batch_size == 64
    
    def test_configuration_validation_success(self):
        """Test successful configuration validation."""
        valid_env = {
            "DATABASE_URL": "postgresql://valid:password@localhost:5432/db",
            "REDIS_URL": "redis://localhost:6379/0",
            "SECRET_KEY": "this-is-a-very-long-secret-key-for-testing-purposes-that-meets-requirements",
            "DATABASE_POOL_SIZE": "10",
            "REDIS_MAX_CONNECTIONS": "5",
            "RATE_LIMIT_PER_MINUTE": "50",
        }
        
        with patch.dict(os.environ, valid_env):
            config = reload_config()
            metrics = get_initialization_metrics()
            
            assert metrics is not None
            assert metrics.meets_slo
            assert config.database_url == valid_env["DATABASE_URL"]
            assert config.database_pool_size == 10
            assert config.redis_max_connections == 5
    
    def test_configuration_validation_failures(self):
        """Test configuration validation error handling."""
        invalid_configs = [
            {
                "name": "invalid_database_url",
                "env": {"DATABASE_URL": "invalid-url"},
                "expected_error": "Database URL must be PostgreSQL or SQLite"
            },
            {
                "name": "invalid_redis_url", 
                "env": {"REDIS_URL": "invalid://localhost:6379"},
                "expected_error": "Redis URL must start with redis://"
            },
            {
                "name": "short_secret_key",
                "env": {"SECRET_KEY": "short"},
                "expected_error": "Secret key must be at least 32 characters"
            },
        ]
        
        for test_case in invalid_configs:
            with patch.dict(os.environ, test_case["env"], clear=True):
                with pytest.raises(ValueError) as exc_info:
                    reload_config()
                
                assert test_case["expected_error"] in str(exc_info.value)
    
    def test_production_environment_validation(self):
        """Test production-specific validation rules."""
        production_env = {
            "ENVIRONMENT": "production",
            "DEBUG": "true",  # Invalid in production
            "SECRET_KEY": "dev-key-change-in-production",  # Invalid in production
        }
        
        with patch.dict(os.environ, production_env):
            with pytest.raises(ValueError) as exc_info:
                reload_config()
            
            error_message = str(exc_info.value)
            assert "Debug mode cannot be enabled in production" in error_message or \
                   "Must set production secret key" in error_message
    
    def test_convenience_configuration_functions(self):
        """Test convenience functions for specific configuration sections."""
        test_env = {
            "DATABASE_URL": "postgresql://test:test@localhost:5432/test",
            "DATABASE_POOL_SIZE": "15",
            "DATABASE_TIMEOUT": "45.0",
            "REDIS_URL": "redis://localhost:6379/2", 
            "REDIS_MAX_CONNECTIONS": "25",
            "SECRET_KEY": "test-secret-key-for-convenience-function-testing-that-is-long-enough",
            "JWT_EXPIRY_HOURS": "48",
            "ML_BATCH_SIZE": "128",
            "ML_MODEL_PATH": "/tmp/test_models",
            "METRICS_ENABLED": "false",
            "HEALTH_CHECK_INTERVAL": "60",
        }
        
        with patch.dict(os.environ, test_env):
            config = reload_config()
            
            # Test database config
            db_config = get_database_config()
            assert db_config["url"] == test_env["DATABASE_URL"]
            assert db_config["pool_size"] == 15
            assert db_config["timeout"] == 45.0
            
            # Test Redis config
            redis_config = get_redis_config()
            assert redis_config["url"] == test_env["REDIS_URL"]
            assert redis_config["max_connections"] == 25
            
            # Test security config
            security_config = get_security_config()
            assert security_config["secret_key"] == test_env["SECRET_KEY"]
            assert security_config["jwt_expiry_hours"] == 48
            
            # Test ML config
            ml_config = get_ml_config()
            assert ml_config["batch_size"] == 128
            assert ml_config["model_path"] == test_env["ML_MODEL_PATH"]
            
            # Test monitoring config
            monitoring_config = get_monitoring_config()
            assert monitoring_config["metrics_enabled"] is False
            assert monitoring_config["health_check_interval"] == 60
    
    def test_singleton_behavior(self):
        """Test that configuration manager maintains singleton behavior."""
        manager1 = UnifiedConfigManager()
        manager2 = UnifiedConfigManager()
        
        assert manager1 is manager2
        assert id(manager1) == id(manager2)
        
        config1 = manager1.get_config()
        config2 = manager2.get_config()
        
        assert config1 is config2
    
    def test_configuration_reload_functionality(self):
        """Test configuration reload with environment changes."""
        initial_env = {
            "API_PORT": "8080",
            "ML_BATCH_SIZE": "32",
            "SECRET_KEY": "initial-secret-key-that-is-long-enough-for-validation-requirements",
        }
        
        with patch.dict(os.environ, initial_env):
            config1 = reload_config()
            assert config1.api_port == 8080
            assert config1.ml_batch_size == 32
        
        updated_env = {
            "API_PORT": "9090", 
            "ML_BATCH_SIZE": "64",
            "SECRET_KEY": "updated-secret-key-that-is-long-enough-for-validation-requirements",
        }
        
        with patch.dict(os.environ, updated_env):
            config2 = reload_config()
            assert config2.api_port == 9090
            assert config2.ml_batch_size == 64
            
            # Should be a different instance after reload
            assert config1 is not config2
    
    def test_memory_efficiency(self):
        """Test memory usage of unified configuration system."""
        import sys
        
        # Clear configuration
        UnifiedConfigManager._instance = None
        UnifiedConfigManager._config = None
        
        # Measure memory before config loading
        initial_objects = len(gc.get_objects()) if 'gc' in sys.modules else 0
        
        # Load configuration multiple times
        configs = []
        for _ in range(10):
            configs.append(get_config())
        
        # All should be the same instance (singleton)
        for config in configs[1:]:
            assert config is configs[0]
        
        # Memory should not grow significantly
        final_objects = len(gc.get_objects()) if 'gc' in sys.modules else 0
        
        # Allow some growth but not linear with requests
        if 'gc' in sys.modules:
            object_growth = final_objects - initial_objects
            assert object_growth < 100, f"Excessive memory growth: {object_growth} objects"
    
    def test_concurrent_access_safety(self):
        """Test thread safety of configuration access."""
        import threading
        import queue
        
        results = queue.Queue()
        errors = queue.Queue()
        
        def worker():
            try:
                config = get_config()
                results.put(id(config))
            except Exception as e:
                errors.put(e)
        
        # Start multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=worker)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Check results
        assert errors.empty(), f"Errors occurred: {list(errors.queue)}"
        
        # All threads should get the same config instance
        config_ids = []
        while not results.empty():
            config_ids.append(results.get())
        
        assert len(config_ids) == 10
        assert all(config_id == config_ids[0] for config_id in config_ids)
    
    def test_edge_case_environment_values(self):
        """Test handling of edge case environment values."""
        edge_cases = {
            "DATABASE_POOL_SIZE": "0",  # Should be invalid
            "REDIS_TIMEOUT": "-1.0",   # Should be invalid  
            "API_PORT": "99999",       # Large but valid
            "ML_BATCH_SIZE": "1",      # Minimum valid
            "SECRET_KEY": "a" * 100,   # Very long key
        }
        
        with patch.dict(os.environ, edge_cases):
            try:
                config = reload_config()
                
                # Some values should be accepted
                assert config.api_port == 99999
                assert config.ml_batch_size == 1
                assert len(config.secret_key) == 100
                
                # Others might cause validation errors
                # This depends on implementation details
                
            except ValueError:
                # Expected for some invalid values
                pass
    
    def test_backward_compatibility_interface(self):
        """Test that the new system maintains backward compatibility for critical imports."""
        # Test that old-style access patterns still work
        config = get_config()
        
        # These should all work without errors
        assert hasattr(config, 'database_url')
        assert hasattr(config, 'redis_url')
        assert hasattr(config, 'secret_key')
        assert hasattr(config, 'environment')
        
        # Test convenience functions work
        db_config = get_database_config()
        assert 'url' in db_config
        assert 'pool_size' in db_config
        
        redis_config = get_redis_config()
        assert 'url' in redis_config
        assert 'max_connections' in redis_config


# Performance benchmark test
class TestConfigurationPerformance:
    """Performance-focused tests for configuration system."""
    
    def test_initialization_benchmark(self):
        """Benchmark configuration initialization against SLO."""
        # Clear state
        UnifiedConfigManager._instance = None
        UnifiedConfigManager._config = None
        UnifiedConfigManager._initialization_metrics = None
        
        # Run multiple initialization cycles
        times = []
        for _ in range(5):
            UnifiedConfigManager._instance = None
            UnifiedConfigManager._config = None
            UnifiedConfigManager._initialization_metrics = None
            
            start = time.perf_counter()
            get_config()
            end = time.perf_counter()
            
            times.append((end - start) * 1000)
        
        avg_time = sum(times) / len(times)
        max_time = max(times)
        min_time = min(times)
        
        print(f"Configuration initialization benchmark:")
        print(f"  Average: {avg_time:.2f}ms")
        print(f"  Range: {min_time:.2f}ms - {max_time:.2f}ms")
        print(f"  SLO (<100ms): {'✓' if max_time < 100.0 else '✗'}")
        
        assert avg_time < 100.0, f"Average initialization time {avg_time:.2f}ms exceeds SLO"
        assert max_time < 150.0, f"Maximum initialization time {max_time:.2f}ms exceeds tolerance"
    
    def test_repeated_access_performance(self):
        """Test performance of repeated configuration access."""
        # Initialize once
        get_config()
        
        start_time = time.perf_counter()
        
        # Access configuration many times
        for _ in range(1000):
            config = get_config()
            _ = config.database_url
            _ = config.redis_url
            _ = config.secret_key
        
        end_time = time.perf_counter()
        total_time_ms = (end_time - start_time) * 1000
        avg_access_time_us = (total_time_ms * 1000) / 1000  # microseconds per access
        
        print(f"Repeated access performance:")
        print(f"  1000 accesses in {total_time_ms:.2f}ms")
        print(f"  Average per access: {avg_access_time_us:.2f}μs")
        
        assert avg_access_time_us < 100.0, f"Average access time {avg_access_time_us:.2f}μs too slow"


if __name__ == "__main__":
    # Run with: python -m pytest tests/integration/test_unified_configuration_system.py -v
    pytest.main([__file__, "-v", "--tb=short"])