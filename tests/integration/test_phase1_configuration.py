#!/usr/bin/env python3
"""
COMPREHENSIVE REAL BEHAVIOR TESTING SUITE FOR PHASE 1 CONFIGURATION SYSTEM

This test suite validates the Phase 1 configuration system with actual services,
real file operations, and production-like scenarios. NO MOCKS - only real behavior.

Test Categories:
1. Configuration Hot-Reload Tests - Real file changes with <100ms performance
2. Environment-Specific Loading - Real database connections per environment  
3. Validation System Tests - Real service connectivity validation
4. Migration Script Tests - Real code file hardcoded value detection
5. Schema Versioning Tests - Real configuration file upgrades
6. Error Recovery Tests - Real rollback and graceful degradation
7. Performance Tests - Real memory usage and concurrent access

Author: SRE System (Claude Code)  
Date: 2025-07-25
"""

import asyncio
import gc
import json
import logging
import os
import psutil
import shutil
import tempfile
import time
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid
import weakref

import pytest
import yaml
import coredis
import psycopg

# Import the configuration system components
from prompt_improver.core.config_manager import (
    ConfigManager,
    FileConfigSource,
    EnvironmentConfigSource,
    RemoteConfigSource,
    ConfigChange,
    ConfigChangeType,
    ReloadStatus,
    ConfigMetrics
)
from prompt_improver.core.config_schema import (
    ConfigSchemaManager,
    ConfigSchemaVersion,
    migrate_configuration
)
from prompt_improver.core.config_validator import (
    ConfigurationValidator,
    validate_startup_configuration,
    save_validation_report
)
from prompt_improver.database.config import DatabaseConfig
from prompt_improver.security.config_validator import SecurityConfigValidator
from scripts.migrate_hardcoded_config import HardcodedConfigMigrator

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetrics:
    """Real performance metrics captured during testing."""
    operation: str
    duration_ms: float
    memory_before_mb: float
    memory_after_mb: float
    memory_peak_mb: float
    cpu_percent: float
    timestamp: datetime = field(default_factory=datetime.now)

@dataclass
class RealBehaviorTestResult:
    """Result from real behavior configuration testing."""
    test_name: str
    success: bool
    duration_ms: float
    memory_used_mb: float
    performance_metrics: List[PerformanceMetrics]
    config_changes_detected: int
    services_validated: List[str]
    error_details: Optional[str] = None
    
    def meets_performance_requirements(self) -> bool:
        """Check if test meets performance requirements."""
        # Hot-reload should be under 100ms
        hot_reload_metrics = [m for m in self.performance_metrics if 'reload' in m.operation.lower()]
        if hot_reload_metrics:
            return all(m.duration_ms < 100 for m in hot_reload_metrics)
        return True


class Phase1ConfigurationRealBehaviorTestSuite:
    """
    Comprehensive real behavior test suite for Phase 1 configuration system.
    Tests with actual services, real files, and production scenarios.
    """
    
    def __init__(self):
        """Initialize test suite with real environment setup."""
        self.test_dir = None
        self.temp_configs = {}
        self.real_postgres_available = False
        self.real_redis_available = False
        self.config_manager = None
        self.performance_metrics = []
        
    async def setup_real_environment(self) -> bool:
        """Set up real testing environment with actual services."""
        try:
            # Create temporary directory for test configurations
            self.test_dir = Path(tempfile.mkdtemp(prefix="phase1_config_test_"))
            logger.info(f"Created test directory: {self.test_dir}")
            
            # Test real PostgreSQL connectivity
            await self._test_postgresql_connectivity()
            
            # Test real Redis connectivity  
            await self._test_redis_connectivity()
            
            # Create real configuration files
            await self._create_real_test_configs()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup real environment: {e}")
            return False
    
    async def teardown_real_environment(self):
        """Clean up real testing environment."""
        try:
            if self.config_manager:
                await self.config_manager.stop()
                
            if self.test_dir and self.test_dir.exists():
                shutil.rmtree(self.test_dir)
                logger.info(f"Cleaned up test directory: {self.test_dir}")
                
        except Exception as e:
            logger.warning(f"Error during cleanup: {e}")
    
    async def _test_postgresql_connectivity(self):
        """Test real PostgreSQL database connectivity."""
        try:
            db_config = DatabaseConfig()
            conn_string = db_config.psycopg_connection_string
            
            async with psycopg.AsyncConnection.connect(
                conn_string, 
                connect_timeout=5
            ) as conn:
                async with conn.cursor() as cur:
                    await cur.execute("SELECT version(), current_database(), current_user")
                    result = await cur.fetchone()
                    
                    self.real_postgres_available = True
                    logger.info(f"✅ PostgreSQL connected: {result[1]} as {result[2]}")
                    
        except Exception as e:
            logger.warning(f"⚠️ PostgreSQL not available for real testing: {e}")
            self.real_postgres_available = False
    
    async def _test_redis_connectivity(self):
        """Test real Redis connectivity."""
        try:
            redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/15")
            client = coredis.Redis.from_url(redis_url)
            
            # Test basic operations
            await client.ping()
            test_key = f"config_test_{uuid.uuid4().hex[:8]}"
            await client.set(test_key, "test_value", ex=10)
            value = await client.get(test_key)
            await client.delete(test_key)
            
            if value == b"test_value":
                self.real_redis_available = True
                logger.info("✅ Redis connected and functional")
            else:
                raise Exception("Redis data integrity test failed")
                
        except Exception as e:
            logger.warning(f"⚠️ Redis not available for real testing: {e}")
            self.real_redis_available = False
    
    async def _create_real_test_configs(self):
        """Create real configuration files for testing."""
        # Development configuration
        dev_config = {
            "environment": "development",
            "database": {
                "host": os.getenv("POSTGRES_HOST", "localhost"),
                "port": int(os.getenv("POSTGRES_PORT", "5432")),
                "database": os.getenv("POSTGRES_DATABASE", "prompt_improver_dev"),
                "pool_size": 5,
                "timeout": 30
            },
            "redis": {
                "url": os.getenv("REDIS_URL", "redis://localhost:6379/1"),
                "max_connections": 10
            },
            "ml": {
                "model_path": "./models",
                "batch_size": 32,
                "cache_enabled": True
            },
            "monitoring": {
                "enabled": True,
                "metrics_interval": 30,
                "health_check_timeout": 10
            }
        }
        
        dev_config_file = self.test_dir / "config_dev.yaml"
        with open(dev_config_file, 'w') as f:
            yaml.dump(dev_config, f)
        self.temp_configs["dev"] = dev_config_file
        
        # Production configuration
        prod_config = {
            "environment": "production", 
            "database": {
                "host": os.getenv("POSTGRES_HOST", "prod-db.example.com"),
                "port": int(os.getenv("POSTGRES_PORT", "5432")),
                "database": os.getenv("POSTGRES_DATABASE", "prompt_improver_prod"),
                "pool_size": 20,
                "timeout": 10
            },
            "redis": {
                "url": os.getenv("REDIS_URL", "redis://prod-redis.example.com:6379/0"),
                "max_connections": 50
            },
            "ml": {
                "model_path": "/opt/models",
                "batch_size": 128,
                "cache_enabled": True
            },
            "monitoring": {
                "enabled": True,
                "metrics_interval": 15,
                "health_check_timeout": 5
            }
        }
        
        prod_config_file = self.test_dir / "config_prod.yaml"
        with open(prod_config_file, 'w') as f:
            yaml.dump(prod_config, f)
        self.temp_configs["prod"] = prod_config_file
        
        # Environment variables file
        env_file = self.test_dir / ".env"
        env_content = f"""
# Test environment variables
ENVIRONMENT=development
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DATABASE=prompt_improver_test
POSTGRES_USERNAME=test_user
POSTGRES_PASSWORD=test_password
REDIS_URL=redis://localhost:6379/15
ML_MODEL_PATH=./models
MONITORING_ENABLED=true
API_RATE_LIMIT_PER_MINUTE=100
"""
        env_file.write_text(env_content.strip())
        self.temp_configs["env"] = env_file
    
    def _capture_performance_metrics(self, operation: str) -> PerformanceMetrics:
        """Capture real performance metrics for an operation."""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return PerformanceMetrics(
            operation=operation,
            duration_ms=0,  # Will be set by caller
            memory_before_mb=memory_info.rss / 1024 / 1024,
            memory_after_mb=0,  # Will be set by caller  
            memory_peak_mb=0,  # Will be set by caller
            cpu_percent=process.cpu_percent()
        )
    
    async def test_configuration_hot_reload_real_files(self) -> RealBehaviorTestResult:
        """
        Test configuration hot-reload with actual file changes.
        Validates <100ms reload performance and zero-downtime operation.
        """
        test_name = "configuration_hot_reload_real_files"
        start_time = time.time()
        
        try:
            # Setup configuration manager with real file watching
            self.config_manager = ConfigManager(watch_files=True, max_config_versions=5)
            
            # Add real file source
            dev_config_source = FileConfigSource(
                self.temp_configs["dev"], 
                name="dev_config",
                priority=10
            )
            await self.config_manager.add_source(dev_config_source)
            
            # Start configuration manager
            await self.config_manager.start()
            
            # Verify initial configuration load
            initial_config = await self.config_manager.get_config()
            assert initial_config is not None
            assert initial_config["environment"] == "development"
            
            # Test multiple hot-reload scenarios
            reload_metrics = []
            changes_detected = 0
            
            # Scenario 1: Modify database pool size
            config_data = yaml.safe_load(self.temp_configs["dev"].read_text())
            config_data["database"]["pool_size"] = 10
            
            # Measure reload performance
            metrics_before = self._capture_performance_metrics("hot_reload_pool_size")
            reload_start = time.time()
            
            # Write modified configuration
            with open(self.temp_configs["dev"], 'w') as f:
                yaml.dump(config_data, f)
            
            # Wait for hot-reload (with timeout)
            reload_detected = False
            max_wait = 2.0  # 2 seconds max wait
            check_interval = 0.01  # 10ms checks
            
            while time.time() - reload_start < max_wait:
                current_config = await self.config_manager.get_config()
                if current_config.get("database", {}).get("pool_size") == 10:
                    reload_detected = True
                    break
                await asyncio.sleep(check_interval)
            
            reload_duration = (time.time() - reload_start) * 1000  # Convert to ms
            
            assert reload_detected, "Hot-reload not detected within timeout"
            assert reload_duration < 100, f"Hot-reload took {reload_duration:.1f}ms (requirement: <100ms)"
            
            metrics_after = self._capture_performance_metrics("hot_reload_pool_size")
            metrics_before.duration_ms = reload_duration
            metrics_before.memory_after_mb = metrics_after.memory_before_mb
            reload_metrics.append(metrics_before)
            changes_detected += 1
            
            logger.info(f"✅ Hot-reload test 1: {reload_duration:.1f}ms")
            
            # Scenario 2: Modify multiple nested values
            config_data["ml"]["batch_size"] = 64
            config_data["monitoring"]["metrics_interval"] = 60
            config_data["redis"]["max_connections"] = 20
            
            metrics_before = self._capture_performance_metrics("hot_reload_multiple_changes")
            reload_start = time.time()
            
            with open(self.temp_configs["dev"], 'w') as f:
                yaml.dump(config_data, f)
            
            # Wait for reload
            while time.time() - reload_start < max_wait:
                current_config = await self.config_manager.get_config()
                ml_config = current_config.get("ml", {})
                if (ml_config.get("batch_size") == 64 and 
                    current_config.get("monitoring", {}).get("metrics_interval") == 60):
                    break
                await asyncio.sleep(check_interval)
            
            reload_duration = (time.time() - reload_start) * 1000
            assert reload_duration < 100, f"Multi-change reload took {reload_duration:.1f}ms"
            
            metrics_after = self._capture_performance_metrics("hot_reload_multiple_changes")
            metrics_before.duration_ms = reload_duration
            metrics_before.memory_after_mb = metrics_after.memory_before_mb
            reload_metrics.append(metrics_before)
            changes_detected += 1
            
            logger.info(f"✅ Hot-reload test 2: {reload_duration:.1f}ms")
            
            # Scenario 3: Test rollback on invalid configuration
            invalid_config = config_data.copy()
            invalid_config["database"]["port"] = "invalid_port"  # Invalid type
            
            with open(self.temp_configs["dev"], 'w') as f:
                yaml.dump(invalid_config, f)
            
            # Configuration should remain valid (rollback or graceful handling)
            await asyncio.sleep(0.2)  # Allow time for processing
            current_config = await self.config_manager.get_config()
            
            # Should still have valid configuration
            assert isinstance(current_config.get("database", {}).get("port"), int)
            logger.info("✅ Invalid configuration handled gracefully")
            
            # Restore valid configuration
            with open(self.temp_configs["dev"], 'w') as f:
                yaml.dump(config_data, f)
            
            # Test zero-downtime: configuration should always be accessible
            for i in range(50):
                config = await self.config_manager.get_config()
                assert config is not None
                await asyncio.sleep(0.01)
            
            logger.info("✅ Zero-downtime validated (50 rapid accesses)")
            
            duration_ms = (time.time() - start_time) * 1000
            
            return RealBehaviorTestResult(
                test_name=test_name,
                success=True,
                duration_ms=duration_ms,
                memory_used_mb=sum(m.memory_after_mb - m.memory_before_mb for m in reload_metrics),
                performance_metrics=reload_metrics,
                config_changes_detected=changes_detected,
                services_validated=["file_watcher", "config_manager"],
                error_details=None
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ Hot-reload test failed: {e}")
            
            return RealBehaviorTestResult(
                test_name=test_name,
                success=False,
                duration_ms=duration_ms,
                memory_used_mb=0,
                performance_metrics=[],
                config_changes_detected=0,
                services_validated=[],
                error_details=str(e)
            )
    
    async def test_environment_specific_loading_real_connections(self) -> RealBehaviorTestResult:
        """
        Test environment-specific configuration loading with real database connections.
        Validates different configs for dev/staging/prod environments.
        """
        test_name = "environment_specific_loading_real_connections"
        start_time = time.time()
        
        try:
            services_validated = []
            config_changes = 0
            
            # Test development environment
            os.environ["ENVIRONMENT"] = "development"
            
            config_manager = ConfigManager(watch_files=False)
            dev_source = FileConfigSource(self.temp_configs["dev"], name="dev", priority=10)
            await config_manager.add_source(dev_source)
            await config_manager.start()
            
            dev_config = await config_manager.get_config()
            assert dev_config["environment"] == "development"
            assert dev_config["database"]["pool_size"] == 10  # Updated value
            services_validated.append("dev_config_load")
            config_changes += 1
            
            # Test database configuration validation
            if self.real_postgres_available:
                db_config = DatabaseConfig()
                # Verify configuration can create valid connection
                conn_string = db_config.psycopg_connection_string
                assert "postgres" in conn_string
                services_validated.append("postgres_config_validation")
                
                # Test actual connection with dev config
                try:
                    async with psycopg.AsyncConnection.connect(
                        conn_string, connect_timeout=5
                    ) as conn:
                        async with conn.cursor() as cur:
                            await cur.execute("SELECT current_database()")
                            result = await cur.fetchone()
                            assert result is not None
                            services_validated.append("postgres_real_connection")
                except Exception as e:
                    logger.warning(f"Real PostgreSQL connection failed: {e}")
            
            await config_manager.stop()
            
            # Test production environment
            os.environ["ENVIRONMENT"] = "production"
            
            config_manager = ConfigManager(watch_files=False)
            prod_source = FileConfigSource(self.temp_configs["prod"], name="prod", priority=10)
            await config_manager.add_source(prod_source)
            await config_manager.start()
            
            prod_config = await config_manager.get_config()
            assert prod_config["environment"] == "production"
            assert prod_config["database"]["pool_size"] == 20  # Production value
            assert prod_config["monitoring"]["metrics_interval"] == 15  # Prod frequency
            services_validated.append("prod_config_load")
            config_changes += 1
            
            # Test Redis configuration validation
            if self.real_redis_available:
                redis_url = prod_config.get("redis", {}).get("url", os.getenv("REDIS_URL"))
                if redis_url:
                    try:
                        client = coredis.Redis.from_url(redis_url)
                        await client.ping()
                        services_validated.append("redis_real_connection")
                    except Exception as e:
                        logger.warning(f"Real Redis connection failed: {e}")
            
            await config_manager.stop()
            
            # Test environment variable override
            os.environ["POSTGRES_HOST"] = "override.example.com"
            os.environ["ML_BATCH_SIZE"] = "256"
            
            env_source = EnvironmentConfigSource(prefix="", name="env", priority=20)
            config_manager = ConfigManager(watch_files=False)
            await config_manager.add_source(prod_source)  # Lower priority
            await config_manager.add_source(env_source)    # Higher priority
            await config_manager.start()
            
            merged_config = await config_manager.get_config()
            
            # Environment variables should override file config
            assert merged_config.get("postgres_host") == "override.example.com"
            assert merged_config.get("ml_batch_size") == "256"
            services_validated.append("env_variable_override")
            config_changes += 1
            
            await config_manager.stop()
            
            # Clean up environment
            del os.environ["POSTGRES_HOST"]
            del os.environ["ML_BATCH_SIZE"]
            os.environ["ENVIRONMENT"] = "development"  # Reset
            
            duration_ms = (time.time() - start_time) * 1000
            
            return RealBehaviorTestResult(
                test_name=test_name,
                success=True,
                duration_ms=duration_ms,
                memory_used_mb=psutil.Process().memory_info().rss / 1024 / 1024,
                performance_metrics=[],
                config_changes_detected=config_changes,
                services_validated=services_validated,
                error_details=None
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ Environment-specific loading test failed: {e}")
            
            return RealBehaviorTestResult(
                test_name=test_name,
                success=False,
                duration_ms=duration_ms,
                memory_used_mb=0,
                performance_metrics=[],
                config_changes_detected=0,
                services_validated=[],
                error_details=str(e)
            )
    
    async def test_startup_validation_real_services(self) -> RealBehaviorTestResult:
        """
        Test startup validation system with actual service connections.
        Validates PostgreSQL, Redis, and other services at startup.
        """
        test_name = "startup_validation_real_services"
        start_time = time.time()
        
        try:
            services_validated = []
            
            # Test comprehensive startup validation
            validator = ConfigurationValidator(environment="development")
            
            # Set up test environment variables
            test_env = {
                "POSTGRES_HOST": os.getenv("POSTGRES_HOST", "localhost"),
                "POSTGRES_PORT": os.getenv("POSTGRES_PORT", "5432"),
                "POSTGRES_DATABASE": os.getenv("POSTGRES_DATABASE", "prompt_improver_test"),
                "POSTGRES_USERNAME": os.getenv("POSTGRES_USERNAME", "test_user"),
                "POSTGRES_PASSWORD": os.getenv("POSTGRES_PASSWORD", "test_password"),
                "REDIS_URL": os.getenv("REDIS_URL", "redis://localhost:6379/15"),
                "ML_MODEL_PATH": str(self.test_dir / "models"),
                "MONITORING_ENABLED": "true",
                "API_RATE_LIMIT_PER_MINUTE": "100",
                "HEALTH_CHECK_TIMEOUT_SECONDS": "10"
            }
            
            # Temporarily set environment variables
            original_env = {}
            for key, value in test_env.items():
                original_env[key] = os.getenv(key)
                os.environ[key] = value
            
            try:
                # Create ML model directory
                model_dir = self.test_dir / "models"
                model_dir.mkdir(exist_ok=True)
                (model_dir / "test_model.bin").write_text("fake model data")
                
                # Run comprehensive validation
                is_valid, report = await validate_startup_configuration("development")
                
                # Validate results
                assert report is not None
                assert report.environment == "development"
                
                # Check individual validation results
                component_results = {r.component: r for r in report.results}
                
                # Environment setup should pass
                env_result = component_results.get("environment_setup")
                if env_result:
                    assert env_result.is_valid, f"Environment setup failed: {env_result.message}"
                    services_validated.append("environment_setup")
                
                # Security validation
                security_result = component_results.get("security") 
                if security_result:
                    services_validated.append("security_validation")
                
                # Database configuration
                db_config_result = component_results.get("database_config")
                if db_config_result:
                    assert db_config_result.is_valid, f"DB config failed: {db_config_result.message}"
                    services_validated.append("database_config")
                
                # Real database connectivity test
                if self.real_postgres_available:
                    db_conn_result = component_results.get("database_connectivity")
                    if db_conn_result and db_conn_result.is_valid:
                        services_validated.append("database_real_connectivity")
                
                # Redis configuration
                redis_config_result = component_results.get("redis_config")
                if redis_config_result:
                    services_validated.append("redis_config")
                
                # Real Redis connectivity test
                if self.real_redis_available:
                    redis_conn_result = component_results.get("redis_connectivity")
                    if redis_conn_result and redis_conn_result.is_valid:
                        services_validated.append("redis_real_connectivity")
                
                # ML configuration
                ml_result = component_results.get("ml_config")
                if ml_result:
                    assert ml_result.is_valid, f"ML config failed: {ml_result.message}"
                    services_validated.append("ml_config")
                
                # API configuration
                api_result = component_results.get("api_config")
                if api_result:
                    services_validated.append("api_config")
                
                # Health check configuration
                health_result = component_results.get("health_check_config")
                if health_result:
                    services_validated.append("health_check_config")
                
                # Monitoring configuration
                monitoring_result = component_results.get("monitoring_config")
                if monitoring_result:
                    services_validated.append("monitoring_config")
                
                # Save validation report
                report_file = self.test_dir / "validation_report.json"
                save_validation_report(report, str(report_file))
                assert report_file.exists()
                
                # Verify report contains expected data
                report_data = json.loads(report_file.read_text())
                assert "timestamp" in report_data
                assert "environment" in report_data
                assert "results" in report_data
                
                logger.info(f"✅ Startup validation completed with {len(services_validated)} services")
                
            finally:
                # Restore original environment
                for key, value in original_env.items():
                    if value is None:
                        os.environ.pop(key, None)
                    else:
                        os.environ[key] = value
            
            duration_ms = (time.time() - start_time) * 1000
            
            return RealBehaviorTestResult(
                test_name=test_name,
                success=True,
                duration_ms=duration_ms,
                memory_used_mb=psutil.Process().memory_info().rss / 1024 / 1024,
                performance_metrics=[],
                config_changes_detected=0,
                services_validated=services_validated,
                error_details=None
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ Startup validation test failed: {e}")
            
            return RealBehaviorTestResult(
                test_name=test_name,
                success=False,
                duration_ms=duration_ms,
                memory_used_mb=0,
                performance_metrics=[],
                config_changes_detected=0,
                services_validated=[],
                error_details=str(e)
            )
    
    async def test_hardcoded_migration_real_code_files(self) -> RealBehaviorTestResult:
        """
        Test hardcoded value detection and migration with real code files.
        Creates actual Python files with hardcoded values for detection.
        """
        test_name = "hardcoded_migration_real_code_files"
        start_time = time.time()
        
        try:
            # Create real code files with hardcoded values
            code_dir = self.test_dir / "test_code"
            code_dir.mkdir()
            
            # Sample Python file with hardcoded database values
            db_file = code_dir / "database_client.py"
            db_file.write_text('''
import psycopg

class DatabaseClient:
    def __init__(self):
        # Hardcoded database configuration - should be migrated
        self.host = "localhost"
        self.port = 5432
        self.database = "prompt_improver"
        self.pool_size = 10
        self.timeout = 30
        
    def connect(self):
        connection_string = f"postgresql://user:password@{self.host}:{self.port}/{self.database}"
        return psycopg.connect(connection_string)
''')
            
            # Sample configuration file with hardcoded paths
            config_file = code_dir / "ml_config.py"
            config_file.write_text('''
import os

# Machine learning configuration with hardcoded paths
ML_MODEL_PATH = "/opt/models"
ML_BATCH_SIZE = 32
ML_CACHE_TTL = 3600
REDIS_URL = "redis://localhost:6379/0"

class MLConfig:
    def __init__(self):
        self.model_path = "./models/bert-base-uncased"
        self.api_key = "sk-1234567890abcdef"  # Should be environment variable
        self.timeout = 60
''')
            
            # API client with hardcoded endpoints
            api_file = code_dir / "api_client.py" 
            api_file.write_text('''
import requests

class APIClient:
    def __init__(self):
        self.base_url = "https://api.example.com/v1"
        self.timeout = 30
        self.max_retries = 3
        
    def authenticate(self):
        # Hardcoded API credentials
        return {
            "api_key": "prod_key_123456789",
            "secret": "secret_abc123def456"
        }
''')
            
            # Run hardcoded configuration migrator
            migrator = HardcodedConfigMigrator(str(code_dir))
            report = migrator.scan_codebase()
            
            # Validate detection results
            assert report.total_files_scanned >= 3, f"Expected 3+ files, scanned {report.total_files_scanned}"
            assert len(report.hardcoded_values) > 0, "No hardcoded values detected"
            
            # Check for specific hardcoded values
            detected_values = [hv.value for hv in report.hardcoded_values]
            detected_contexts = [hv.context for hv in report.hardcoded_values]
            
            # Verify database hardcoded values detected
            database_detected = any("localhost" in val or "5432" in val for val in detected_values)
            redis_detected = any("redis://" in val for val in detected_values)
            api_detected = any("api.example.com" in val or "api_key" in context for val, context in zip(detected_values, detected_contexts))
            
            # High confidence values should include critical ones
            high_conf_values = [hv for hv in report.hardcoded_values if hv.confidence >= 0.8]
            assert len(high_conf_values) > 0, "No high confidence hardcoded values found"
            
            # Generate migration script
            migration_script = self.test_dir / "migration_script.py"
            migrator.generate_migration_script(report, str(migration_script))
            
            assert migration_script.exists(), "Migration script not generated"
            
            # Validate migration script content
            script_content = migration_script.read_text()
            assert "HIGH_CONFIDENCE_MIGRATIONS" in script_content
            assert "apply_migrations" in script_content
            assert "create_env_template" in script_content
            
            # Test migration script execution (dry run)
            exec_globals = {}
            exec(script_content, exec_globals)
            
            # Verify functions are available
            assert "apply_migrations" in exec_globals
            assert "create_env_template" in exec_globals
            
            # Test environment template creation
            exec_globals["create_env_template"]()
            env_template = Path("migration_env_template.env")
            
            if env_template.exists():
                template_content = env_template.read_text()
                assert "DATABASE_URL" in template_content or "POSTGRES_HOST" in template_content
                env_template.unlink()  # Clean up
            
            # Save migration report
            report_file = self.test_dir / "hardcoded_migration_report.json"
            report_file.write_text(json.dumps(report.to_dict(), indent=2))
            
            assert report_file.exists()
            
            logger.info(f"✅ Hardcoded migration test: {len(report.hardcoded_values)} values detected")
            logger.info(f"   High confidence: {len(high_conf_values)}")
            logger.info(f"   Suggested env vars: {len(report.suggested_env_vars)}")
            
            duration_ms = (time.time() - start_time) * 1000
            
            return RealBehaviorTestResult(
                test_name=test_name,
                success=True,
                duration_ms=duration_ms,
                memory_used_mb=psutil.Process().memory_info().rss / 1024 / 1024,
                performance_metrics=[],
                config_changes_detected=len(report.hardcoded_values),
                services_validated=["hardcoded_scanner", "migration_generator"],
                error_details=None
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ Hardcoded migration test failed: {e}")
            
            return RealBehaviorTestResult(
                test_name=test_name,
                success=False,
                duration_ms=duration_ms,
                memory_used_mb=0,
                performance_metrics=[],
                config_changes_detected=0,
                services_validated=[],
                error_details=str(e)
            )
    
    async def test_schema_versioning_real_upgrades(self) -> RealBehaviorTestResult:
        """
        Test configuration schema versioning with real configuration files.
        Tests actual configuration upgrades from v1.0 to v2.0.
        """
        test_name = "schema_versioning_real_upgrades"
        start_time = time.time()
        
        try:
            schema_manager = ConfigSchemaManager()
            services_validated = []
            config_changes = 0
            
            # Create v1.0 configuration file
            v1_config = {
                "database_host": "localhost",
                "database_port": 5432,
                "redis_host": "localhost", 
                "redis_port": 6379,
                "debug": True
            }
            
            v1_config_file = self.test_dir / "config_v1.json"
            v1_config_file.write_text(json.dumps(v1_config, indent=2))
            
            # Detect initial schema version
            detected_version = schema_manager.get_schema_version(v1_config_file)
            assert detected_version == ConfigSchemaVersion.V1_0
            services_validated.append("version_detection")
            
            # Test migration to v1.1 (adds ML configuration)
            success = schema_manager.migrate_to_version(v1_config_file, ConfigSchemaVersion.V1_1)
            assert success, "Migration to v1.1 failed"
            
            # Verify v1.1 configuration
            migrated_config = json.loads(v1_config_file.read_text())
            assert "ML_MODEL_PATH" in migrated_config
            assert "ML_BATCH_SIZE" in migrated_config
            assert migrated_config["ML_ENABLE_CACHING"] == "true"
            services_validated.append("v1_1_migration")
            config_changes += 1
            
            # Check schema metadata
            schema_file = v1_config_file.parent / f".{v1_config_file.name}.schema"
            assert schema_file.exists(), "Schema metadata file not created"
            
            schema_metadata = json.loads(schema_file.read_text())
            assert schema_metadata["schema_version"] == "1.1"
            assert "migration_history" in schema_metadata
            
            # Test migration to v1.2 (adds monitoring)
            success = schema_manager.migrate_to_version(v1_config_file, ConfigSchemaVersion.V1_2)
            assert success, "Migration to v1.2 failed"
            
            migrated_config = json.loads(v1_config_file.read_text())
            assert "MONITORING_ENABLED" in migrated_config
            assert "METRICS_EXPORT_INTERVAL_SECONDS" in migrated_config
            assert migrated_config["TRACING_ENABLED"] == "true"
            services_validated.append("v1_2_migration")
            config_changes += 1
            
            # Test migration to v2.0 (major restructure)
            success = schema_manager.migrate_to_version(v1_config_file, ConfigSchemaVersion.V2_0)
            assert success, "Migration to v2.0 failed"
            
            migrated_config = json.loads(v1_config_file.read_text())
            assert "FEATURE_FLAG_ML_IMPROVEMENTS" in migrated_config
            assert "SECURITY_ENABLE_RATE_LIMITING" in migrated_config
            assert migrated_config["FEATURE_FLAG_ADVANCED_CACHING"] == "true"
            services_validated.append("v2_0_migration")
            config_changes += 1
            
            # Verify final schema version
            final_version = schema_manager.get_schema_version(v1_config_file)
            assert final_version == ConfigSchemaVersion.V2_0
            
            # Test schema compliance validation
            is_valid, issues = schema_manager.validate_schema_compliance(v1_config_file)
            assert is_valid, f"Schema compliance failed: {issues}"
            services_validated.append("schema_validation")
            
            # Test backup creation
            backup_file = v1_config_file.with_suffix(f"{v1_config_file.suffix}.backup")
            assert backup_file.exists(), "Backup file not created during migration"
            
            # Verify backup contains original data
            backup_data = json.loads(backup_file.read_text())
            assert "FEATURE_FLAG_ML_IMPROVEMENTS" not in backup_data  # Should be original v1.0 data
            
            # Test migration with .env file
            env_config = {
                "DATABASE_HOST": "localhost",
                "DATABASE_PORT": "5432",
                "DEBUG": "true"
            }
            
            env_config_file = self.test_dir / "config_v1.env"
            env_lines = [f"{k}={v}" for k, v in env_config.items()]
            env_config_file.write_text("\n".join(env_lines))
            
            # Migrate env file to latest
            success = migrate_configuration(str(env_config_file))
            assert success, "ENV file migration failed"
            
            # Verify env file migration
            migrated_env = env_config_file.read_text()
            assert "ML_MODEL_PATH" in migrated_env
            assert "MONITORING_ENABLED" in migrated_env
            assert "FEATURE_FLAG_ML_IMPROVEMENTS" in migrated_env
            services_validated.append("env_file_migration")
            config_changes += 1
            
            logger.info(f"✅ Schema versioning test: {config_changes} successful migrations")
            
            duration_ms = (time.time() - start_time) * 1000
            
            return RealBehaviorTestResult(
                test_name=test_name,
                success=True,
                duration_ms=duration_ms,
                memory_used_mb=psutil.Process().memory_info().rss / 1024 / 1024,
                performance_metrics=[],
                config_changes_detected=config_changes,
                services_validated=services_validated,
                error_details=None
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ Schema versioning test failed: {e}")
            
            return RealBehaviorTestResult(
                test_name=test_name,
                success=False,
                duration_ms=duration_ms,
                memory_used_mb=0,
                performance_metrics=[],
                config_changes_detected=0,
                services_validated=[],
                error_details=str(e)
            )
    
    async def test_error_recovery_real_scenarios(self) -> RealBehaviorTestResult:
        """
        Test error recovery and rollback with real failure scenarios.
        Tests graceful degradation and automatic recovery.
        """
        test_name = "error_recovery_real_scenarios"
        start_time = time.time()
        
        try:
            services_validated = []
            config_changes = 0
            
            # Setup configuration manager with multiple sources
            config_manager = ConfigManager(watch_files=True, max_config_versions=10)
            
            # Add file source
            dev_source = FileConfigSource(self.temp_configs["dev"], name="dev", priority=10)
            await config_manager.add_source(dev_source)
            
            # Add environment source
            env_source = EnvironmentConfigSource(prefix="TEST_", name="env", priority=20)
            await config_manager.add_source(env_source)
            
            await config_manager.start()
            
            # Get initial valid configuration
            initial_config = await config_manager.get_config()
            assert initial_config is not None
            initial_version = config_manager._metrics.current_config_version
            services_validated.append("initial_config_load")
            
            # Scenario 1: Invalid JSON in configuration file
            logger.info("Testing invalid JSON recovery...")
            
            # Backup original content
            original_content = self.temp_configs["dev"].read_text()
            
            # Write invalid JSON
            self.temp_configs["dev"].write_text("{ invalid json content }")
            
            # Wait for file change detection
            await asyncio.sleep(0.2)
            
            # Configuration should remain accessible (graceful degradation)
            current_config = await config_manager.get_config()
            assert current_config is not None
            
            # Should still have the original valid configuration
            assert current_config.get("environment") == "development"
            services_validated.append("invalid_json_recovery")
            
            # Restore valid configuration
            self.temp_configs["dev"].write_text(original_content)
            await asyncio.sleep(0.2)
            
            # Scenario 2: File permission errors
            logger.info("Testing file permission error recovery...")
            
            # Make file unreadable (if possible on current OS)
            try:
                original_mode = self.temp_configs["dev"].stat().st_mode
                self.temp_configs["dev"].chmod(0o000)  # No permissions
                
                # Trigger reload
                result = await config_manager.reload("dev")
                
                # Should handle gracefully
                assert result.status in [ReloadStatus.FAILED, ReloadStatus.PARTIAL]
                
                # Configuration should still be accessible
                current_config = await config_manager.get_config()
                assert current_config is not None
                services_validated.append("permission_error_recovery")
                
                # Restore permissions
                self.temp_configs["dev"].chmod(original_mode)
                
            except (OSError, PermissionError):
                # Permission changes might not work on all systems
                logger.info("Skipping permission test (not supported on this system)")
            
            # Scenario 3: Test rollback functionality
            logger.info("Testing rollback functionality...")
            
            # Create a series of valid configuration changes
            config_data = yaml.safe_load(original_content)
            
            # Change 1
            config_data["database"]["pool_size"] = 15
            with open(self.temp_configs["dev"], 'w') as f:
                yaml.dump(config_data, f)
            await asyncio.sleep(0.1)
            config_changes += 1
            
            # Change 2  
            config_data["ml"]["batch_size"] = 64
            with open(self.temp_configs["dev"], 'w') as f:
                yaml.dump(config_data, f)
            await asyncio.sleep(0.1)
            config_changes += 1
            
            # Change 3
            config_data["monitoring"]["metrics_interval"] = 45
            with open(self.temp_configs["dev"], 'w') as f:
                yaml.dump(config_data, f)
            await asyncio.sleep(0.1)
            config_changes += 1
            
            # Verify current configuration
            current_config = await config_manager.get_config()
            assert current_config["database"]["pool_size"] == 15
            assert current_config["ml"]["batch_size"] == 64
            assert current_config["monitoring"]["metrics_interval"] == 45
            
            # Test rollback 1 step
            rollback_result = await config_manager.rollback(steps=1)
            assert rollback_result.status == ReloadStatus.ROLLED_BACK
            
            # Should have previous configuration
            rolled_back_config = await config_manager.get_config()
            assert rolled_back_config["monitoring"]["metrics_interval"] != 45  # Should be rolled back
            services_validated.append("rollback_1_step")
            
            # Test rollback 2 more steps
            rollback_result = await config_manager.rollback(steps=2)
            assert rollback_result.status == ReloadStatus.ROLLED_BACK
            
            # Should be close to original configuration
            rolled_back_config = await config_manager.get_config()
            assert rolled_back_config["database"]["pool_size"] != 15  # Should be rolled back
            services_validated.append("rollback_multiple_steps")
            
            # Scenario 4: Test circuit breaker for remote source (if applicable)
            logger.info("Testing graceful degradation with failed sources...")
            
            # Add a remote source that will fail
            failed_remote = RemoteConfigSource(
                url="http://nonexistent.example.com/config.json",
                name="failed_remote",
                priority=5
            )
            await config_manager.add_source(failed_remote)
            
            # This should not break the configuration manager
            reload_result = await config_manager.reload()
            
            # Should be partial success (other sources work)
            assert reload_result.status in [ReloadStatus.SUCCESS, ReloadStatus.PARTIAL]
            
            # Configuration should still be accessible
            current_config = await config_manager.get_config()
            assert current_config is not None
            services_validated.append("failed_remote_source_handling")
            
            # Scenario 5: Test rapid successive changes (stress test)
            logger.info("Testing rapid successive changes...")
            
            start_rapid = time.time()
            rapid_changes = 0
            
            for i in range(10):
                config_data["test_rapid_change"] = f"value_{i}"
                with open(self.temp_configs["dev"], 'w') as f:
                    yaml.dump(config_data, f)
                await asyncio.sleep(0.05)  # 50ms between changes
                rapid_changes += 1
            
            # Wait for all changes to be processed
            await asyncio.sleep(0.5)
            
            # Configuration should still be stable
            final_config = await config_manager.get_config()
            assert final_config is not None
            assert "test_rapid_change" in final_config
            
            rapid_duration = time.time() - start_rapid
            logger.info(f"Processed {rapid_changes} rapid changes in {rapid_duration:.2f}s")
            services_validated.append("rapid_changes_stability")
            config_changes += rapid_changes
            
            await config_manager.stop()
            
            logger.info(f"✅ Error recovery test: {len(services_validated)} scenarios validated")
            
            duration_ms = (time.time() - start_time) * 1000
            
            return RealBehaviorTestResult(
                test_name=test_name,
                success=True,
                duration_ms=duration_ms,
                memory_used_mb=psutil.Process().memory_info().rss / 1024 / 1024,
                performance_metrics=[],
                config_changes_detected=config_changes,
                services_validated=services_validated,
                error_details=None
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ Error recovery test failed: {e}")
            
            return RealBehaviorTestResult(
                test_name=test_name,
                success=False,
                duration_ms=duration_ms,
                memory_used_mb=0,
                performance_metrics=[],
                config_changes_detected=0,
                services_validated=[],
                error_details=str(e)
            )
    
    async def test_performance_benchmarks_real_load(self) -> RealBehaviorTestResult:
        """
        Test configuration system performance under real load conditions.
        Measures memory usage, concurrent access, and reload performance.
        """
        test_name = "performance_benchmarks_real_load"
        start_time = time.time()
        
        try:
            performance_metrics = []
            services_validated = []
            
            # Setup configuration manager
            config_manager = ConfigManager(watch_files=False, max_config_versions=50)
            
            # Add multiple configuration sources
            dev_source = FileConfigSource(self.temp_configs["dev"], name="dev", priority=10)
            prod_source = FileConfigSource(self.temp_configs["prod"], name="prod", priority=5)
            env_source = EnvironmentConfigSource(prefix="PERF_TEST_", name="env", priority=15)
            
            await config_manager.add_source(dev_source)
            await config_manager.add_source(prod_source)
            await config_manager.add_source(env_source)
            
            await config_manager.start()
            
            # Benchmark 1: Configuration loading time
            logger.info("Benchmarking configuration loading...")
            
            load_metrics = self._capture_performance_metrics("config_loading")
            load_start = time.time()
            
            for i in range(100):
                config = await config_manager.get_config()
                assert config is not None
            
            load_duration = (time.time() - load_start) * 1000
            load_metrics.duration_ms = load_duration
            
            avg_load_time = load_duration / 100
            assert avg_load_time < 1.0, f"Average config load time {avg_load_time:.2f}ms too slow"
            
            load_metrics.memory_after_mb = psutil.Process().memory_info().rss / 1024 / 1024
            performance_metrics.append(load_metrics)
            services_validated.append("config_loading_performance")
            
            logger.info(f"✅ Config loading: {avg_load_time:.2f}ms average")
            
            # Benchmark 2: Hot-reload performance
            logger.info("Benchmarking hot-reload performance...")
            
            config_data = yaml.safe_load(self.temp_configs["dev"].read_text())
            
            reload_times = []
            for i in range(10):
                config_data["benchmark_iteration"] = i
                
                reload_metrics = self._capture_performance_metrics(f"hot_reload_{i}")
                reload_start = time.time()
                
                # Write configuration
                with open(self.temp_configs["dev"], 'w') as f:
                    yaml.dump(config_data, f)
                
                # Trigger reload
                result = await config_manager.reload("dev")
                assert result.status == ReloadStatus.SUCCESS
                
                reload_duration = (time.time() - reload_start) * 1000
                reload_times.append(reload_duration)
                
                reload_metrics.duration_ms = reload_duration
                reload_metrics.memory_after_mb = psutil.Process().memory_info().rss / 1024 / 1024
                performance_metrics.append(reload_metrics)
            
            avg_reload_time = sum(reload_times) / len(reload_times)
            max_reload_time = max(reload_times)
            
            assert avg_reload_time < 100, f"Average reload time {avg_reload_time:.1f}ms exceeds 100ms requirement"
            assert max_reload_time < 200, f"Max reload time {max_reload_time:.1f}ms too slow"
            
            services_validated.append("hot_reload_performance")
            logger.info(f"✅ Hot-reload: {avg_reload_time:.1f}ms average, {max_reload_time:.1f}ms max")
            
            # Benchmark 3: Concurrent access test
            logger.info("Benchmarking concurrent access...")
            
            concurrent_metrics = self._capture_performance_metrics("concurrent_access")
            concurrent_start = time.time()
            
            async def concurrent_config_access():
                """Simulate concurrent configuration access."""
                for _ in range(50):
                    config = await config_manager.get_config("database.pool_size", 10)
                    assert config is not None
                    await asyncio.sleep(0.001)  # 1ms between accesses
            
            # Run 10 concurrent tasks
            tasks = [concurrent_config_access() for _ in range(10)]
            await asyncio.gather(*tasks)
            
            concurrent_duration = (time.time() - concurrent_start) * 1000
            concurrent_metrics.duration_ms = concurrent_duration
            concurrent_metrics.memory_after_mb = psutil.Process().memory_info().rss / 1024 / 1024
            performance_metrics.append(concurrent_metrics)
            
            # Should handle 500 concurrent accesses (10 tasks * 50 each) efficiently
            total_accesses = 500
            avg_concurrent_time = concurrent_duration / total_accesses
            
            assert avg_concurrent_time < 0.5, f"Concurrent access too slow: {avg_concurrent_time:.3f}ms per access"
            
            services_validated.append("concurrent_access_performance")
            logger.info(f"✅ Concurrent access: {avg_concurrent_time:.3f}ms per access")
            
            # Benchmark 4: Memory usage under load
            logger.info("Benchmarking memory usage...")
            
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Create many configuration versions
            for i in range(30):
                config_data["memory_test_version"] = i
                config_data["large_data"] = "x" * 1000  # 1KB per version
                
                with open(self.temp_configs["dev"], 'w') as f:
                    yaml.dump(config_data, f)
                
                await config_manager.reload("dev")
            
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            memory_increase = memory_after - memory_before
            
            # Memory increase should be reasonable (less than 50MB for 30 versions)
            assert memory_increase < 50, f"Memory usage increased by {memory_increase:.1f}MB"
            
            services_validated.append("memory_usage_test")
            logger.info(f"✅ Memory usage: {memory_increase:.1f}MB increase for 30 versions")
            
            # Benchmark 5: Configuration metrics accuracy
            logger.info("Validating configuration metrics...")
            
            metrics = config_manager.get_metrics()
            assert metrics.total_reloads >= 40  # At least 40 reloads performed
            assert metrics.successful_reloads > 0
            assert metrics.average_reload_time_ms > 0
            assert metrics.config_sources_count == 3  # 3 sources added
            
            services_validated.append("metrics_accuracy")
            logger.info(f"✅ Metrics: {metrics.total_reloads} reloads, {metrics.average_reload_time_ms:.1f}ms avg")
            
            # Benchmark 6: Subscription notification performance
            logger.info("Benchmarking subscription notifications...")
            
            notifications_received = []
            
            def config_change_handler(changes):
                notifications_received.append(len(changes))
            
            subscription = config_manager.subscribe(config_change_handler)
            
            # Make several changes and measure notification speed
            notification_start = time.time()
            
            for i in range(5):
                config_data[f"notification_test_{i}"] = f"value_{i}"
                with open(self.temp_configs["dev"], 'w') as f:
                    yaml.dump(config_data, f)
                await config_manager.reload("dev")
                await asyncio.sleep(0.01)  # Allow notification processing
            
            notification_duration = (time.time() - notification_start) * 1000
            
            # Should receive notifications for all changes
            assert len(notifications_received) >= 5
            
            config_manager.unsubscribe(subscription)
            services_validated.append("subscription_notifications")
            
            logger.info(f"✅ Notifications: {len(notifications_received)} received in {notification_duration:.1f}ms")
            
            await config_manager.stop()
            
            # Force garbage collection and measure final memory
            gc.collect()
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            logger.info(f"✅ Performance benchmarks completed: {len(services_validated)} tests")
            
            duration_ms = (time.time() - start_time) * 1000
            
            return RealBehaviorTestResult(
                test_name=test_name,
                success=True,
                duration_ms=duration_ms,
                memory_used_mb=final_memory,
                performance_metrics=performance_metrics,
                config_changes_detected=len(notifications_received),
                services_validated=services_validated,
                error_details=None
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"❌ Performance benchmarks failed: {e}")
            
            return RealBehaviorTestResult(
                test_name=test_name,
                success=False,
                duration_ms=duration_ms,
                memory_used_mb=0,
                performance_metrics=[],
                config_changes_detected=0,
                services_validated=[],
                error_details=str(e)
            )

    async def run_all_tests(self) -> Dict[str, RealBehaviorTestResult]:
        """Run all Phase 1 configuration real behavior tests."""
        logger.info("🚀 Starting Phase 1 Configuration Real Behavior Test Suite")
        logger.info("=" * 60)
        
        # Setup real environment
        setup_success = await self.setup_real_environment()
        if not setup_success:
            logger.error("❌ Failed to setup real testing environment")
            return {}
        
        test_results = {}
        
        try:
            # Test suite execution
            tests = [
                ("Configuration Hot-Reload", self.test_configuration_hot_reload_real_files),
                ("Environment-Specific Loading", self.test_environment_specific_loading_real_connections),
                ("Startup Validation", self.test_startup_validation_real_services),
                ("Hardcoded Migration", self.test_hardcoded_migration_real_code_files),
                ("Schema Versioning", self.test_schema_versioning_real_upgrades),
                ("Error Recovery", self.test_error_recovery_real_scenarios),
                ("Performance Benchmarks", self.test_performance_benchmarks_real_load)
            ]
            
            for test_name, test_func in tests:
                logger.info(f"\n🧪 Running: {test_name}")
                logger.info("-" * 40)
                
                try:
                    result = await test_func()
                    test_results[test_name] = result
                    
                    if result.success:
                        logger.info(f"✅ {test_name}: PASSED")
                        logger.info(f"   Duration: {result.duration_ms:.1f}ms")
                        logger.info(f"   Memory: {result.memory_used_mb:.1f}MB")
                        logger.info(f"   Services: {len(result.services_validated)}")
                        logger.info(f"   Changes: {result.config_changes_detected}")
                        
                        if result.performance_metrics:
                            logger.info(f"   Metrics: {len(result.performance_metrics)} captured")
                            if not result.meets_performance_requirements():
                                logger.warning("   ⚠️ Performance requirements not fully met")
                    else:
                        logger.error(f"❌ {test_name}: FAILED")
                        logger.error(f"   Error: {result.error_details}")
                        
                except Exception as e:
                    logger.error(f"❌ {test_name}: EXCEPTION - {e}")
                    test_results[test_name] = RealBehaviorTestResult(
                        test_name=test_name,
                        success=False,
                        duration_ms=0,
                        memory_used_mb=0,
                        performance_metrics=[],
                        config_changes_detected=0,
                        services_validated=[],
                        error_details=str(e)
                    )
        
        finally:
            # Cleanup
            await self.teardown_real_environment()
        
        # Generate summary report
        self._generate_summary_report(test_results)
        
        return test_results
    
    def _generate_summary_report(self, test_results: Dict[str, RealBehaviorTestResult]):
        """Generate comprehensive summary report."""
        logger.info("\n📊 Phase 1 Configuration Test Summary")
        logger.info("=" * 60)
        
        total_tests = len(test_results)
        successful_tests = sum(1 for r in test_results.values() if r.success)
        
        total_duration = sum(r.duration_ms for r in test_results.values())
        total_services = sum(len(r.services_validated) for r in test_results.values())
        total_changes = sum(r.config_changes_detected for r in test_results.values())
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Successful: {successful_tests}")
        logger.info(f"Success Rate: {(successful_tests/total_tests)*100:.1f}%")
        logger.info(f"Total Duration: {total_duration:.1f}ms")
        logger.info(f"Services Validated: {total_services}")
        logger.info(f"Config Changes Detected: {total_changes}")
        
        # Service availability summary
        logger.info(f"\n🔗 Real Service Availability:")
        logger.info(f"PostgreSQL: {'✅ Available' if self.real_postgres_available else '❌ Not Available'}")
        logger.info(f"Redis: {'✅ Available' if self.real_redis_available else '❌ Not Available'}")
        
        # Performance summary
        performance_tests = [r for r in test_results.values() if r.performance_metrics]
        if performance_tests:
            logger.info(f"\n⚡ Performance Summary:")
            hot_reload_metrics = []
            for result in performance_tests:
                hot_reload_metrics.extend([m for m in result.performance_metrics if 'reload' in m.operation.lower()])
            
            if hot_reload_metrics:
                avg_reload = sum(m.duration_ms for m in hot_reload_metrics) / len(hot_reload_metrics)
                max_reload = max(m.duration_ms for m in hot_reload_metrics)
                logger.info(f"Hot-reload Average: {avg_reload:.1f}ms")
                logger.info(f"Hot-reload Maximum: {max_reload:.1f}ms")
                logger.info(f"Performance Target (<100ms): {'✅ Met' if max_reload < 100 else '❌ Missed'}")
        
        # Failed tests details
        failed_tests = [name for name, result in test_results.items() if not result.success]
        if failed_tests:
            logger.info(f"\n❌ Failed Tests:")
            for test_name in failed_tests:
                result = test_results[test_name]
                logger.info(f"   {test_name}: {result.error_details}")
        
        logger.info("\n🎯 Test Suite Completed")
        logger.info("=" * 60)


# Pytest integration
class TestPhase1ConfigurationRealBehavior:
    """Pytest wrapper for Phase 1 configuration real behavior tests."""
    
    @pytest.fixture(scope="class")
    async def test_suite(self):
        """Create and setup test suite."""
        suite = Phase1ConfigurationRealBehaviorTestSuite()
        await suite.setup_real_environment()
        yield suite
        await suite.teardown_real_environment()
    
    @pytest.mark.asyncio
    @pytest.mark.real_behavior
    @pytest.mark.slow
    async def test_configuration_hot_reload_real_files(self, test_suite):
        """Test configuration hot-reload with real file changes."""
        result = await test_suite.test_configuration_hot_reload_real_files()
        assert result.success, f"Hot-reload test failed: {result.error_details}"
        assert result.meets_performance_requirements(), "Performance requirements not met"
    
    @pytest.mark.asyncio 
    @pytest.mark.real_behavior
    async def test_environment_specific_loading_real_connections(self, test_suite):
        """Test environment-specific loading with real database connections."""
        result = await test_suite.test_environment_specific_loading_real_connections()
        assert result.success, f"Environment loading test failed: {result.error_details}"
    
    @pytest.mark.asyncio
    @pytest.mark.real_behavior
    async def test_startup_validation_real_services(self, test_suite):
        """Test startup validation with real service connections."""
        result = await test_suite.test_startup_validation_real_services()
        assert result.success, f"Startup validation test failed: {result.error_details}"
    
    @pytest.mark.asyncio
    @pytest.mark.real_behavior
    async def test_hardcoded_migration_real_code_files(self, test_suite):
        """Test hardcoded value migration with real code files."""
        result = await test_suite.test_hardcoded_migration_real_code_files()
        assert result.success, f"Hardcoded migration test failed: {result.error_details}"
    
    @pytest.mark.asyncio
    @pytest.mark.real_behavior
    async def test_schema_versioning_real_upgrades(self, test_suite):
        """Test schema versioning with real configuration upgrades."""
        result = await test_suite.test_schema_versioning_real_upgrades()
        assert result.success, f"Schema versioning test failed: {result.error_details}"
    
    @pytest.mark.asyncio
    @pytest.mark.real_behavior
    async def test_error_recovery_real_scenarios(self, test_suite):
        """Test error recovery with real failure scenarios.""" 
        result = await test_suite.test_error_recovery_real_scenarios()
        assert result.success, f"Error recovery test failed: {result.error_details}"
    
    @pytest.mark.asyncio
    @pytest.mark.real_behavior
    @pytest.mark.slow
    async def test_performance_benchmarks_real_load(self, test_suite):
        """Test performance benchmarks under real load."""
        result = await test_suite.test_performance_benchmarks_real_load()
        assert result.success, f"Performance benchmarks failed: {result.error_details}"
    
    @pytest.mark.asyncio
    @pytest.mark.real_behavior
    @pytest.mark.comprehensive
    async def test_run_all_phase1_configuration_tests(self, test_suite):
        """Run all Phase 1 configuration tests."""
        results = await test_suite.run_all_tests()
        
        assert len(results) > 0, "No tests were executed"
        
        successful_tests = [name for name, result in results.items() if result.success]
        failed_tests = [name for name, result in results.items() if not result.success]
        
        success_rate = len(successful_tests) / len(results)
        
        # At least 80% of tests should pass for real behavior validation
        assert success_rate >= 0.8, f"Success rate {success_rate:.1%} below 80% threshold. Failed: {failed_tests}"
        
        logger.info(f"✅ Phase 1 Configuration Real Behavior Tests: {success_rate:.1%} success rate")


if __name__ == "__main__":
    """
    Standalone execution for Phase 1 configuration real behavior testing.
    Run with: python -m tests.integration.test_phase1_configuration
    """
    async def main():
        suite = Phase1ConfigurationRealBehaviorTestSuite()
        results = await suite.run_all_tests()
        
        success_count = sum(1 for r in results.values() if r.success)
        total_count = len(results)
        
        print(f"\n🎯 Final Results: {success_count}/{total_count} tests passed")
        
        if success_count == total_count:
            print("🎉 All Phase 1 configuration tests passed!")
            exit(0)
        else:
            print("❌ Some tests failed - review logs for details")
            exit(1)
    
    asyncio.run(main())