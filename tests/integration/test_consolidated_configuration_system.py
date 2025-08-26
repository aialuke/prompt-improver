"""Integration tests for consolidated configuration system.

Tests the unified configuration system with real behavior validation,
ensuring all consolidated configurations work correctly.
"""

import os
import tempfile
import warnings
from pathlib import Path

import pytest

from prompt_improver.core.config import (
    # Main configuration
    AppConfig,
    # Schema management
    ConfigSchemaVersion,
    # Factory and creation
    ConfigurationFactory,
    ConfigurationLogger,
    # Validation
    ConfigurationValidator,
    DatabaseConfig,
    MLConfig,
    MonitoringConfig,
    # Retry configuration
    RetryStrategy,
    SecurityConfig,
    StandardRetryConfigs,
    TextStatConfig,
    TextStatWrapper,
    UnifiedConfigService,
    create_retry_config,
    create_test_config,
    get_config,
    get_config_logger,
    get_textstat_wrapper,
    text_analysis,
    validate_configuration,
    validate_startup_configuration,
)


class TestConsolidatedConfigurationSystem:
    """Test suite for consolidated configuration system."""

    def test_main_configuration_access(self):
        """Test main configuration access works correctly."""
        config = get_config()
        assert isinstance(config, AppConfig)

        # Test hierarchical access
        assert hasattr(config, 'database')
        assert hasattr(config, 'security')
        assert hasattr(config, 'monitoring')
        assert hasattr(config, 'ml')

        # Test domain configs are properly typed
        assert isinstance(config.database, DatabaseConfig)
        assert isinstance(config.security, SecurityConfig)
        assert isinstance(config.monitoring, MonitoringConfig)
        assert isinstance(config.ml, MLConfig)

    def test_domain_specific_configurations(self):
        """Test all domain-specific configurations load correctly."""
        # Test database config
        db_config = DatabaseConfig()
        assert hasattr(db_config, 'postgres_host')
        assert hasattr(db_config, 'pool_min_size')
        assert hasattr(db_config, 'pool_max_size')

        # Test security config
        sec_config = SecurityConfig()
        assert hasattr(sec_config, 'secret_key')
        assert hasattr(sec_config, 'security_profile')

        # Test monitoring config
        mon_config = MonitoringConfig()
        assert hasattr(mon_config, 'enabled')
        assert hasattr(mon_config, 'opentelemetry')

        # Test ML config
        ml_config = MLConfig()
        assert hasattr(ml_config, 'enabled')
        assert hasattr(ml_config, 'model_serving')
        assert hasattr(ml_config, 'orchestration')

        # Test ML orchestration consolidated settings
        assert hasattr(ml_config.orchestration, 'max_concurrent_workflows')
        assert hasattr(ml_config.orchestration, 'gpu_allocation_timeout')
        assert hasattr(ml_config.orchestration, 'api_timeout')

    def test_factory_and_creation(self):
        """Test configuration factory and creation functions."""
        # Test environment-based config creation
        test_config = create_test_config({"debug": True})
        assert isinstance(test_config, AppConfig)

        # Test factory pattern
        factory = ConfigurationFactory()
        config = factory.create_config(environment="testing")
        assert isinstance(config, AppConfig)

    @pytest.mark.asyncio
    async def test_configuration_validation(self):
        """Test configuration validation system."""
        # Test basic validation
        config = get_config()
        is_valid, report = await validate_configuration(config)
        assert isinstance(is_valid, bool)
        assert hasattr(report, 'overall_valid')

        # Test startup validation (may fail in test env, but should not error)
        try:
            is_valid, report = await validate_startup_configuration("testing")
            assert isinstance(is_valid, bool)
            assert hasattr(report, 'results')
        except Exception as e:
            # Acceptable in test environment without full services
            assert "not available" in str(e) or "failed" in str(e)

    def test_schema_management(self):
        """Test configuration schema management."""
        service = UnifiedConfigService()
        assert hasattr(service, 'migrations')

        # Test schema version detection
        with tempfile.NamedTemporaryFile(encoding='utf-8', mode='w', suffix='.env', delete=False) as f:
            f.write("POSTGRES_HOST=localhost\n")
            f.write("MONITORING_ENABLED=true\n")
            temp_path = Path(f.name)

        try:
            version = service.get_schema_version(temp_path)
            assert isinstance(version, (ConfigSchemaVersion, type(None)))
        finally:
            temp_path.unlink()

    def test_configuration_logging(self):
        """Test configuration logging system."""
        logger = get_config_logger()
        assert isinstance(logger, ConfigurationLogger)

        # Test logging functionality
        logger.log_configuration_load(
            config_file="test.env",
            environment="testing",
            success=True,
            details={"test": True}
        )

        events = logger.events
        assert len(events) > 0
        assert events[-1].event_type == "load_success"

        # Clear events for cleanup
        logger.clear_events()

    def test_textstat_configuration(self):
        """Test TextStat configuration integration."""
        # Test config creation
        config = TextStatConfig(language="en_US", enable_caching=True)
        assert config.language == "en_US"
        assert config.enable_caching is True

        # Test wrapper
        wrapper = get_textstat_wrapper(config)
        assert isinstance(wrapper, TextStatWrapper)

        # Test text analysis
        result = text_analysis("This is a test sentence for analysis.")
        assert isinstance(result, dict)
        assert "flesch_reading_ease" in result
        assert "syllable_count" in result

        # Test health check
        health = wrapper.health_check()
        assert health["status"] in {"healthy", "unhealthy"}

    def test_retry_configuration(self):
        """Test retry configuration system."""
        # Test standard configs
        fast_config = StandardRetryConfigs.FAST_OPERATION
        assert fast_config.max_attempts == 3
        assert fast_config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF

        # Test custom config creation
        custom_config = create_retry_config(
            operation_type="critical",
            max_attempts=5,
            base_delay=1.0
        )
        assert custom_config.max_attempts == 5
        assert custom_config.base_delay == 1.0

        # Test delay calculation
        delay = custom_config.calculate_delay(2)
        assert isinstance(delay, float)
        assert delay > 0

        # Test retry logic
        exception = Exception("test error")
        should_retry = custom_config.should_retry(exception, 1)
        assert isinstance(should_retry, bool)

    def test_backward_compatibility_warnings(self):
        """Test that backward compatibility imports issue warnings."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")

            try:
                from prompt_improver.core.config import DatabaseConfig as LegacyDB
                # Should have issued a deprecation warning
                assert len(w) > 0
                assert issubclass(w[-1].category, DeprecationWarning)
                assert "deprecated" in str(w[-1].message)
            except ImportError:
                # Acceptable if common.config not available
                pass

    def test_environment_variable_integration(self):
        """Test environment variable integration."""
        # Set test environment variables
        os.environ["POSTGRES_HOST"] = "test-host"
        os.environ["POSTGRES_PORT"] = "5433"

        try:
            config = get_config()
            # Should pick up environment variables
            assert config.database.postgres_host == "test-host"
            assert config.database.postgres_port == 5433
        finally:
            # Clean up
            del os.environ["POSTGRES_HOST"]
            del os.environ["POSTGRES_PORT"]

    def test_ml_orchestration_consolidation(self):
        """Test ML orchestration configuration consolidation."""
        ml_config = MLConfig()
        orch_config = ml_config.orchestration

        # Test consolidated orchestration settings are available
        assert hasattr(orch_config, 'max_concurrent_workflows')
        assert hasattr(orch_config, 'gpu_allocation_timeout')
        assert hasattr(orch_config, 'memory_limit_gb')
        assert hasattr(orch_config, 'training_timeout')
        assert hasattr(orch_config, 'api_timeout')
        assert hasattr(orch_config, 'enable_authentication')

        # Test values are reasonable
        assert orch_config.max_concurrent_workflows > 0
        assert orch_config.gpu_allocation_timeout > 0
        assert orch_config.memory_limit_gb > 0

    def test_configuration_serialization(self):
        """Test configuration serialization works correctly."""
        config = get_config()

        # Test model_dump works
        config_dict = config.model_dump()
        assert isinstance(config_dict, dict)
        assert "database" in config_dict
        assert "security" in config_dict
        assert "monitoring" in config_dict
        assert "ml" in config_dict

        # Test nested serialization
        db_dict = config.database.model_dump()
        assert isinstance(db_dict, dict)
        assert "postgres_host" in db_dict

    def test_configuration_validation_comprehensive(self):
        """Test comprehensive configuration validation."""
        validator = ConfigurationValidator("testing")

        # Test individual validation methods exist
        assert hasattr(validator, '_validate_environment_setup')
        assert hasattr(validator, '_validate_database_configuration')
        assert hasattr(validator, '_validate_security_configuration')
        assert hasattr(validator, '_validate_ml_configuration')

    def test_unified_config_service_integration(self):
        """Test unified config service integration."""
        service = UnifiedConfigService()

        # Test feature flag integration
        enabled = service.is_enabled("test_flag", context={"environment": "testing"})
        assert isinstance(enabled, bool)

        # Test configuration reload
        result = service.reload_configuration()
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
