"""
Configuration Migration Example
Demonstrates how to migrate from old database config to new centralized config system.

This file shows examples of how to update existing code to use the new
centralized Pydantic configuration system.
"""

import asyncio
import logging

# New centralized configuration system
from prompt_improver.core.config import (
    get_config,
    reload_config,
    add_config_reload_callback,
    config_lifespan,
    validate_config_environment,
    get_config_summary
)

# Old database config (for comparison)
from prompt_improver.database.config import get_database_config

logger = logging.getLogger(__name__)

# ============================================================================
# BEFORE: Old Configuration Usage Pattern
# ============================================================================

def old_database_setup():
    """Example of old database configuration usage."""
    # OLD WAY - importing specific config
    from prompt_improver.database.config import DatabaseConfig
    
    db_config = DatabaseConfig()
    database_url = db_config.database_url
    pool_settings = {
        'min_size': db_config.pool_min_size,
        'max_size': db_config.pool_max_size,
        'timeout': db_config.pool_timeout
    }
    return database_url, pool_settings

# ============================================================================
# AFTER: New Configuration Usage Pattern
# ============================================================================

def new_configuration_setup():
    """Example of new centralized configuration usage."""
    # NEW WAY - single centralized config
    config = get_config()
    
    # Access database settings
    database_url = config.get_database_url(async_driver=True)
    pool_settings = {
        'min_size': config.database.pool_min_size,
        'max_size': config.database.pool_max_size,
        'timeout': config.database.pool_timeout
    }
    
    # Access Redis settings
    redis_url = config.get_redis_url()
    redis_settings = {
        'max_connections': config.redis.max_connections,
        'timeout': config.redis.connection_timeout
    }
    
    # Access API settings
    api_settings = {
        'host': config.api.host,
        'port': config.api.port,
        'rate_limit_enabled': config.api.rate_limit_enabled
    }
    
    # Check environment-specific features
    if config.environment.is_production:
        logger.info("Running in production mode")
        if not config.api.enable_https:
            logger.warning("HTTPS not enabled in production")
    
    return {
        'database_url': database_url,
        'redis_url': redis_url,
        'pool_settings': pool_settings,
        'redis_settings': redis_settings,
        'api_settings': api_settings
    }

# ============================================================================
# Advanced Usage Examples
# ============================================================================

async def application_startup_example():
    """Example of using configuration in application startup."""
    # Load and validate configuration
    config = get_config()
    
    # Validate startup requirements
    validation_issues = config.validate_startup_requirements()
    if validation_issues:
        for issue in validation_issues:
            logger.error(f"Startup validation error: {issue}")
        raise RuntimeError("Configuration validation failed")
    
    # Environment validation
    env_validation = validate_config_environment()
    if not env_validation["valid"]:
        for error in env_validation["errors"]:
            logger.error(f"Environment validation error: {error}")
        raise RuntimeError("Environment validation failed")
    
    # Log warnings and recommendations
    for warning in env_validation["warnings"]:
        logger.warning(f"Configuration warning: {warning}")
    
    for rec in env_validation["recommendations"]:
        logger.info(f"Recommendation: {rec}")
    
    # Log configuration summary (safe for logs)
    config_summary = get_config_summary()
    logger.info(f"Configuration loaded: {config_summary}")
    
    # Setup feature flags
    features = {
        'rate_limiting': config.is_feature_enabled('rate_limiting'),
        'metrics': config.is_feature_enabled('metrics'),
        'model_serving': config.is_feature_enabled('model_serving'),
    }
    logger.info(f"Feature flags: {features}")

def configuration_hot_reload_example():
    """Example of setting up configuration hot-reload."""
    
    def on_config_reload(new_config):
        """Callback function called when configuration is reloaded."""
        logger.info("Configuration reloaded!")
        logger.info(f"New environment: {new_config.environment.environment}")
        
        # Update application components based on new config
        if new_config.environment.is_production != get_config().environment.is_production:
            logger.warning("Environment type changed - may require restart")
        
        # Example: Update rate limiting if it changed
        if new_config.api.rate_limit_enabled != get_config().api.rate_limit_enabled:
            logger.info(f"Rate limiting changed to: {new_config.api.rate_limit_enabled}")
    
    # Add callback for configuration changes
    add_config_reload_callback(on_config_reload)
    
    # Manual reload example
    try:
        updated_config = reload_config()
        logger.info("Configuration manually reloaded")
    except Exception as e:
        logger.error(f"Failed to reload configuration: {e}")

async def async_configuration_lifespan_example():
    """Example of using configuration with async lifespan management."""
    
    async with config_lifespan() as config:
        logger.info(f"Application started with environment: {config.environment.environment}")
        
        # Initialize database connection
        db_url = config.get_database_url()
        logger.info(f"Connecting to database: {config.database.host}:{config.database.port}")
        
        # Initialize Redis connection
        redis_url = config.get_redis_url()
        logger.info(f"Connecting to Redis: {config.redis.host}:{config.redis.port}")
        
        # Initialize ML models if enabled
        if config.ml.model_serving_enabled:
            logger.info(f"Loading ML models from: {config.ml.model_storage_path}")
        
        # Start health checks
        logger.info(f"Starting health checks every {config.health.health_check_interval}s")
        
        # Simulate application running
        await asyncio.sleep(1)
        
        logger.info("Application shutting down gracefully")

# ============================================================================
# Migration Helpers
# ============================================================================

def compare_old_vs_new_config():
    """Compare old and new configuration systems."""
    print("=== Configuration System Comparison ===\n")
    
    # Old system
    print("OLD SYSTEM:")
    try:
        old_config = get_database_config()
        print(f"  Database URL: {old_config.database_url}")
        print(f"  Pool settings: {old_config.pool_min_size}-{old_config.pool_max_size}")
        print(f"  Echo SQL: {old_config.echo_sql}")
    except Exception as e:
        print(f"  Error loading old config: {e}")
    
    print("\nNEW SYSTEM:")
    try:
        new_config = get_config()
        print(f"  Environment: {new_config.environment.environment}")
        print(f"  Database URL: {new_config.get_database_url()}")
        print(f"  Pool settings: {new_config.database.pool_min_size}-{new_config.database.pool_max_size}")
        print(f"  Redis URL: {new_config.get_redis_url()}")
        print(f"  API: {new_config.api.host}:{new_config.api.port}")
        print(f"  Features enabled: rate_limiting={new_config.is_feature_enabled('rate_limiting')}")
    except Exception as e:
        print(f"  Error loading new config: {e}")

def migration_checklist():
    """Print migration checklist for developers."""
    print("=== Configuration Migration Checklist ===\n")
    
    checklist = [
        "1. Install watchdog dependency: pip install watchdog>=4.0.0",
        "2. Update .env file with new configuration options",
        "3. Replace direct database config imports with centralized config",
        "4. Update application startup to use config validation",
        "5. Add configuration hot-reload callbacks if needed",
        "6. Update health checks to use new health config",
        "7. Test configuration in all environments (dev/staging/prod)",
        "8. Update documentation and deployment scripts",
    ]
    
    for item in checklist:
        print(f"  {item}")
    
    print("\n=== Code Changes Required ===")
    
    examples = [
        "# OLD: from prompt_improver.database.config import DatabaseConfig",
        "# NEW: from prompt_improver.core.config import get_config",
        "",
        "# OLD: db_config = DatabaseConfig()",
        "# NEW: config = get_config()",
        "",
        "# OLD: url = db_config.database_url",
        "# NEW: url = config.get_database_url()",
        "",
        "# NEW: Access Redis, API, ML settings from same config object",
        "# NEW: redis_url = config.get_redis_url()",
        "# NEW: api_port = config.api.port",
        "# NEW: model_timeout = config.ml.inference_timeout",
    ]
    
    for example in examples:
        print(f"  {example}")

# ============================================================================
# Testing and Validation Functions
# ============================================================================

def test_configuration_loading():
    """Test configuration loading and validation."""
    print("=== Configuration Testing ===\n")
    
    try:
        # Load configuration
        config = get_config()
        print("✓ Configuration loaded successfully")
        
        # Test database connection string
        db_url = config.get_database_url()
        print(f"✓ Database URL generated: {db_url[:20]}...")
        
        # Test Redis connection string
        redis_url = config.get_redis_url()
        print(f"✓ Redis URL generated: {redis_url}")
        
        # Test validation
        issues = config.validate_startup_requirements()
        if issues:
            print("⚠ Validation issues found:")
            for issue in issues:
                print(f"    - {issue}")
        else:
            print("✓ No validation issues found")
        
        # Test environment validation
        env_validation = validate_config_environment()
        print(f"✓ Environment validation: {'PASS' if env_validation['valid'] else 'FAIL'}")
        
        if env_validation['warnings']:
            print("⚠ Warnings:")
            for warning in env_validation['warnings']:
                print(f"    - {warning}")
        
        if env_validation['recommendations']:
            print("💡 Recommendations:")
            for rec in env_validation['recommendations']:
                print(f"    - {rec}")
        
        print("\n✓ All tests passed!")
        
    except Exception as e:
        print(f"✗ Configuration test failed: {e}")
        return False
    
    return True

# ============================================================================
# Main Example Runner
# ============================================================================

async def main():
    """Main example runner."""
    print("Prompt Improver Configuration System Examples\n")
    print("=" * 50)
    
    # Compare old vs new
    compare_old_vs_new_config()
    print()
    
    # Show migration checklist
    migration_checklist()
    print()
    
    # Test configuration
    test_result = test_configuration_loading()
    print()
    
    if test_result:
        # Demo application startup
        print("=== Application Startup Demo ===")
        try:
            await application_startup_example()
            print("✓ Startup demo completed successfully")
        except Exception as e:
            print(f"✗ Startup demo failed: {e}")
        
        print()
        
        # Demo configuration hot-reload
        print("=== Hot-Reload Demo ===")
        configuration_hot_reload_example()
        print("✓ Hot-reload callbacks configured")
        
        print()
        
        # Demo async lifespan
        print("=== Async Lifespan Demo ===")
        try:
            await async_configuration_lifespan_example()
            print("✓ Async lifespan demo completed")
        except Exception as e:
            print(f"✗ Async lifespan demo failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())