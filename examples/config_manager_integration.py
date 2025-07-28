"""
Configuration Manager Integration Examples

This file demonstrates how to integrate the new ConfigManager with existing
systems in the Prompt Improver project, showing practical usage patterns
and best practices for SRE operations.

Examples include:
- Integrating with existing DatabaseConfig
- Hot-reload for feature flags replacement
- Multi-environment configuration management
- Monitoring and observability integration
- Production deployment patterns

Author: SRE System (Claude Code)
Date: 2025-07-25
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Dict, Any, Optional

# Existing imports from the project
from prompt_improver.core.config_manager import (
    ConfigManager,
    FileConfigSource,
    EnvironmentConfigSource,
    RemoteConfigSource,
    initialize_config_manager,
    get_config,
    get_config_section
)
from prompt_improver.database.config import DatabaseConfig
from prompt_improver.security.config_validator import validate_security_configuration

logger = logging.getLogger(__name__)


class EnhancedDatabaseConfig(DatabaseConfig):
    """Enhanced database configuration with hot-reload support."""

    _config_manager: Optional[ConfigManager] = None

    @classmethod
    async def initialize_with_hot_reload(
        cls,
        config_files: list[str] = None,
        watch_files: bool = True
    ) -> 'EnhancedDatabaseConfig':
        """Initialize database config with hot-reload capability."""

        # Set up configuration sources
        sources = []

        # Add file sources
        if config_files:
            for config_file in config_files:
                source = FileConfigSource(config_file, priority=5)
                sources.append(source)

        # Add environment source (highest priority)
        env_source = EnvironmentConfigSource(prefix="POSTGRES_", priority=10)
        sources.append(env_source)

        # Initialize config manager
        cls._config_manager = await initialize_config_manager(sources, watch_files)

        # Create initial instance
        return await cls._create_from_config_manager()

    @classmethod
    async def _create_from_config_manager(cls) -> 'EnhancedDatabaseConfig':
        """Create database config instance from config manager."""
        if not cls._config_manager:
            raise RuntimeError("Config manager not initialized")

        # Get database configuration section
        db_config = await get_config_section("database")

        # Merge with environment variables (maintaining existing pattern)
        config_dict = {
            "postgres_host": db_config.get("host", os.getenv("POSTGRES_HOST", "localhost")),
            "postgres_port": db_config.get("port", int(os.getenv("POSTGRES_PORT", "5432"))),
            "postgres_database": db_config.get("database", os.getenv("POSTGRES_DATABASE", "apes_production")),
            "postgres_username": db_config.get("username", os.getenv("POSTGRES_USERNAME", "apes_user")),
            "postgres_password": os.getenv("POSTGRES_PASSWORD"),  # Always from env for security
            "pool_min_size": db_config.get("pool", {}).get("min_connections", 4),
            "pool_max_size": db_config.get("pool", {}).get("max_connections", 16),
            "pool_timeout": db_config.get("pool", {}).get("connection_timeout", 10),
        }

        # Remove None values
        config_dict = {k: v for k, v in config_dict.items() if v is not None}

        return cls(**config_dict)

    async def reload_config(self) -> bool:
        """Reload configuration from sources."""
        if not self._config_manager:
            return False

        try:
            result = await self._config_manager.reload()
            if result.status.value in ["success", "partial"]:
                # Update self with new configuration
                new_instance = await self._create_from_config_manager()

                # Update current instance attributes
                for field_name, field_value in new_instance.__dict__.items():
                    setattr(self, field_name, field_value)

                logger.info(f"Database configuration reloaded successfully in {result.reload_time_ms:.1f}ms")
                return True
            else:
                logger.error(f"Database configuration reload failed: {result.errors}")
                return False

        except Exception as e:
            logger.error(f"Error reloading database configuration: {e}")
            return False


async def setup_production_config_manager() -> ConfigManager:
    """Set up configuration manager for production deployment."""

    # Production configuration sources in priority order
    sources = []

    # 1. Remote configuration service (highest priority)
    remote_config_url = os.getenv("CONFIG_SERVICE_URL")
    if remote_config_url:
        headers = {}
        auth_token = os.getenv("CONFIG_SERVICE_TOKEN")
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        remote_source = RemoteConfigSource(
            url=remote_config_url,
            headers=headers,
            timeout=10,
            priority=20
        )
        sources.append(remote_source)
        logger.info("Added remote configuration source")

    # 2. Local configuration files (medium priority)
    config_dir = Path(os.getenv("CONFIG_DIR", "/etc/prompt-improver"))
    config_files = [
        config_dir / "database.yaml",
        config_dir / "application.yaml",
        config_dir / "features.yaml"
    ]

    for config_file in config_files:
        if config_file.exists():
            source = FileConfigSource(config_file, priority=10)
            sources.append(source)
            logger.info(f"Added file configuration source: {config_file}")

    # 3. Environment variables (lowest priority, but always available)
    env_source = EnvironmentConfigSource(prefix="APP_", priority=5)
    sources.append(env_source)

    # Initialize config manager
    manager = await initialize_config_manager(sources, watch_files=True)

    # Set up configuration change monitoring
    def config_change_handler(changes):
        logger.info(f"Configuration changed: {len(changes)} changes detected")
        for change in changes[:5]:  # Log first 5 changes
            logger.info(f"  {change.change_type.value}: {change.path} = {change.new_value}")

    manager.subscribe(config_change_handler)

    return manager


async def feature_flags_migration_example():
    """Example of migrating from existing feature flags to ConfigManager."""

    # Set up config manager with feature flags configuration
    feature_config_path = Path("config/feature_flags.yaml")
    if not feature_config_path.exists():
        # Create example feature flags config
        feature_config_path.parent.mkdir(exist_ok=True)
        example_config = {
            "features": {
                "advanced_ml_pipeline": {
                    "enabled": True,
                    "rollout_percentage": 50,
                    "description": "Advanced ML pipeline with enhanced optimization"
                },
                "new_ui_components": {
                    "enabled": False,
                    "rollout_percentage": 0,
                    "description": "New UI components for improved user experience"
                },
                "enhanced_security": {
                    "enabled": True,
                    "rollout_percentage": 100,
                    "description": "Enhanced security features"
                }
            }
        }

        import yaml
        with open(feature_config_path, 'w') as f:
            yaml.dump(example_config, f, default_flow_style=False)

    # Initialize config manager
    source = FileConfigSource(feature_config_path)
    manager = await initialize_config_manager([source])

    # Usage examples
    async def is_feature_enabled(feature_name: str, user_id: str = None) -> bool:
        """Check if a feature is enabled for a user."""
        feature_config = await get_config(f"features.{feature_name}")
        if not feature_config:
            return False

        if not feature_config.get("enabled", False):
            return False

        rollout_percentage = feature_config.get("rollout_percentage", 0)
        if rollout_percentage >= 100:
            return True

        if user_id and rollout_percentage > 0:
            # Simple hash-based rollout (in production, use more sophisticated logic)
            import hashlib
            hash_value = int(hashlib.sha256(f"{feature_name}:{user_id}".encode()).hexdigest()[:8], 16)
            user_percentage = (hash_value % 100) + 1
            return user_percentage <= rollout_percentage

        return False

    # Test feature flag functionality
    print("Feature Flag Examples:")
    print(f"Advanced ML Pipeline: {await is_feature_enabled('advanced_ml_pipeline', 'user123')}")
    print(f"New UI Components: {await is_feature_enabled('new_ui_components', 'user123')}")
    print(f"Enhanced Security: {await is_feature_enabled('enhanced_security', 'user123')}")

    return manager


async def monitoring_integration_example():
    """Example of integrating ConfigManager with monitoring systems."""

    # Create a monitoring-aware config manager
    config_file = Path("config/monitoring.yaml")
    config_file.parent.mkdir(exist_ok=True)

    monitoring_config = {
        "monitoring": {
            "prometheus": {
                "enabled": True,
                "port": 9090,
                "metrics_path": "/metrics"
            },
            "logging": {
                "level": "INFO",
                "format": "json",
                "output": "stdout"
            },
            "health_checks": {
                "enabled": True,
                "interval_seconds": 30,
                "timeout_seconds": 5
            }
        },
        "alerts": {
            "config_reload_failures": {
                "enabled": True,
                "threshold": 3,
                "window_minutes": 10
            },
            "slow_config_reloads": {
                "enabled": True,
                "threshold_ms": 100
            }
        }
    }

    import yaml
    with open(config_file, 'w') as f:
        yaml.dump(monitoring_config, f, default_flow_style=False)

    # Initialize with monitoring
    source = FileConfigSource(config_file)
    manager = await initialize_config_manager([source])

    # Monitoring integration
    class ConfigManagerMonitoring:
        def __init__(self, config_manager: ConfigManager):
            self.manager = config_manager
            self.metrics = {
                "reload_count": 0,
                "reload_failures": 0,
                "slow_reloads": 0,
                "last_reload_time": None
            }

            # Subscribe to config changes
            self.manager.subscribe(self._handle_config_change)

        def _handle_config_change(self, changes):
            """Handle configuration changes for monitoring."""
            self.metrics["reload_count"] += 1

            # Check for monitoring configuration changes
            monitoring_changes = [c for c in changes if c.path.startswith("monitoring")]
            if monitoring_changes:
                asyncio.create_task(self._update_monitoring_config())

        async def _update_monitoring_config(self):
            """Update monitoring configuration."""
            monitoring_config = await get_config_section("monitoring")

            # Update logging level
            log_level = monitoring_config.get("logging", {}).get("level", "INFO")
            logging.getLogger().setLevel(getattr(logging, log_level))

            logger.info(f"Updated monitoring configuration: log_level={log_level}")

        async def collect_metrics(self) -> Dict[str, Any]:
            """Collect configuration management metrics."""
            manager_metrics = self.manager.get_metrics()

            return {
                "config_manager": {
                    "total_reloads": manager_metrics.total_reloads,
                    "successful_reloads": manager_metrics.successful_reloads,
                    "failed_reloads": manager_metrics.failed_reloads,
                    "average_reload_time_ms": manager_metrics.average_reload_time_ms,
                    "active_subscriptions": manager_metrics.active_subscriptions,
                    "config_sources_count": manager_metrics.config_sources_count
                },
                "custom_metrics": self.metrics
            }

    # Set up monitoring
    monitoring = ConfigManagerMonitoring(manager)

    # Example: Collect and display metrics
    metrics = await monitoring.collect_metrics()
    print("Configuration Manager Metrics:")
    for category, values in metrics.items():
        print(f"  {category}:")
        for key, value in values.items():
            print(f"    {key}: {value}")

    return manager, monitoring


async def high_availability_config_example():
    """Example of high-availability configuration setup."""

    # Multiple configuration sources for redundancy
    sources = []

    # Primary remote source
    primary_remote = RemoteConfigSource(
        url="https://config-primary.example.com/config.json",
        headers={"Authorization": "Bearer primary-token"},
        priority=30,
        name="primary-remote"
    )
    sources.append(primary_remote)

    # Backup remote source
    backup_remote = RemoteConfigSource(
        url="https://config-backup.example.com/config.json",
        headers={"Authorization": "Bearer backup-token"},
        priority=25,
        name="backup-remote"
    )
    sources.append(backup_remote)

    # Local cache file (always available)
    cache_file = Path("/var/cache/prompt-improver/config.yaml")
    cache_file.parent.mkdir(parents=True, exist_ok=True)

    # Create cache file if it doesn't exist
    if not cache_file.exists():
        default_config = {
            "application": {
                "name": "prompt-improver",
                "version": "1.0.0",
                "debug": False
            },
            "database": {
                "host": "localhost",
                "port": 5432,
                "pool": {
                    "min_connections": 2,
                    "max_connections": 10
                }
            }
        }

        import yaml
        with open(cache_file, 'w') as f:
            yaml.dump(default_config, f)

    cache_source = FileConfigSource(cache_file, priority=20, name="cache-file")
    sources.append(cache_source)

    # Environment variables (fallback)
    env_source = EnvironmentConfigSource(prefix="APP_", priority=10, name="environment")
    sources.append(env_source)

    # Initialize HA config manager
    manager = ConfigManager(watch_files=True)

    for source in sources:
        await manager.add_source(source)

    await manager.start()

    # HA monitoring
    class HAConfigMonitor:
        def __init__(self, config_manager: ConfigManager):
            self.manager = config_manager
            self.source_health = {}

        async def check_source_health(self):
            """Check health of all configuration sources."""
            sources = self.manager.get_sources()

            for source in sources:
                try:
                    # Test if source can be accessed
                    if hasattr(source, 'is_modified'):
                        await source.is_modified()
                    self.source_health[source.name] = True
                except Exception as e:
                    self.source_health[source.name] = False
                    logger.warning(f"Config source {source.name} unhealthy: {e}")

            healthy_sources = sum(1 for status in self.source_health.values() if status)
            logger.info(f"Configuration sources health: {healthy_sources}/{len(sources)} healthy")

            return self.source_health

        async def force_failover(self, failed_source_name: str):
            """Force failover from a failed source."""
            logger.warning(f"Forcing failover from source: {failed_source_name}")

            # In production, you might disable the failed source temporarily
            # and trigger a reload from remaining sources
            result = await self.manager.reload()

            if result.status.value == "success":
                logger.info("Failover successful")
            else:
                logger.error(f"Failover failed: {result.errors}")

    ha_monitor = HAConfigMonitor(manager)
    await ha_monitor.check_source_health()

    return manager, ha_monitor


async def main():
    """Main example runner."""
    print("🔧 Configuration Manager Integration Examples\n")

    try:
        # Example 1: Enhanced Database Configuration
        print("1. Enhanced Database Configuration with Hot-reload")
        print("-" * 50)

        # Create example database config file
        config_dir = Path("config")
        config_dir.mkdir(exist_ok=True)

        db_config_file = config_dir / "database.yaml"
        if not db_config_file.exists():
            import yaml
            example_db_config = {
                "database": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "apes_production",
                    "username": "apes_user",
                    "pool": {
                        "min_connections": 4,
                        "max_connections": 20,
                        "connection_timeout": 30
                    }
                }
            }
            with open(db_config_file, 'w') as f:
                yaml.dump(example_db_config, f)

        # Initialize enhanced database config
        os.environ.setdefault("POSTGRES_PASSWORD", "example_password_123")
        db_config = await EnhancedDatabaseConfig.initialize_with_hot_reload(
            config_files=[str(db_config_file)]
        )

        print(f"Database URL: {db_config.database_url}")
        print(f"Pool settings: min={db_config.pool_min_size}, max={db_config.pool_max_size}")
        print()

        # Example 2: Feature Flags Migration
        print("2. Feature Flags Migration Example")
        print("-" * 50)
        feature_manager = await feature_flags_migration_example()
        print()

        # Example 3: Monitoring Integration
        print("3. Monitoring Integration Example")
        print("-" * 50)
        monitor_manager, monitoring = await monitoring_integration_example()
        print()

        # Example 4: Performance Test
        print("4. Performance Test (<100ms reload requirement)")
        print("-" * 50)

        test_config_file = config_dir / "performance_test.yaml"
        test_config = {"performance": {"test": "initial"}}

        import yaml
        with open(test_config_file, 'w') as f:
            yaml.dump(test_config, f)

        perf_source = FileConfigSource(test_config_file)
        perf_manager = await initialize_config_manager([perf_source], watch_files=False)

        # Measure reload performance
        import time
        reload_times = []

        for i in range(5):
            # Update config
            test_config["performance"]["test"] = f"value_{i}"
            test_config["performance"]["iteration"] = i

            with open(test_config_file, 'w') as f:
                yaml.dump(test_config, f)

            # Measure reload time
            start_time = time.time()
            result = await perf_manager.reload()
            end_time = time.time()

            reload_time_ms = (end_time - start_time) * 1000
            reload_times.append(reload_time_ms)

            print(f"  Reload {i+1}: {reload_time_ms:.1f}ms - {'✅' if reload_time_ms < 100 else '❌'}")

        avg_time = sum(reload_times) / len(reload_times)
        print(f"  Average reload time: {avg_time:.1f}ms")
        print(f"  Performance requirement (<100ms): {'✅ PASSED' if avg_time < 100 else '❌ FAILED'}")

        # Cleanup
        await perf_manager.stop()
        await monitor_manager.stop()
        await feature_manager.stop()

        print("\n✅ All examples completed successfully!")

    except Exception as e:
        logger.error(f"Example failed: {e}")
        print(f"\n❌ Example failed: {e}")
        raise


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Run examples
    asyncio.run(main())
