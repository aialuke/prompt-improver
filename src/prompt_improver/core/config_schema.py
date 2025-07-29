"""
Configuration schema versioning system for future updates.
Ensures backward compatibility and smooth transitions between configuration versions.
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, Tuple
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

class ConfigSchemaVersion(Enum):
    """Enumeration of configuration schema versions."""
    V1_0 = "1.0"  # Initial version
    V1_1 = "1.1"  # Added ML configuration
    V1_2 = "1.2"  # Added monitoring and observability
    V2_0 = "2.0"  # Major restructure with environment-specific configs
    
    @classmethod
    def latest(cls) -> "ConfigSchemaVersion":
        """Get the latest schema version."""
        return cls.V2_0
    
    @classmethod
    def from_string(cls, version_str: str) -> "ConfigSchemaVersion":
        """Create version from string."""
        for version in cls:
            if version.value == version_str:
                return version
        raise ValueError(f"Unknown schema version: {version_str}")

@dataclass
class ConfigMigration:
    """Represents a configuration migration between versions."""
    from_version: ConfigSchemaVersion
    to_version: ConfigSchemaVersion
    description: str
    migration_func: Callable[[Dict[str, Any]], Dict[str, Any]]
    is_breaking: bool = False
    
    def apply(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply this migration to configuration data."""
        logger.info(f"Applying migration {self.from_version.value} -> {self.to_version.value}: {self.description}")
        return self.migration_func(config_data)

class ConfigurationSchema(BaseModel):
    """Pydantic model for configuration schema metadata."""
    schema_version: str = Field(..., description="Configuration schema version")
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    environment: str = Field(..., description="Target environment")
    config_hash: Optional[str] = Field(None, description="Hash of configuration content")
    migration_history: List[str] = Field(default_factory=list, description="Applied migrations")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class ConfigSchemaManager:
    """Manages configuration schema versions and migrations."""
    
    def __init__(self):
        """Initialize schema manager with available migrations."""
        self.migrations: List[ConfigMigration] = []
        self._register_migrations()
    
    def _register_migrations(self) -> None:
        """Register all available migrations."""
        
        # Migration from V1.0 to V1.1: Add ML configuration
        self.migrations.append(ConfigMigration(
            from_version=ConfigSchemaVersion.V1_0,
            to_version=ConfigSchemaVersion.V1_1,
            description="Add ML model configuration",
            migration_func=self._migrate_v1_0_to_v1_1
        ))
        
        # Migration from V1.1 to V1.2: Add monitoring
        self.migrations.append(ConfigMigration(
            from_version=ConfigSchemaVersion.V1_1,
            to_version=ConfigSchemaVersion.V1_2,
            description="Add monitoring and observability configuration",
            migration_func=self._migrate_v1_1_to_v1_2
        ))
        
        # Migration from V1.2 to V2.0: Environment-specific configs
        self.migrations.append(ConfigMigration(
            from_version=ConfigSchemaVersion.V1_2,
            to_version=ConfigSchemaVersion.V2_0,
            description="Restructure to environment-specific configuration files",
            migration_func=self._migrate_v1_2_to_v2_0,
            is_breaking=True
        ))
    
    def get_schema_version(self, config_path: Path) -> Optional[ConfigSchemaVersion]:
        """Detect the schema version of a configuration file."""
        if not config_path.exists():
            return None
        
        try:
            # Look for schema metadata file
            schema_file = config_path.parent / f".{config_path.name}.schema"
            if schema_file.exists():
                schema_data = json.loads(schema_file.read_text())
                return ConfigSchemaVersion.from_string(schema_data["schema_version"])
            
            # Fallback: Analyze configuration content to guess version
            return self._detect_version_from_content(config_path)
            
        except Exception as e:
            logger.warning(f"Could not detect schema version for {config_path}: {e}")
            return None
    
    def _detect_version_from_content(self, config_path: Path) -> ConfigSchemaVersion:
        """Detect version by analyzing configuration content."""
        try:
            content = config_path.read_text()
            
            # V2.0 indicators: environment-specific variables
            if any(var in content for var in ["MONITORING_ENABLED", "FEATURE_FLAG_", "COMPLIANCE_"]):
                return ConfigSchemaVersion.V2_0
            
            # V1.2 indicators: monitoring variables
            if any(var in content for var in ["METRICS_EXPORT_INTERVAL", "TRACING_ENABLED"]):
                return ConfigSchemaVersion.V1_2
            
            # V1.1 indicators: ML variables
            if any(var in content for var in ["ML_MODEL_PATH", "ML_BATCH_SIZE"]):
                return ConfigSchemaVersion.V1_1
            
            # Default to V1.0
            return ConfigSchemaVersion.V1_0
            
        except Exception:
            return ConfigSchemaVersion.V1_0
    
    def migrate_to_latest(self, config_path: Path) -> bool:
        """Migrate configuration to the latest schema version."""
        current_version = self.get_schema_version(config_path)
        if current_version is None:
            logger.error(f"Could not determine schema version for {config_path}")
            return False
        
        latest_version = ConfigSchemaVersion.latest()
        if current_version == latest_version:
            logger.info(f"Configuration {config_path} is already at latest version {latest_version.value}")
            return True
        
        return self.migrate_to_version(config_path, latest_version)
    
    def migrate_to_version(self, config_path: Path, target_version: ConfigSchemaVersion) -> bool:
        """Migrate configuration to a specific schema version."""
        current_version = self.get_schema_version(config_path)
        if current_version is None:
            logger.error(f"Could not determine schema version for {config_path}")
            return False
        
        if current_version == target_version:
            logger.info(f"Configuration {config_path} is already at version {target_version.value}")
            return True
        
        # Find migration path
        migration_path = self._find_migration_path(current_version, target_version)
        if not migration_path:
            logger.error(f"No migration path found from {current_version.value} to {target_version.value}")
            return False
        
        # Load current configuration
        try:
            if config_path.suffix in ['.json']:
                config_data = json.loads(config_path.read_text())
            else:
                # For .env files, convert to dict
                config_data = self._parse_env_file(config_path)
        except Exception as e:
            logger.error(f"Could not load configuration from {config_path}: {e}")
            return False
        
        # Apply migrations
        applied_migrations = []
        for migration in migration_path:
            try:
                config_data = migration.apply(config_data)
                applied_migrations.append(f"{migration.from_version.value}->{migration.to_version.value}")
            except Exception as e:
                logger.error(f"Migration failed: {migration.description}: {e}")
                return False
        
        # Save migrated configuration
        try:
            # Create backup
            backup_path = config_path.with_suffix(f"{config_path.suffix}.backup")
            backup_path.write_text(config_path.read_text())
            logger.info(f"Created backup: {backup_path}")
            
            # Save migrated config
            if config_path.suffix in ['.json']:
                config_path.write_text(json.dumps(config_data, indent=2))
            else:
                self._write_env_file(config_path, config_data)
            
            # Save schema metadata
            schema = ConfigurationSchema(
                schema_version=target_version.value,
                environment=self._detect_environment_from_path(config_path),
                migration_history=applied_migrations
            )
            
            schema_file = config_path.parent / f".{config_path.name}.schema"
            schema_file.write_text(schema.json(indent=2))
            
            logger.info(f"Successfully migrated {config_path} to version {target_version.value}")
            return True
            
        except Exception as e:
            logger.error(f"Could not save migrated configuration: {e}")
            return False
    
    def _find_migration_path(self, from_version: ConfigSchemaVersion, 
                           to_version: ConfigSchemaVersion) -> Optional[List[ConfigMigration]]:
        """Find a path of migrations between two versions."""
        # Simple linear search for now - could be enhanced with graph algorithms
        current_version = from_version
        path = []
        
        while current_version != to_version:
            next_migration = None
            
            for migration in self.migrations:
                if migration.from_version == current_version:
                    next_migration = migration
                    break
            
            if next_migration is None:
                return None  # No migration path found
            
            path.append(next_migration)
            current_version = next_migration.to_version
        
        return path
    
    def _parse_env_file(self, config_path: Path) -> Dict[str, Any]:
        """Parse .env file into dictionary."""
        config_data = {}
        
        for line in config_path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                config_data[key.strip()] = value.strip()
        
        return config_data
    
    def _write_env_file(self, config_path: Path, config_data: Dict[str, Any]) -> None:
        """Write dictionary to .env file format."""
        lines = []
        
        for key, value in sorted(config_data.items()):
            lines.append(f"{key}={value}")
        
        config_path.write_text('\n'.join(lines))
    
    def _detect_environment_from_path(self, config_path: Path) -> str:
        """Detect environment from file path."""
        filename = config_path.name.lower()
        
        if 'production' in filename or 'prod' in filename:
            return 'production'
        elif 'staging' in filename or 'stage' in filename:
            return 'staging'
        elif 'development' in filename or 'dev' in filename:
            return 'development'
        elif 'test' in filename:
            return 'test'
        else:
            return 'unknown'
    
    # Migration functions
    def _migrate_v1_0_to_v1_1(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from V1.0 to V1.1: Add ML configuration."""
        ml_defaults = {
            "ML_MODEL_PATH": "./models",
            "ML_ENABLE_CACHING": "true",
            "ML_CACHE_TTL_SECONDS": "3600",
            "ML_BATCH_SIZE": "32",
            "ML_MAX_WORKERS": "4"
        }
        
        # Add ML configuration if not present
        for key, default_value in ml_defaults.items():
            if key not in config_data:
                config_data[key] = default_value
        
        return config_data
    
    def _migrate_v1_1_to_v1_2(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from V1.1 to V1.2: Add monitoring configuration."""
        monitoring_defaults = {
            "MONITORING_ENABLED": "true",
            "METRICS_EXPORT_INTERVAL_SECONDS": "30",
            "TRACING_ENABLED": "true",
            "HEALTH_CHECK_TIMEOUT_SECONDS": "10"
        }
        
        # Add monitoring configuration if not present
        for key, default_value in monitoring_defaults.items():
            if key not in config_data:
                config_data[key] = default_value
        
        return config_data
    
    def _migrate_v1_2_to_v2_0(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Migrate from V1.2 to V2.0: Environment-specific restructure."""
        v2_defaults = {
            "FEATURE_FLAG_ML_IMPROVEMENTS": "true",
            "FEATURE_FLAG_ADVANCED_CACHING": "true",
            "FEATURE_FLAG_EXPERIMENTAL_RULES": "false",
            "SECURITY_ENABLE_RATE_LIMITING": "true",
            "SECURITY_PASSWORD_MIN_LENGTH": "12"
        }
        
        # Add V2.0 configuration if not present
        for key, default_value in v2_defaults.items():
            if key not in config_data:
                config_data[key] = default_value
        
        return config_data
    
    def validate_schema_compliance(self, config_path: Path) -> Tuple[bool, List[str]]:
        """Validate that configuration complies with its schema version."""
        issues = []
        
        current_version = self.get_schema_version(config_path)
        if current_version is None:
            issues.append("Could not determine schema version")
            return False, issues
        
        try:
            config_data = self._parse_env_file(config_path)
            
            # Version-specific validation
            if current_version == ConfigSchemaVersion.V2_0:
                issues.extend(self._validate_v2_0_schema(config_data))
            elif current_version == ConfigSchemaVersion.V1_2:
                issues.extend(self._validate_v1_2_schema(config_data))
            # Add more version-specific validations as needed
            
        except Exception as e:
            issues.append(f"Error validating schema: {e}")
        
        return len(issues) == 0, issues
    
    def _validate_v2_0_schema(self, config_data: Dict[str, Any]) -> List[str]:
        """Validate V2.0 schema compliance."""
        issues = []
        
        required_v2_vars = [
            "FEATURE_FLAG_ML_IMPROVEMENTS",
            "SECURITY_ENABLE_RATE_LIMITING",
            "MONITORING_ENABLED"
        ]
        
        for var in required_v2_vars:
            if var not in config_data:
                issues.append(f"Missing required V2.0 variable: {var}")
        
        return issues
    
    def _validate_v1_2_schema(self, config_data: Dict[str, Any]) -> List[str]:
        """Validate V1.2 schema compliance."""
        issues = []
        
        required_v1_2_vars = [
            "MONITORING_ENABLED",
            "ML_MODEL_PATH"
        ]
        
        for var in required_v1_2_vars:
            if var not in config_data:
                issues.append(f"Missing required V1.2 variable: {var}")
        
        return issues

# Global schema manager instance
schema_manager = ConfigSchemaManager()

def migrate_configuration(config_path: str, target_version: Optional[str] = None) -> bool:
    """
    Convenience function to migrate configuration file.
    
    Args:
        config_path: Path to configuration file
        target_version: Target schema version (defaults to latest)
        
    Returns:
        True if migration successful, False otherwise
    """
    path = Path(config_path)
    
    if target_version:
        target = ConfigSchemaVersion.from_string(target_version)
        return schema_manager.migrate_to_version(path, target)
    else:
        return schema_manager.migrate_to_latest(path)

if __name__ == "__main__":
    """CLI entry point for schema management."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Manage configuration schema versions")
    parser.add_argument("config_file", help="Configuration file to process")
    parser.add_argument("--migrate", action="store_true", help="Migrate to latest version")
    parser.add_argument("--target-version", help="Target version for migration")
    parser.add_argument("--validate", action="store_true", help="Validate schema compliance")
    parser.add_argument("--show-version", action="store_true", help="Show current schema version")
    
    args = parser.parse_args()
    
    config_path = Path(args.config_file)
    
    if args.show_version:
        version = schema_manager.get_schema_version(config_path)
        print(f"Schema version: {version.value if version else 'unknown'}")
    
    if args.validate:
        is_valid, issues = schema_manager.validate_schema_compliance(config_path)
        if is_valid:
            print("✅ Schema validation passed")
        else:
            print("❌ Schema validation failed:")
            for issue in issues:
                print(f"  - {issue}")
    
    if args.migrate:
        if args.target_version:
            success = migrate_configuration(str(config_path), args.target_version)
        else:
            success = migrate_configuration(str(config_path))
        
        if success:
            print("✅ Migration completed successfully")
        else:
            print("❌ Migration failed")