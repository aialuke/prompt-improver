#!/usr/bin/env python3
"""
Quick validation script for Phase 1 Configuration System setup.

This script performs basic validation to ensure the configuration system
is properly set up and ready for comprehensive testing.

Usage:
    python scripts/validate_phase1_config_setup.py
"""

import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

async def validate_imports():
    """Validate that all required modules can be imported."""
    print("🔍 Validating imports...")
    
    try:
        from prompt_improver.core.config_manager import ConfigManager
        print("✅ ConfigManager import successful")
        
        from prompt_improver.core.config_schema import ConfigSchemaManager
        print("✅ ConfigSchemaManager import successful")
        
        from prompt_improver.core.config_validator import ConfigurationValidator
        print("✅ ConfigurationValidator import successful")
        
        from scripts.migrate_hardcoded_config import HardcodedConfigMigrator
        print("✅ HardcodedConfigMigrator import successful")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

async def validate_basic_functionality():
    """Validate basic configuration system functionality."""
    print("\n🧪 Validating basic functionality...")
    
    try:
        from prompt_improver.core.config_manager import ConfigManager
        from prompt_improver.core.config_schema import ConfigSchemaManager
        from prompt_improver.core.config_validator import ConfigurationValidator
        from scripts.migrate_hardcoded_config import HardcodedConfigMigrator
        
        # Test ConfigManager instantiation
        config_manager = ConfigManager(watch_files=False)
        print("✅ ConfigManager instantiation successful")
        
        # Test schema manager
        schema_manager = ConfigSchemaManager()
        latest_version = schema_manager.schema_manager if hasattr(schema_manager, 'schema_manager') else "Available"
        print(f"✅ ConfigSchemaManager instantiation successful - {latest_version}")
        
        # Test validator
        validator = ConfigurationValidator()
        print("✅ ConfigurationValidator instantiation successful")
        
        # Test hardcoded migrator
        migrator = HardcodedConfigMigrator(".")
        print("✅ HardcodedConfigMigrator instantiation successful")
        
        return True
    except Exception as e:
        print(f"❌ Functionality error: {e}")
        return False

async def validate_dependencies():
    """Validate that required dependencies are available."""
    print("\n📦 Validating dependencies...")
    
    try:
        import yaml
        print("✅ PyYAML available")
        
        import psycopg
        print("✅ psycopg available")
        
        import coredis
        print("✅ coredis available")
        
        import watchdog
        print("✅ watchdog available")
        
        import psutil
        print("✅ psutil available")
        
        import aiohttp
        print("✅ aiohttp available")
        
        return True
    except ImportError as e:
        print(f"❌ Dependency missing: {e}")
        return False

async def validate_environment():
    """Validate environment configuration."""
    print("\n🌍 Validating environment...")
    
    # Check for basic environment variables
    required_vars = [
        "POSTGRES_HOST", "POSTGRES_PORT", "POSTGRES_DATABASE",
        "POSTGRES_USERNAME", "POSTGRES_PASSWORD"
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"⚠️  Missing environment variables: {', '.join(missing_vars)}")
        print("   These can be set for testing or will use defaults")
    else:
        print("✅ All required environment variables present")
    
    # Check optional environment variables
    optional_vars = {
        "REDIS_URL": "redis://localhost:6379/15",
        "ML_MODEL_PATH": "./models",
        "MONITORING_ENABLED": "true"
    }
    
    for var, default in optional_vars.items():
        value = os.getenv(var, default)
        print(f"   {var}: {value}")
    
    return True

async def test_basic_config_operations():
    """Test basic configuration operations."""
    print("\n⚙️  Testing basic configuration operations...")
    
    try:
        from prompt_improver.core.config_manager import (
            ConfigManager, FileConfigSource, EnvironmentConfigSource
        )
        
        # Create temporary config file
        import tempfile
        import json
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            config_data = {
                "test": True,
                "database": {"host": "localhost", "port": 5432},
                "redis": {"url": "redis://localhost:6379"}
            }
            json.dump(config_data, f)
            temp_config_path = f.name
        
        try:
            # Test file source
            file_source = FileConfigSource(temp_config_path, name="test")
            config_data_loaded = await file_source.load_config()
            assert config_data_loaded["test"] is True
            print("✅ File configuration source working")
            
            # Test environment source
            os.environ["TEST_CONFIG_VAR"] = "test_value"
            env_source = EnvironmentConfigSource(prefix="TEST_", name="test_env")
            env_data = await env_source.load_config()
            assert "config_var" in env_data
            print("✅ Environment configuration source working")
            
            # Test config manager
            config_manager = ConfigManager(watch_files=False)
            await config_manager.add_source(file_source)
            await config_manager.start()
            
            config = await config_manager.get_config()
            assert config is not None
            assert config["test"] is True
            print("✅ Configuration manager working")
            
            await config_manager.stop()
            
        finally:
            # Cleanup
            os.unlink(temp_config_path)
            os.environ.pop("TEST_CONFIG_VAR", None)
        
        return True
    except Exception as e:
        print(f"❌ Configuration operations error: {e}")
        return False

async def validate_test_infrastructure():
    """Validate test infrastructure setup."""
    print("\n🧪 Validating test infrastructure...")
    
    try:
        import pytest
        print("✅ pytest available")
        
        # Check test file exists
        test_file = project_root / "tests" / "integration" / "test_phase1_configuration.py"
        if test_file.exists():
            print("✅ Main test file exists")
        else:
            print("❌ Main test file missing")
            return False
        
        # Check test runner exists
        runner_file = project_root / "scripts" / "run_phase1_config_tests.py"
        if runner_file.exists():
            print("✅ Test runner script exists")
        else:
            print("❌ Test runner script missing")
            return False
        
        # Check test directory structure
        test_reports_dir = project_root / "test_reports"
        if not test_reports_dir.exists():
            test_reports_dir.mkdir(parents=True)
            print("✅ Created test reports directory")
        else:
            print("✅ Test reports directory exists")
        
        return True
    except ImportError as e:
        print(f"❌ Test infrastructure error: {e}")
        return False

async def main():
    """Main validation routine."""
    print("🚀 Phase 1 Configuration System Setup Validation")
    print("=" * 55)
    
    validation_results = []
    
    # Run all validations
    validations = [
        ("Imports", validate_imports),
        ("Dependencies", validate_dependencies),
        ("Environment", validate_environment),
        ("Basic Functionality", validate_basic_functionality),
        ("Config Operations", test_basic_config_operations),
        ("Test Infrastructure", validate_test_infrastructure)
    ]
    
    for name, validation_func in validations:
        try:
            result = await validation_func()
            validation_results.append((name, result))
        except Exception as e:
            print(f"❌ {name} validation failed with exception: {e}")
            validation_results.append((name, False))
    
    # Summary
    print("\n📊 Validation Summary")
    print("=" * 25)
    
    passed = 0
    total = len(validation_results)
    
    for name, result in validation_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{name}: {status}")
        if result:
            passed += 1
    
    success_rate = (passed / total) * 100
    print(f"\nOverall: {passed}/{total} validations passed ({success_rate:.1f}%)")
    
    if success_rate >= 85:
        print("\n🎉 Phase 1 Configuration System is ready for comprehensive testing!")
        print("\nNext steps:")
        print("  1. Run quick test: python scripts/run_phase1_config_tests.py --test-type hot-reload")
        print("  2. Run all tests: python scripts/run_phase1_config_tests.py")
        print("  3. Run with pytest: pytest tests/integration/test_phase1_configuration.py -v")
        return 0
    elif success_rate >= 70:
        print("\n⚠️  Phase 1 Configuration System has some issues but may work with limitations")
        print("Review the failed validations above and install missing dependencies")
        return 1
    else:
        print("\n❌ Phase 1 Configuration System is not ready for testing")
        print("Please resolve the validation failures before running tests")
        return 2

if __name__ == "__main__":
    exit(asyncio.run(main()))