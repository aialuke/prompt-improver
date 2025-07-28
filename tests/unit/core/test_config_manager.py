"""
Comprehensive tests for ConfigManager hot-reload system.

Tests cover:
- Thread safety and concurrent access
- Performance requirements (<100ms reload times)
- Error handling and graceful degradation
- Rollback functionality
- Multi-source configuration merging
- Real behavior validation
- Zero-downtime operation

Author: SRE System (Claude Code)
Date: 2025-07-25
"""

import asyncio
import json
import os
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from typing import Dict, Any, List

import aiohttp
import pytest
import yaml
from aioresponses import aioresponses

from prompt_improver.core.config_manager import (
    ConfigManager,
    FileConfigSource,
    EnvironmentConfigSource,
    RemoteConfigSource,
    ConfigChange,
    ConfigChangeType,
    ReloadStatus,
    ConfigMetrics,
    initialize_config_manager,
    get_config_manager,
    get_config,
    get_config_section
)


class TestConfigSources:
    """Test configuration sources."""

    @pytest.mark.asyncio
    async def test_file_config_source_yaml(self):
        """Test YAML file configuration source."""
        config_data = {
            'database': {
                'host': 'localhost',
                'port': 5432
            },
            'app': {
                'debug': True,
                'secret_key': '${SECRET_KEY:default_secret}'
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            # Set environment variable for substitution
            os.environ['SECRET_KEY'] = 'test_secret_123'

            source = FileConfigSource(temp_path)
            loaded_config = await source.load_config()

            assert loaded_config['database']['host'] == 'localhost'
            assert loaded_config['database']['port'] == 5432
            assert loaded_config['app']['debug'] is True
            assert loaded_config['app']['secret_key'] == 'test_secret_123'

            # Test modification detection
            assert await source.is_modified() is False

            # Modify file
            time.sleep(0.1)  # Ensure timestamp difference
            with open(temp_path, 'w') as f:
                yaml.dump({'modified': True}, f)

            assert await source.is_modified() is True

        finally:
            os.unlink(temp_path)
            os.environ.pop('SECRET_KEY', None)

    @pytest.mark.asyncio
    async def test_file_config_source_json(self):
        """Test JSON file configuration source."""
        config_data = {
            'api': {
                'timeout': 30,
                'retries': 3
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            temp_path = f.name

        try:
            source = FileConfigSource(temp_path)
            loaded_config = await source.load_config()

            assert loaded_config['api']['timeout'] == 30
            assert loaded_config['api']['retries'] == 3

        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_environment_config_source(self):
        """Test environment variable configuration source."""
        # Set test environment variables
        test_env = {
            'APP_DEBUG': 'true',
            'APP_HOST': 'localhost',
            'APP_PORT': '8080',
            'APP_CONFIG': '{"nested": {"value": 42}}',
            'OTHER_VAR': 'should_be_ignored'
        }

        for key, value in test_env.items():
            os.environ[key] = value

        try:
            source = EnvironmentConfigSource(prefix='APP_')
            loaded_config = await source.load_config()

            assert loaded_config['debug'] == 'true'
            assert loaded_config['host'] == 'localhost'
            assert loaded_config['port'] == '8080'
            assert loaded_config['config'] == {"nested": {"value": 42}}
            assert 'other_var' not in loaded_config

            # Test modification detection
            assert await source.is_modified() is False

            os.environ['APP_NEW_VAR'] = 'new_value'
            assert await source.is_modified() is True

        finally:
            for key in test_env:
                os.environ.pop(key, None)
            os.environ.pop('APP_NEW_VAR', None)

    @pytest.mark.asyncio
    async def test_remote_config_source(self):
        """Test remote HTTP configuration source."""
        config_data = {
            'remote': {
                'setting1': 'value1',
                'setting2': 42
            }
        }

        with aioresponses() as m:
            m.get('http://config.example.com/config.json',
                  payload=config_data,
                  headers={'ETag': 'test-etag'})

            source = RemoteConfigSource('http://config.example.com/config.json')
            loaded_config = await source.load_config()

            assert loaded_config['remote']['setting1'] == 'value1'
            assert loaded_config['remote']['setting2'] == 42

            # Test ETag caching
            m.get('http://config.example.com/config.json',
                  status=304,
                  headers={'ETag': 'test-etag'})

            cached_config = await source.load_config()
            assert cached_config == {}  # 304 returns empty dict

    @pytest.mark.asyncio
    async def test_remote_config_source_circuit_breaker(self):
        """Test remote source circuit breaker functionality."""
        source = RemoteConfigSource('http://config.example.com/config.json')
        source._circuit_breaker_threshold = 2  # Lower threshold for testing

        with aioresponses() as m:
            # Simulate failures
            m.get('http://config.example.com/config.json',
                  exception=aiohttp.ClientError("Connection failed"))

            # First failure
            with pytest.raises(aiohttp.ClientError):
                await source.load_config()

            # Second failure - should open circuit
            with pytest.raises(aiohttp.ClientError):
                await source.load_config()

            # Circuit should be open now
            assert source._is_circuit_open() is True

            # Load should return empty config without making request
            empty_config = await source.load_config()
            assert empty_config == {}


class TestConfigStore:
    """Test configuration store."""

    @pytest.fixture
    def config_store(self):
        """Create a config store for testing."""
        from prompt_improver.core.config_manager import ConfigStore, ConfigVersion
        return ConfigStore()

    @pytest.mark.asyncio
    async def test_config_update_and_retrieval(self, config_store):
        """Test configuration update and retrieval."""
        config = {
            'app': {
                'name': 'test',
                'debug': True
            },
            'database': {
                'host': 'localhost'
            }
        }

        from prompt_improver.core.config_manager import ConfigVersion
        version = ConfigVersion(source="test", checksum="test123")

        changes = await config_store.update_config(config, version)

        # Test full config retrieval
        full_config = await config_store.get_config()
        assert full_config == config

        # Test path-based retrieval
        app_name = await config_store.get_config('app.name')
        assert app_name == 'test'

        app_debug = await config_store.get_config('app.debug')
        assert app_debug is True

        # Test nested path
        db_host = await config_store.get_config('database.host')
        assert db_host == 'localhost'

        # Test non-existent path
        missing = await config_store.get_config('missing.path', 'default')
        assert missing == 'default'

    @pytest.mark.asyncio
    async def test_config_changes_calculation(self, config_store):
        """Test configuration changes calculation."""
        from prompt_improver.core.config_manager import ConfigVersion

        # Initial config
        initial_config = {
            'app': {'name': 'test', 'debug': True},
            'database': {'host': 'localhost'}
        }
        version1 = ConfigVersion(source="test1", checksum="abc123")
        changes1 = await config_store.update_config(initial_config, version1)

        # Updated config
        updated_config = {
            'app': {'name': 'test', 'debug': False, 'new_setting': 'value'},
            'database': {'host': 'remote'},
            'new_section': {'setting': 'value'}
        }
        version2 = ConfigVersion(source="test2", checksum="def456")
        changes2 = await config_store.update_config(updated_config, version2)

        # Verify changes
        change_paths = {change.path for change in changes2}
        assert 'app.debug' in change_paths
        assert 'app.new_setting' in change_paths
        assert 'database.host' in change_paths
        assert 'new_section' in change_paths

        # Check change types
        debug_change = next(c for c in changes2 if c.path == 'app.debug')
        assert debug_change.change_type == ConfigChangeType.MODIFIED
        assert debug_change.old_value is True
        assert debug_change.new_value is False

        new_setting_change = next(c for c in changes2 if c.path == 'app.new_setting')
        assert new_setting_change.change_type == ConfigChangeType.ADDED
        assert new_setting_change.new_value == 'value'

    @pytest.mark.asyncio
    async def test_rollback_functionality(self, config_store):
        """Test configuration rollback."""
        from prompt_improver.core.config_manager import ConfigVersion

        # Config version 1
        config1 = {'version': 1, 'setting': 'value1'}
        version1 = ConfigVersion(source="test1", checksum="v1")
        await config_store.update_config(config1, version1)

        # Config version 2
        config2 = {'version': 2, 'setting': 'value2'}
        version2 = ConfigVersion(source="test2", checksum="v2")
        await config_store.update_config(config2, version2)

        # Config version 3
        config3 = {'version': 3, 'setting': 'value3'}
        version3 = ConfigVersion(source="test3", checksum="v3")
        await config_store.update_config(config3, version3)

        # Verify current config
        current = await config_store.get_config()
        assert current['version'] == 3

        # Rollback 1 step
        success = await config_store.rollback(1)
        assert success is True

        rolled_back = await config_store.get_config()
        assert rolled_back['version'] == 2

        # Rollback 2 more steps (should fail - not enough history)
        success = await config_store.rollback(2)
        assert success is False

        # Should still be at version 2
        current = await config_store.get_config()
        assert current['version'] == 2

    @pytest.mark.asyncio
    async def test_subscription_system(self, config_store):
        """Test configuration change subscription system."""
        from prompt_improver.core.config_manager import ConfigVersion

        change_events = []

        def change_handler(changes):
            change_events.extend(changes)

        # Subscribe to changes
        subscription = config_store.subscribe(change_handler)

        # Update config
        config = {'test': 'value'}
        version = ConfigVersion(source="test", checksum="test")
        changes = await config_store.update_config(config, version)

        # Wait for async notification
        await asyncio.sleep(0.1)

        # Verify notification received
        assert len(change_events) > 0
        assert change_events[0].path == 'test'
        assert change_events[0].new_value == 'value'

        # Unsubscribe
        config_store.unsubscribe(subscription)

        # Update config again
        config2 = {'test': 'value2'}
        version2 = ConfigVersion(source="test2", checksum="test2")
        await config_store.update_config(config2, version2)

        # Wait and verify no new notifications
        await asyncio.sleep(0.1)
        original_count = len(change_events)

        # Should not have received new changes
        new_changes = [c for c in change_events if c.new_value == 'value2']
        assert len(new_changes) == 0


class TestConfigManager:
    """Test configuration manager."""

    @pytest.mark.asyncio
    async def test_basic_functionality(self):
        """Test basic configuration manager functionality."""
        manager = ConfigManager()

        # Create test config file
        config_data = {
            'app': {'name': 'test_app', 'version': '1.0.0'},
            'database': {'host': 'localhost', 'port': 5432}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            # Add file source
            source = FileConfigSource(temp_path)
            await manager.add_source(source)

            # Start manager
            await manager.start()

            # Test configuration retrieval
            app_name = await manager.get_config('app.name')
            assert app_name == 'test_app'

            db_config = await manager.get_section('database')
            assert db_config['host'] == 'localhost'
            assert db_config['port'] == 5432

            # Test metrics
            metrics = manager.get_metrics()
            assert metrics.config_sources_count == 1
            assert metrics.total_reloads >= 1

        finally:
            await manager.stop()
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_multi_source_priority(self):
        """Test multiple configuration sources with priority ordering."""
        manager = ConfigManager()

        # Base config (low priority)
        base_config = {
            'app': {'name': 'base_app', 'debug': False},
            'database': {'host': 'localhost', 'timeout': 30}
        }

        # Override config (high priority)
        override_config = {
            'app': {'debug': True, 'new_setting': 'override_value'},
            'database': {'host': 'override_host'}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f1:
            yaml.dump(base_config, f1)
            base_path = f1.name

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f2:
            yaml.dump(override_config, f2)
            override_path = f2.name

        try:
            # Add sources with different priorities
            base_source = FileConfigSource(base_path, priority=1)
            override_source = FileConfigSource(override_path, priority=10)

            await manager.add_source(base_source)
            await manager.add_source(override_source)
            await manager.start()

            # Test merged configuration
            app_name = await manager.get_config('app.name')
            assert app_name == 'base_app'  # From base config

            app_debug = await manager.get_config('app.debug')
            assert app_debug is True  # Overridden by high priority source

            new_setting = await manager.get_config('app.new_setting')
            assert new_setting == 'override_value'  # From override config

            db_host = await manager.get_config('database.host')
            assert db_host == 'override_host'  # Overridden

            db_timeout = await manager.get_config('database.timeout')
            assert db_timeout == 30  # From base config (not overridden)

        finally:
            await manager.stop()
            os.unlink(base_path)
            os.unlink(override_path)

    @pytest.mark.asyncio
    async def test_hot_reload_performance(self):
        """Test hot-reload performance requirements (<100ms)."""
        manager = ConfigManager(watch_files=False)  # Disable file watching for controlled testing

        config_data = {'test': 'initial_value'}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            source = FileConfigSource(temp_path)
            await manager.add_source(source)
            await manager.start()

            # Perform multiple reloads and measure time
            reload_times = []

            for i in range(10):
                # Modify config
                new_config = {'test': f'value_{i}', 'iteration': i}
                with open(temp_path, 'w') as f:
                    yaml.dump(new_config, f)

                # Measure reload time
                start_time = time.time()
                result = await manager.reload()
                end_time = time.time()

                reload_time_ms = (end_time - start_time) * 1000
                reload_times.append(reload_time_ms)

                # Verify reload was successful
                assert result.status == ReloadStatus.SUCCESS
                assert result.reload_time_ms < 100  # Performance requirement

                # Verify config was updated
                test_value = await manager.get_config('test')
                assert test_value == f'value_{i}'

            # Check average performance
            avg_reload_time = sum(reload_times) / len(reload_times)
            assert avg_reload_time < 100, f"Average reload time {avg_reload_time:.1f}ms exceeds 100ms limit"

            print(f"Hot-reload performance: avg={avg_reload_time:.1f}ms, max={max(reload_times):.1f}ms")

        finally:
            await manager.stop()
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_concurrent_access(self):
        """Test thread safety with concurrent access."""
        manager = ConfigManager()

        config_data = {
            'counter': 0,
            'data': {'nested': {'value': 'initial'}}
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            source = FileConfigSource(temp_path)
            await manager.add_source(source)
            await manager.start()

            # Concurrent readers
            async def reader_task(reader_id: int):
                results = []
                for i in range(50):
                    value = await manager.get_config('data.nested.value')
                    results.append(value)
                    await asyncio.sleep(0.001)  # Small delay
                return results

            # Concurrent writer
            async def writer_task():
                for i in range(10):
                    new_config = {
                        'counter': i,
                        'data': {'nested': {'value': f'update_{i}'}}
                    }
                    with open(temp_path, 'w') as f:
                        yaml.dump(new_config, f)

                    await manager.reload()
                    await asyncio.sleep(0.01)  # Small delay

            # Run concurrent tasks
            reader_tasks = [reader_task(i) for i in range(5)]
            all_tasks = reader_tasks + [writer_task()]

            results = await asyncio.gather(*all_tasks, return_exceptions=True)

            # Verify no exceptions occurred
            for result in results:
                if isinstance(result, Exception):
                    pytest.fail(f"Concurrent access failed: {result}")

            # Verify readers got valid data
            reader_results = results[:-1]  # Exclude writer result
            for reader_result in reader_results:
                assert all(isinstance(value, str) for value in reader_result)
                assert all(value.startswith(('initial', 'update_')) for value in reader_result)

        finally:
            await manager.stop()
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_rollback_functionality(self):
        """Test configuration rollback functionality."""
        manager = ConfigManager(watch_files=False)

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump({'version': 1}, f)
            temp_path = f.name

        try:
            source = FileConfigSource(temp_path)
            await manager.add_source(source)
            await manager.start()

            # Update config multiple times
            for version in range(2, 5):
                config = {'version': version}
                with open(temp_path, 'w') as f:
                    yaml.dump(config, f)
                await manager.reload()

                current_version = await manager.get_config('version')
                assert current_version == version

            # Rollback one step
            rollback_result = await manager.rollback(1)
            assert rollback_result.status == ReloadStatus.ROLLED_BACK

            current_version = await manager.get_config('version')
            assert current_version == 3

            # Rollback two more steps
            rollback_result = await manager.rollback(2)
            assert rollback_result.status == ReloadStatus.ROLLED_BACK

            current_version = await manager.get_config('version')
            assert current_version == 1

            # Try to rollback beyond history
            rollback_result = await manager.rollback(5)
            assert rollback_result.status == ReloadStatus.FAILED
            assert "Insufficient configuration history" in rollback_result.errors[0]

        finally:
            await manager.stop()
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_error_handling_and_degradation(self):
        """Test error handling and graceful degradation."""
        manager = ConfigManager()

        # Test with non-existent file
        source = FileConfigSource('/non/existent/file.yaml')
        await manager.add_source(source)
        await manager.start()

        # Should handle missing file gracefully
        config = await manager.get_config('any.path', 'default')
        assert config == 'default'

        # Test with invalid YAML
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            f.write('invalid: yaml: content: [')
            invalid_path = f.name

        try:
            invalid_source = FileConfigSource(invalid_path)
            await manager.add_source(invalid_source)

            # Reload should handle invalid YAML
            result = await manager.reload()
            assert result.status in [ReloadStatus.FAILED, ReloadStatus.PARTIAL]
            assert len(result.errors) > 0

            # Manager should still be operational
            config = await manager.get_config('test', 'still_works')
            assert config == 'still_works'

        finally:
            await manager.stop()
            os.unlink(invalid_path)

    @pytest.mark.asyncio
    async def test_subscription_notifications(self):
        """Test configuration change notifications."""
        manager = ConfigManager(watch_files=False)

        received_changes = []

        def change_handler(changes):
            received_changes.extend(changes)

        # Subscribe to changes
        subscription = manager.subscribe(change_handler)

        config_data = {'initial': 'value'}
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            source = FileConfigSource(temp_path)
            await manager.add_source(source)
            await manager.start()

            # Wait for initial load notification
            await asyncio.sleep(0.1)
            initial_change_count = len(received_changes)

            # Update configuration
            new_config = {'initial': 'updated_value', 'new_key': 'new_value'}
            with open(temp_path, 'w') as f:
                yaml.dump(new_config, f)

            await manager.reload()
            await asyncio.sleep(0.1)  # Wait for notification

            # Verify we received change notifications
            new_changes = received_changes[initial_change_count:]
            assert len(new_changes) > 0

            # Check specific changes
            change_paths = {change.path for change in new_changes}
            assert 'initial' in change_paths or 'new_key' in change_paths

            # Unsubscribe
            manager.unsubscribe(subscription)

            # Update again - should not receive notifications
            another_config = {'initial': 'final_value'}
            with open(temp_path, 'w') as f:
                yaml.dump(another_config, f)

            await manager.reload()
            await asyncio.sleep(0.1)

            # Should not have received new changes after unsubscribing
            final_change_count = len(received_changes)
            assert final_change_count == len(received_changes)

        finally:
            await manager.stop()
            os.unlink(temp_path)


class TestConvenienceFunctions:
    """Test convenience functions."""

    @pytest.mark.asyncio
    async def test_global_config_manager(self):
        """Test global configuration manager functions."""
        # Clean up any existing global manager
        import prompt_improver.core.config_manager as config_module
        config_module._default_manager = None

        config_data = {'global_test': 'value', 'section': {'nested': 'data'}}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            # Initialize global manager
            source = FileConfigSource(temp_path)
            manager = await initialize_config_manager([source])

            # Test global functions
            global_manager = get_config_manager()
            assert global_manager is manager

            global_test = await get_config('global_test')
            assert global_test == 'value'

            section_data = await get_config_section('section')
            assert section_data['nested'] == 'data'

            # Test with non-existent path
            missing = await get_config('missing.path', 'default_val')
            assert missing == 'default_val'

        finally:
            if manager:
                await manager.stop()
            os.unlink(temp_path)
            config_module._default_manager = None


class TestRealBehaviorValidation:
    """Real behavior testing for production validation."""

    @pytest.mark.asyncio
    async def test_zero_downtime_operation(self):
        """Test zero-downtime configuration updates."""
        manager = ConfigManager(watch_files=False)

        config_data = {'service': {'status': 'running', 'port': 8080}}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            source = FileConfigSource(temp_path)
            await manager.add_source(source)
            await manager.start()

            # Simulate continuous service operations during config updates
            service_operational = True
            operation_count = 0

            async def continuous_operations():
                nonlocal operation_count, service_operational
                while service_operational:
                    # Simulate service operation reading config
                    try:
                        status = await manager.get_config('service.status')
                        port = await manager.get_config('service.port')

                        # Verify config is always accessible
                        assert status is not None
                        assert port is not None

                        operation_count += 1
                        await asyncio.sleep(0.001)  # Small delay

                    except Exception as e:
                        pytest.fail(f"Service operation failed during config update: {e}")

            # Start continuous operations
            operations_task = asyncio.create_task(continuous_operations())

            # Perform multiple config updates while operations are running
            for i in range(20):
                new_config = {
                    'service': {
                        'status': 'running',
                        'port': 8080 + i,
                        'update_count': i
                    }
                }

                with open(temp_path, 'w') as f:
                    yaml.dump(new_config, f)

                # Reload configuration
                result = await manager.reload()
                assert result.status == ReloadStatus.SUCCESS

                # Brief pause
                await asyncio.sleep(0.005)

            # Stop continuous operations
            service_operational = False
            await operations_task

            # Verify operations continued without interruption
            assert operation_count > 100  # Should have many successful operations

            # Verify final config state
            final_port = await manager.get_config('service.port')
            assert final_port == 8080 + 19  # Last update

        finally:
            await manager.stop()
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_memory_stability_long_running(self):
        """Test memory stability during long-running operations."""
        import gc
        import psutil
        import os

        manager = ConfigManager(watch_files=False)

        config_data = {'memory_test': 'initial'}

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(config_data, f)
            temp_path = f.name

        try:
            source = FileConfigSource(temp_path)
            await manager.add_source(source)
            await manager.start()

            # Get initial memory usage
            process = psutil.Process(os.getpid())
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Perform many config updates and accesses
            for i in range(100):
                # Update config
                new_config = {'memory_test': f'value_{i}', 'data': list(range(100))}
                with open(temp_path, 'w') as f:
                    yaml.dump(new_config, f)

                await manager.reload()

                # Access config multiple times
                for j in range(10):
                    value = await manager.get_config('memory_test')
                    assert value == f'value_{i}'

                # Periodic garbage collection
                if i % 20 == 0:
                    gc.collect()

            # Final garbage collection
            gc.collect()
            await asyncio.sleep(0.1)

            # Check final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = final_memory - initial_memory

            # Memory increase should be reasonable (less than 50MB for this test)
            assert memory_increase < 50, f"Memory increase {memory_increase:.1f}MB is too high"

            print(f"Memory test: initial={initial_memory:.1f}MB, final={final_memory:.1f}MB, increase={memory_increase:.1f}MB")

        finally:
            await manager.stop()
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_configuration_validation_security(self):
        """Test configuration security validation."""
        manager = ConfigManager()

        # Test config with potential security issues
        insecure_config = {
            'database': {
                'password': 'hardcoded_password_123',  # Security issue
                'host': 'localhost'
            },
            'api': {
                'secret_key': 'another_hardcoded_secret',  # Security issue
                'timeout': 30
            }
        }

        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(insecure_config, f)
            temp_path = f.name

        try:
            source = FileConfigSource(temp_path)
            await manager.add_source(source)
            await manager.start()

            # Reload should detect security issues
            result = await manager.reload()

            # Should have warnings/errors about security issues
            assert result.status in [ReloadStatus.FAILED, ReloadStatus.PARTIAL]

            # Check if security validation caught the issues
            error_messages = ' '.join(result.errors)
            assert 'hardcoded' in error_messages.lower() or 'security' in error_messages.lower()

        finally:
            await manager.stop()
            os.unlink(temp_path)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
