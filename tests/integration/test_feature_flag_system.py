"""
Integration Tests for Feature Flag System

Comprehensive tests for the feature flag system covering:
- Configuration loading and validation
- Hot-reload functionality
- Percentage-based rollouts
- Technical debt phase management
- Error handling and recovery
- Performance and thread safety
"""

import asyncio
import json
import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Dict, List
from unittest.mock import patch, MagicMock

import pytest
import yaml

from prompt_improver.core.feature_flags import (
    FeatureFlagManager,
    FeatureFlagDefinition,
    EvaluationContext,
    EvaluationResult,
    FlagState,
    RolloutStrategy,
    RolloutConfig
)
from prompt_improver.core.feature_flag_init import (
    initialize_feature_flag_system,
    get_config_path
)


class TestFeatureFlagIntegration(unittest.TestCase):
    """Integration tests for the complete feature flag system."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_feature_flags.yaml"
        
        # Create test configuration
        self.test_config = {
            "version": "1.0.0",
            "schema_version": "2025.1",
            "global": {
                "default_rollout_percentage": 10.0,
                "sticky_bucketing": True,
                "evaluation_timeout_ms": 100,
                "metrics_enabled": True
            },
            "flags": {
                "phase1_config_externalization": {
                    "state": "rollout",
                    "default_variant": "off",
                    "variants": {"on": True, "off": False},
                    "rollout": {
                        "strategy": "percentage",
                        "percentage": 25.0,
                        "sticky": True
                    },
                    "targeting_rules": [
                        {
                            "name": "admin_users",
                            "condition": "user_type == 'admin'",
                            "variant": "on",
                            "priority": 100
                        }
                    ],
                    "metadata": {
                        "phase": 1,
                        "description": "Configuration externalization"
                    }
                },
                "phase2_health_checks": {
                    "state": "rollout",
                    "default_variant": "off",
                    "variants": {"on": True, "off": False, "partial": "basic_only"},
                    "rollout": {
                        "strategy": "percentage",
                        "percentage": 15.0,
                        "sticky": True
                    },
                    "metadata": {
                        "phase": 2,
                        "dependencies": ["phase1_config_externalization"]
                    }
                },
                "test_user_list_flag": {
                    "state": "rollout",
                    "default_variant": "off",
                    "variants": {"on": True, "off": False},
                    "rollout": {
                        "strategy": "user_list",
                        "user_list": ["admin001", "tester001"],
                        "sticky": True
                    }
                }
            }
        }
        
        # Write test configuration
        with open(self.config_path, 'w') as f:
            yaml.dump(self.test_config, f)
    
    def tearDown(self):
        """Clean up test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_feature_flag_manager_initialization(self):
        """Test feature flag manager initialization."""
        manager = FeatureFlagManager(self.config_path, watch_files=False)
        
        # Test configuration loading
        config_info = manager.get_configuration_info()
        self.assertEqual(config_info["flags_count"], 3)
        self.assertIsNotNone(config_info["loaded_at"])
        self.assertEqual(config_info["version"], "1.0.0")
        
        # Test flag retrieval
        all_flags = manager.get_all_flags()
        self.assertIn("phase1_config_externalization", all_flags)
        self.assertIn("phase2_health_checks", all_flags)
        
        manager.shutdown()
    
    def test_percentage_based_rollout(self):
        """Test percentage-based rollout functionality."""
        manager = FeatureFlagManager(self.config_path, watch_files=False)
        
        # Test multiple users to verify percentage distribution
        results = []
        for i in range(100):
            context = EvaluationContext(
                user_id=f"user_{i:03d}",
                environment="test"
            )
            result = manager.evaluate_flag("phase1_config_externalization", context)
            results.append(result.value)
        
        # Should be approximately 25% enabled (allowing for variance)
        enabled_count = sum(1 for r in results if r)
        enabled_percentage = enabled_count / len(results) * 100
        
        # Allow 10% variance from expected 25%
        self.assertGreater(enabled_percentage, 15)
        self.assertLess(enabled_percentage, 35)
        
        manager.shutdown()
    
    def test_user_list_rollout(self):
        """Test user list rollout strategy."""
        manager = FeatureFlagManager(self.config_path, watch_files=False)
        
        # Test users in the list
        allowed_users = ["admin001", "tester001"]
        for user_id in allowed_users:
            context = EvaluationContext(user_id=user_id, environment="test")
            result = manager.evaluate_flag("test_user_list_flag", context)
            self.assertTrue(result.value, f"User {user_id} should be enabled")
            self.assertEqual(result.reason, "ROLLOUT_MATCH")
        
        # Test users not in the list
        denied_users = ["user001", "guest001"]
        for user_id in denied_users:
            context = EvaluationContext(user_id=user_id, environment="test")
            result = manager.evaluate_flag("test_user_list_flag", context)
            self.assertFalse(result.value, f"User {user_id} should be disabled")
            self.assertEqual(result.reason, "DEFAULT")
        
        manager.shutdown()
    
    def test_targeting_rules(self):
        """Test targeting rule evaluation."""
        manager = FeatureFlagManager(self.config_path, watch_files=False)
        
        # Test admin user (should match targeting rule)
        admin_context = EvaluationContext(
            user_id="admin_user",
            user_type="admin",
            environment="test"
        )
        result = manager.evaluate_flag("phase1_config_externalization", admin_context)
        self.assertTrue(result.value)
        # Note: targeting rules take precedence over rollout
        self.assertEqual(result.reason, "TARGETING_MATCH")
        
        # Test regular user (should follow rollout percentage)
        user_context = EvaluationContext(
            user_id="regular_user",
            user_type="user",
            environment="test"
        )
        result = manager.evaluate_flag("phase1_config_externalization", user_context)
        # Result depends on user bucketing, but reason should be rollout or default
        self.assertIn(result.reason, ["ROLLOUT_MATCH", "DEFAULT"])
        
        manager.shutdown()
    
    def test_sticky_bucketing(self):
        """Test that users get consistent results (sticky bucketing)."""
        manager = FeatureFlagManager(self.config_path, watch_files=False)
        
        user_context = EvaluationContext(
            user_id="consistent_user",
            environment="test"
        )
        
        # Evaluate flag multiple times
        results = []
        for _ in range(10):
            result = manager.evaluate_flag("phase1_config_externalization", user_context)
            results.append(result.value)
        
        # All results should be the same (sticky)
        self.assertTrue(all(r == results[0] for r in results),
                       "User should get consistent results")
        
        manager.shutdown()
    
    def test_hot_reload_functionality(self):
        """Test hot-reload configuration changes."""
        manager = FeatureFlagManager(self.config_path, watch_files=False)
        
        # Initial state
        context = EvaluationContext(user_id="test_user", environment="test")
        initial_result = manager.evaluate_flag("phase1_config_externalization", context)
        
        # Modify configuration
        modified_config = self.test_config.copy()
        modified_config["flags"]["phase1_config_externalization"]["state"] = "enabled"
        
        with open(self.config_path, 'w') as f:
            yaml.dump(modified_config, f)
        
        # Manually reload (simulating file watcher)
        manager.reload_configuration()
        
        # Check if change was applied
        new_result = manager.evaluate_flag("phase1_config_externalization", context)
        self.assertTrue(new_result.value, "Flag should be enabled after reload")
        self.assertEqual(new_result.reason, "DEFAULT", "Should use default variant when enabled")
        
        manager.shutdown()
    
    def test_metrics_collection(self):
        """Test metrics collection and reporting."""
        manager = FeatureFlagManager(self.config_path, watch_files=False)
        
        context = EvaluationContext(user_id="metrics_user", environment="test")
        
        # Perform multiple evaluations
        for _ in range(5):
            manager.evaluate_flag("phase1_config_externalization", context)
            manager.evaluate_flag("phase2_health_checks", context)
        
        # Check metrics
        metrics = manager.get_metrics()
        
        self.assertIn("phase1_config_externalization", metrics)
        self.assertIn("phase2_health_checks", metrics)
        
        phase1_metrics = metrics["phase1_config_externalization"]
        self.assertEqual(phase1_metrics.evaluations_count, 5)
        self.assertIsNotNone(phase1_metrics.last_evaluated)
        
        manager.shutdown()
    
    def test_error_handling(self):
        """Test error handling in various scenarios."""
        manager = FeatureFlagManager(self.config_path, watch_files=False)
        
        context = EvaluationContext(user_id="error_user", environment="test")
        
        # Test non-existent flag
        result = manager.evaluate_flag("non_existent_flag", context, "default_value")
        self.assertEqual(result.value, "default_value")
        self.assertEqual(result.reason, "FLAG_NOT_FOUND")
        
        # Check error metrics
        metrics = manager.get_metrics("non_existent_flag")
        self.assertEqual(metrics.evaluations_count, 1)
        
        manager.shutdown()
    
    def test_disabled_flag_handling(self):
        """Test handling of disabled flags."""
        # Create config with disabled flag
        disabled_config = self.test_config.copy()
        disabled_config["flags"]["disabled_flag"] = {
            "state": "disabled",
            "default_variant": "off",
            "variants": {"on": True, "off": False}
        }
        
        disabled_config_path = Path(self.temp_dir) / "disabled_config.yaml"
        with open(disabled_config_path, 'w') as f:
            yaml.dump(disabled_config, f)
        
        manager = FeatureFlagManager(disabled_config_path, watch_files=False)
        
        context = EvaluationContext(user_id="test_user", environment="test")
        result = manager.evaluate_flag("disabled_flag", context)
        
        self.assertFalse(result.value)
        self.assertEqual(result.reason, "FLAG_DISABLED")
        
        manager.shutdown()
    
    def test_concurrent_evaluations(self):
        """Test thread safety with concurrent evaluations."""
        manager = FeatureFlagManager(self.config_path, watch_files=False)
        
        results = []
        errors = []
        
        def evaluate_flags():
            try:
                for i in range(10):
                    context = EvaluationContext(
                        user_id=f"concurrent_user_{threading.current_thread().ident}_{i}",
                        environment="test"
                    )
                    result = manager.evaluate_flag("phase1_config_externalization", context)
                    results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for _ in range(5):
            thread = threading.Thread(target=evaluate_flags)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Verify no errors occurred
        self.assertEqual(len(errors), 0, f"Concurrent evaluation errors: {errors}")
        self.assertEqual(len(results), 50, "Should have 50 evaluation results")
        
        # Verify metrics are consistent
        metrics = manager.get_metrics("phase1_config_externalization")
        self.assertEqual(metrics.evaluations_count, 50)
        
        manager.shutdown()
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        # Test invalid configuration
        invalid_config_path = Path(self.temp_dir) / "invalid_config.yaml"
        
        invalid_config = {
            "flags": {
                "invalid_flag": {
                    "state": "invalid_state",  # Invalid state
                    "default_variant": "missing_variant",  # Variant doesn't exist
                    "variants": {"on": True, "off": False}
                }
            }
        }
        
        with open(invalid_config_path, 'w') as f:
            yaml.dump(invalid_config, f)
        
        # Manager should handle invalid config gracefully
        manager = FeatureFlagManager(invalid_config_path, watch_files=False)
        
        # Invalid flag should not be loaded
        all_flags = manager.get_all_flags()
        self.assertNotIn("invalid_flag", all_flags)
        
        manager.shutdown()
    
    def test_technical_debt_phase_dependencies(self):
        """Test technical debt phase dependency logic."""
        manager = FeatureFlagManager(self.config_path, watch_files=False)
        
        # Check that phase metadata is preserved
        all_flags = manager.get_all_flags()
        
        phase1_flag = all_flags["phase1_config_externalization"]
        self.assertEqual(phase1_flag.metadata["phase"], 1)
        
        phase2_flag = all_flags["phase2_health_checks"]
        self.assertEqual(phase2_flag.metadata["phase"], 2)
        self.assertIn("phase1_config_externalization", phase2_flag.metadata["dependencies"])
        
        manager.shutdown()


class TestFeatureFlagInitialization(unittest.TestCase):
    """Test feature flag initialization utilities."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        os.chdir(self.temp_dir)
        
        # Create mock project structure
        self.config_dir = Path("config")
        self.config_dir.mkdir()
        
        # Create test configs
        configs = {
            "feature_flags.yaml": {"version": "1.0.0", "flags": {}},
            "feature_flags_development.yaml": {"version": "1.0.0", "environment": "development", "flags": {}},
            "feature_flags_production.yaml": {"version": "1.0.0", "environment": "production", "flags": {}}
        }
        
        for filename, config in configs.items():
            with open(self.config_dir / filename, 'w') as f:
                yaml.dump(config, f)
    
    def tearDown(self):
        """Clean up test environment."""
        os.chdir(self.original_cwd)
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_get_config_path_environment_specific(self):
        """Test getting environment-specific configuration paths."""
        # Test development environment
        dev_path = get_config_path("development")
        self.assertTrue(dev_path.name == "feature_flags_development.yaml")
        
        # Test production environment
        prod_path = get_config_path("production")
        self.assertTrue(prod_path.name == "feature_flags_production.yaml")
        
        # Test fallback to default
        staging_path = get_config_path("staging")  # No staging config exists
        self.assertTrue(staging_path.name == "feature_flags.yaml")
    
    def test_initialize_feature_flag_system(self):
        """Test feature flag system initialization."""
        manager = initialize_feature_flag_system("development")
        
        self.assertIsNotNone(manager)
        config_info = manager.get_configuration_info()
        self.assertEqual(config_info["flags_count"], 0)  # Empty test config
        
        manager.shutdown()
    
    @patch.dict(os.environ, {"ENVIRONMENT": "production"})
    def test_environment_variable_detection(self):
        """Test environment detection from environment variables."""
        path = get_config_path()
        self.assertTrue(path.name == "feature_flags_production.yaml")


class TestFeatureFlagPerformance(unittest.TestCase):
    """Performance tests for feature flag system."""
    
    def setUp(self):
        """Set up performance test environment."""
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "perf_test_config.yaml"
        
        # Create performance test configuration with many flags
        flags = {}
        for i in range(100):
            flags[f"perf_flag_{i}"] = {
                "state": "rollout",
                "default_variant": "off",
                "variants": {"on": True, "off": False},
                "rollout": {
                    "strategy": "percentage", 
                    "percentage": 10.0,
                    "sticky": True
                }
            }
        
        config = {"version": "1.0.0", "flags": flags}
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f)
    
    def tearDown(self):
        """Clean up performance test environment."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_evaluation_performance(self):
        """Test feature flag evaluation performance."""
        manager = FeatureFlagManager(self.config_path, watch_files=False)
        
        context = EvaluationContext(user_id="perf_user", environment="test")
        
        # Warm up
        for i in range(10):
            manager.evaluate_flag(f"perf_flag_{i}", context)
        
        # Performance test
        start_time = time.time()
        evaluations = 1000
        
        for i in range(evaluations):
            flag_key = f"perf_flag_{i % 100}"
            manager.evaluate_flag(flag_key, context)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should complete 1000 evaluations in reasonable time
        self.assertLess(duration, 1.0, f"1000 evaluations took {duration:.3f}s, should be under 1s")
        
        # Calculate evaluations per second
        evaluations_per_second = evaluations / duration
        self.assertGreater(evaluations_per_second, 1000, 
                          f"Performance: {evaluations_per_second:.0f} eval/s, should be >1000")
        
        manager.shutdown()
    
    def test_concurrent_performance(self):
        """Test performance under concurrent load."""
        manager = FeatureFlagManager(self.config_path, watch_files=False)
        
        results = []
        start_time = time.time()
        
        def concurrent_evaluations():
            thread_results = []
            context = EvaluationContext(
                user_id=f"user_{threading.current_thread().ident}",
                environment="test"
            )
            
            for i in range(100):
                flag_key = f"perf_flag_{i % 100}"
                result = manager.evaluate_flag(flag_key, context)
                thread_results.append(result)
            
            results.extend(thread_results)
        
        # Create concurrent threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=concurrent_evaluations)
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Should handle 1000 concurrent evaluations efficiently
        self.assertEqual(len(results), 1000)
        self.assertLess(duration, 2.0, f"Concurrent test took {duration:.3f}s, should be under 2s")
        
        manager.shutdown()


if __name__ == "__main__":
    # Set up test environment
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))
    
    # Run tests
    unittest.main(verbosity=2)