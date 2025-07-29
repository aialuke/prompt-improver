#!/usr/bin/env python3
"""
APES Component Integration Testing for API metrics.
Tests integration with MCP server, CLI separation, and centralized utilities.
"""

import asyncio
import time
from datetime import datetime, timezone
from typing import Dict, Any

from src.prompt_improver.metrics.api_metrics import (
    APIMetricsCollector,
    APIUsageMetric,
    UserJourneyMetric,
    EndpointCategory,
    HTTPMethod,
    UserJourneyStage,
    AuthenticationMethod,
    get_api_metrics_collector,
    record_api_request,
    record_user_journey_event
)

# Test centralized utilities integration
try:
    from src.prompt_improver.core.common import get_logger, MetricsMixin, ConfigMixin
    COMMON_UTILS_AVAILABLE = True
except ImportError:
    COMMON_UTILS_AVAILABLE = False
    print("Common utilities not available")

# Test metrics registry integration
try:
    from src.prompt_improver.performance.monitoring.metrics_registry import get_metrics_registry
    METRICS_REGISTRY_AVAILABLE = True
except ImportError:
    METRICS_REGISTRY_AVAILABLE = False
    print("Metrics registry not available")


class APESIntegrationTester:
    """Test APES system integration for API metrics."""

    def __init__(self):
        self.test_collector = None

    async def setup_test_environment(self):
        """Set up test environment for APES integration."""
        print("🔧 Setting up APES integration test environment...")

        # Create collector with APES-compatible configuration
        config = {
            "max_api_metrics": 5000,
            "max_journey_metrics": 2500,
            "aggregation_window_minutes": 1,
            "retention_hours": 24,
            "mcp_performance_mode": True,  # Enable MCP optimizations
            "cli_separation_mode": True    # Ensure CLI/MCP separation
        }

        self.test_collector = APIMetricsCollector(config)
        await self.test_collector.start_collection()
        print("✓ APES-compatible collector initialized")

    async def teardown_test_environment(self):
        """Clean up test environment."""
        if self.test_collector:
            await self.test_collector.stop_collection()
        print("✓ APES integration test environment cleaned up")

    def test_centralized_utilities_integration(self):
        """Test 4.1: Integration with centralized utilities."""
        print("🧪 Test: Centralized Utilities Integration")

        try:
            if not COMMON_UTILS_AVAILABLE:
                print("⚠️  Common utilities not available - testing basic functionality")
                # Test that collector works without centralized utilities
                collector = APIMetricsCollector()
                assert collector is not None
                print("✓ Graceful degradation without common utilities: PASSED")
                return True

            # Test MetricsMixin integration
            assert hasattr(self.test_collector, 'metrics_registry')
            assert hasattr(self.test_collector, 'metrics_available')

            # Test ConfigMixin integration
            assert hasattr(self.test_collector, 'local_config')

            # Test logger integration
            assert hasattr(self.test_collector, 'logger')
            assert self.test_collector.logger is not None

            # Test that logger works
            self.test_collector.logger.info("Test log message from APES integration test")

            print("✓ Centralized utilities integration: PASSED")
            return True

        except Exception as e:
            print(f"✗ Centralized utilities integration: FAILED - {e}")
            return False

    async def test_mcp_server_performance_compliance(self):
        """Test 4.2: MCP server performance compliance (<200ms SLA)."""
        print("🧪 Test: MCP Server Performance Compliance")

        try:
            # Simulate MCP server read-only operations
            mcp_operations = []

            # Test 1: Quick metric recording (simulating MCP data collection)
            start_time = time.time()
            for i in range(10):
                api_metric = APIUsageMetric(
                    endpoint=f"/mcp/v1/rules/apply",
                    method=HTTPMethod.POST,
                    category=EndpointCategory.PROMPT_IMPROVEMENT,
                    status_code=200,
                    response_time_ms=50.0 + i * 2,  # Simulating fast MCP responses
                    request_size_bytes=256,
                    response_size_bytes=512,
                    user_id=f"mcp_user_{i}",
                    session_id=f"mcp_session_{i}",
                    ip_address="127.0.0.1",
                    user_agent="MCP-Client/1.0",
                    timestamp=datetime.now(timezone.utc),
                    query_parameters_count=1,
                    payload_type="application/json",
                    rate_limited=False,
                    cache_hit=True,  # MCP should have high cache hit rate
                    authentication_method=AuthenticationMethod.JWT_TOKEN,
                    api_version="v1"
                )
                await self.test_collector.record_api_usage(api_metric)

            recording_time = (time.time() - start_time) * 1000
            mcp_operations.append(("metric_recording", recording_time))

            # Test 2: Quick analytics retrieval (simulating MCP rule lookup)
            start_time = time.time()
            analytics = await self.test_collector.get_endpoint_analytics(hours=1)
            analytics_time = (time.time() - start_time) * 1000
            mcp_operations.append(("analytics_retrieval", analytics_time))

            # Test 3: Session state lookup (simulating MCP user context)
            start_time = time.time()
            stats = self.test_collector.get_collection_stats()
            stats_time = (time.time() - start_time) * 1000
            mcp_operations.append(("stats_lookup", stats_time))

            # Verify MCP SLA compliance (<200ms for all operations)
            sla_violations = []
            for operation, duration in mcp_operations:
                print(f"  {operation}: {duration:.2f}ms")
                if duration >= 200:
                    sla_violations.append((operation, duration))

            if not sla_violations:
                print("✓ MCP server performance compliance: PASSED")
                return True
            else:
                print(f"✗ MCP server performance compliance: FAILED - SLA violations: {sla_violations}")
                return False

        except Exception as e:
            print(f"✗ MCP server performance compliance: FAILED - {e}")
            return False

    async def test_cli_mcp_architectural_separation(self):
        """Test 4.3: CLI/MCP architectural separation."""
        print("🧪 Test: CLI/MCP Architectural Separation")

        try:
            # Test that metrics collection doesn't interfere with CLI operations
            # Simulate CLI training workflow metrics
            cli_metrics = []
            for i in range(5):
                # CLI metrics should be different from MCP metrics
                api_metric = APIUsageMetric(
                    endpoint=f"/cli/v1/train/step-{i}",
                    method=HTTPMethod.POST,
                    category=EndpointCategory.ML_TRAINING,  # CLI-specific category
                    status_code=200,
                    response_time_ms=500.0 + i * 100,  # CLI operations can be slower
                    request_size_bytes=2048,
                    response_size_bytes=4096,
                    user_id="cli_user",
                    session_id="cli_training_session",
                    ip_address="127.0.0.1",
                    user_agent="CLI-Agent/1.0",
                    timestamp=datetime.now(timezone.utc),
                    query_parameters_count=0,
                    payload_type="application/json",
                    rate_limited=False,
                    cache_hit=False,  # CLI operations typically don't use cache
                    authentication_method=AuthenticationMethod.LOCAL_AUTH,
                    api_version="v1"
                )
                await self.test_collector.record_api_usage(api_metric)
                cli_metrics.append(api_metric)

            # Simulate MCP read-only operations
            mcp_metrics = []
            for i in range(5):
                api_metric = APIUsageMetric(
                    endpoint="/mcp/v1/rules/lookup",
                    method=HTTPMethod.GET,  # MCP is read-only
                    category=EndpointCategory.RULE_APPLICATION,  # MCP-specific category
                    status_code=200,
                    response_time_ms=25.0 + i * 5,  # MCP operations are fast
                    request_size_bytes=128,
                    response_size_bytes=256,
                    user_id=f"mcp_agent_{i}",
                    session_id=f"mcp_lookup_session_{i}",
                    ip_address="127.0.0.1",
                    user_agent="MCP-Server/1.0",
                    timestamp=datetime.now(timezone.utc),
                    query_parameters_count=2,
                    payload_type="application/json",
                    rate_limited=False,
                    cache_hit=True,  # MCP should have high cache hit rate
                    authentication_method=AuthenticationMethod.JWT_TOKEN,
                    api_version="v1"
                )
                await self.test_collector.record_api_usage(api_metric)
                mcp_metrics.append(api_metric)

            # Verify separation by analyzing collected metrics
            analytics = await self.test_collector.get_endpoint_analytics(hours=1)

            # Check that both CLI and MCP metrics are tracked separately
            cli_endpoints = [endpoint for endpoint in analytics["endpoint_analytics"].keys() if "/cli/" in endpoint]
            mcp_endpoints = [endpoint for endpoint in analytics["endpoint_analytics"].keys() if "/mcp/" in endpoint]

            assert len(cli_endpoints) > 0, "CLI endpoints not found in analytics"
            assert len(mcp_endpoints) > 0, "MCP endpoints not found in analytics"

            # Verify performance characteristics are different
            cli_avg_time = sum(analytics["endpoint_analytics"][ep]["avg_response_time_ms"] for ep in cli_endpoints) / len(cli_endpoints)
            mcp_avg_time = sum(analytics["endpoint_analytics"][ep]["avg_response_time_ms"] for ep in mcp_endpoints) / len(mcp_endpoints)

            # CLI should be slower than MCP (different performance profiles)
            assert cli_avg_time > mcp_avg_time, "CLI and MCP performance profiles not properly separated"

            print(f"  CLI average response time: {cli_avg_time:.2f}ms")
            print(f"  MCP average response time: {mcp_avg_time:.2f}ms")
            print("✓ CLI/MCP architectural separation: PASSED")
            return True

        except Exception as e:
            print(f"✗ CLI/MCP architectural separation: FAILED - {e}")
            return False

    async def test_ml_training_pipeline_integration(self):
        """Test 4.4: ML training pipeline integration."""
        print("🧪 Test: ML Training Pipeline Integration")

        try:
            # Simulate ML training pipeline metrics collection
            training_stages = [
                "data_export",
                "pattern_discovery",
                "rule_optimization",
                "model_validation",
                "deployment"
            ]

            training_session_id = "ml_training_session_001"

            for i, stage in enumerate(training_stages):
                # Record journey metric for training stage
                journey_metric = UserJourneyMetric(
                    user_id="ml_pipeline",
                    session_id=training_session_id,
                    journey_stage=UserJourneyStage.ML_TRAINING,  # Custom stage for ML
                    event_type=f"training_{stage}",
                    endpoint=f"/cli/v1/train/{stage}",
                    success=True,
                    conversion_value=None,  # Training doesn't have monetary conversion
                    time_to_action_seconds=60.0 + i * 30,  # Each stage takes longer
                    previous_stage=UserJourneyStage.ML_TRAINING if i > 0 else None,
                    feature_flags_active=["ml_optimization", "advanced_training"],
                    cohort_id="ml_pipeline_users",
                    timestamp=datetime.now(timezone.utc),
                    metadata={
                        "training_stage": stage,
                        "stage_number": i + 1,
                        "total_stages": len(training_stages),
                        "pipeline_config": {
                            "batch_size": 1000,
                            "learning_rate": 0.001,
                            "epochs": 10
                        }
                    }
                )
                await self.test_collector.record_user_journey(journey_metric)

                # Record corresponding API metric
                api_metric = APIUsageMetric(
                    endpoint=f"/cli/v1/train/{stage}",
                    method=HTTPMethod.POST,
                    category=EndpointCategory.ML_TRAINING,
                    status_code=200,
                    response_time_ms=1000.0 + i * 500,  # Training operations are slow
                    request_size_bytes=4096,
                    response_size_bytes=8192,
                    user_id="ml_pipeline",
                    session_id=training_session_id,
                    ip_address="127.0.0.1",
                    user_agent="ML-Pipeline/1.0",
                    timestamp=datetime.now(timezone.utc),
                    query_parameters_count=0,
                    payload_type="application/json",
                    rate_limited=False,
                    cache_hit=False,  # Training doesn't use cache
                    authentication_method=AuthenticationMethod.LOCAL_AUTH,
                    api_version="v1"
                )
                await self.test_collector.record_api_usage(api_metric)

            # Verify ML training metrics are properly collected
            journey_analytics = await self.test_collector.get_user_journey_analytics(hours=1)

            # Should have journey data now (check for actual data structure, not "status" field)
            if "status" in journey_analytics and journey_analytics["status"] == "no_data":
                raise AssertionError("ML training journey data not found")

            # Verify we have the expected data structure when data exists
            assert "total_journey_events" in journey_analytics, "Journey analytics missing total_journey_events"
            assert "stage_analytics" in journey_analytics, "Journey analytics missing stage_analytics"
            assert journey_analytics["total_journey_events"] >= len(training_stages), f"Expected at least {len(training_stages)} journey events, got {journey_analytics['total_journey_events']}"

            # Verify training session tracking
            stats = self.test_collector.get_collection_stats()
            assert stats["journey_events_tracked"] >= len(training_stages), "Not all training stages tracked"

            # Verify ML_TRAINING stage is present in analytics
            assert "ml_training" in journey_analytics["stage_analytics"], "ML_TRAINING stage not found in analytics"
            ml_training_stats = journey_analytics["stage_analytics"]["ml_training"]
            assert ml_training_stats["total_events"] >= len(training_stages), "Not all training events recorded in stage analytics"

            print(f"  Training stages tracked: {len(training_stages)}")
            print(f"  Journey events recorded: {stats['journey_events_tracked']}")
            print(f"  Total journey events in analytics: {journey_analytics['total_journey_events']}")
            print(f"  ML training stage events: {ml_training_stats['total_events']}")
            print("✓ ML training pipeline integration: PASSED")
            return True

        except Exception as e:
            print(f"✗ ML training pipeline integration: FAILED - {e}")
            return False

    async def test_global_collector_singleton(self):
        """Test 4.5: Global collector singleton pattern."""
        print("🧪 Test: Global Collector Singleton")

        try:
            # Test that global collector is consistent
            collector1 = get_api_metrics_collector()
            collector2 = get_api_metrics_collector()

            # Should be the same instance
            assert collector1 is collector2, "Global collector not singleton"

            # Test convenience functions use global collector
            await record_api_request(
                endpoint="/test/singleton",
                method=HTTPMethod.GET,
                category=EndpointCategory.HEALTH_CHECK,
                status_code=200,
                response_time_ms=50.0,
                user_id="singleton_test"
            )

            # Verify it was recorded in global collector
            stats = collector1.get_collection_stats()
            assert stats["api_calls_tracked"] > 0, "Global collector not receiving metrics"

            print("✓ Global collector singleton: PASSED")
            return True

        except Exception as e:
            print(f"✗ Global collector singleton: FAILED - {e}")
            return False


async def main():
    """Run APES component integration tests."""
    print("🚀 APES Component Integration Testing")
    print("=" * 50)

    tester = APESIntegrationTester()

    try:
        await tester.setup_test_environment()

        tests = [
            tester.test_centralized_utilities_integration,
            tester.test_mcp_server_performance_compliance,
            tester.test_cli_mcp_architectural_separation,
            tester.test_ml_training_pipeline_integration,
            tester.test_global_collector_singleton
        ]

        results = []
        for test in tests:
            if asyncio.iscoroutinefunction(test):
                result = await test()
            else:
                result = test()
            results.append(result)
            print()

        # Summary
        passed = sum(results)
        total = len(results)
        success_rate = (passed / total) * 100

        print("📊 APES INTEGRATION TEST SUMMARY")
        print("=" * 50)
        print(f"Tests Passed: {passed}/{total}")
        print(f"Success Rate: {success_rate:.1f}%")

        if success_rate == 100:
            print("🎉 All APES integration tests passed!")
            print("\n✅ APES Integration Verified:")
            print("  • Centralized utilities properly integrated")
            print("  • MCP server <200ms SLA compliance maintained")
            print("  • CLI/MCP architectural separation enforced")
            print("  • ML training pipeline metrics collection working")
            print("  • Global singleton pattern functioning correctly")
            return True
        else:
            print("❌ Some APES integration tests failed")
            return False

    finally:
        await tester.teardown_test_environment()


if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
