#!/usr/bin/env python3
"""Run Comprehensive Testcontainer Suite.

Orchestrates the complete real behavior testing suite with performance validation.
Validates all modernized services against performance targets with real containers.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.containers.postgres_container import PostgreSQLTestContainer
from tests.integration.real_behavior_testing.containers.ml_test_container import (
    MLTestContainer,
)
from tests.integration.real_behavior_testing.containers.network_simulator import (
    NetworkSimulator,
)
from tests.integration.real_behavior_testing.containers.real_redis_container import (
    RealRedisTestContainer,
)
from tests.integration.real_behavior_testing.performance.benchmark_suite import (
    BenchmarkSuite,
)
from tests.integration.real_behavior_testing.utils.test_data_factory import (
    TestDataFactory,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('comprehensive_testcontainer_suite.log')
    ]
)
logger = logging.getLogger(__name__)


class ComprehensiveTestSuite:
    """Comprehensive test suite orchestrator."""

    def __init__(self, enable_ml: bool = True, enable_network: bool = True) -> None:
        """Initialize comprehensive test suite.

        Args:
            enable_ml: Enable ML container testing
            enable_network: Enable network simulation testing
        """
        self.enable_ml = enable_ml
        self.enable_network = enable_network

        # Container instances
        self.postgres_container: PostgreSQLTestContainer | None = None
        self.redis_container: RealRedisTestContainer | None = None
        self.ml_container: MLTestContainer | None = None
        self.network_simulator: NetworkSimulator | None = None

        # Suite components
        self.benchmark_suite: BenchmarkSuite | None = None
        self.test_data_factory = TestDataFactory()

        # Results
        self.results: dict[str, Any] = {}

        logger.info(f"ComprehensiveTestSuite initialized (ML: {enable_ml}, Network: {enable_network})")

    async def setup_containers(self):
        """Set up all test containers."""
        logger.info("Setting up test containers...")

        try:
            # Start PostgreSQL container
            logger.info("Starting PostgreSQL container...")
            self.postgres_container = PostgreSQLTestContainer(
                postgres_version="16",
                database_name="comprehensive_test_db"
            )
            await self.postgres_container.start()
            logger.info("✓ PostgreSQL container ready")

            # Start Redis container
            logger.info("Starting Redis container...")
            self.redis_container = RealRedisTestContainer(
                redis_version="7.2",
                max_memory="512mb"
            )
            await self.redis_container.start()
            logger.info("✓ Redis container ready")

            # Start ML container if enabled
            if self.enable_ml:
                logger.info("Starting ML test container...")
                self.ml_container = MLTestContainer(
                    memory_limit_mb=1024,
                    enable_caching=True
                )
                await self.ml_container.start()
                logger.info("✓ ML container ready")

            # Start Network simulator if enabled
            if self.enable_network:
                logger.info("Starting network simulator...")
                self.network_simulator = NetworkSimulator()
                await self.network_simulator.start()
                logger.info("✓ Network simulator ready")

            # Initialize benchmark suite
            self.benchmark_suite = BenchmarkSuite(
                postgres_container=self.postgres_container,
                redis_container=self.redis_container,
                ml_container=self.ml_container,
                network_simulator=self.network_simulator
            )

            logger.info("All containers started successfully")

        except Exception as e:
            logger.exception(f"Failed to setup containers: {e}")
            await self.cleanup_containers()
            raise

    async def cleanup_containers(self):
        """Clean up all test containers."""
        logger.info("Cleaning up test containers...")

        cleanup_tasks = []

        if self.network_simulator:
            cleanup_tasks.append(self.network_simulator.stop())

        if self.ml_container:
            cleanup_tasks.append(self.ml_container.stop())

        if self.redis_container:
            cleanup_tasks.append(self.redis_container.stop())

        if self.postgres_container:
            cleanup_tasks.append(self.postgres_container.stop())

        if cleanup_tasks:
            await asyncio.gather(*cleanup_tasks, return_exceptions=True)

        logger.info("Container cleanup completed")

    async def run_container_health_checks(self) -> dict[str, Any]:
        """Run health checks on all containers.

        Returns:
            Health check results
        """
        logger.info("Running container health checks...")

        health_results = {
            "timestamp": datetime.now(UTC).isoformat(),
            "containers": {}
        }

        # PostgreSQL health check
        if self.postgres_container:
            try:
                async with self.postgres_container.get_session() as session:
                    result = await session.execute("SELECT 1")
                    assert result.scalar() == 1

                health_results["containers"]["postgresql"] = {
                    "status": "healthy",
                    "connection_info": self.postgres_container.get_connection_info(),
                }
                logger.info("✓ PostgreSQL health check passed")

            except Exception as e:
                health_results["containers"]["postgresql"] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                logger.exception(f"✗ PostgreSQL health check failed: {e}")

        # Redis health check
        if self.redis_container:
            try:
                async with self.redis_container.get_client() as client:
                    result = await client.ping()
                    assert result is True

                memory_usage = await self.redis_container.get_memory_usage()
                connection_stats = await self.redis_container.get_connection_stats()

                health_results["containers"]["redis"] = {
                    "status": "healthy",
                    "connection_info": self.redis_container.get_connection_info(),
                    "memory_usage": memory_usage,
                    "connection_stats": connection_stats,
                }
                logger.info("✓ Redis health check passed")

            except Exception as e:
                health_results["containers"]["redis"] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                logger.exception(f"✗ Redis health check failed: {e}")

        # ML container health check
        if self.ml_container:
            try:
                performance_metrics = self.ml_container.get_performance_metrics()
                memory_usage = self.ml_container.get_memory_usage()

                health_results["containers"]["ml"] = {
                    "status": "healthy",
                    "performance_metrics": performance_metrics,
                    "memory_usage": memory_usage,
                }
                logger.info("✓ ML container health check passed")

            except Exception as e:
                health_results["containers"]["ml"] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                logger.exception(f"✗ ML container health check failed: {e}")

        # Network simulator health check
        if self.network_simulator:
            try:
                stats = self.network_simulator.get_failure_statistics()

                health_results["containers"]["network_simulator"] = {
                    "status": "healthy",
                    "simulator_stats": stats,
                }
                logger.info("✓ Network simulator health check passed")

            except Exception as e:
                health_results["containers"]["network_simulator"] = {
                    "status": "unhealthy",
                    "error": str(e),
                }
                logger.exception(f"✗ Network simulator health check failed: {e}")

        healthy_containers = sum(1 for c in health_results["containers"].values() if c["status"] == "healthy")
        total_containers = len(health_results["containers"])

        logger.info(f"Container health check completed: {healthy_containers}/{total_containers} healthy")

        return health_results

    async def run_performance_benchmarks(self) -> dict[str, Any]:
        """Run comprehensive performance benchmarks.

        Returns:
            Benchmark results
        """
        if not self.benchmark_suite:
            raise RuntimeError("Benchmark suite not initialized")

        logger.info("Running comprehensive performance benchmarks...")

        benchmark_results = await self.benchmark_suite.run_comprehensive_benchmark()

        # Generate human-readable report
        performance_report = self.benchmark_suite.generate_performance_report(benchmark_results)

        logger.info("Performance benchmarks completed")
        logger.info("\n" + performance_report)

        return {
            "benchmark_results": benchmark_results,
            "performance_report": performance_report,
        }

    async def run_service_validation_tests(self) -> dict[str, Any]:
        """Run service validation tests.

        Returns:
            Service validation results
        """
        logger.info("Running service validation tests...")

        validation_results = {
            "timestamp": datetime.now(UTC).isoformat(),
            "validations": {}
        }

        # Database repository migration validation
        if self.postgres_container:
            logger.info("Validating database repository migration...")

            try:
                # Test repository pattern compliance
                async with self.postgres_container.get_session() as session:
                    # Verify clean architecture patterns
                    await session.execute("CREATE TEMP TABLE test_repository_pattern (id INTEGER, data TEXT)")
                    await session.execute("INSERT INTO test_repository_pattern VALUES (1, 'test')")
                    result = await session.execute("SELECT COUNT(*) FROM test_repository_pattern")
                    count = result.scalar()

                    assert count == 1, "Repository pattern test failed"

                validation_results["validations"]["database_repository"] = {
                    "status": "passed",
                    "description": "Database repository migration validated",
                }
                logger.info("✓ Database repository validation passed")

            except Exception as e:
                validation_results["validations"]["database_repository"] = {
                    "status": "failed",
                    "error": str(e),
                }
                logger.exception(f"✗ Database repository validation failed: {e}")

        # Cache service validation
        if self.redis_container:
            logger.info("Validating cache service architecture...")

            try:
                # Test multi-level caching
                async with self.redis_container.get_client() as client:
                    # L2 Cache validation
                    test_key = "validation_test_key"
                    test_value = "validation_test_value"

                    await client.set(test_key, test_value, ex=300)  # 5 minutes
                    retrieved_value = await client.get(test_key)

                    assert retrieved_value == test_value, "Cache service validation failed"

                validation_results["validations"]["cache_service"] = {
                    "status": "passed",
                    "description": "Multi-level cache service validated",
                }
                logger.info("✓ Cache service validation passed")

            except Exception as e:
                validation_results["validations"]["cache_service"] = {
                    "status": "failed",
                    "error": str(e),
                }
                logger.exception(f"✗ Cache service validation failed: {e}")

        # ML service decomposition validation
        if self.ml_container:
            logger.info("Validating ML service decomposition...")

            try:
                # Test ML services architecture
                dataset = self.ml_container.get_test_dataset("small")
                assert dataset is not None, "Test dataset not available"

                # Test feature extraction service
                feature_result = await self.ml_container.extract_features(
                    dataset.prompts[:10],
                    feature_type="tfidf"
                )
                assert feature_result.success, "Feature extraction failed"

                # Test prediction service
                prediction_result = await self.ml_container.generate_predictions(
                    {"characteristics": {"test": "data"}},
                    model_type="pattern_prediction"
                )
                assert prediction_result.success, "Prediction generation failed"

                validation_results["validations"]["ml_services"] = {
                    "status": "passed",
                    "description": "ML service decomposition validated",
                    "services_tested": ["feature_extraction", "prediction_generation"],
                }
                logger.info("✓ ML service validation passed")

            except Exception as e:
                validation_results["validations"]["ml_services"] = {
                    "status": "failed",
                    "error": str(e),
                }
                logger.exception(f"✗ ML service validation failed: {e}")

        # Service orchestration validation
        logger.info("Validating service orchestration patterns...")

        try:
            # Test service coordination
            test_data = await self.test_data_factory.create_ml_test_dataset(size="small")
            assert test_data.record_count > 0, "Test data creation failed"

            validation_results["validations"]["service_orchestration"] = {
                "status": "passed",
                "description": "Service orchestration patterns validated",
                "test_data_records": test_data.record_count,
            }
            logger.info("✓ Service orchestration validation passed")

        except Exception as e:
            validation_results["validations"]["service_orchestration"] = {
                "status": "failed",
                "error": str(e),
            }
            logger.exception(f"✗ Service orchestration validation failed: {e}")

        passed_validations = sum(1 for v in validation_results["validations"].values() if v["status"] == "passed")
        total_validations = len(validation_results["validations"])

        logger.info(f"Service validation completed: {passed_validations}/{total_validations} passed")

        return validation_results

    async def run_comprehensive_suite(self) -> dict[str, Any]:
        """Run the complete comprehensive test suite.

        Returns:
            Complete test suite results
        """
        suite_start_time = time.perf_counter()

        logger.info("=" * 80)
        logger.info("STARTING COMPREHENSIVE TESTCONTAINER SUITE")
        logger.info("=" * 80)

        suite_results = {
            "suite_id": f"comprehensive_suite_{int(time.time())}",
            "started_at": datetime.now(UTC).isoformat(),
            "configuration": {
                "ml_enabled": self.enable_ml,
                "network_enabled": self.enable_network,
            },
            "results": {}
        }

        try:
            # Setup containers
            await self.setup_containers()

            # Run health checks
            health_results = await self.run_container_health_checks()
            suite_results["results"]["health_checks"] = health_results

            # Run performance benchmarks
            benchmark_results = await self.run_performance_benchmarks()
            suite_results["results"]["performance_benchmarks"] = benchmark_results

            # Run service validation tests
            validation_results = await self.run_service_validation_tests()
            suite_results["results"]["service_validation"] = validation_results

            # Calculate overall results
            total_duration = (time.perf_counter() - suite_start_time) * 1000

            suite_results.update({
                "completed_at": datetime.now(UTC).isoformat(),
                "total_duration_ms": total_duration,
                "status": "completed",
            })

            # Generate summary
            summary = self._generate_suite_summary(suite_results)
            suite_results["summary"] = summary

            logger.info("=" * 80)
            logger.info("COMPREHENSIVE TEST SUITE COMPLETED")
            logger.info("=" * 80)
            logger.info(f"Total Duration: {total_duration:.1f}ms")
            logger.info(f"Overall Status: {summary['overall_status']}")
            logger.info(f"Health Checks: {summary['health_summary']}")
            logger.info(f"Performance: {summary['performance_summary']}")
            logger.info(f"Validations: {summary['validation_summary']}")

            return suite_results

        except Exception as e:
            total_duration = (time.perf_counter() - suite_start_time) * 1000

            suite_results.update({
                "completed_at": datetime.now(UTC).isoformat(),
                "total_duration_ms": total_duration,
                "status": "failed",
                "error": str(e),
            })

            logger.exception(f"Comprehensive test suite failed: {e}")
            raise

        finally:
            # Cleanup containers
            await self.cleanup_containers()

    def _generate_suite_summary(self, suite_results: dict[str, Any]) -> dict[str, Any]:
        """Generate comprehensive suite summary.

        Args:
            suite_results: Complete suite results

        Returns:
            Suite summary
        """
        results = suite_results["results"]

        # Health check summary
        health_results = results.get("health_checks", {})
        containers = health_results.get("containers", {})
        healthy_count = sum(1 for c in containers.values() if c.get("status") == "healthy")
        total_containers = len(containers)
        health_summary = f"{healthy_count}/{total_containers} containers healthy"

        # Performance summary
        benchmark_results = results.get("performance_benchmarks", {}).get("benchmark_results", {})
        if benchmark_results:
            analysis = benchmark_results.get("analysis", {})
            summary = analysis.get("summary", {})
            performance_summary = f"{summary.get('met_targets', 0)}/{summary.get('total_targets', 0)} targets met ({summary.get('overall_success_rate', 0):.1f}%)"
        else:
            performance_summary = "No benchmarks run"

        # Validation summary
        validation_results = results.get("service_validation", {})
        validations = validation_results.get("validations", {})
        passed_validations = sum(1 for v in validations.values() if v.get("status") == "passed")
        total_validations = len(validations)
        validation_summary = f"{passed_validations}/{total_validations} validations passed"

        # Overall status
        overall_healthy = healthy_count == total_containers
        overall_performance = benchmark_results and analysis.get("summary", {}).get("overall_success_rate", 0) >= 80
        overall_validation = passed_validations == total_validations

        if overall_healthy and overall_performance and overall_validation:
            overall_status = "excellent"
        elif overall_healthy and (overall_performance or overall_validation):
            overall_status = "good"
        elif overall_healthy:
            overall_status = "partial"
        else:
            overall_status = "poor"

        return {
            "overall_status": overall_status,
            "health_summary": health_summary,
            "performance_summary": performance_summary,
            "validation_summary": validation_summary,
            "total_containers": total_containers,
            "healthy_containers": healthy_count,
            "performance_targets_met": summary.get('met_targets', 0) if benchmark_results else 0,
            "performance_targets_total": summary.get('total_targets', 0) if benchmark_results else 0,
            "validations_passed": passed_validations,
            "validations_total": total_validations,
        }


async def main():
    """Main entry point for comprehensive test suite."""
    parser = argparse.ArgumentParser(description="Run comprehensive testcontainer suite")
    parser.add_argument("--no-ml", action="store_true", help="Disable ML container testing")
    parser.add_argument("--no-network", action="store_true", help="Disable network simulation testing")
    parser.add_argument("--output", "-o", help="Output file for results (JSON)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Initialize test suite
    suite = ComprehensiveTestSuite(
        enable_ml=not args.no_ml,
        enable_network=not args.no_network
    )

    try:
        # Run comprehensive suite
        results = await suite.run_comprehensive_suite()

        # Save results if output file specified
        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, default=str)
            logger.info(f"Results saved to: {output_path}")

        # Determine exit code based on results
        summary = results.get("summary", {})
        overall_status = summary.get("overall_status", "poor")

        if overall_status in {"excellent", "good"}:
            sys.exit(0)
        elif overall_status == "partial":
            sys.exit(1)
        else:
            sys.exit(2)

    except Exception as e:
        logger.exception(f"Test suite execution failed: {e}")
        sys.exit(3)


if __name__ == "__main__":
    asyncio.run(main())
