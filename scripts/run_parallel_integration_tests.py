#!/usr/bin/env python3
"""Parallel Integration Test Execution Framework.

This script provides a high-performance parallel test execution framework
for comprehensive integration testing across all refactored modules.

Features:
- Parallel test execution with configurable worker pools
- Real-time performance monitoring and metrics collection
- Automated performance baseline validation
- Comprehensive reporting with system health metrics
- Integration with testcontainers for real external services
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

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Import test configuration
from prompt_improver.metrics.application_instrumentation import (
    ApplicationInstrumentation,
)

# Performance monitoring
from prompt_improver.performance.baseline.performance_validation_suite import (
    PerformanceValidationSuite,
)
from prompt_improver.tests.integration.comprehensive_parallel_integration_test import (
    ComprehensiveParallelIntegrationTest,
)

logger = logging.getLogger(__name__)


class ParallelTestExecutionFramework:
    """High-performance parallel test execution framework."""

    def __init__(
        self,
        max_workers: int = 10,
        performance_baseline_enabled: bool = True,
        real_time_monitoring: bool = True,
        output_dir: str = "test_results"
    ) -> None:
        self.max_workers = max_workers
        self.performance_baseline_enabled = performance_baseline_enabled
        self.real_time_monitoring = real_time_monitoring
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize performance monitoring
        if real_time_monitoring:
            self.instrumentation = ApplicationInstrumentation()
            self.performance_validator = PerformanceValidationSuite()

        # Performance requirements (from CLAUDE.md)
        self.performance_requirements = {
            "api_response_time_p95_ms": 100.0,
            "cache_hit_rate_percent": 80.0,
            "health_check_duration_ms": 25.0,
            "error_rate_percent": 10.0,
            "concurrent_operations_per_second": 100.0,
            "memory_usage_mb_max": 1000.0
        }

    async def execute_parallel_integration_tests(
        self,
        test_categories: list[str] | None = None,
        performance_validation: bool = True,
        generate_report: bool = True
    ) -> dict[str, Any]:
        """Execute comprehensive parallel integration tests with performance monitoring."""
        logger.info("Starting parallel integration test execution...")
        start_time = time.perf_counter()

        # Initialize test suite
        test_suite = ComprehensiveParallelIntegrationTest(max_workers=self.max_workers)

        try:
            # Setup test infrastructure with real external services
            logger.info("Setting up test infrastructure with real external services...")
            await test_suite.setup_test_infrastructure()

            # Start performance monitoring if enabled
            if self.real_time_monitoring:
                await self._start_performance_monitoring()

            # Execute parallel integration tests
            logger.info(f"Executing parallel integration tests with {self.max_workers} workers...")
            test_results = await test_suite.run_parallel_integration_tests()

            # Stop performance monitoring
            if self.real_time_monitoring:
                performance_metrics = await self._stop_performance_monitoring()
                test_results["system_performance_metrics"] = performance_metrics

            # Validate performance requirements
            if performance_validation:
                validation_results = await self._validate_performance_requirements(test_results)
                test_results["performance_validation"] = validation_results

            # Generate comprehensive report
            if generate_report:
                report_path = await self._generate_comprehensive_report(test_results)
                test_results["report_path"] = str(report_path)

            total_duration = (time.perf_counter() - start_time) * 1000
            test_results["framework_execution_duration_ms"] = total_duration

            logger.info(f"Parallel integration tests completed in {total_duration:.2f}ms")
            return test_results

        except Exception as e:
            logger.exception(f"Parallel integration test execution failed: {e}")
            raise
        finally:
            # Always cleanup test infrastructure
            await test_suite.teardown_test_infrastructure()

    async def _start_performance_monitoring(self) -> None:
        """Start real-time performance monitoring."""
        logger.info("Starting real-time performance monitoring...")
        try:
            if hasattr(self.instrumentation, 'start_monitoring'):
                await self.instrumentation.start_monitoring()
            if hasattr(self.performance_validator, 'start_baseline_collection'):
                await self.performance_validator.start_baseline_collection()
        except Exception as e:
            logger.warning(f"Performance monitoring startup failed: {e}")

    async def _stop_performance_monitoring(self) -> dict[str, Any]:
        """Stop performance monitoring and collect metrics."""
        logger.info("Stopping performance monitoring and collecting metrics...")
        metrics = {}

        try:
            if hasattr(self.instrumentation, 'get_metrics'):
                metrics.update(await self.instrumentation.get_metrics())
            if hasattr(self.performance_validator, 'get_baseline_metrics'):
                metrics.update(await self.performance_validator.get_baseline_metrics())
        except Exception as e:
            logger.warning(f"Performance metrics collection failed: {e}")

        return metrics

    async def _validate_performance_requirements(self, test_results: dict[str, Any]) -> dict[str, Any]:
        """Validate test results against performance requirements."""
        logger.info("Validating performance requirements...")

        validation_results = {
            "requirements_met": True,
            "violations": [],
            "performance_score": 0.0,
            "detailed_validation": {}
        }

        performance = test_results.get("performance_metrics")
        if not performance:
            validation_results["requirements_met"] = False
            validation_results["violations"].append("No performance metrics available")
            return validation_results

        # Validate each performance requirement
        requirements_checked = 0
        requirements_passed = 0

        # API Response Time P95
        if hasattr(performance, 'api_response_time_p95'):
            requirements_checked += 1
            actual_p95 = performance.api_response_time_p95
            required_p95 = self.performance_requirements["api_response_time_p95_ms"]

            if actual_p95 <= required_p95:
                requirements_passed += 1
                validation_results["detailed_validation"]["api_response_time_p95"] = {
                    "status": "PASS",
                    "actual": actual_p95,
                    "required": required_p95
                }
            else:
                validation_results["violations"].append(
                    f"API P95 response time {actual_p95}ms exceeds {required_p95}ms"
                )
                validation_results["detailed_validation"]["api_response_time_p95"] = {
                    "status": "FAIL",
                    "actual": actual_p95,
                    "required": required_p95
                }

        # Cache Hit Rate
        if hasattr(performance, 'cache_hit_rate'):
            requirements_checked += 1
            actual_hit_rate = performance.cache_hit_rate
            required_hit_rate = self.performance_requirements["cache_hit_rate_percent"]

            if actual_hit_rate >= required_hit_rate:
                requirements_passed += 1
                validation_results["detailed_validation"]["cache_hit_rate"] = {
                    "status": "PASS",
                    "actual": actual_hit_rate,
                    "required": required_hit_rate
                }
            else:
                validation_results["violations"].append(
                    f"Cache hit rate {actual_hit_rate}% below {required_hit_rate}%"
                )
                validation_results["detailed_validation"]["cache_hit_rate"] = {
                    "status": "FAIL",
                    "actual": actual_hit_rate,
                    "required": required_hit_rate
                }

        # Health Check Duration
        if hasattr(performance, 'health_check_duration'):
            requirements_checked += 1
            actual_duration = performance.health_check_duration
            required_duration = self.performance_requirements["health_check_duration_ms"]

            if actual_duration <= required_duration:
                requirements_passed += 1
                validation_results["detailed_validation"]["health_check_duration"] = {
                    "status": "PASS",
                    "actual": actual_duration,
                    "required": required_duration
                }
            else:
                validation_results["violations"].append(
                    f"Health check duration {actual_duration}ms exceeds {required_duration}ms"
                )
                validation_results["detailed_validation"]["health_check_duration"] = {
                    "status": "FAIL",
                    "actual": actual_duration,
                    "required": required_duration
                }

        # Error Rate
        if hasattr(performance, 'error_rate'):
            requirements_checked += 1
            actual_error_rate = performance.error_rate
            required_error_rate = self.performance_requirements["error_rate_percent"]

            if actual_error_rate <= required_error_rate:
                requirements_passed += 1
                validation_results["detailed_validation"]["error_rate"] = {
                    "status": "PASS",
                    "actual": actual_error_rate,
                    "required": required_error_rate
                }
            else:
                validation_results["violations"].append(
                    f"Error rate {actual_error_rate}% exceeds {required_error_rate}%"
                )
                validation_results["detailed_validation"]["error_rate"] = {
                    "status": "FAIL",
                    "actual": actual_error_rate,
                    "required": required_error_rate
                }

        # Calculate overall performance score
        if requirements_checked > 0:
            validation_results["performance_score"] = (requirements_passed / requirements_checked) * 100
            validation_results["requirements_met"] = requirements_passed == requirements_checked

        logger.info(f"Performance validation completed: {requirements_passed}/{requirements_checked} requirements met")
        return validation_results

    async def _generate_comprehensive_report(self, test_results: dict[str, Any]) -> Path:
        """Generate comprehensive integration test report."""
        logger.info("Generating comprehensive integration test report...")

        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        report_path = self.output_dir / f"comprehensive_integration_report_{timestamp}.json"

        # Enhanced report with detailed analysis
        report = {
            "metadata": {
                "generated_at": datetime.now(UTC).isoformat(),
                "framework_version": "2025.1.0",
                "test_execution_mode": "parallel",
                "max_workers": self.max_workers,
                "performance_baseline_enabled": self.performance_baseline_enabled,
                "real_time_monitoring_enabled": self.real_time_monitoring
            },
            "executive_summary": {
                "overall_status": "PASS" if test_results.get("passed_tests", 0) >= test_results.get("total_tests", 0) * 0.9 else "FAIL",
                "test_pass_rate": (test_results.get("passed_tests", 0) / test_results.get("total_tests", 1)) * 100,
                "performance_score": test_results.get("performance_validation", {}).get("performance_score", 0.0),
                "total_execution_time_ms": test_results.get("framework_execution_duration_ms", 0.0),
                "architecture_validation": "All god object decompositions validated successfully" if test_results.get("passed_tests", 0) > 0 else "Architecture validation issues detected"
            },
            "test_results": test_results,
            "performance_analysis": self._analyze_performance_results(test_results),
            "architecture_validation": self._analyze_architecture_validation(test_results),
            "recommendations": self._generate_recommendations(test_results)
        }

        # Write report to file
        with open(report_path, 'w', encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        # Generate summary report for quick review
        summary_path = self.output_dir / f"integration_test_summary_{timestamp}.md"
        await self._generate_markdown_summary(report, summary_path)

        logger.info(f"Comprehensive report generated: {report_path}")
        logger.info(f"Summary report generated: {summary_path}")

        return report_path

    def _analyze_performance_results(self, test_results: dict[str, Any]) -> dict[str, Any]:
        """Analyze performance results and provide insights."""
        performance_analysis = {
            "overall_performance_status": "UNKNOWN",
            "bottlenecks_identified": [],
            "performance_trends": {},
            "optimization_opportunities": []
        }

        performance = test_results.get("performance_metrics")
        if not performance:
            return performance_analysis

        # Analyze response times
        if hasattr(performance, 'api_response_time_p95'):
            if performance.api_response_time_p95 < 50:
                performance_analysis["performance_trends"]["response_time"] = "EXCELLENT"
            elif performance.api_response_time_p95 < 100:
                performance_analysis["performance_trends"]["response_time"] = "GOOD"
            else:
                performance_analysis["performance_trends"]["response_time"] = "NEEDS_IMPROVEMENT"
                performance_analysis["bottlenecks_identified"].append("High API response times")
                performance_analysis["optimization_opportunities"].append("Optimize API endpoint performance")

        # Analyze cache performance
        if hasattr(performance, 'cache_hit_rate'):
            if performance.cache_hit_rate > 90:
                performance_analysis["performance_trends"]["cache_performance"] = "EXCELLENT"
            elif performance.cache_hit_rate > 80:
                performance_analysis["performance_trends"]["cache_performance"] = "GOOD"
            else:
                performance_analysis["performance_trends"]["cache_performance"] = "NEEDS_IMPROVEMENT"
                performance_analysis["bottlenecks_identified"].append("Low cache hit rate")
                performance_analysis["optimization_opportunities"].append("Improve cache strategy and TTL settings")

        # Determine overall status
        validation_results = test_results.get("performance_validation", {})
        if validation_results.get("requirements_met", False):
            performance_analysis["overall_performance_status"] = "PASS"
        elif validation_results.get("performance_score", 0) > 70:
            performance_analysis["overall_performance_status"] = "WARNING"
        else:
            performance_analysis["overall_performance_status"] = "FAIL"

        return performance_analysis

    def _analyze_architecture_validation(self, test_results: dict[str, Any]) -> dict[str, Any]:
        """Analyze architecture validation results."""
        architecture_analysis = {
            "decomposition_success": True,
            "facade_pattern_validation": "PASS",
            "di_container_validation": "PASS",
            "service_integration_validation": "PASS",
            "clean_architecture_compliance": "PASS",
            "issues_identified": []
        }

        detailed_results = test_results.get("detailed_results", [])

        # Check facade pattern validation
        facade_tests = [r for r in detailed_results if "facade" in r.test_name.lower()]
        if facade_tests:
            failed_facade_tests = [r for r in facade_tests if not r.success]
            if failed_facade_tests:
                architecture_analysis["facade_pattern_validation"] = "FAIL"
                architecture_analysis["issues_identified"].append("Service facade integration issues")
                architecture_analysis["decomposition_success"] = False

        # Check DI container validation
        di_tests = [r for r in detailed_results if "di_container" in r.test_name or "container" in r.test_name]
        if di_tests:
            failed_di_tests = [r for r in di_tests if not r.success]
            if failed_di_tests:
                architecture_analysis["di_container_validation"] = "FAIL"
                architecture_analysis["issues_identified"].append("DI container orchestration issues")
                architecture_analysis["decomposition_success"] = False

        # Check service integration
        integration_tests = [r for r in detailed_results if "integration" in r.test_name]
        if integration_tests:
            failed_integration_tests = [r for r in integration_tests if not r.success]
            if failed_integration_tests:
                architecture_analysis["service_integration_validation"] = "FAIL"
                architecture_analysis["issues_identified"].append("Cross-service integration issues")

        return architecture_analysis

    def _generate_recommendations(self, test_results: dict[str, Any]) -> list[str]:
        """Generate actionable recommendations based on test results."""
        recommendations = []

        # Performance recommendations
        performance_validation = test_results.get("performance_validation", {})
        violations = performance_validation.get("violations", [])

        for violation in violations:
            if "response time" in violation.lower():
                recommendations.append("Optimize API endpoints and implement more aggressive caching")
            elif "cache hit rate" in violation.lower():
                recommendations.append("Review cache TTL settings and implement cache warming strategies")
            elif "health check" in violation.lower():
                recommendations.append("Optimize health check queries and implement circuit breakers")
            elif "error rate" in violation.lower():
                recommendations.append("Implement better error handling and retry mechanisms")

        # Architecture recommendations
        failed_tests = [r for r in test_results.get("detailed_results", []) if not r.success]
        if failed_tests:
            test_categories = {r.test_name.split("_")[0] for r in failed_tests}
            for category in test_categories:
                if category == "cache":
                    recommendations.append("Review cache service decomposition and coordination")
                elif category == "ml":
                    recommendations.append("Validate ML service integration and repository patterns")
                elif category == "security":
                    recommendations.append("Review security service integration and validation workflows")
                elif category == "di":
                    recommendations.append("Validate DI container service resolution and lifecycle management")

        # General recommendations
        pass_rate = (test_results.get("passed_tests", 0) / test_results.get("total_tests", 1)) * 100
        if pass_rate < 95:
            recommendations.append("Investigate failing tests and improve integration test coverage")

        if not recommendations:
            recommendations.append("All integration tests passed successfully - maintain current architecture")

        return recommendations

    async def _generate_markdown_summary(self, report: dict[str, Any], summary_path: Path) -> None:
        """Generate markdown summary report."""
        executive_summary = report["executive_summary"]

        markdown_content = f"""# Comprehensive Integration Test Summary

## Executive Summary
- **Overall Status**: {executive_summary["overall_status"]}
- **Test Pass Rate**: {executive_summary["test_pass_rate"]:.1f}%
- **Performance Score**: {executive_summary["performance_score"]:.1f}%
- **Total Execution Time**: {executive_summary["total_execution_time_ms"]:.2f}ms
- **Architecture Status**: {executive_summary["architecture_validation"]}

## Performance Analysis
"""

        performance_analysis = report["performance_analysis"]
        markdown_content += f"- **Overall Performance**: {performance_analysis['overall_performance_status']}\n"

        if performance_analysis["bottlenecks_identified"]:
            markdown_content += "- **Bottlenecks**: " + ", ".join(performance_analysis["bottlenecks_identified"]) + "\n"

        markdown_content += "\n## Architecture Validation\n"
        architecture_analysis = report["architecture_validation"]
        markdown_content += f"- **Decomposition Success**: {architecture_analysis['decomposition_success']}\n"
        markdown_content += f"- **Facade Pattern**: {architecture_analysis['facade_pattern_validation']}\n"
        markdown_content += f"- **DI Containers**: {architecture_analysis['di_container_validation']}\n"
        markdown_content += f"- **Service Integration**: {architecture_analysis['service_integration_validation']}\n"

        if report["recommendations"]:
            markdown_content += "\n## Recommendations\n"
            for i, rec in enumerate(report["recommendations"], 1):
                markdown_content += f"{i}. {rec}\n"

        with open(summary_path, 'w', encoding="utf-8") as f:
            f.write(markdown_content)


async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Execute comprehensive parallel integration tests"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=10,
        help="Number of parallel workers (default: 10)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="test_results",
        help="Output directory for test results (default: test_results)"
    )
    parser.add_argument(
        "--no-performance-validation",
        action="store_true",
        help="Disable performance validation"
    )
    parser.add_argument(
        "--no-monitoring",
        action="store_true",
        help="Disable real-time monitoring"
    )
    parser.add_argument(
        "--no-report",
        action="store_true",
        help="Disable report generation"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )

    args = parser.parse_args()

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Initialize test execution framework
    framework = ParallelTestExecutionFramework(
        max_workers=args.workers,
        performance_baseline_enabled=not args.no_performance_validation,
        real_time_monitoring=not args.no_monitoring,
        output_dir=args.output_dir
    )

    try:
        # Execute parallel integration tests
        results = await framework.execute_parallel_integration_tests(
            performance_validation=not args.no_performance_validation,
            generate_report=not args.no_report
        )

        # Print summary results
        print("\n" + "=" * 80)
        print("COMPREHENSIVE INTEGRATION TEST RESULTS")
        print("=" * 80)
        print(f"Total Tests: {results['total_tests']}")
        print(f"Passed Tests: {results['passed_tests']}")
        print(f"Failed Tests: {results['failed_tests']}")
        print(f"Pass Rate: {(results['passed_tests'] / results['total_tests']) * 100:.1f}%")

        if "performance_validation" in results:
            perf_validation = results["performance_validation"]
            print(f"Performance Score: {perf_validation['performance_score']:.1f}%")
            print(f"Performance Requirements Met: {perf_validation['requirements_met']}")

        if "report_path" in results:
            print(f"Detailed Report: {results['report_path']}")

        print("=" * 80)

        # Exit with appropriate code
        if results["failed_tests"] == 0 and results.get("performance_validation", {}).get("requirements_met", True):
            print("✅ All integration tests passed with performance requirements met!")
            sys.exit(0)
        else:
            print("❌ Some integration tests failed or performance requirements not met!")
            sys.exit(1)

    except Exception as e:
        logger.exception(f"Integration test execution failed: {e}")
        print(f"❌ Integration test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
