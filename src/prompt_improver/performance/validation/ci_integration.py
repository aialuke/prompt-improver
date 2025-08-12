"""CI/CD Integration for Performance Monitoring

This module provides comprehensive CI/CD integration for performance monitoring,
regression detection, and automated performance gating for deployment pipelines.

Features:
1. Pre-commit hooks for performance validation
2. CI pipeline integration with exit codes
3. Performance budgets and thresholds management
4. Automated reporting and alerting
5. Deployment gating based on performance criteria
"""

import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiofiles

from .comprehensive_benchmark import (
    ValidationBenchmarkFramework,
    ValidationBenchmarkResult,
)
from .memory_profiler import MemoryLeakDetector
from .regression_detector import PerformanceRegressionDetector, RegressionSeverity

logger = logging.getLogger(__name__)


class CIStage(Enum):
    """CI/CD pipeline stages."""

    PRE_COMMIT = "pre_commit"
    UNIT_TESTS = "unit_tests"
    INTEGRATION_TESTS = "integration_tests"
    PERFORMANCE_TESTS = "performance_tests"
    PRE_DEPLOYMENT = "pre_deployment"
    POST_DEPLOYMENT = "post_deployment"


class PerformanceGate(Enum):
    """Performance gate decision results."""

    PASS = "pass"
    WARN = "warn"
    FAIL = "fail"
    BLOCK = "block"  # Critical issues that block deployment


@dataclass
class PerformanceBudget:
    """Performance budget definition."""

    operation_name: str
    max_latency_us: float
    max_memory_kb: float
    min_success_rate: float
    max_regression_percent: float = 10.0
    enable_memory_leak_check: bool = True
    stage_requirements: dict[CIStage, bool] = field(default_factory=dict)

    def model_dump(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "operation_name": self.operation_name,
            "max_latency_us": self.max_latency_us,
            "max_memory_kb": self.max_memory_kb,
            "min_success_rate": self.min_success_rate,
            "max_regression_percent": self.max_regression_percent,
            "enable_memory_leak_check": self.enable_memory_leak_check,
            "stage_requirements": {
                stage.value: required
                for stage, required in self.stage_requirements.items()
            },
        }


@dataclass
class PerformanceGateResult:
    """Result from performance gate evaluation."""

    stage: CIStage
    gate_decision: PerformanceGate
    overall_score: float  # 0.0 to 100.0
    passed_budgets: int
    total_budgets: int
    critical_violations: list[str]
    warnings: list[str]
    performance_summary: dict[str, Any]
    recommendations: list[str]
    execution_time_seconds: float
    timestamp: datetime

    def model_dump(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "stage": self.stage.value,
            "gate_decision": self.gate_decision.value,
            "overall_score": self.overall_score,
            "passed_budgets": self.passed_budgets,
            "total_budgets": self.total_budgets,
            "critical_violations": self.critical_violations,
            "warnings": self.warnings,
            "performance_summary": self.performance_summary,
            "recommendations": self.recommendations,
            "execution_time_seconds": self.execution_time_seconds,
            "timestamp": self.timestamp.isoformat(),
        }


class PerformanceCIIntegration:
    """CI/CD integration for performance monitoring."""

    def __init__(self, config_dir: Path | None = None):
        self.config_dir = config_dir or Path(".performance")
        self.config_dir.mkdir(exist_ok=True)

        # Initialize components
        self.benchmark_framework = ValidationBenchmarkFramework()
        self.regression_detector = PerformanceRegressionDetector(
            self.config_dir / "regression_data"
        )
        self.memory_detector = MemoryLeakDetector(self.config_dir / "memory_data")

        # Performance budgets
        self.budgets: list[PerformanceBudget] = []

        # CI/CD settings
        self.ci_settings = {
            "enable_performance_gates": True,
            "fail_on_critical_regression": True,
            "fail_on_memory_leaks": True,
            "baseline_comparison_days": 7,
            "min_samples_for_comparison": 10,
            "performance_test_timeout_minutes": 30,
        }

    async def initialize(self) -> None:
        """Initialize CI integration with configuration."""
        await self._load_performance_budgets()
        await self._load_ci_settings()
        await self.regression_detector.initialize()
        logger.info("Performance CI integration initialized")

    async def run_performance_gate(
        self,
        stage: CIStage,
        quick_check: bool = False,
        operations_limit: int | None = None,
    ) -> PerformanceGateResult:
        """Run performance gate for CI/CD stage.

        Args:
            stage: CI/CD stage being evaluated
            quick_check: Run reduced operations for faster feedback
            operations_limit: Limit operations for this run

        Returns:
            Performance gate result with pass/fail decision
        """
        start_time = datetime.now(UTC)
        logger.info(f"Running performance gate for stage: {stage.value}")

        try:
            # Determine test parameters based on stage and quick_check
            test_params = self._get_test_parameters(
                stage, quick_check, operations_limit
            )

            # Run performance benchmarks
            benchmark_results = (
                await self.benchmark_framework.run_comprehensive_benchmark(
                    operations=test_params["operations"],
                    memory_operations=test_params["memory_operations"],
                    concurrent_sessions=test_params["concurrent_sessions"],
                )
            )

            # Check for regressions
            regression_results = await self._check_all_regressions()

            # Memory leak detection (if enabled for this stage)
            memory_results = None
            if test_params["check_memory_leaks"]:
                memory_results = (
                    await self.memory_detector.run_comprehensive_leak_detection(
                        operations=test_params["memory_operations"]
                    )
                )

            # Evaluate against performance budgets
            gate_result = await self._evaluate_performance_gate(
                stage=stage,
                benchmark_results=benchmark_results,
                regression_results=regression_results,
                memory_results=memory_results,
            )

            # Save results for historical tracking
            await self._save_gate_results(gate_result)

            # Generate reports
            await self._generate_ci_report(gate_result)

            execution_time = (datetime.now(UTC) - start_time).total_seconds()
            gate_result.execution_time_seconds = execution_time

            logger.info(
                f"Performance gate completed - Stage: {stage.value}, "
                f"Decision: {gate_result.gate_decision.value}, "
                f"Score: {gate_result.overall_score:.1f}%, "
                f"Time: {execution_time:.1f}s"
            )

            return gate_result

        except Exception as e:
            logger.error(f"Performance gate failed with error: {e}")
            # Return failure result
            return PerformanceGateResult(
                stage=stage,
                gate_decision=PerformanceGate.FAIL,
                overall_score=0.0,
                passed_budgets=0,
                total_budgets=len(self.budgets),
                critical_violations=[f"Performance gate execution failed: {e!s}"],
                warnings=[],
                performance_summary={"error": str(e)},
                recommendations=[
                    "Fix performance gate execution issues before deployment"
                ],
                execution_time_seconds=(datetime.now(UTC) - start_time).total_seconds(),
                timestamp=datetime.now(UTC),
            )

    def _get_test_parameters(
        self, stage: CIStage, quick_check: bool, operations_limit: int | None
    ) -> dict[str, Any]:
        """Get test parameters based on CI stage and constraints."""
        # Base parameters for different stages
        stage_params = {
            CIStage.PRE_COMMIT: {
                "operations": 1000,
                "memory_operations": 5000,
                "concurrent_sessions": 10,
                "check_memory_leaks": False,
            },
            CIStage.UNIT_TESTS: {
                "operations": 5000,
                "memory_operations": 20000,
                "concurrent_sessions": 25,
                "check_memory_leaks": True,
            },
            CIStage.INTEGRATION_TESTS: {
                "operations": 10000,
                "memory_operations": 50000,
                "concurrent_sessions": 50,
                "check_memory_leaks": True,
            },
            CIStage.PERFORMANCE_TESTS: {
                "operations": 50000,
                "memory_operations": 200000,
                "concurrent_sessions": 200,
                "check_memory_leaks": True,
            },
            CIStage.PRE_DEPLOYMENT: {
                "operations": 25000,
                "memory_operations": 100000,
                "concurrent_sessions": 100,
                "check_memory_leaks": True,
            },
            CIStage.POST_DEPLOYMENT: {
                "operations": 10000,
                "memory_operations": 50000,
                "concurrent_sessions": 50,
                "check_memory_leaks": False,  # Focus on quick validation
            },
        }

        params = stage_params.get(stage, stage_params[CIStage.INTEGRATION_TESTS])

        # Apply quick check reductions
        if quick_check:
            params = {
                "operations": params["operations"] // 5,
                "memory_operations": params["memory_operations"] // 10,
                "concurrent_sessions": params["concurrent_sessions"] // 2,
                "check_memory_leaks": False,  # Skip memory leak checks for quick runs
            }

        # Apply operations limit override
        if operations_limit is not None:
            params["operations"] = min(params["operations"], operations_limit)
            params["memory_operations"] = min(
                params["memory_operations"], operations_limit * 2
            )

        return params

    async def _check_all_regressions(self) -> dict[str, Any]:
        """Check for performance regressions across all operations."""
        regression_summary = {
            "total_operations": 0,
            "operations_with_regressions": 0,
            "critical_regressions": [],
            "high_regressions": [],
            "medium_regressions": [],
            "low_regressions": [],
        }

        # Check each operation that has performance budgets
        for budget in self.budgets:
            operation_name = budget.operation_name
            regression_summary["total_operations"] += 1

            try:
                alerts = await self.regression_detector.check_for_regressions(
                    operation_name
                )

                if alerts:
                    regression_summary["operations_with_regressions"] += 1

                    for alert in alerts:
                        alert_info = {
                            "operation": operation_name,
                            "severity": alert.severity.value,
                            "degradation_percent": alert.degradation_percent,
                            "affected_metrics": alert.affected_metrics,
                        }

                        if alert.severity == RegressionSeverity.CRITICAL:
                            regression_summary["critical_regressions"].append(
                                alert_info
                            )
                        elif alert.severity == RegressionSeverity.HIGH:
                            regression_summary["high_regressions"].append(alert_info)
                        elif alert.severity == RegressionSeverity.MEDIUM:
                            regression_summary["medium_regressions"].append(alert_info)
                        else:
                            regression_summary["low_regressions"].append(alert_info)

            except Exception as e:
                logger.warning(f"Failed to check regressions for {operation_name}: {e}")

        return regression_summary

    async def _evaluate_performance_gate(
        self,
        stage: CIStage,
        benchmark_results: dict[str, ValidationBenchmarkResult],
        regression_results: dict[str, Any],
        memory_results: dict[str, Any] | None,
    ) -> PerformanceGateResult:
        """Evaluate performance gate based on budgets and results."""
        critical_violations = []
        warnings = []
        passed_budgets = 0
        score_components = []

        # Evaluate each budget
        for budget in self.budgets:
            # Skip if budget not required for this stage
            if not budget.stage_requirements.get(stage, True):
                continue

            operation_name = budget.operation_name
            budget_passed = True
            budget_score = 100.0

            # Check if we have benchmark results for this operation
            if operation_name in benchmark_results:
                result = benchmark_results[operation_name]

                # Latency check
                if result.current_performance_us > budget.max_latency_us:
                    budget_passed = False
                    violation_percent = (
                        (result.current_performance_us - budget.max_latency_us)
                        / budget.max_latency_us
                    ) * 100
                    critical_violations.append(
                        f"{operation_name}: Latency {result.current_performance_us:.1f}μs exceeds budget {budget.max_latency_us:.1f}μs ({violation_percent:+.1f}%)"
                    )
                    budget_score = max(0, budget_score - violation_percent)

                # Memory check
                if result.memory_usage_kb > budget.max_memory_kb:
                    budget_passed = False
                    violation_percent = (
                        (result.memory_usage_kb - budget.max_memory_kb)
                        / budget.max_memory_kb
                    ) * 100
                    critical_violations.append(
                        f"{operation_name}: Memory {result.memory_usage_kb:.1f}KB exceeds budget {budget.max_memory_kb:.1f}KB ({violation_percent:+.1f}%)"
                    )
                    budget_score = max(0, budget_score - violation_percent)

                # Success rate check
                if result.success_rate < budget.min_success_rate:
                    budget_passed = False
                    critical_violations.append(
                        f"{operation_name}: Success rate {result.success_rate:.1%} below budget {budget.min_success_rate:.1%}"
                    )
                    budget_score = max(
                        0, budget_score - 50
                    )  # Major penalty for reliability issues

            # Check regression results
            regressions = []
            for severity_list in [
                "critical_regressions",
                "high_regressions",
                "medium_regressions",
            ]:
                regressions.extend([
                    r
                    for r in regression_results.get(severity_list, [])
                    if r["operation"] == operation_name
                ])

            for regression in regressions:
                degradation = regression["degradation_percent"]
                if degradation > budget.max_regression_percent:
                    if regression["severity"] in ["critical", "high"]:
                        budget_passed = False
                        critical_violations.append(
                            f"{operation_name}: {regression['severity'].upper()} regression {degradation:.1f}% exceeds budget {budget.max_regression_percent:.1f}%"
                        )
                        budget_score = max(0, budget_score - degradation)
                    else:
                        warnings.append(
                            f"{operation_name}: {regression['severity']} regression {degradation:.1f}%"
                        )
                        budget_score = max(0, budget_score - (degradation / 2))

            # Memory leak check
            if budget.enable_memory_leak_check and memory_results:
                operation_memory_result = memory_results.get(
                    "detailed_results", {}
                ).get(operation_name)
                if operation_memory_result and operation_memory_result.get(
                    "leak_analysis", {}
                ).get("leak_suspected"):
                    leak_patterns = operation_memory_result["leak_analysis"].get(
                        "leak_patterns_detected", []
                    )
                    critical_leaks = [
                        p
                        for p in leak_patterns
                        if p.get("severity") in ["CRITICAL", "HIGH"]
                    ]

                    if critical_leaks:
                        budget_passed = False
                        critical_violations.append(
                            f"{operation_name}: Memory leak detected - {len(critical_leaks)} critical patterns"
                        )
                        budget_score = max(0, budget_score - 30)
                    else:
                        warnings.append(
                            f"{operation_name}: Minor memory growth patterns detected"
                        )
                        budget_score = max(0, budget_score - 10)

            if budget_passed:
                passed_budgets += 1

            score_components.append(budget_score)

        # Calculate overall score and gate decision
        overall_score = statistics.mean(score_components) if score_components else 0.0

        # Determine gate decision
        if critical_violations:
            if (
                len(critical_violations) > len(self.budgets) * 0.5
            ):  # More than 50% critical violations
                gate_decision = PerformanceGate.BLOCK
            else:
                gate_decision = PerformanceGate.FAIL
        elif warnings and overall_score < 80:
            gate_decision = PerformanceGate.WARN
        else:
            gate_decision = PerformanceGate.PASS

        # Generate recommendations
        recommendations = self._generate_gate_recommendations(
            gate_decision, critical_violations, warnings, overall_score
        )

        return PerformanceGateResult(
            stage=stage,
            gate_decision=gate_decision,
            overall_score=overall_score,
            passed_budgets=passed_budgets,
            total_budgets=len(self.budgets),
            critical_violations=critical_violations,
            warnings=warnings,
            performance_summary={
                "benchmark_results_count": len(benchmark_results),
                "regression_summary": regression_results,
                "memory_analysis_available": memory_results is not None,
            },
            recommendations=recommendations,
            execution_time_seconds=0,  # Will be set by caller
            timestamp=datetime.now(UTC),
        )

    def _generate_gate_recommendations(
        self,
        gate_decision: PerformanceGate,
        critical_violations: list[str],
        warnings: list[str],
        overall_score: float,
    ) -> list[str]:
        """Generate recommendations based on gate results."""
        recommendations = []

        if gate_decision == PerformanceGate.BLOCK:
            recommendations.extend([
                "DEPLOYMENT BLOCKED: Critical performance issues detected",
                "Address all critical violations before proceeding",
                "Consider rollback if these issues are newly introduced",
                "Review recent changes for performance impact",
            ])
        elif gate_decision == PerformanceGate.FAIL:
            recommendations.extend([
                "DEPLOYMENT FAILED: Performance budgets not met",
                "Fix critical violations before deployment",
                "Consider performance optimization sprint",
                "Update performance budgets if requirements have changed",
            ])
        elif gate_decision == PerformanceGate.WARN:
            recommendations.extend([
                "DEPLOYMENT WARNING: Performance concerns detected",
                "Monitor performance closely in production",
                "Plan optimization work for next sprint",
                "Consider tightening performance budgets",
            ])
        else:  # PASS
            recommendations.extend([
                "Performance gate passed successfully",
                "All performance budgets met",
                "Continue with deployment pipeline",
            ])

        # Add specific recommendations based on violations
        if len(critical_violations) > 0:
            recommendations.append(
                f"Address {len(critical_violations)} critical performance violations"
            )

        if len(warnings) > 0:
            recommendations.append(f"Review {len(warnings)} performance warnings")

        if overall_score < 70:
            recommendations.append(
                "Overall performance score below 70% - comprehensive review needed"
            )

        return recommendations

    async def _save_gate_results(self, result: PerformanceGateResult) -> None:
        """Save gate results for historical tracking."""
        results_file = self.config_dir / f"gate_results_{result.stage.value}.jsonl"

        # Append result to JSONL file
        async with aiofiles.open(results_file, "a") as f:
            await f.write(json.dumps(result.model_dump()) + "\n")

    async def _generate_ci_report(self, result: PerformanceGateResult) -> None:
        """Generate CI-friendly report formats."""
        # JSON report for machine processing
        json_report = (
            self.config_dir
            / f"performance_gate_report_{result.stage.value}_{int(result.timestamp.timestamp())}.json"
        )
        async with aiofiles.open(json_report, "w") as f:
            await f.write(json.dumps(result.model_dump(), indent=2))

        # Human-readable report
        text_report = (
            self.config_dir
            / f"performance_gate_summary_{result.stage.value}_{int(result.timestamp.timestamp())}.txt"
        )

        lines = [
            "=" * 80,
            f"PERFORMANCE GATE REPORT - {result.stage.value.upper()}",
            "=" * 80,
            f"Timestamp: {result.timestamp.isoformat()}",
            f"Decision: {result.gate_decision.value.upper()}",
            f"Overall Score: {result.overall_score:.1f}%",
            f"Budgets Passed: {result.passed_budgets}/{result.total_budgets}",
            f"Execution Time: {result.execution_time_seconds:.1f}s",
            "",
        ]

        if result.critical_violations:
            lines.extend(["CRITICAL VIOLATIONS:", "-" * 20])
            for violation in result.critical_violations:
                lines.append(f"  ❌ {violation}")
            lines.append("")

        if result.warnings:
            lines.extend(["WARNINGS:", "-" * 20])
            for warning in result.warnings:
                lines.append(f"  ⚠️  {warning}")
            lines.append("")

        lines.extend(["RECOMMENDATIONS:", "-" * 20])
        for recommendation in result.recommendations:
            lines.append(f"  • {recommendation}")

        lines.extend(["", "=" * 80])

        async with aiofiles.open(text_report, "w") as f:
            await f.write("\n".join(lines))

        logger.info(f"Performance gate reports saved: {json_report}, {text_report}")

    async def _load_performance_budgets(self) -> None:
        """Load performance budgets from configuration."""
        budgets_file = self.config_dir / "performance_budgets.json"

        if not budgets_file.exists():
            # Create default budgets based on Validation_Consolidation.md targets
            default_budgets = [
                PerformanceBudget(
                    operation_name="mcp_message_decode",
                    max_latency_us=6.4,  # Target from analysis
                    max_memory_kb=400,  # 84% reduction target
                    min_success_rate=0.999,  # 99.9%
                    max_regression_percent=15.0,
                    stage_requirements={
                        CIStage.INTEGRATION_TESTS: True,
                        CIStage.PERFORMANCE_TESTS: True,
                        CIStage.PRE_DEPLOYMENT: True,
                    },
                ),
                PerformanceBudget(
                    operation_name="config_instantiation",
                    max_latency_us=8.4,  # Target from analysis
                    max_memory_kb=200,
                    min_success_rate=0.995,  # 99.5%
                    max_regression_percent=10.0,
                    stage_requirements={
                        CIStage.UNIT_TESTS: True,
                        CIStage.INTEGRATION_TESTS: True,
                        CIStage.PRE_DEPLOYMENT: True,
                    },
                ),
                PerformanceBudget(
                    operation_name="metrics_collection",
                    max_latency_us=1.0,  # Target from analysis
                    max_memory_kb=100,
                    min_success_rate=0.999,  # 99.9%
                    max_regression_percent=20.0,
                    stage_requirements={
                        CIStage.INTEGRATION_TESTS: True,
                        CIStage.PERFORMANCE_TESTS: True,
                        CIStage.PRE_DEPLOYMENT: True,
                    },
                ),
            ]

            await self._save_performance_budgets(default_budgets)
            self.budgets = default_budgets
        else:
            try:
                async with aiofiles.open(budgets_file) as f:
                    content = await f.read()
                    budgets_data = json.loads(content)

                self.budgets = []
                for budget_data in budgets_data:
                    stage_requirements = {}
                    for stage_name, required in budget_data.get(
                        "stage_requirements", {}
                    ).items():
                        try:
                            stage_requirements[CIStage(stage_name)] = required
                        except ValueError:
                            logger.warning(f"Unknown CI stage in budget: {stage_name}")

                    budget = PerformanceBudget(
                        operation_name=budget_data["operation_name"],
                        max_latency_us=budget_data["max_latency_us"],
                        max_memory_kb=budget_data["max_memory_kb"],
                        min_success_rate=budget_data["min_success_rate"],
                        max_regression_percent=budget_data.get(
                            "max_regression_percent", 10.0
                        ),
                        enable_memory_leak_check=budget_data.get(
                            "enable_memory_leak_check", True
                        ),
                        stage_requirements=stage_requirements,
                    )
                    self.budgets.append(budget)

                logger.info(f"Loaded {len(self.budgets)} performance budgets")
            except Exception as e:
                logger.error(f"Failed to load performance budgets: {e}")
                self.budgets = []

    async def _save_performance_budgets(self, budgets: list[PerformanceBudget]) -> None:
        """Save performance budgets to configuration file."""
        budgets_file = self.config_dir / "performance_budgets.json"
        budgets_data = [budget.model_dump() for budget in budgets]

        async with aiofiles.open(budgets_file, "w") as f:
            await f.write(json.dumps(budgets_data, indent=2))

    async def _load_ci_settings(self) -> None:
        """Load CI settings from configuration."""
        settings_file = self.config_dir / "ci_settings.json"

        if settings_file.exists():
            try:
                async with aiofiles.open(settings_file) as f:
                    content = await f.read()
                    loaded_settings = json.loads(content)
                    self.ci_settings.update(loaded_settings)
                logger.info("Loaded CI settings from configuration")
            except Exception as e:
                logger.warning(f"Failed to load CI settings: {e}")

        # Save current settings
        async with aiofiles.open(settings_file, "w") as f:
            await f.write(json.dumps(self.ci_settings, indent=2))

    def get_exit_code(self, gate_result: PerformanceGateResult) -> int:
        """Get appropriate exit code for CI/CD pipeline.

        Returns:
            0: Success (PASS)
            1: Warning (WARN) - non-blocking
            2: Failure (FAIL) - should block deployment
            3: Critical (BLOCK) - must block deployment
        """
        exit_codes = {
            PerformanceGate.PASS: 0,
            PerformanceGate.WARN: 1,
            PerformanceGate.FAIL: 2,
            PerformanceGate.BLOCK: 3,
        }
        return exit_codes.get(gate_result.gate_decision, 2)


# Factory function
def get_ci_integration(config_dir: Path | None = None) -> PerformanceCIIntegration:
    """Get or create CI integration instance."""
    return PerformanceCIIntegration(config_dir)


# CLI interface for CI/CD integration
async def run_performance_gate_cli(
    stage: str,
    quick: bool = False,
    operations_limit: int | None = None,
    config_dir: str | None = None,
) -> int:
    """Run performance gate from CLI for CI/CD integration.

    Args:
        stage: CI stage name
        quick: Enable quick check mode
        operations_limit: Limit number of operations
        config_dir: Configuration directory path

    Returns:
        Exit code for CI/CD pipeline
    """
    try:
        ci_stage = CIStage(stage.lower())
    except ValueError:
        logger.error(f"Invalid CI stage: {stage}")
        print(f"Valid stages: {[s.value for s in CIStage]}")
        return 2

    ci_integration = get_ci_integration(Path(config_dir) if config_dir else None)
    await ci_integration.initialize()

    gate_result = await ci_integration.run_performance_gate(
        stage=ci_stage, quick_check=quick, operations_limit=operations_limit
    )

    # Print summary for CI logs
    print(
        json.dumps(
            {
                "stage": gate_result.stage.value,
                "decision": gate_result.gate_decision.value,
                "score": gate_result.overall_score,
                "budgets_passed": f"{gate_result.passed_budgets}/{gate_result.total_budgets}",
                "critical_violations": len(gate_result.critical_violations),
                "warnings": len(gate_result.warnings),
                "execution_time_seconds": gate_result.execution_time_seconds,
            },
            indent=2,
        )
    )

    # Print recommendations
    if gate_result.recommendations:
        print("\nRecommendations:")
        for rec in gate_result.recommendations:
            print(f"  • {rec}")

    return ci_integration.get_exit_code(gate_result)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Performance CI/CD Integration")
    parser.add_argument("stage", choices=[s.value for s in CIStage], help="CI/CD stage")
    parser.add_argument(
        "--quick", action="store_true", help="Quick check mode (reduced operations)"
    )
    parser.add_argument(
        "--operations-limit", type=int, help="Limit number of operations"
    )
    parser.add_argument(
        "--config-dir", type=str, default=".performance", help="Configuration directory"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.INFO)

    async def main():
        exit_code = await run_performance_gate_cli(
            stage=args.stage,
            quick=args.quick,
            operations_limit=args.operations_limit,
            config_dir=args.config_dir,
        )
        sys.exit(exit_code)

    asyncio.run(main())
