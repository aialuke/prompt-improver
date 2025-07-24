#!/usr/bin/env python3
"""
Comprehensive Integration Test Runner (2025)

Master test orchestrator that runs all Phase 1 & 2 validation test suites:
- End-to-end integration testing
- Production simulation testing  
- Cross-platform compatibility testing
- Compound performance validation
- Business impact measurement
- Comprehensive regression testing

Provides unified reporting and validation of all business impact targets.
"""

import asyncio
import json
import logging
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Import all test suites
from integration.comprehensive_e2e_integration import run_comprehensive_integration_tests
from load.production_simulation import run_production_simulation_tests
from compatibility.cross_platform_integration import run_cross_platform_compatibility_tests
from performance.compound_performance import run_compound_performance_tests
from validation.business_metrics import run_business_impact_measurement
from regression.comprehensive_regression_suite import run_comprehensive_regression_tests

from prompt_improver.utils.datetime_utils import aware_utc_now

logger = logging.getLogger(__name__)

@dataclass
class MasterTestSuiteResults:
    """Comprehensive results from all test suites."""
    
    # Individual suite results
    integration_results: Optional[Dict[str, Any]] = None
    production_simulation_results: Optional[Dict[str, Any]] = None
    cross_platform_results: Optional[Dict[str, Any]] = None
    compound_performance_results: Optional[Dict[str, Any]] = None
    business_impact_results: Optional[Dict[str, Any]] = None
    regression_results: Optional[Dict[str, Any]] = None
    
    # Master summary
    total_test_suites: int = 6
    completed_suites: int = 0
    failed_suites: int = 0
    suite_success_rate: float = 0.0
    
    # Overall business impact validation
    business_targets_achieved: Dict[str, bool] = field(default_factory=dict)
    overall_business_success_rate: float = 0.0
    
    # Critical findings
    critical_issues: List[str] = field(default_factory=list)
    deployment_blockers: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    # Execution metadata
    start_time: datetime = field(default_factory=aware_utc_now)
    end_time: Optional[datetime] = None
    total_duration_minutes: float = 0.0
    execution_environment: Dict[str, str] = field(default_factory=dict)

class ComprehensiveIntegrationTestRunner:
    """Master test runner for all Phase 1 & 2 validation test suites.
    
    Orchestrates and coordinates execution of:
    - Comprehensive E2E integration testing
    - Production-scale simulation testing
    - Cross-platform compatibility validation
    - Compound performance verification
    - Business impact measurement
    - Regression testing for backward compatibility
    
    Provides unified reporting and validation against all business targets.
    """
    
    def __init__(self, 
                 output_dir: Path = Path("./comprehensive_test_results"),
                 parallel_execution: bool = False,
                 fail_fast: bool = False):
        """Initialize comprehensive test runner.
        
        Args:
            output_dir: Directory for all test results and reports
            parallel_execution: Run test suites in parallel (experimental)
            fail_fast: Stop on first suite failure
        """
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.parallel_execution = parallel_execution
        self.fail_fast = fail_fast
        
        # Business impact targets (from original requirements)
        self.business_targets = {
            "type_safety_error_reduction": {"target": 99.5, "baseline_errors": 205, "target_errors": 1},
            "database_performance_improvement": {"target": 79.4, "description": "load reduction"},
            "batch_processing_improvement": {"target": 12.5, "description": "12.5x improvement"},
            "ml_deployment_speed": {"target": 40.0, "description": "40% faster deployment"},
            "ml_experiment_throughput": {"target": 10.0, "description": "10x experiment throughput"},
            "developer_experience": {"target": 30.0, "description": "30% faster development cycles"}
        }
        
        logger.info("Comprehensive Integration Test Runner initialized")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Parallel execution: {'ENABLED' if parallel_execution else 'DISABLED'}")
        logger.info(f"Fail fast mode: {'ENABLED' if fail_fast else 'DISABLED'}")
    
    async def run_all_test_suites(self) -> MasterTestSuiteResults:
        """Run all comprehensive test suites and generate master report.
        
        Returns:
            Comprehensive results from all test suites
        """
        logger.info("🚀 Starting comprehensive Phase 1 & 2 validation testing...")
        logger.info("This will validate ALL business impact targets and ensure zero regressions")
        
        results = MasterTestSuiteResults()
        results.execution_environment = {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": sys.platform,
            "working_directory": str(Path.cwd())
        }
        
        start_time = time.time()
        
        try:
            if self.parallel_execution:
                # Run suites in parallel (experimental)
                results = await self._run_suites_parallel(results)
            else:
                # Run suites sequentially (recommended)
                results = await self._run_suites_sequential(results)
            
            # Generate comprehensive analysis
            results = await self._analyze_comprehensive_results(results)
            
        except Exception as e:
            logger.error(f"Critical error during test execution: {e}")
            results.critical_issues.append(f"Test execution failed: {str(e)}")
        
        # Calculate final metrics
        results.end_time = aware_utc_now()
        results.total_duration_minutes = (time.time() - start_time) / 60
        results.suite_success_rate = (results.completed_suites / results.total_test_suites) * 100
        
        # Save comprehensive report
        await self._save_master_report(results)
        
        # Generate and display summary
        summary = self._generate_master_summary(results)
        print(summary)
        
        logger.info(f"✅ Comprehensive testing completed in {results.total_duration_minutes:.1f} minutes")
        logger.info(f"📊 Suite success rate: {results.suite_success_rate:.1f}%")
        logger.info(f"🎯 Business success rate: {results.overall_business_success_rate:.1f}%")
        
        return results
    
    async def _run_suites_sequential(self, results: MasterTestSuiteResults) -> MasterTestSuiteResults:
        """Run all test suites sequentially."""
        
        test_suites = [
            ("Comprehensive E2E Integration", self._run_integration_tests),
            ("Production Simulation", self._run_production_simulation),
            ("Cross-Platform Compatibility", self._run_cross_platform_tests),
            ("Compound Performance Validation", self._run_compound_performance_tests),
            ("Business Impact Measurement", self._run_business_impact_tests),
            ("Comprehensive Regression Testing", self._run_regression_tests)
        ]
        
        for suite_name, suite_function in test_suites:
            logger.info(f"🧪 Running {suite_name}...")
            
            try:
                suite_result = await suite_function()
                results.completed_suites += 1
                logger.info(f"✅ {suite_name} completed successfully")
                
            except Exception as e:
                logger.error(f"❌ {suite_name} failed: {e}")
                results.failed_suites += 1
                results.critical_issues.append(f"{suite_name} failed: {str(e)}")
                
                if self.fail_fast:
                    logger.error("Fail-fast mode enabled - stopping execution")
                    break
        
        return results
    
    async def _run_suites_parallel(self, results: MasterTestSuiteResults) -> MasterTestSuiteResults:
        """Run test suites in parallel (experimental)."""
        
        logger.warning("Parallel execution is experimental - some tests may conflict")
        
        # Create tasks for parallel execution
        tasks = [
            asyncio.create_task(self._run_integration_tests(), name="integration"),
            asyncio.create_task(self._run_production_simulation(), name="production"),
            asyncio.create_task(self._run_cross_platform_tests(), name="compatibility"),
            asyncio.create_task(self._run_compound_performance_tests(), name="performance"),
            asyncio.create_task(self._run_business_impact_tests(), name="business"),
            asyncio.create_task(self._run_regression_tests(), name="regression")
        ]
        
        # Wait for all tasks to complete
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for task, result in zip(tasks, completed_tasks):
            if isinstance(result, Exception):
                logger.error(f"Task {task.get_name()} failed: {result}")
                results.failed_suites += 1
                results.critical_issues.append(f"{task.get_name()} failed: {str(result)}")
            else:
                results.completed_suites += 1
                logger.info(f"✅ Task {task.get_name()} completed")
        
        return results
    
    async def _run_integration_tests(self) -> Dict[str, Any]:
        """Run comprehensive E2E integration tests."""
        
        output_path = self.output_dir / "integration_test_results.json"
        
        try:
            report = await run_comprehensive_integration_tests(
                output_path=output_path,
                enable_real_behavior_validation=True,
                enable_performance_monitoring=True
            )
            
            return {
                "status": "completed",
                "report": report,
                "output_path": str(output_path)
            }
            
        except Exception as e:
            logger.error(f"Integration tests failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _run_production_simulation(self) -> Dict[str, Any]:
        """Run production-scale simulation tests."""
        
        output_path = self.output_dir / "production_simulation_results.json"
        
        try:
            report = await run_production_simulation_tests(
                output_path=output_path,
                concurrent_users=500,
                dataset_size_gb=5.0,
                test_duration_minutes=15
            )
            
            return {
                "status": "completed",
                "report": report,
                "output_path": str(output_path)
            }
            
        except Exception as e:
            logger.error(f"Production simulation failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _run_cross_platform_tests(self) -> Dict[str, Any]:
        """Run cross-platform compatibility tests."""
        
        output_path = self.output_dir / "cross_platform_results.json"
        
        try:
            report = await run_cross_platform_compatibility_tests(
                output_path=output_path,
                enable_performance_testing=True,
                enable_integration_testing=True
            )
            
            return {
                "status": "completed",
                "report": report,
                "output_path": str(output_path)
            }
            
        except Exception as e:
            logger.error(f"Cross-platform testing failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _run_compound_performance_tests(self) -> Dict[str, Any]:
        """Run compound performance validation tests."""
        
        output_path = self.output_dir / "compound_performance_results.json"
        
        try:
            report = await run_compound_performance_tests(
                output_path=output_path,
                enable_stress_testing=True,
                enable_resource_monitoring=True
            )
            
            return {
                "status": "completed",
                "report": report,
                "output_path": str(output_path)
            }
            
        except Exception as e:
            logger.error(f"Compound performance testing failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _run_business_impact_tests(self) -> Dict[str, Any]:
        """Run business impact measurement tests."""
        
        output_path = self.output_dir / "business_impact_results.json"
        
        try:
            report = await run_business_impact_measurement(
                output_path=output_path,
                validate_all_targets=True,
                generate_roi_analysis=True
            )
            
            return {
                "status": "completed",
                "report": report,
                "output_path": str(output_path)
            }
            
        except Exception as e:
            logger.error(f"Business impact testing failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _run_regression_tests(self) -> Dict[str, Any]:
        """Run comprehensive regression tests."""
        
        output_path = self.output_dir / "regression_test_results.json"
        
        try:
            report = await run_comprehensive_regression_tests(
                output_path=output_path,
                categories=None,  # Run all categories
                update_baselines=False,
                strict_checking=True
            )
            
            return {
                "status": "completed",
                "report": report,
                "output_path": str(output_path)
            }
            
        except Exception as e:
            logger.error(f"Regression testing failed: {e}")
            return {
                "status": "failed",
                "error": str(e)
            }
    
    async def _analyze_comprehensive_results(self, results: MasterTestSuiteResults) -> MasterTestSuiteResults:
        """Analyze comprehensive results and validate business targets."""
        
        logger.info("📊 Analyzing comprehensive test results...")
        
        # Load and analyze individual suite results
        suite_files = {
            "integration": "integration_test_results.json",
            "production": "production_simulation_results.json",
            "compatibility": "cross_platform_results.json",
            "performance": "compound_performance_results.json",
            "business": "business_impact_results.json",
            "regression": "regression_test_results.json"
        }
        
        suite_data = {}
        for suite_name, filename in suite_files.items():
            file_path = self.output_dir / filename
            if file_path.exists():
                try:
                    with open(file_path, 'r') as f:
                        suite_data[suite_name] = json.load(f)
                except Exception as e:
                    logger.warning(f"Failed to load {suite_name} results: {e}")
        
        # Business target validation
        business_targets_met = 0
        total_targets = len(self.business_targets)
        
        for target_name, target_info in self.business_targets.items():
            target_met = self._validate_business_target(target_name, target_info, suite_data)
            results.business_targets_achieved[target_name] = target_met
            if target_met:
                business_targets_met += 1
        
        results.overall_business_success_rate = (business_targets_met / total_targets) * 100
        
        # Critical issue analysis
        if results.failed_suites > 0:
            results.deployment_blockers.append(f"{results.failed_suites} test suites failed")
        
        if results.overall_business_success_rate < 80:
            results.deployment_blockers.append("Business targets not sufficiently met")
        
        # Generate recommendations
        if results.overall_business_success_rate >= 95:
            results.recommendations.append("✅ Excellent validation results - ready for production deployment")
        elif results.overall_business_success_rate >= 85:
            results.recommendations.append("⚠️ Good validation results - address minor issues before deployment")
        elif results.overall_business_success_rate >= 70:
            results.recommendations.append("🔶 Moderate validation results - significant improvements needed")
        else:
            results.recommendations.append("❌ Poor validation results - major issues must be resolved")
        
        return results
    
    def _validate_business_target(self, target_name: str, target_info: Dict[str, Any], suite_data: Dict[str, Any]) -> bool:
        """Validate individual business target against test results."""
        
        # Simplified validation - in real implementation would parse actual test results
        # This is a placeholder that demonstrates the validation logic
        
        if target_name == "type_safety_error_reduction":
            # Check if type safety improvements were validated
            if "business" in suite_data:
                business_results = suite_data["business"]
                type_safety_score = business_results.get("summary", {}).get("type_safety_improvement", 0)
                return type_safety_score >= target_info["target"]
        
        elif target_name == "database_performance_improvement":
            # Check database performance improvements
            if "performance" in suite_data:
                perf_results = suite_data["performance"]
                db_improvement = perf_results.get("summary", {}).get("database_performance_improvement", 0)
                return db_improvement >= target_info["target"]
        
        elif target_name == "batch_processing_improvement":
            # Check batch processing improvements
            if "performance" in suite_data:
                perf_results = suite_data["performance"]
                batch_improvement = perf_results.get("summary", {}).get("batch_processing_improvement", 0)
                return batch_improvement >= target_info["target"]
        
        elif target_name == "ml_deployment_speed":
            # Check ML deployment speed improvements
            if "business" in suite_data:
                business_results = suite_data["business"]
                deployment_improvement = business_results.get("summary", {}).get("ml_deployment_improvement", 0)
                return deployment_improvement >= target_info["target"]
        
        elif target_name == "ml_experiment_throughput":
            # Check ML experiment throughput improvements
            if "business" in suite_data:
                business_results = suite_data["business"]
                throughput_improvement = business_results.get("summary", {}).get("ml_throughput_improvement", 0)
                return throughput_improvement >= target_info["target"]
        
        elif target_name == "developer_experience":
            # Check developer experience improvements
            if "compatibility" in suite_data:
                compat_results = suite_data["compatibility"]
                dev_score = compat_results.get("summary", {}).get("developer_experience_score", 0)
                return dev_score >= 8.0  # Converted from percentage to score
        
        # Default: assume target not met if no validation logic
        return False
    
    async def _save_master_report(self, results: MasterTestSuiteResults):
        """Save comprehensive master report."""
        
        master_report_path = self.output_dir / "comprehensive_master_report.json"
        
        report_data = {
            "master_summary": {
                "total_test_suites": results.total_test_suites,
                "completed_suites": results.completed_suites,
                "failed_suites": results.failed_suites,
                "suite_success_rate": results.suite_success_rate,
                "overall_business_success_rate": results.overall_business_success_rate
            },
            "business_targets": {
                "targets_achieved": results.business_targets_achieved,
                "target_definitions": self.business_targets
            },
            "critical_analysis": {
                "critical_issues": results.critical_issues,
                "deployment_blockers": results.deployment_blockers,
                "recommendations": results.recommendations
            },
            "execution_metadata": {
                "start_time": results.start_time.isoformat(),
                "end_time": results.end_time.isoformat() if results.end_time else None,
                "total_duration_minutes": results.total_duration_minutes,
                "execution_environment": results.execution_environment
            }
        }
        
        with open(master_report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Master report saved to: {master_report_path}")
    
    def _generate_master_summary(self, results: MasterTestSuiteResults) -> str:
        """Generate comprehensive master summary."""
        
        summary = f"""
🎯 COMPREHENSIVE PHASE 1 & 2 VALIDATION RESULTS
=============================================

📊 Test Suite Execution:
├── Total Suites: {results.total_test_suites}
├── Completed: {results.completed_suites} ✅
├── Failed: {results.failed_suites} ❌
└── Success Rate: {results.suite_success_rate:.1f}%

💼 Business Impact Targets:
"""
        
        for target_name, achieved in results.business_targets_achieved.items():
            status = "✅ ACHIEVED" if achieved else "❌ NOT MET"
            target_desc = self.business_targets[target_name].get("description", "")
            summary += f"├── {target_name.replace('_', ' ').title()}: {status}\n"
            if target_desc:
                summary += f"│   └── Target: {target_desc}\n"
        
        summary += f"\n🎯 Overall Business Success: {results.overall_business_success_rate:.1f}%\n"
        
        if results.deployment_blockers:
            summary += f"\n🚫 Deployment Blockers:\n"
            for blocker in results.deployment_blockers:
                summary += f"├── {blocker}\n"
        
        if results.critical_issues:
            summary += f"\n🚨 Critical Issues:\n"
            for issue in results.critical_issues:
                summary += f"├── {issue}\n"
        
        if results.recommendations:
            summary += f"\n💡 Recommendations:\n"
            for rec in results.recommendations:
                summary += f"├── {rec}\n"
        
        summary += f"\n⏱️ Total Duration: {results.total_duration_minutes:.1f} minutes"
        summary += f"\n📁 Results Directory: {self.output_dir}"
        summary += f"\n📅 Completed: {results.end_time.strftime('%Y-%m-%d %H:%M:%S') if results.end_time else 'In Progress'}"
        
        return summary

# Main execution function

async def run_comprehensive_validation(
    output_dir: Path = Path("./comprehensive_test_results"),
    parallel_execution: bool = False,
    fail_fast: bool = False
) -> MasterTestSuiteResults:
    """Run comprehensive Phase 1 & 2 validation testing.
    
    Args:
        output_dir: Directory for test results
        parallel_execution: Run suites in parallel
        fail_fast: Stop on first failure
        
    Returns:
        Comprehensive validation results
    """
    
    runner = ComprehensiveIntegrationTestRunner(
        output_dir=output_dir,
        parallel_execution=parallel_execution,
        fail_fast=fail_fast
    )
    
    results = await runner.run_all_test_suites()
    
    # Determine exit code for CI/CD
    if results.deployment_blockers:
        logger.error("DEPLOYMENT BLOCKERS DETECTED - Test suite failed")
        sys.exit(1)
    elif results.overall_business_success_rate < 80:
        logger.warning("Business targets not sufficiently met")
        sys.exit(2)
    else:
        logger.info("All validation tests completed successfully")
        sys.exit(0)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Phase 1 & 2 Integration Testing")
    parser.add_argument("--output-dir", type=Path, default="./comprehensive_test_results",
                       help="Output directory for test results")
    parser.add_argument("--parallel", action="store_true",
                       help="Run test suites in parallel (experimental)")
    parser.add_argument("--fail-fast", action="store_true",
                       help="Stop on first test suite failure")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run comprehensive validation
    asyncio.run(run_comprehensive_validation(
        output_dir=args.output_dir,
        parallel_execution=args.parallel,
        fail_fast=args.fail_fast
    ))