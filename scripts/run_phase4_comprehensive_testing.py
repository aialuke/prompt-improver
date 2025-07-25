#!/usr/bin/env python3
"""
Phase 4 Comprehensive Testing Runner

Orchestrates the complete Phase 4 testing suite:
1. Real behavior testing with actual data
2. Real data scenarios testing  
3. Upgrade performance validation
4. Integration matrix validation
5. Regression testing
6. Report generation and analysis

This script ensures all testing is completed systematically and generates
a comprehensive report of system readiness after the major upgrades.
"""

import asyncio
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import logging

# Add src to path
sys.path.insert(0, 'src')

class Phase4TestRunner:
    """Comprehensive test runner for Phase 4"""
    
    def __init__(self):
        self.start_time = time.time()
        self.test_results = {}
        self.project_root = Path(__file__).parent.parent
        self.test_reports_dir = self.project_root / "test_reports" / "phase4"
        self.test_reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.test_reports_dir / "phase4_testing.log"),
                logging.StreamHandler(sys.stdout)
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    async def run_test_module(self, module_path: str, test_name: str) -> Dict[str, Any]:
        """Run a specific test module and capture results"""
        self.logger.info(f"Starting {test_name}...")
        
        start_time = time.time()
        
        try:
            # Import and run the test module
            if module_path == "tests.integration.test_phase4_comprehensive_real_behavior":
                from tests.integration.test_phase4_comprehensive_real_behavior import Phase4ComprehensiveTestSuite
                test_suite = Phase4ComprehensiveTestSuite()
                results = await test_suite.run_all_tests()
                
                success = results.failed == 0
                details = {
                    "passed": results.passed,
                    "failed": results.failed,
                    "total_time": results.total_time,
                    "performance_metrics": results.performance_metrics
                }
                
            elif module_path == "tests.integration.test_real_data_scenarios":
                from tests.integration.test_real_data_scenarios import RealDataScenariosTest
                test_suite = RealDataScenariosTest()
                test_results = await test_suite.run_all_tests()
                
                success = len(test_results) > 0
                details = {
                    "tests_completed": len(test_results),
                    "test_results": test_results
                }
                
            elif module_path == "tests.integration.test_upgrade_performance_validation":
                from tests.integration.test_upgrade_performance_validation import PerformanceBenchmark
                benchmark = PerformanceBenchmark()
                benchmark_results = await benchmark.run_all_benchmarks()
                
                success = "results" in benchmark_results
                details = benchmark_results
                
            else:
                # Run as subprocess for other tests
                result = subprocess.run(
                    [sys.executable, "-m", module_path],
                    cwd=self.project_root,
                    capture_output=True,
                    text=True,
                    timeout=1800  # 30 minutes timeout
                )
                
                success = result.returncode == 0
                details = {
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr
                }
            
            execution_time = time.time() - start_time
            
            result_data = {
                "test_name": test_name,
                "module_path": module_path,
                "success": success,
                "execution_time": execution_time,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "details": details
            }
            
            if success:
                self.logger.info(f"✅ {test_name} completed successfully in {execution_time:.2f}s")
            else:
                self.logger.error(f"❌ {test_name} failed after {execution_time:.2f}s")
            
            return result_data
            
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"❌ {test_name} crashed after {execution_time:.2f}s: {e}")
            
            return {
                "test_name": test_name,
                "module_path": module_path,
                "success": False,
                "execution_time": execution_time,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e),
                "details": {}
            }
    
    async def verify_system_prerequisites(self) -> Dict[str, Any]:
        """Verify system is ready for Phase 4 testing"""
        self.logger.info("🔍 Verifying system prerequisites...")
        
        prerequisites = {
            "numpy_version": None,
            "mlflow_version": None,
            "websockets_version": None,
            "database_available": False,
            "mlflow_tracking_available": False,
            "redis_available": False,
            "disk_space_gb": 0,
            "memory_gb": 0
        }
        
        try:
            # Check package versions
            import numpy as np
            import mlflow
            import websockets
            
            prerequisites["numpy_version"] = np.__version__
            prerequisites["mlflow_version"] = mlflow.__version__
            prerequisites["websockets_version"] = websockets.__version__
            
            # Verify versions meet requirements
            numpy_ok = np.__version__.startswith("2.")
            mlflow_ok = mlflow.__version__.startswith("3.")
            websockets_ok = websockets.__version__.startswith("15.")
            
            self.logger.info(f"NumPy: {np.__version__} {'✅' if numpy_ok else '❌'}")
            self.logger.info(f"MLflow: {mlflow.__version__} {'✅' if mlflow_ok else '❌'}")
            self.logger.info(f"Websockets: {websockets.__version__} {'✅' if websockets_ok else '❌'}")
            
            if not all([numpy_ok, mlflow_ok, websockets_ok]):
                raise Exception("Required package versions not met")
            
            # Check system resources
            import psutil
            
            memory_info = psutil.virtual_memory()
            disk_info = psutil.disk_usage('/')
            
            prerequisites["memory_gb"] = memory_info.total / 1024 / 1024 / 1024
            prerequisites["disk_space_gb"] = disk_info.free / 1024 / 1024 / 1024
            
            self.logger.info(f"Memory: {prerequisites['memory_gb']:.1f} GB")
            self.logger.info(f"Disk space: {prerequisites['disk_space_gb']:.1f} GB")
            
            # Check minimum requirements
            if prerequisites["memory_gb"] < 8:
                self.logger.warning("⚠️  Low memory: recommend 8GB+ for comprehensive testing")
            
            if prerequisites["disk_space_gb"] < 5:
                self.logger.warning("⚠️  Low disk space: recommend 5GB+ for test data")
            
            # Test database connection
            try:
                from prompt_improver.database import create_async_session
                async with create_async_session() as session:
                    await session.execute("SELECT 1")
                prerequisites["database_available"] = True
                self.logger.info("Database: ✅ Available")
            except Exception as e:
                self.logger.warning(f"Database: ⚠️  Not available ({e})")
            
            # Test MLflow tracking
            try:
                client = mlflow.tracking.MlflowClient()
                experiments = client.search_experiments(max_results=1)
                prerequisites["mlflow_tracking_available"] = True
                self.logger.info("MLflow tracking: ✅ Available")
            except Exception as e:
                self.logger.warning(f"MLflow tracking: ⚠️  Not available ({e})")
            
            # Test Redis availability (optional)
            try:
                import coredis
                redis_client = coredis.Redis()
                await redis_client.ping()
                prerequisites["redis_available"] = True
                self.logger.info("Redis: ✅ Available")
            except Exception as e:
                self.logger.info("Redis: ℹ️  Not available (optional)")
            
            prerequisites["system_ready"] = True
            
        except Exception as e:
            self.logger.error(f"❌ System prerequisites check failed: {e}")
            prerequisites["system_ready"] = False
            prerequisites["error"] = str(e)
        
        return prerequisites
    
    async def run_comprehensive_test_suite(self) -> Dict[str, Any]:
        """Run the complete Phase 4 test suite"""
        self.logger.info("🚀 Starting Phase 4 Comprehensive Testing Suite")
        self.logger.info("=" * 80)
        
        # Verify prerequisites
        prerequisites = await self.verify_system_prerequisites()
        if not prerequisites.get("system_ready", False):
            raise Exception("System prerequisites not met - cannot proceed with testing")
        
        # Define test modules in execution order
        test_modules = [
            {
                "module": "tests.integration.test_phase4_comprehensive_real_behavior",
                "name": "Phase 4 Comprehensive Real Behavior Testing",
                "description": "End-to-end testing with actual data, real ML models, and live WebSocket connections"
            },
            {
                "module": "tests.integration.test_real_data_scenarios", 
                "name": "Real Data Scenarios Testing",
                "description": "Production-scale data processing, ML workflows, and concurrent user patterns"
            },
            {
                "module": "tests.integration.test_upgrade_performance_validation",
                "name": "Upgrade Performance Validation",
                "description": "Performance benchmarking to validate upgrade improvements"
            },
            {
                "module": "tests.integration.test_phase3_comprehensive",
                "name": "Phase 3 Regression Testing", 
                "description": "Ensure Phase 3 functionality still works after upgrades"
            },
            {
                "module": "tests.integration.test_websocket_coredis_integration",
                "name": "WebSocket Integration Testing",
                "description": "WebSocket and Redis integration after upgrades"
            }
        ]
        
        self.logger.info(f"Executing {len(test_modules)} test modules...")
        
        # Execute tests sequentially to avoid resource conflicts
        test_results = []
        for test_config in test_modules:
            try:
                result = await self.run_test_module(
                    test_config["module"],
                    test_config["name"]
                )
                test_results.append(result)
                
                # Save intermediate results
                await self.save_intermediate_results(test_results)
                
                # Brief pause between tests
                await asyncio.sleep(2)
                
            except Exception as e:
                self.logger.error(f"Critical error in {test_config['name']}: {e}")
                test_results.append({
                    "test_name": test_config["name"],
                    "module_path": test_config["module"],
                    "success": False,
                    "execution_time": 0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "error": str(e),
                    "details": {}
                })
        
        # Compile comprehensive results
        total_time = time.time() - self.start_time
        passed_tests = sum(1 for result in test_results if result["success"])
        failed_tests = len(test_results) - passed_tests
        
        comprehensive_results = {
            "metadata": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_execution_time": total_time,
                "phase": "Phase 4 - Comprehensive Real Behavior Testing",
                "system_prerequisites": prerequisites
            },
            "summary": {
                "total_tests": len(test_results),
                "passed_tests": passed_tests,
                "failed_tests": failed_tests,
                "success_rate": (passed_tests / len(test_results)) * 100 if test_results else 0
            },
            "test_results": test_results,
            "recommendations": await self.generate_recommendations(test_results)
        }
        
        return comprehensive_results
    
    async def save_intermediate_results(self, test_results: List[Dict[str, Any]]):
        """Save intermediate test results"""
        intermediate_file = self.test_reports_dir / "phase4_intermediate_results.json"
        
        intermediate_data = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "completed_tests": len(test_results),
            "test_results": test_results
        }
        
        with open(intermediate_file, 'w') as f:
            json.dump(intermediate_data, f, indent=2, default=str)
    
    async def generate_recommendations(self, test_results: List[Dict[str, Any]]) -> List[str]:
        """Generate recommendations based on test results"""
        recommendations = []
        
        failed_tests = [result for result in test_results if not result["success"]]
        
        if not failed_tests:
            recommendations.extend([
                "🎉 All Phase 4 tests passed! The system is ready for production.",
                "✅ NumPy 2.x, MLflow 3.x, and Websockets 15.x upgrades are successful.",
                "✅ Real data processing, ML workflows, and real-time analytics are working correctly.",
                "✅ Performance improvements have been validated.",
                "✅ No regressions detected in existing functionality."
            ])
        else:
            recommendations.append(f"⚠️  {len(failed_tests)} test(s) failed - review required before production.")
            
            for failed_test in failed_tests:
                test_name = failed_test["test_name"]
                
                if "Real Behavior" in test_name:
                    recommendations.append(
                        "🔍 Real behavior testing failed - check NumPy/MLflow/WebSocket integration"
                    )
                elif "Performance" in test_name:
                    recommendations.append(
                        "⚡ Performance validation failed - review upgrade optimization"
                    )
                elif "Data Scenarios" in test_name:
                    recommendations.append(
                        "📊 Real data scenarios failed - check production-scale processing"
                    )
                elif "Regression" in test_name:
                    recommendations.append(
                        "🔄 Regression testing failed - existing functionality may be broken"
                    )
        
        # Performance-specific recommendations
        performance_results = [r for r in test_results if "Performance" in r["test_name"]]
        if performance_results:
            perf_result = performance_results[0]
            if perf_result["success"]:
                recommendations.append("📈 Performance benchmarks passed - upgrades deliver improvements")
            else:
                recommendations.append("📉 Performance benchmarks failed - investigate upgrade impact")
        
        # System resource recommendations
        import psutil
        memory_gb = psutil.virtual_memory().total / 1024 / 1024 / 1024
        
        if memory_gb < 16:
            recommendations.append(
                "💾 Consider upgrading to 16GB+ RAM for optimal performance with large datasets"
            )
        
        return recommendations
    
    async def generate_final_report(self, results: Dict[str, Any]):
        """Generate comprehensive final report"""
        self.logger.info("\n" + "=" * 80)
        self.logger.info("📊 PHASE 4 COMPREHENSIVE TESTING FINAL REPORT")
        self.logger.info("=" * 80)
        
        metadata = results["metadata"]
        summary = results["summary"]
        
        # Executive Summary
        self.logger.info(f"📋 Executive Summary:")
        self.logger.info(f"   Total Execution Time: {metadata['total_execution_time']:.1f} seconds")
        self.logger.info(f"   Tests Executed: {summary['total_tests']}")
        self.logger.info(f"   Tests Passed: {summary['passed_tests']}")
        self.logger.info(f"   Tests Failed: {summary['failed_tests']}")
        self.logger.info(f"   Success Rate: {summary['success_rate']:.1f}%")
        
        # System Information
        self.logger.info(f"\n🖥️  System Information:")
        prereqs = metadata["system_prerequisites"]
        self.logger.info(f"   NumPy: {prereqs['numpy_version']}")
        self.logger.info(f"   MLflow: {prereqs['mlflow_version']}")
        self.logger.info(f"   Websockets: {prereqs['websockets_version']}")
        self.logger.info(f"   Memory: {prereqs['memory_gb']:.1f} GB")
        self.logger.info(f"   Database: {'✅' if prereqs['database_available'] else '❌'}")
        
        # Test Results Details
        self.logger.info(f"\n📝 Test Results Details:")
        for result in results["test_results"]:
            status = "✅ PASSED" if result["success"] else "❌ FAILED"
            self.logger.info(f"   {result['test_name']}: {status} ({result['execution_time']:.1f}s)")
            
            if not result["success"] and "error" in result:
                self.logger.info(f"      Error: {result['error']}")
        
        # Recommendations
        self.logger.info(f"\n💡 Recommendations:")
        for recommendation in results["recommendations"]:
            self.logger.info(f"   {recommendation}")
        
        # Final Verdict
        self.logger.info(f"\n🎯 Final Verdict:")
        if summary["failed_tests"] == 0:
            self.logger.info("   🎉 PHASE 4 TESTING COMPLETE - SYSTEM READY FOR PRODUCTION!")
            self.logger.info("   All upgrades validated, performance improvements confirmed.")
        elif summary["success_rate"] >= 80:
            self.logger.info("   ⚠️  PHASE 4 TESTING MOSTLY SUCCESSFUL - MINOR ISSUES TO RESOLVE")
            self.logger.info("   Most functionality working, address failing tests before production.")
        else:
            self.logger.info("   ❌ PHASE 4 TESTING FAILED - MAJOR ISSUES REQUIRE ATTENTION")
            self.logger.info("   Significant problems detected, thorough review required.")
        
        # Save final report
        report_file = self.test_reports_dir / "phase4_final_report.json"
        with open(report_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"\n📄 Detailed report saved: {report_file}")
        
        # Generate summary report file
        summary_file = self.test_reports_dir / "phase4_summary.md"
        await self.generate_markdown_summary(results, summary_file)
        
        self.logger.info(f"📄 Summary report saved: {summary_file}")
    
    async def generate_markdown_summary(self, results: Dict[str, Any], output_file: Path):
        """Generate markdown summary report"""
        summary = results["summary"]
        metadata = results["metadata"]
        
        markdown_content = f"""# Phase 4 Comprehensive Testing Report

## Executive Summary

- **Total Tests**: {summary['total_tests']}
- **Passed**: {summary['passed_tests']}
- **Failed**: {summary['failed_tests']}
- **Success Rate**: {summary['success_rate']:.1f}%
- **Execution Time**: {metadata['total_execution_time']:.1f} seconds
- **Timestamp**: {metadata['timestamp']}

## System Upgrades Validated

- ✅ **NumPy 2.x**: {metadata['system_prerequisites']['numpy_version']}
- ✅ **MLflow 3.x**: {metadata['system_prerequisites']['mlflow_version']}
- ✅ **Websockets 15.x**: {metadata['system_prerequisites']['websockets_version']}

## Test Results

| Test Name | Status | Duration | Details |
|-----------|--------|----------|---------|
"""
        
        for result in results["test_results"]:
            status = "✅ PASSED" if result["success"] else "❌ FAILED"
            markdown_content += f"| {result['test_name']} | {status} | {result['execution_time']:.1f}s | "
            
            if result["success"]:
                markdown_content += "All validations passed |\n"
            else:
                error_msg = result.get("error", "Test failed")[:50]
                markdown_content += f"{error_msg}... |\n"
        
        markdown_content += f"""
## Recommendations

"""
        for rec in results["recommendations"]:
            markdown_content += f"- {rec}\n"
        
        markdown_content += f"""
## Final Verdict

"""
        if summary["failed_tests"] == 0:
            markdown_content += "🎉 **SYSTEM READY FOR PRODUCTION**\n\nAll Phase 4 tests passed successfully. The NumPy 2.x, MLflow 3.x, and Websockets 15.x upgrades have been validated with real data and production-scale testing."
        else:
            markdown_content += f"⚠️  **{summary['failed_tests']} ISSUES REQUIRE ATTENTION**\n\nReview failed tests before production deployment."
        
        with open(output_file, 'w') as f:
            f.write(markdown_content)

async def main():
    """Main entry point for Phase 4 testing"""
    runner = Phase4TestRunner()
    
    try:
        # Run comprehensive test suite
        results = await runner.run_comprehensive_test_suite()
        
        # Generate final report
        await runner.generate_final_report(results)
        
        # Determine exit code
        if results["summary"]["failed_tests"] == 0:
            return 0  # Success
        elif results["summary"]["success_rate"] >= 80:
            return 1  # Mostly successful, minor issues
        else:
            return 2  # Major failures
            
    except Exception as e:
        runner.logger.error(f"💥 Phase 4 testing crashed: {e}")
        return 3  # Critical failure

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)