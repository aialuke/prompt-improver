#!/usr/bin/env python3
"""
System Metrics Test Runner

This script runs the comprehensive test suite for system_metrics.py validation:
1. Real behavior tests with actual database and Prometheus integration
2. Performance validation tests with <1ms overhead verification
3. APES component integration tests
4. Comprehensive reporting with performance metrics

Usage:
    python scripts/run_system_metrics_tests.py [--performance] [--integration] [--all]
"""

import asyncio
import sys
import time
import argparse
import subprocess
from pathlib import Path
from typing import Dict, List, Any


class SystemMetricsTestRunner:
    """Comprehensive test runner for system metrics validation."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.test_results: Dict[str, Any] = {}
        
    def run_command(self, command: List[str], description: str) -> Dict[str, Any]:
        """Run a command and capture results."""
        print(f"\n🚀 {description}")
        print(f"   Command: {' '.join(command)}")
        
        start_time = time.perf_counter()
        
        try:
            result = subprocess.run(
                command,
                cwd=self.project_root,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            end_time = time.perf_counter()
            duration = (end_time - start_time) * 1000  # Convert to ms
            
            success = result.returncode == 0
            
            if success:
                print(f"   ✅ {description} completed successfully ({duration:.1f}ms)")
            else:
                print(f"   ❌ {description} failed ({duration:.1f}ms)")
                print(f"   Error: {result.stderr}")
            
            return {
                "success": success,
                "duration_ms": duration,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            print(f"   ⏰ {description} timed out after 5 minutes")
            return {
                "success": False,
                "duration_ms": 300000,
                "stdout": "",
                "stderr": "Test timed out",
                "returncode": -1
            }
        except Exception as e:
            print(f"   💥 {description} failed with exception: {e}")
            return {
                "success": False,
                "duration_ms": 0,
                "stdout": "",
                "stderr": str(e),
                "returncode": -1
            }
    
    def run_real_behavior_tests(self) -> bool:
        """Run real behavior tests with actual components."""
        print("\n" + "="*60)
        print("🧪 RUNNING REAL BEHAVIOR TESTS")
        print("="*60)
        
        command = [
            "python", "-m", "pytest",
            "tests/integration/test_system_metrics_real_behavior.py",
            "-v", "--tb=short", "--no-header",
            "--disable-warnings"
        ]
        
        result = self.run_command(command, "Real Behavior Tests")
        self.test_results["real_behavior"] = result
        
        return result["success"]
    
    def run_performance_tests(self) -> bool:
        """Run performance validation tests."""
        print("\n" + "="*60)
        print("⚡ RUNNING PERFORMANCE VALIDATION TESTS")
        print("="*60)
        
        command = [
            "python", "-m", "pytest",
            "tests/performance/test_system_metrics_performance.py",
            "-v", "--tb=short", "--no-header",
            "-m", "performance",
            "--disable-warnings"
        ]
        
        result = self.run_command(command, "Performance Validation Tests")
        self.test_results["performance"] = result
        
        return result["success"]
    
    def run_integration_tests(self) -> bool:
        """Run APES component integration tests."""
        print("\n" + "="*60)
        print("🔗 RUNNING APES INTEGRATION TESTS")
        print("="*60)
        
        command = [
            "python", "-m", "pytest",
            "tests/integration/test_apes_system_metrics_integration.py",
            "-v", "--tb=short", "--no-header",
            "-m", "integration",
            "--disable-warnings"
        ]
        
        result = self.run_command(command, "APES Integration Tests")
        self.test_results["integration"] = result
        
        return result["success"]
    
    def run_type_checking(self) -> bool:
        """Run type checking validation."""
        print("\n" + "="*60)
        print("🔍 RUNNING TYPE CHECKING VALIDATION")
        print("="*60)
        
        command = [
            "python", "-m", "mypy",
            "src/prompt_improver/metrics/system_metrics.py",
            "--strict", "--no-error-summary"
        ]
        
        result = self.run_command(command, "Type Checking Validation")
        self.test_results["type_checking"] = result
        
        return result["success"]
    
    def run_code_quality_checks(self) -> bool:
        """Run code quality validation."""
        print("\n" + "="*60)
        print("📊 RUNNING CODE QUALITY CHECKS")
        print("="*60)
        
        # Run flake8 for code style
        flake8_command = [
            "python", "-m", "flake8",
            "src/prompt_improver/metrics/system_metrics.py",
            "--max-line-length=120",
            "--ignore=E203,W503"
        ]
        
        flake8_result = self.run_command(flake8_command, "Flake8 Code Style Check")
        
        # Run bandit for security
        bandit_command = [
            "python", "-m", "bandit",
            "src/prompt_improver/metrics/system_metrics.py",
            "-f", "txt"
        ]
        
        bandit_result = self.run_command(bandit_command, "Bandit Security Check")
        
        success = flake8_result["success"] and bandit_result["success"]
        
        self.test_results["code_quality"] = {
            "success": success,
            "flake8": flake8_result,
            "bandit": bandit_result
        }
        
        return success
    
    def generate_report(self) -> None:
        """Generate comprehensive test report."""
        print("\n" + "="*60)
        print("📋 COMPREHENSIVE TEST REPORT")
        print("="*60)
        
        total_tests = len(self.test_results)
        passed_tests = sum(1 for result in self.test_results.values() 
                          if isinstance(result, dict) and result.get("success", False))
        
        print(f"\n📊 Overall Results: {passed_tests}/{total_tests} test suites passed")
        
        # Detailed results
        for test_name, result in self.test_results.items():
            if isinstance(result, dict):
                status = "✅ PASSED" if result.get("success", False) else "❌ FAILED"
                duration = result.get("duration_ms", 0)
                print(f"   {status} {test_name.replace('_', ' ').title()}: {duration:.1f}ms")
                
                if not result.get("success", False) and result.get("stderr"):
                    print(f"      Error: {result['stderr'][:200]}...")
        
        # Performance summary
        if "performance" in self.test_results:
            perf_result = self.test_results["performance"]
            if perf_result.get("success", False):
                print(f"\n⚡ Performance Validation:")
                print(f"   ✅ <1ms overhead target verified")
                print(f"   ✅ Concurrent operations validated")
                print(f"   ✅ Memory efficiency confirmed")
        
        # Integration summary
        if "integration" in self.test_results:
            int_result = self.test_results["integration"]
            if int_result.get("success", False):
                print(f"\n🔗 APES Integration:")
                print(f"   ✅ PostgreSQL database compatibility")
                print(f"   ✅ MCP server architecture compatibility")
                print(f"   ✅ CLI component integration")
        
        # Overall assessment
        if passed_tests == total_tests:
            print(f"\n🎉 ALL TESTS PASSED - System metrics implementation is production-ready!")
            print(f"   ✅ Real behavior validation complete")
            print(f"   ✅ Performance targets met (<1ms overhead)")
            print(f"   ✅ APES integration verified")
            print(f"   ✅ Code quality standards met")
        else:
            print(f"\n⚠️  {total_tests - passed_tests} test suite(s) failed - Review required")
            print(f"   Please address failing tests before production deployment")
    
    def run_all_tests(self, include_performance: bool = True, 
                     include_integration: bool = True) -> bool:
        """Run all test suites."""
        print("🚀 Starting Comprehensive System Metrics Test Suite")
        print(f"   Project root: {self.project_root}")
        print(f"   Performance tests: {'Enabled' if include_performance else 'Disabled'}")
        print(f"   Integration tests: {'Enabled' if include_integration else 'Disabled'}")
        
        all_passed = True
        
        # Always run real behavior tests
        if not self.run_real_behavior_tests():
            all_passed = False
        
        # Run performance tests if requested
        if include_performance:
            if not self.run_performance_tests():
                all_passed = False
        
        # Run integration tests if requested
        if include_integration:
            if not self.run_integration_tests():
                all_passed = False
        
        # Always run type checking and code quality
        if not self.run_type_checking():
            all_passed = False
        
        if not self.run_code_quality_checks():
            all_passed = False
        
        # Generate comprehensive report
        self.generate_report()
        
        return all_passed


def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="System Metrics Test Runner")
    parser.add_argument("--performance", action="store_true", 
                       help="Run performance validation tests")
    parser.add_argument("--integration", action="store_true",
                       help="Run APES integration tests")
    parser.add_argument("--all", action="store_true",
                       help="Run all test suites")
    
    args = parser.parse_args()
    
    # Default to running all tests if no specific flags
    if not any([args.performance, args.integration, args.all]):
        args.all = True
    
    runner = SystemMetricsTestRunner()
    
    if args.all:
        success = runner.run_all_tests(include_performance=True, include_integration=True)
    else:
        success = runner.run_all_tests(
            include_performance=args.performance,
            include_integration=args.integration
        )
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
