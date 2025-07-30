#!/usr/bin/env python3
"""
Real Behavior Test Runner for OpenTelemetry Migration

Runs the comprehensive real behavior test suite following 2025 best practices.
Uses actual infrastructure (PostgreSQL, Redis, OpenTelemetry) instead of mocks
to validate the migrated ML components and monitoring framework.
"""

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

# Test configuration
TEST_MODULES = [
    "tests/integration/test_real_behavior_ml_components.py",
    "tests/integration/test_ml_pipeline_end_to_end.py", 
    "tests/integration/test_opentelemetry_ml_monitoring.py",
    "tests/unit/utils/test_redis_cache.py"  # Updated for OpenTelemetry
]

REQUIRED_SERVICES = [
    "postgresql",
    "redis"
]

class RealBehaviorTestRunner:
    """
    Test runner for real behavior testing with actual infrastructure.
    
    Manages test execution with proper setup/teardown of services
    and comprehensive reporting following 2025 best practices.
    """
    
    def __init__(self, verbose: bool = True, fail_fast: bool = False):
        self.verbose = verbose
        self.fail_fast = fail_fast
        self.test_results: Dict[str, Any] = {}
        self.start_time: Optional[float] = None
        
    def check_prerequisites(self) -> bool:
        """Check that all prerequisites are available."""
        print("🔍 Checking prerequisites...")
        
        # Check Python dependencies
        required_packages = [
            "pytest",
            "pytest-asyncio", 
            "testcontainers",
            "psycopg",
            "coredis",
            "opentelemetry-api",
            "opentelemetry-sdk"
        ]
        
        missing_packages = []
        for package in required_packages:
            try:
                __import__(package.replace("-", "_"))
            except ImportError:
                missing_packages.append(package)
        
        if missing_packages:
            print(f"❌ Missing required packages: {', '.join(missing_packages)}")
            print("   Install with: pip install " + " ".join(missing_packages))
            return False
        
        # Check Docker availability (for testcontainers)
        try:
            result = subprocess.run(
                ["docker", "version"], 
                capture_output=True, 
                text=True, 
                timeout=10
            )
            if result.returncode != 0:
                print("❌ Docker is not running or not accessible")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print("❌ Docker is not installed or not accessible")
            return False
        
        print("✅ All prerequisites satisfied")
        return True
    
    def setup_test_environment(self) -> bool:
        """Setup test environment variables and configuration."""
        print("⚙️  Setting up test environment...")
        
        # Set environment variables for testing
        test_env = {
            "PYTEST_CURRENT_TEST": "real_behavior_tests",
            "OPENTELEMETRY_ENVIRONMENT": "test",
            "REDIS_URL": "redis://localhost:6379/15",  # Use test database
            "DATABASE_URL": "postgresql://localhost:5432/apes_test",
            "LOG_LEVEL": "INFO" if self.verbose else "WARNING",
            "SKIP_SLOW_TESTS": "false",
            "ENABLE_REAL_BEHAVIOR_TESTS": "true"
        }
        
        for key, value in test_env.items():
            os.environ[key] = value
        
        print("✅ Test environment configured")
        return True
    
    def run_test_module(self, test_module: str) -> Dict[str, Any]:
        """Run a single test module and return results."""
        print(f"\n🧪 Running {test_module}...")
        
        start_time = time.time()
        
        # Build pytest command
        cmd = [
            sys.executable, "-m", "pytest",
            test_module,
            "-v" if self.verbose else "-q",
            "--tb=short",
            "--asyncio-mode=auto",
            "--disable-warnings" if not self.verbose else "",
            "-x" if self.fail_fast else ""
        ]
        
        # Remove empty strings
        cmd = [arg for arg in cmd if arg]
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout per module
            )
            
            duration = time.time() - start_time
            
            # Parse pytest output for test counts
            output_lines = result.stdout.split('\n')
            test_count = 0
            passed_count = 0
            failed_count = 0
            
            for line in output_lines:
                if "passed" in line and "failed" in line:
                    # Parse line like "5 passed, 2 failed in 10.5s"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed":
                            passed_count = int(parts[i-1])
                        elif part == "failed":
                            failed_count = int(parts[i-1])
                    test_count = passed_count + failed_count
                elif "passed" in line and "failed" not in line:
                    # Parse line like "10 passed in 5.2s"
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part == "passed":
                            passed_count = int(parts[i-1])
                    test_count = passed_count
            
            return {
                "module": test_module,
                "success": result.returncode == 0,
                "duration": duration,
                "test_count": test_count,
                "passed": passed_count,
                "failed": failed_count,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "return_code": result.returncode
            }
            
        except subprocess.TimeoutExpired:
            return {
                "module": test_module,
                "success": False,
                "duration": 300,
                "test_count": 0,
                "passed": 0,
                "failed": 0,
                "stdout": "",
                "stderr": "Test module timed out after 5 minutes",
                "return_code": -1
            }
        except Exception as e:
            return {
                "module": test_module,
                "success": False,
                "duration": time.time() - start_time,
                "test_count": 0,
                "passed": 0,
                "failed": 0,
                "stdout": "",
                "stderr": f"Error running test: {e}",
                "return_code": -1
            }
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all test modules and return comprehensive results."""
        print("🚀 Starting Real Behavior Test Suite")
        print("=" * 60)
        
        self.start_time = time.time()
        
        # Check prerequisites
        if not self.check_prerequisites():
            return {
                "status": "failed",
                "error": "Prerequisites not satisfied",
                "results": []
            }
        
        # Setup environment
        if not self.setup_test_environment():
            return {
                "status": "failed", 
                "error": "Environment setup failed",
                "results": []
            }
        
        # Run tests
        results = []
        total_passed = 0
        total_failed = 0
        total_tests = 0
        
        for test_module in TEST_MODULES:
            if not Path(test_module).exists():
                print(f"⚠️  Test module not found: {test_module}")
                continue
                
            result = self.run_test_module(test_module)
            results.append(result)
            
            total_tests += result["test_count"]
            total_passed += result["passed"]
            total_failed += result["failed"]
            
            # Print immediate feedback
            status_emoji = "✅" if result["success"] else "❌"
            print(f"{status_emoji} {test_module}: {result['passed']} passed, {result['failed']} failed ({result['duration']:.1f}s)")
            
            # Stop on first failure if fail_fast is enabled
            if self.fail_fast and not result["success"]:
                print("🛑 Stopping on first failure (fail-fast mode)")
                break
        
        total_duration = time.time() - self.start_time
        overall_success = total_failed == 0 and total_tests > 0
        
        return {
            "status": "completed",
            "overall_success": overall_success,
            "total_duration": total_duration,
            "total_tests": total_tests,
            "total_passed": total_passed,
            "total_failed": total_failed,
            "success_rate": total_passed / total_tests if total_tests > 0 else 0,
            "results": results
        }
    
    def print_summary_report(self, results: Dict[str, Any]) -> None:
        """Print comprehensive test summary report."""
        print("\n" + "=" * 60)
        print("📊 REAL BEHAVIOR TEST SUITE SUMMARY")
        print("=" * 60)
        
        if results["status"] != "completed":
            print(f"❌ Test suite failed: {results.get('error', 'Unknown error')}")
            return
        
        # Overall statistics
        status_emoji = "✅" if results["overall_success"] else "❌"
        print(f"{status_emoji} Overall Status: {'PASSED' if results['overall_success'] else 'FAILED'}")
        print(f"📈 Success Rate: {results['success_rate']:.1%}")
        print(f"🧪 Total Tests: {results['total_tests']}")
        print(f"✅ Passed: {results['total_passed']}")
        print(f"❌ Failed: {results['total_failed']}")
        print(f"⏱️  Total Duration: {results['total_duration']:.1f} seconds")
        
        # Per-module breakdown
        print(f"\n📋 Module Results:")
        for result in results["results"]:
            status_emoji = "✅" if result["success"] else "❌"
            module_name = Path(result["module"]).name
            print(f"  {status_emoji} {module_name}")
            print(f"     Tests: {result['test_count']} | Passed: {result['passed']} | Failed: {result['failed']}")
            print(f"     Duration: {result['duration']:.1f}s")
            
            if not result["success"] and result["stderr"]:
                print(f"     Error: {result['stderr'][:100]}...")
        
        # Recommendations
        print(f"\n💡 Recommendations:")
        if results["overall_success"]:
            print("  🎉 All tests passed! The OpenTelemetry migration is working correctly.")
            print("  🚀 Ready for production deployment with real behavior validation.")
        else:
            print("  ⚠️  Some tests failed. Review the failures above.")
            print("  🔧 Fix issues before deploying to production.")
            print("  📝 Consider running individual modules for detailed debugging.")
        
        print("=" * 60)


def main():
    """Main entry point for test runner."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run real behavior tests for OpenTelemetry migration"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Run in quiet mode with minimal output"
    )
    parser.add_argument(
        "--fail-fast", "-x",
        action="store_true", 
        help="Stop on first test failure"
    )
    parser.add_argument(
        "--module", "-m",
        help="Run specific test module only"
    )
    
    args = parser.parse_args()
    
    # Create test runner
    runner = RealBehaviorTestRunner(
        verbose=not args.quiet,
        fail_fast=args.fail_fast
    )
    
    # Run specific module or all tests
    if args.module:
        if Path(args.module).exists():
            result = runner.run_test_module(args.module)
            success = result["success"]
            print(f"\n{'✅' if success else '❌'} {args.module}: {result['passed']} passed, {result['failed']} failed")
        else:
            print(f"❌ Test module not found: {args.module}")
            success = False
    else:
        results = runner.run_all_tests()
        runner.print_summary_report(results)
        success = results.get("overall_success", False)
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
