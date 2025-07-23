#!/usr/bin/env python3
"""
Test performance profiling script for APES test suite.

This script runs pytest with detailed timing information and generates
performance reports to identify slow tests and optimization opportunities.
"""

import subprocess
import sys
import time
import json
from pathlib import Path


def run_pytest_with_profiling():
    """Run pytest with detailed profiling information."""
    print("🔍 Running pytest with performance profiling...")
    
    # Run pytest with detailed timing and profiling
    cmd = [
        sys.executable, "-m", "pytest", 
        "-v", 
        "--tb=short",
        "--durations=20",  # Show 20 slowest tests
        "--collect-only",  # First, just collect to check for issues
        "-q"
    ]
    
    print(f"Running: {' '.join(cmd)}")
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        collection_time = time.time() - start_time
        
        print(f"✅ Test collection completed in {collection_time:.2f}s")
        print(f"Total tests found: {result.stdout.count('::')}")
        
        if result.returncode != 0:
            print("❌ Test collection failed:")
            print(result.stdout)
            print(result.stderr)
            return False
            
        # Now run actual tests with profiling
        cmd = [
            sys.executable, "-m", "pytest", 
            "-v",
            "--tb=short", 
            "--durations=20",
            "--maxfail=10",  # Stop after 10 failures
            "-x",  # Stop on first failure for faster feedback
            "--timeout=10",  # 10 second timeout per test
        ]
        
        print(f"Running actual tests: {' '.join(cmd)}")
        start_time = time.time()
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        execution_time = time.time() - start_time
        
        print(f"✅ Test execution completed in {execution_time:.2f}s")
        
        # Parse and display results
        lines = result.stdout.split('\n')
        
        # Find test summary
        summary_started = False
        for line in lines:
            if "test session starts" in line:
                print(f"📊 {line}")
            elif "collected" in line and "items" in line:
                print(f"🔢 {line}")
            elif line.startswith("="):
                if "FAILURES" in line or "ERRORS" in line:
                    summary_started = True
                elif summary_started and ("passed" in line or "failed" in line or "error" in line):
                    print(f"📈 {line}")
                    break
        
        # Extract slowest tests
        durations_started = False
        slowest_tests = []
        
        for line in lines:
            if "slowest durations" in line:
                durations_started = True
                print(f"\n🐌 {line}")
            elif durations_started and line.strip():
                if line.startswith("="):
                    break
                if "::" in line and "s" in line:
                    print(f"   {line}")
                    slowest_tests.append(line.strip())
        
        # Display errors/failures summary
        if result.returncode != 0:
            print("\n❌ Test failures/errors detected:")
            in_summary = False
            for line in lines:
                if "FAILURES" in line or "ERRORS" in line:
                    in_summary = True
                elif in_summary and line.startswith("="):
                    in_summary = False
                elif in_summary and line.strip():
                    print(f"   {line}")
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("❌ Test run timed out")
        return False
    except Exception as e:
        print(f"❌ Error running tests: {e}")
        return False


def generate_performance_report():
    """Generate a performance optimization report."""
    print("\n📋 Performance Optimization Report")
    print("=" * 50)
    
    report = {
        "recommendations": [
            "🔧 Use deterministic fixtures with reduced sample sizes",
            "⚡ Implement parallel test execution for independent tests",
            "📦 Cache expensive fixtures at session scope when possible",
            "🗄️ Use PostgreSQL containers for unit tests instead of mocks",
            "🎯 Mock external dependencies to reduce I/O overhead",
            "📊 Profile database operations and optimize queries",
            "🔄 Implement test markers to categorize fast vs slow tests",
            "🚀 Use pytest-xdist for parallel execution"
        ],
        "configuration_improvements": [
            "Enable pytest-benchmark for performance regression testing",
            "Set up CI/CD pipeline with performance monitoring",
            "Configure test timeouts to prevent hanging tests",
            "Use pytest-cov for coverage-guided optimization",
            "Implement test result caching for unchanged code"
        ]
    }
    
    for category, items in report.items():
        print(f"\n{category.replace('_', ' ').title()}:")
        for item in items:
            print(f"  {item}")
    
    return report


def main():
    """Main function to run performance profiling."""
    print("🧪 APES Test Suite Performance Profiler")
    print("=" * 50)
    
    # Run pytest with profiling
    success = run_pytest_with_profiling()
    
    # Generate performance report
    report = generate_performance_report()
    
    # Summary
    print(f"\n{'✅ SUCCESS' if success else '❌ ISSUES DETECTED'}")
    print(f"Status: {'Tests passed' if success else 'Tests failed - see output above'}")
    print(f"Next steps: {'Ready for CI/CD' if success else 'Fix failing tests first'}")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
