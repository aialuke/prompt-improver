#!/usr/bin/env python3
"""
Test runner for MultiLevelCache comprehensive real behavior testing.

This script provides a convenient way to run the comprehensive cache testing
suite with various configurations, filters, and reporting options.

Usage:
    python scripts/run_cache_tests.py --help
    python scripts/run_cache_tests.py --test-type integration
    python scripts/run_cache_tests.py --test-type performance --verbose
    python scripts/run_cache_tests.py --test-class TestBasicCacheOperations
    python scripts/run_cache_tests.py --benchmark --report-file cache_benchmark.json
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class CacheTestRunner:
    """
    Comprehensive test runner for MultiLevelCache testing.
    
    Provides orchestration for running different types of cache tests
    with configurable parameters, reporting, and performance analysis.
    """
    
    def __init__(self):
        self.project_root = project_root
        self.test_file = self.project_root / "tests" / "integration" / "test_multi_level_cache_real_behavior.py"
        self.results: Dict[str, Any] = {}
    
    def run_tests(
        self,
        test_type: str = "integration",
        test_class: Optional[str] = None,
        test_method: Optional[str] = None,
        verbose: bool = False,
        capture_output: bool = True,
        markers: Optional[List[str]] = None,
        report_file: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Run cache tests with specified parameters.
        
        Args:
            test_type: Type of tests to run ('all', 'integration', 'performance', 'load')
            test_class: Specific test class to run
            test_method: Specific test method to run
            verbose: Enable verbose output
            capture_output: Capture test output
            markers: Pytest markers to include/exclude
            report_file: File to save test results
            
        Returns:
            Test execution results
        """
        logger.info(f"Starting cache tests - Type: {test_type}")
        
        # Build pytest command
        cmd = self._build_pytest_command(
            test_type, test_class, test_method, verbose, markers
        )
        
        # Execute tests
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                cwd=self.project_root,
                capture_output=capture_output,
                text=True,
                timeout=1800  # 30 minute timeout
            )
            
            execution_time = time.time() - start_time
            
            # Process results
            test_results = {
                'command': ' '.join(cmd),
                'return_code': result.returncode,
                'execution_time_seconds': execution_time,
                'stdout': result.stdout if capture_output else '',
                'stderr': result.stderr if capture_output else '',
                'success': result.returncode == 0,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
                'test_type': test_type,
                'test_class': test_class,
                'test_method': test_method
            }
            
            # Parse pytest output for additional metrics
            if result.stdout:
                test_results.update(self._parse_pytest_output(result.stdout))
            
            self.results = test_results
            
            # Save results if requested
            if report_file:
                self._save_results(report_file, test_results)
            
            # Log summary
            self._log_test_summary(test_results)
            
            return test_results
            
        except subprocess.TimeoutExpired:
            logger.error("Test execution timed out after 30 minutes")
            return {
                'success': False,
                'error': 'Test execution timeout',
                'execution_time_seconds': time.time() - start_time
            }
        except Exception as e:
            logger.error(f"Test execution failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'execution_time_seconds': time.time() - start_time
            }
    
    def _build_pytest_command(
        self,
        test_type: str,
        test_class: Optional[str],
        test_method: Optional[str],
        verbose: bool,
        markers: Optional[List[str]]
    ) -> List[str]:
        """Build pytest command with appropriate parameters."""
        cmd = ['python', '-m', 'pytest']
        
        # Add test file
        cmd.append(str(self.test_file))
        
        # Add specific test selection
        if test_class and test_method:
            cmd.append(f'::{test_class}::{test_method}')
        elif test_class:
            cmd.append(f'::{test_class}')
        
        # Add verbosity
        if verbose:
            cmd.extend(['-v', '-s'])
        
        # Add markers based on test type
        if test_type == 'performance':
            cmd.extend(['-m', 'performance'])
        elif test_type == 'integration':
            cmd.extend(['-m', 'integration or not performance'])
        elif test_type == 'load':
            cmd.extend(['-k', 'load or concurrent or throughput'])
        
        # Add custom markers
        if markers:
            for marker in markers:
                cmd.extend(['-m', marker])
        
        # Add reporting options
        cmd.extend([
            '--tb=short',
            '--strict-markers',
            '--disable-warnings'
        ])
        
        # Add coverage if available
        try:
            import coverage
            cmd.extend(['--cov=src/prompt_improver/utils/multi_level_cache'])
        except ImportError:
            pass
        
        return cmd
    
    def _parse_pytest_output(self, stdout: str) -> Dict[str, Any]:
        """Parse pytest output for metrics."""
        metrics = {}
        
        lines = stdout.split('\n')
        for line in lines:
            # Parse test results summary
            if 'passed' in line and 'failed' in line:
                # Example: "5 passed, 2 failed, 1 skipped in 45.67s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'passed' and i > 0:
                        metrics['tests_passed'] = int(parts[i-1])
                    elif part == 'failed' and i > 0:
                        metrics['tests_failed'] = int(parts[i-1])
                    elif part == 'skipped' and i > 0:
                        metrics['tests_skipped'] = int(parts[i-1])
            
            # Parse execution time
            if 'in' in line and 's' in line:
                # Extract time from "in 45.67s"
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == 'in' and i < len(parts) - 1:
                        time_str = parts[i+1].rstrip('s')
                        try:
                            metrics['pytest_execution_time'] = float(time_str)
                        except ValueError:
                            pass
        
        return metrics
    
    def _save_results(self, report_file: str, results: Dict[str, Any]):
        """Save test results to file."""
        try:
            report_path = Path(report_file)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Test results saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")
    
    def _log_test_summary(self, results: Dict[str, Any]):
        """Log test execution summary."""
        success = results.get('success', False)
        execution_time = results.get('execution_time_seconds', 0)
        tests_passed = results.get('tests_passed', 0)
        tests_failed = results.get('tests_failed', 0)
        tests_skipped = results.get('tests_skipped', 0)
        
        status = "PASSED" if success else "FAILED"
        logger.info(f"Test execution {status}")
        logger.info(f"Execution time: {execution_time:.2f} seconds")
        
        if tests_passed or tests_failed or tests_skipped:
            logger.info(f"Results: {tests_passed} passed, {tests_failed} failed, {tests_skipped} skipped")


class BenchmarkRunner:
    """
    Benchmark runner for cache performance analysis.
    
    Provides specialized benchmarking capabilities with performance
    analysis, regression detection, and detailed reporting.
    """
    
    def __init__(self, test_runner: CacheTestRunner):
        self.test_runner = test_runner
        self.benchmark_results: Dict[str, Any] = {}
    
    def run_benchmark_suite(self, report_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Run comprehensive benchmark suite.
        
        Args:
            report_file: File to save benchmark results
            
        Returns:
            Benchmark results
        """
        logger.info("Starting comprehensive cache benchmark suite")
        
        benchmark_configs = [
            ('basic_operations', 'TestBasicCacheOperations', None),
            ('multi_level_behavior', 'TestMultiLevelBehavior', None),
            ('performance_characteristics', 'TestPerformanceCharacteristics', None),
            ('performance_benchmarking', 'TestPerformanceBenchmarking', None)
        ]
        
        results = {
            'benchmark_suite': 'MultiLevelCache Comprehensive Benchmark',
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'results': {},
            'summary': {}
        }
        
        total_start_time = time.time()
        
        for benchmark_name, test_class, test_method in benchmark_configs:
            logger.info(f"Running benchmark: {benchmark_name}")
            
            benchmark_result = self.test_runner.run_tests(
                test_type='performance',
                test_class=test_class,
                test_method=test_method,
                verbose=True,
                capture_output=True,
                markers=['performance']
            )
            
            results['results'][benchmark_name] = benchmark_result
        
        # Calculate summary metrics
        total_execution_time = time.time() - total_start_time
        total_tests = sum(r.get('tests_passed', 0) + r.get('tests_failed', 0) for r in results['results'].values())
        total_passed = sum(r.get('tests_passed', 0) for r in results['results'].values())
        total_failed = sum(r.get('tests_failed', 0) for r in results['results'].values())
        
        results['summary'] = {
            'total_execution_time_seconds': total_execution_time,
            'total_tests': total_tests,
            'total_passed': total_passed,
            'total_failed': total_failed,
            'success_rate': total_passed / total_tests if total_tests > 0 else 0,
            'overall_success': total_failed == 0
        }
        
        self.benchmark_results = results
        
        # Save results
        if report_file:
            self._save_benchmark_results(report_file, results)
        
        # Log summary
        self._log_benchmark_summary(results)
        
        return results
    
    def _save_benchmark_results(self, report_file: str, results: Dict[str, Any]):
        """Save benchmark results to file."""
        try:
            report_path = Path(report_file)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Benchmark results saved to {report_path}")
        except Exception as e:
            logger.error(f"Failed to save benchmark results: {e}")
    
    def _log_benchmark_summary(self, results: Dict[str, Any]):
        """Log benchmark execution summary."""
        summary = results['summary']
        
        logger.info("=" * 60)
        logger.info("BENCHMARK SUITE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total execution time: {summary['total_execution_time_seconds']:.2f} seconds")
        logger.info(f"Total tests: {summary['total_tests']}")
        logger.info(f"Passed: {summary['total_passed']}")
        logger.info(f"Failed: {summary['total_failed']}")
        logger.info(f"Success rate: {summary['success_rate']:.1%}")
        logger.info(f"Overall result: {'SUCCESS' if summary['overall_success'] else 'FAILURE'}")
        
        # Log individual benchmark results
        for benchmark_name, result in results['results'].items():
            status = "PASS" if result.get('success', False) else "FAIL"
            time_taken = result.get('execution_time_seconds', 0)
            logger.info(f"  {benchmark_name}: {status} ({time_taken:.2f}s)")


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(
        description='Run comprehensive MultiLevelCache tests with real behavior validation'
    )
    
    parser.add_argument(
        '--test-type',
        choices=['all', 'integration', 'performance', 'load', 'basic', 'warming', 'health'],
        default='integration',
        help='Type of tests to run'
    )
    
    parser.add_argument(
        '--test-class',
        help='Specific test class to run'
    )
    
    parser.add_argument(
        '--test-method',
        help='Specific test method to run'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '--report-file',
        help='File to save test results (JSON format)'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Run comprehensive benchmark suite'
    )
    
    parser.add_argument(
        '--markers',
        nargs='*',
        help='Pytest markers to include'
    )
    
    parser.add_argument(
        '--no-capture',
        action='store_true',
        help='Disable output capture (show real-time output)'
    )
    
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Run quick test subset for development'
    )
    
    args = parser.parse_args()
    
    # Initialize test runner
    test_runner = CacheTestRunner()
    
    try:
        if args.benchmark:
            # Run benchmark suite
            benchmark_runner = BenchmarkRunner(test_runner)
            results = benchmark_runner.run_benchmark_suite(args.report_file)
            success = results['summary']['overall_success']
        else:
            # Run regular tests
            if args.quick:
                # Quick test configuration
                test_class = 'TestBasicCacheOperations'
                test_method = 'test_cache_set_and_get_simple_data'
            else:
                test_class = args.test_class
                test_method = args.test_method
            
            results = test_runner.run_tests(
                test_type=args.test_type,
                test_class=test_class,
                test_method=test_method,
                verbose=args.verbose,
                capture_output=not args.no_capture,
                markers=args.markers,
                report_file=args.report_file
            )
            success = results.get('success', False)
        
        # Exit with appropriate code
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("Test execution interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Test runner failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()