#!/usr/bin/env python3
"""
Phase 1 Configuration Real Behavior Test Runner

This script runs the comprehensive Phase 1 configuration testing suite with
proper environment setup and detailed reporting.

Usage:
    python scripts/run_phase1_config_tests.py [options]

Options:
    --test-type: Specific test type to run (hot-reload, validation, etc.)
    --postgres-url: Custom PostgreSQL connection URL
    --redis-url: Custom Redis connection URL
    --output-dir: Directory for test reports and logs
    --verbose: Enable verbose logging
    --performance-only: Run only performance tests
    --skip-service-tests: Skip tests requiring external services
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tests.integration.test_phase1_configuration import Phase1ConfigurationRealBehaviorTestSuite


def setup_logging(verbose: bool = False, output_dir: Optional[Path] = None) -> None:
    """Setup logging configuration for test execution."""
    log_level = logging.DEBUG if verbose else logging.INFO
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    
    # File handler (if output directory specified)
    handlers = [console_handler]
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        log_file = output_dir / f"phase1_config_tests_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(console_formatter)
        handlers.append(file_handler)
        print(f"📄 Detailed logs will be written to: {log_file}")
    
    # Setup root logger
    logging.basicConfig(
        level=log_level,
        handlers=handlers,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Set specific logger levels
    logging.getLogger('watchdog').setLevel(logging.WARNING)
    logging.getLogger('aiohttp').setLevel(logging.WARNING)


def setup_test_environment(postgres_url: Optional[str] = None, redis_url: Optional[str] = None) -> None:
    """Setup test environment variables."""
    test_env = {
        "ENVIRONMENT": "development",
        "POSTGRES_HOST": "localhost",
        "POSTGRES_PORT": "5432", 
        "POSTGRES_DATABASE": "prompt_improver_test",
        "POSTGRES_USERNAME": "test_user",
        "POSTGRES_PASSWORD": "test_password",
        "REDIS_URL": "redis://localhost:6379/15",
        "ML_MODEL_PATH": "./models",
        "MONITORING_ENABLED": "true",
        "API_RATE_LIMIT_PER_MINUTE": "100",
        "HEALTH_CHECK_TIMEOUT_SECONDS": "10",
        "METRICS_EXPORT_INTERVAL_SECONDS": "30",
        "TRACING_ENABLED": "true"
    }
    
    # Override with custom URLs if provided
    if postgres_url:
        test_env["DATABASE_URL"] = postgres_url
    if redis_url:
        test_env["REDIS_URL"] = redis_url
    
    # Set environment variables
    for key, value in test_env.items():
        if key not in os.environ:  # Don't override existing values
            os.environ[key] = value
    
    print("🔧 Test environment configured")
    print(f"   PostgreSQL: {os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}")
    print(f"   Redis: {os.getenv('REDIS_URL')}")
    print(f"   Environment: {os.getenv('ENVIRONMENT')}")


async def run_specific_test(suite: Phase1ConfigurationRealBehaviorTestSuite, test_type: str) -> Dict[str, Any]:
    """Run a specific test type."""
    test_mapping = {
        "hot-reload": suite.test_configuration_hot_reload_real_files,
        "environment": suite.test_environment_specific_loading_real_connections,
        "validation": suite.test_startup_validation_real_services,
        "migration": suite.test_hardcoded_migration_real_code_files,
        "schema": suite.test_schema_versioning_real_upgrades,
        "recovery": suite.test_error_recovery_real_scenarios,
        "performance": suite.test_performance_benchmarks_real_load
    }
    
    if test_type not in test_mapping:
        raise ValueError(f"Unknown test type: {test_type}. Available: {list(test_mapping.keys())}")
    
    print(f"🧪 Running specific test: {test_type}")
    result = await test_mapping[test_type]()
    
    return {test_type: result}


async def run_performance_tests_only(suite: Phase1ConfigurationRealBehaviorTestSuite) -> Dict[str, Any]:
    """Run only performance-related tests."""
    print("⚡ Running performance tests only")
    
    results = {}
    
    # Hot-reload performance test
    print("Testing hot-reload performance...")
    results["hot_reload_performance"] = await suite.test_configuration_hot_reload_real_files()
    
    # General performance benchmarks
    print("Testing performance benchmarks...")
    results["performance_benchmarks"] = await suite.test_performance_benchmarks_real_load()
    
    return results


async def run_without_services(suite: Phase1ConfigurationRealBehaviorTestSuite) -> Dict[str, Any]:
    """Run tests that don't require external services."""
    print("🔧 Running tests without external service dependencies")
    
    results = {}
    
    # Configuration hot-reload (file-based only)
    print("Testing configuration hot-reload...")
    results["hot_reload"] = await suite.test_configuration_hot_reload_real_files()
    
    # Hardcoded migration (file-based)
    print("Testing hardcoded migration...")
    results["migration"] = await suite.test_hardcoded_migration_real_code_files()
    
    # Schema versioning (file-based)
    print("Testing schema versioning...")
    results["schema"] = await suite.test_schema_versioning_real_upgrades()
    
    # Error recovery (configuration-only)
    print("Testing error recovery...")
    results["recovery"] = await suite.test_error_recovery_real_scenarios()
    
    return results


def save_test_results(results: Dict[str, Any], output_dir: Path) -> None:
    """Save test results to JSON file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = output_dir / f"phase1_config_test_results_{timestamp}.json"
    
    # Convert results to serializable format
    serializable_results = {}
    for test_name, result in results.items():
        if hasattr(result, '__dict__'):
            # Convert dataclass to dict
            result_dict = {}
            for field, value in result.__dict__.items():
                if isinstance(value, datetime):
                    result_dict[field] = value.isoformat()
                elif isinstance(value, list) and value and hasattr(value[0], '__dict__'):
                    # Handle list of dataclasses (performance metrics)
                    result_dict[field] = [
                        {k: v.isoformat() if isinstance(v, datetime) else v 
                         for k, v in item.__dict__.items()}
                        for item in value
                    ]
                else:
                    result_dict[field] = value
            serializable_results[test_name] = result_dict
        else:
            serializable_results[test_name] = str(result)
    
    # Add metadata
    test_report = {
        "timestamp": datetime.now().isoformat(),
        "test_suite": "Phase1ConfigurationRealBehavior",
        "environment": os.getenv("ENVIRONMENT", "unknown"),
        "python_version": sys.version,
        "results": serializable_results
    }
    
    with open(results_file, 'w') as f:
        json.dump(test_report, f, indent=2)
    
    print(f"📄 Test results saved to: {results_file}")


def print_results_summary(results: Dict[str, Any]) -> None:
    """Print a summary of test results."""
    print("\n" + "="*60)
    print("📊 PHASE 1 CONFIGURATION TEST RESULTS SUMMARY")
    print("="*60)
    
    total_tests = len(results)
    successful_tests = 0
    total_duration = 0.0
    total_services = 0
    
    for test_name, result in results.items():
        if hasattr(result, 'success') and result.success:
            successful_tests += 1
        
        if hasattr(result, 'duration_ms'):
            total_duration += result.duration_ms
        
        if hasattr(result, 'services_validated'):
            total_services += len(result.services_validated)
    
    success_rate = (successful_tests / total_tests) * 100 if total_tests > 0 else 0
    
    print(f"Total Tests: {total_tests}")
    print(f"Successful: {successful_tests}")
    print(f"Failed: {total_tests - successful_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    print(f"Total Duration: {total_duration:.1f}ms")
    print(f"Services Validated: {total_services}")
    
    # Individual test results
    print(f"\n📋 Individual Test Results:")
    for test_name, result in results.items():
        if hasattr(result, 'success'):
            status = "✅ PASSED" if result.success else "❌ FAILED"
            duration = f"{result.duration_ms:.1f}ms" if hasattr(result, 'duration_ms') else "N/A"
            print(f"   {test_name}: {status} ({duration})")
            
            if hasattr(result, 'error_details') and result.error_details:
                print(f"      Error: {result.error_details}")
        else:
            print(f"   {test_name}: ❓ UNKNOWN")
    
    # Performance highlights
    print(f"\n⚡ Performance Highlights:")
    for test_name, result in results.items():
        if hasattr(result, 'performance_metrics') and result.performance_metrics:
            hot_reload_metrics = [m for m in result.performance_metrics if 'reload' in m.operation.lower()]
            if hot_reload_metrics:
                avg_reload = sum(m.duration_ms for m in hot_reload_metrics) / len(hot_reload_metrics)
                print(f"   {test_name} - Hot-reload: {avg_reload:.1f}ms average")
    
    print("="*60)


async def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description="Phase 1 Configuration Real Behavior Test Runner")
    parser.add_argument("--test-type", help="Specific test type to run", 
                       choices=["hot-reload", "environment", "validation", "migration", "schema", "recovery", "performance"])
    parser.add_argument("--postgres-url", help="Custom PostgreSQL connection URL")
    parser.add_argument("--redis-url", help="Custom Redis connection URL")
    parser.add_argument("--output-dir", type=Path, default=Path("test_reports"), help="Directory for test reports")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    parser.add_argument("--performance-only", action="store_true", help="Run only performance tests")
    parser.add_argument("--skip-service-tests", action="store_true", help="Skip tests requiring external services")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.verbose, args.output_dir)
    
    # Setup test environment
    setup_test_environment(args.postgres_url, args.redis_url)
    
    print("🚀 Starting Phase 1 Configuration Real Behavior Tests")
    print(f"Output directory: {args.output_dir}")
    
    # Create test suite
    suite = Phase1ConfigurationRealBehaviorTestSuite()
    
    try:
        # Setup environment
        setup_success = await suite.setup_real_environment()
        if not setup_success:
            print("❌ Failed to setup test environment")
            return 1
        
        # Run tests based on options
        if args.test_type:
            results = await run_specific_test(suite, args.test_type)
        elif args.performance_only:
            results = await run_performance_tests_only(suite)
        elif args.skip_service_tests:
            results = await run_without_services(suite)
        else:
            # Run all tests
            results = await suite.run_all_tests()
        
        # Save results
        args.output_dir.mkdir(parents=True, exist_ok=True)
        save_test_results(results, args.output_dir)
        
        # Print summary
        print_results_summary(results)
        
        # Determine exit code
        if results:
            failed_tests = [name for name, result in results.items() 
                          if hasattr(result, 'success') and not result.success]
            if not failed_tests:
                print("\n🎉 All tests passed!")
                return 0
            else:
                print(f"\n❌ {len(failed_tests)} test(s) failed: {', '.join(failed_tests)}")
                return 1
        else:
            print("\n❌ No tests were executed")
            return 1
    
    except KeyboardInterrupt:
        print("\n⏹️ Tests interrupted by user")
        return 130
    except Exception as e:
        print(f"\n💥 Test runner failed with exception: {e}")
        logging.exception("Test runner exception")
        return 1
    finally:
        # Cleanup
        await suite.teardown_real_environment()


if __name__ == "__main__":
    exit(asyncio.run(main()))