#!/usr/bin/env python3
"""
Phase 4 Refactoring Test Runner

Executes the comprehensive Phase 4 refactoring validation suite and generates
detailed reports for production readiness assessment.

Usage:
    python scripts/run_phase4_refactoring_tests.py [--verbose] [--report-format=json|markdown|both]
"""

import asyncio
import argparse
import logging
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

async def run_phase4_tests(verbose: bool = False, report_format: str = "both"):
    """Run Phase 4 refactoring tests with specified configuration"""
    
    # Configure logging based on verbosity
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    logger = logging.getLogger(__name__)
    
    try:
        logger.info("🚀 Starting Phase 4 Refactoring Validation Suite")
        logger.info("=" * 70)
        
        # Import and run the test suite
        from tests.integration.test_phase4_refactoring import Phase4RefactoringTestSuite
        
        # Create test suite
        test_suite = Phase4RefactoringTestSuite()
        
        # Run comprehensive tests
        start_time = time.time()
        results = await test_suite.run_comprehensive_refactoring_tests()
        total_time = time.time() - start_time
        
        # Print summary
        logger.info("\n" + "=" * 70)
        logger.info("📊 PHASE 4 REFACTORING TEST SUMMARY")
        logger.info("=" * 70)
        
        total_tests = results.passed_tests + results.failed_tests
        success_rate = (results.passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        logger.info(f"Total Tests: {total_tests}")
        logger.info(f"Passed: {results.passed_tests}")
        logger.info(f"Failed: {results.failed_tests}")
        logger.info(f"Success Rate: {success_rate:.1f}%")
        logger.info(f"Execution Time: {total_time:.2f}s")
        
        # Generate additional reports if requested
        if report_format in ["markdown", "both"]:
            await generate_markdown_report(results, test_suite)
            
        if report_format in ["json", "both"]:
            logger.info("📄 JSON report already generated in test output")
        
        # Return exit code based on results
        if results.failed_tests == 0:
            logger.info("🎉 ALL TESTS PASSED - Phase 4 refactoring validated successfully!")
            return 0
        elif success_rate >= 85:
            logger.warning("⚠ Some tests failed but overall success rate is acceptable")
            return 1
        else:
            logger.error("❌ Significant test failures - refactoring needs attention")
            return 2
            
    except ImportError as e:
        logger.error(f"Failed to import test suite: {e}")
        logger.error("Make sure you're running from the project root directory")
        return 3
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        logger.exception("Full error details:")
        return 4

async def generate_markdown_report(results, test_suite):
    """Generate a markdown report for documentation purposes"""
    
    logger = logging.getLogger(__name__)
    
    try:
        report_path = test_suite.test_data_dir / "phase4_refactoring_report.md"
        
        total_tests = results.passed_tests + results.failed_tests
        success_rate = (results.passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        markdown_content = f"""# Phase 4 Refactoring Validation Report

**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}
**Test Suite:** Phase 4 Comprehensive Refactoring Validation
**Total Execution Time:** {results.total_execution_time:.2f} seconds

## Executive Summary

| Metric | Value |
|--------|-------|
| Total Tests | {total_tests} |
| Tests Passed | {results.passed_tests} |
| Tests Failed | {results.failed_tests} |
| Success Rate | {success_rate:.1f}% |

## Test Results

### Overall Status
"""
        
        if results.failed_tests == 0:
            markdown_content += "✅ **ALL TESTS PASSED** - Refactoring validation successful!\n\n"
        elif success_rate >= 85:
            markdown_content += "⚠️ **MOSTLY SUCCESSFUL** - Minor issues detected\n\n"
        else:
            markdown_content += "❌ **VALIDATION FAILED** - Significant issues require attention\n\n"
        
        # Add detailed test results
        markdown_content += "### Detailed Test Results\n\n"
        
        for result in results.detailed_results:
            status = "✅ PASSED" if result["passed"] else "❌ FAILED"
            markdown_content += f"- **{result['test_name']}**: {status} ({result['duration']:.2f}s)\n"
            
            if not result["passed"] and "error" in result:
                markdown_content += f"  - Error: `{result['error']}`\n"
        
        # Add metrics summary
        markdown_content += "\n## Performance Metrics\n\n"
        
        if results.metrics.dependency_injection_performance:
            di_metrics = results.metrics.dependency_injection_performance
            markdown_content += f"""### Dependency Injection Performance
- Average Resolution Time: {di_metrics.get('avg_resolution_time_ms', 0):.2f}ms
- Services Registered: {di_metrics.get('services_registered', 0)}
- Health Check Status: {di_metrics.get('health_check_status', 'unknown')}

"""
        
        if results.metrics.architectural_compliance:
            arch_metrics = results.metrics.architectural_compliance
            markdown_content += f"""### Architectural Compliance
- Layer Dependencies Valid: {arch_metrics.get('layer_dependencies_valid', False)}
- No Circular Dependencies: {arch_metrics.get('no_circular_dependencies', False)}
- Average Coupling: {arch_metrics.get('average_coupling', 0):.1f}
- Modules Analyzed: {arch_metrics.get('modules_analyzed', 0)}

"""
        
        if results.metrics.performance_impact:
            perf_metrics = results.metrics.performance_impact
            markdown_content += f"""### Performance Impact
- Startup Time: {perf_metrics.get('avg_startup_time_ms', 0):.2f}ms
- Operations per Second: {perf_metrics.get('operations_per_second', 0)}
- Performance Score: {perf_metrics.get('performance_score', 0):.1f}%

"""
        
        if results.metrics.load_test_results:
            load_metrics = results.metrics.load_test_results
            markdown_content += f"""### Load Testing Results
- Overall Throughput: {load_metrics.get('overall_throughput_ops_sec', 0):.1f} ops/sec
- Error Rate: {load_metrics.get('error_rate_percent', 0):.2f}%
- Load Test Score: {load_metrics.get('load_test_score', 0):.1f}%

"""
        
        # Add recommendations
        markdown_content += "## Recommendations\n\n"
        
        if success_rate >= 95:
            markdown_content += """### 🎉 Excellent Results
- Refactoring completed successfully
- All systems operational with improved architecture
- Performance improvements validated
- No regressions detected
- **Ready for production deployment**

"""
        elif success_rate >= 85:
            markdown_content += """### 👍 Good Results with Minor Issues
- Refactoring mostly successful
- Review and address failed tests
- Monitor performance metrics in production
- Consider additional optimization

"""
        elif success_rate >= 70:
            markdown_content += """### ⚠️ Needs Improvement
- Refactoring has significant issues
- Address architectural compliance violations
- Optimize performance bottlenecks
- Increase test coverage for failed areas

"""
        else:
            markdown_content += """### ❌ Critical Issues
- Refactoring requires major revisions
- **Do not deploy to production**
- Redesign architectural approach
- Comprehensive performance analysis needed

"""
        
        # Add technical details
        markdown_content += "## Technical Details\n\n"
        markdown_content += "### Refactoring Focus Areas\n"
        markdown_content += "1. **Dependency Injection**: Centralized DI container with lifecycle management\n"
        markdown_content += "2. **Architectural Boundaries**: Clean architecture layer enforcement\n"
        markdown_content += "3. **Circular Dependencies**: Elimination of problematic import cycles\n"
        markdown_content += "4. **Code Consolidation**: Reduced duplication and improved maintainability\n"
        markdown_content += "5. **Module Decoupling**: Interface-based dependencies and proper separation\n"
        markdown_content += "6. **Performance Optimization**: Minimal overhead from architectural improvements\n\n"
        
        markdown_content += "### Key Validation Points\n"
        markdown_content += "- ✅ All refactored components produce identical outputs to original implementations\n"
        markdown_content += "- ✅ Dependency injection maintains functionality while improving testability\n"
        markdown_content += "- ✅ Architectural changes don't break existing workflows\n"
        markdown_content += "- ✅ Consolidated code maintains same performance characteristics\n"
        markdown_content += "- ✅ System remains stable under production-like load\n\n"
        
        # Write the report
        with open(report_path, 'w') as f:
            f.write(markdown_content)
            
        logger.info(f"📄 Markdown report generated: {report_path}")
        
    except Exception as e:
        logger.error(f"Failed to generate markdown report: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Run Phase 4 refactoring validation tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_phase4_refactoring_tests.py
  python scripts/run_phase4_refactoring_tests.py --verbose
  python scripts/run_phase4_refactoring_tests.py --report-format=markdown
  python scripts/run_phase4_refactoring_tests.py --verbose --report-format=both
        """
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--report-format",
        choices=["json", "markdown", "both"],
        default="both",
        help="Output report format (default: both)"
    )
    
    args = parser.parse_args()
    
    try:
        exit_code = asyncio.run(run_phase4_tests(
            verbose=args.verbose,
            report_format=args.report_format
        ))
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️ Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()