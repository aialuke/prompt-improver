#!/usr/bin/env python3
"""Performance Regression Prevention Framework Demo.
=================================================

Demonstration script for the comprehensive performance regression prevention
framework that protects against reintroduction of:
- 134-1007ms database protocol startup penalties
- TYPE_CHECKING compliance violations
- Heavy dependency contamination during startup

Usage:
    python scripts/regression_prevention_demo.py [--install-hooks] [--check-compliance] [--test-startup]
"""

import asyncio
import logging
import sys
import time
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from prompt_improver.monitoring.regression import (
    ArchitecturalComplianceMonitor,
    CIIntegration,
    RegressionPreventionFramework,
    StartupPerformanceTracker,
    VSCodeDiagnosticsMonitor,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_architectural_compliance():
    """Demonstrate architectural compliance monitoring."""
    print("\n" + "=" * 60)
    print("ARCHITECTURAL COMPLIANCE MONITORING DEMO")
    print("=" * 60)

    monitor = ArchitecturalComplianceMonitor()

    print("Running comprehensive architectural compliance check...")
    report = await monitor.check_compliance(strict=True)

    print("\nCompliance Results:")
    print(f"  Total Protocol Files: {report.total_protocol_files}")
    print(f"  Compliant Files: {report.compliant_files}")
    print(f"  Compliance Ratio: {report.compliance_ratio:.1%}")
    print(f"  Total Violations: {len(report.violations)}")
    print(f"  Estimated Startup Penalty: {report.startup_penalty_estimate_ms}ms")

    if report.violations:
        print("\n⚠️  VIOLATIONS DETECTED:")
        for i, violation in enumerate(report.violations[:5]):  # Show first 5
            print(f"  {i + 1}. {violation.severity.upper()}: {violation.violation_type}")
            print(f"     File: {violation.file_path}:{violation.line_number}")
            print(f"     Description: {violation.description}")
            print(f"     Impact: {violation.impact}")
            print(f"     Suggestion: {violation.suggestion}")
            print()

        if len(report.violations) > 5:
            print(f"     ... and {len(report.violations) - 5} more violations")
    else:
        print("✅ No architectural violations detected!")

    # Domain-specific compliance
    print("\nDomain Compliance Status:")
    for domain, info in report.protocol_domains.items():
        compliance = info["compliant"] / info["files"] if info["files"] > 0 else 1.0
        status = "✅" if compliance == 1.0 else "⚠️" if compliance > 0.5 else "❌"
        print(f"  {status} {domain}: {compliance:.1%} ({info['compliant']}/{info['files']} files)")

    return report


def demo_startup_performance():
    """Demonstrate startup performance tracking."""
    print("\n" + "=" * 60)
    print("STARTUP PERFORMANCE TRACKING DEMO")
    print("=" * 60)

    tracker = StartupPerformanceTracker()

    print("Testing startup performance monitoring...")

    # Simulate startup monitoring (synchronous version)
    with tracker.monitor_startup("demo_application") as profile:
        # Simulate some imports and operations
        import time

        # Simulate work
        time.sleep(0.1)  # 100ms of "work"

        print("  Startup monitoring active...")

    # profile is now populated

    # Get performance summary
    summary = tracker.get_performance_summary()

    print("\nStartup Performance Results:")
    print(f"  Status: {summary['status']}")
    if summary['status'] != 'no_profiles':
        latest = summary['latest_startup']
        print(f"  Startup Time: {latest['duration_seconds']:.3f}s")
        print(f"  Memory Usage: {latest['memory_usage_mb']:.1f}MB")
        print(f"  Total Imports: {latest['total_imports']}")
        print(f"  Heavy Imports: {latest['heavy_imports']}")
        print(f"  Violations: {latest['violations']}")

        if latest['contaminated_dependencies']:
            print(f"  ❌ Contaminated Dependencies: {', '.join(latest['contaminated_dependencies'])}")
        else:
            print("  ✅ No dependency contamination detected")

        # Performance status
        status = summary['performance_status']
        print("\nPerformance Compliance:")
        print(f"  Startup Time: {'✅' if status['startup_time_ok'] else '❌'}")
        print(f"  Memory Usage: {'✅' if status['memory_usage_ok'] else '❌'}")
        print(f"  No Contamination: {'✅' if status['no_contamination'] else '❌'}")
        print(f"  Heavy Imports OK: {'✅' if status['heavy_imports_ok'] else '❌'}")

    # Show slow imports if any
    slow_imports = tracker.get_slow_imports(threshold_seconds=0.01)
    if slow_imports:
        print("\n⚠️  Slow Imports Detected:")
        for imp in slow_imports[:5]:
            print(f"  - {imp['module']}: {imp['duration_seconds']:.3f}s")

    return tracker


async def demo_regression_prevention():
    """Demonstrate the unified regression prevention framework."""
    print("\n" + "=" * 60)
    print("UNIFIED REGRESSION PREVENTION FRAMEWORK DEMO")
    print("=" * 60)

    framework = RegressionPreventionFramework()

    print("Running comprehensive regression check...")
    report = await framework.check_for_regressions(triggered_by="demo")

    print("\nRegression Prevention Results:")
    print(f"  Overall Status: {report.overall_status}")
    print(f"  Framework Health: {report.framework_health:.1%}")
    print(f"  Total Alerts: {len(report.alerts)}")

    if report.alerts:
        print("\n🚨 ALERTS GENERATED:")
        for alert in report.alerts:
            emoji = {"critical": "🔥", "high": "⚠️", "medium": "🟡", "low": "ℹ️"}.get(alert.severity, "🔍")
            print(f"  {emoji} {alert.severity.upper()}: {alert.title}")
            print(f"     Type: {alert.alert_type}")
            print(f"     Description: {alert.description}")
            print(f"     Impact: {alert.impact_estimate}")
            if alert.recommendations:
                print(f"     Recommendations: {'; '.join(alert.recommendations)}")
            if alert.auto_block:
                print("     🚫 AUTO-BLOCKING: This would block CI/CD")
            print()
    else:
        print("✅ No regression alerts generated!")

    # Framework status
    status = framework.get_framework_status()
    print("\nFramework Status:")
    print(f"  Status: {status['status']}")
    print(f"  Active Alerts: {status['active_alerts']}")
    print(f"  Total Checks: {status['total_checks']}")
    print(f"  Last Check: {time.ctime(status['last_check_time']) if status['last_check_time'] else 'Never'}")

    # Performance baselines
    baselines = status['performance_baselines']
    print("\nProtected Performance Baselines:")
    print(f"  Startup Improvement: {baselines['startup_improvement_ratio']:.1%}")
    print(f"  Max Startup Time: {baselines['max_startup_time_ms']}ms")
    print(f"  Protocol Compliance: {baselines['protocol_compliance_ratio']:.1%}")
    print(f"  Zero Critical Violations: {baselines['zero_critical_violations']}")

    return framework, report


def demo_ci_integration():
    """Demonstrate CI/CD integration."""
    print("\n" + "=" * 60)
    print("CI/CD INTEGRATION DEMO")
    print("=" * 60)

    ci_integration = CIIntegration()

    # Show pre-commit config
    print("Generated Pre-commit Configuration:")
    print("-" * 40)
    config = ci_integration.generate_pre_commit_config()
    print(config[:500] + "..." if len(config) > 500 else config)

    print("\n" + "-" * 40)

    # Test startup performance
    print("\nTesting startup performance for CI/CD...")
    result = ci_integration.run_startup_performance_test(max_startup_time=1.0)

    print("Startup Performance Test:")
    print(f"  Result: {'✅ PASS' if result['test_passed'] else '❌ FAIL'}")
    print(f"  Startup Time: {result.get('startup_time_seconds', 'N/A'):.3f}s")
    print(f"  Max Allowed: {result.get('max_allowed_seconds', 'N/A')}s")
    if 'performance_delta' in result:
        delta = result['performance_delta']
        print(f"  Delta: {delta:+.3f}s ({'within target' if delta <= 0 else 'exceeds target'})")

    return ci_integration


async def demo_vscode_integration():
    """Demonstrate VS Code diagnostics integration."""
    print("\n" + "=" * 60)
    print("VS CODE DIAGNOSTICS INTEGRATION DEMO")
    print("=" * 60)

    monitor = VSCodeDiagnosticsMonitor()

    print("VS Code Integration Status:")
    status = monitor.get_monitoring_status()
    print(f"  IDE Available: {'✅' if status['ide_available'] else '❌'}")
    print(f"  Integration Health: {status['integration_health']}")

    if status['ide_available']:
        print("\nAttempting to get current VS Code diagnostics...")
        try:
            diagnostics = await monitor.get_current_diagnostics()
            print(f"  Found diagnostics for {len(diagnostics)} files")

            if diagnostics:
                report = await monitor.analyze_diagnostics_for_regressions(diagnostics)
                print(f"  Regression-related diagnostics: {report.regression_diagnostics}")
                print(f"  Compliance diagnostics: {report.compliance_diagnostics}")
                print(f"  Performance diagnostics: {report.performance_diagnostics}")
        except Exception as e:
            print(f"  Error getting diagnostics: {e}")
    else:
        print("  ⚠️ VS Code diagnostics not available (MCP IDE server not connected)")

        # Demonstrate file-level monitoring instead
        print("\nDemonstrating file-level violation monitoring...")
        from pathlib import Path

        # Find a protocol file to analyze
        protocol_files = list(Path("src/prompt_improver/shared/interfaces/protocols").glob("*.py"))
        if protocol_files:
            sample_file = str(protocol_files[0])
            print(f"  Analyzing: {sample_file}")

            violations = await monitor.monitor_file_for_violations(sample_file)
            if violations:
                print(f"  Found {len(violations)} violations:")
                for v in violations[:3]:
                    print(f"    - {v.severity}: {v.violation_type} at line {v.line_number}")
            else:
                print("  ✅ No violations found in sample file")

    return monitor


async def install_regression_prevention_system():
    """Install the complete regression prevention system."""
    print("\n" + "=" * 60)
    print("INSTALLING REGRESSION PREVENTION SYSTEM")
    print("=" * 60)

    ci_integration = CIIntegration()

    print("Installing pre-commit hooks...")
    success = ci_integration.install_pre_commit_hooks()
    print(f"  Pre-commit hooks: {'✅ Installed' if success else '❌ Failed'}")

    print("Installing GitHub Actions workflow...")
    success = ci_integration.install_github_workflow()
    print(f"  GitHub workflow: {'✅ Installed' if success else '❌ Failed'}")

    print("\n📋 Installation Summary:")
    print("  1. Pre-commit hooks configured for architectural compliance")
    print("  2. GitHub Actions workflow for automated PR checks")
    print("  3. Continuous monitoring framework available")
    print("  4. VS Code integration ready (requires MCP IDE server)")

    print("\n🚀 Next Steps:")
    print("  1. Run 'pre-commit install' to activate hooks")
    print("  2. Commit and push to trigger GitHub Actions")
    print("  3. Configure continuous monitoring in production")
    print("  4. Set up alerting integration with existing monitoring")


async def main():
    """Main demonstration function."""
    import argparse

    parser = argparse.ArgumentParser(description="Performance Regression Prevention Framework Demo")
    parser.add_argument("--install-hooks", action="store_true", help="Install regression prevention system")
    parser.add_argument("--check-compliance", action="store_true", help="Run architectural compliance check only")
    parser.add_argument("--test-startup", action="store_true", help="Test startup performance only")
    parser.add_argument("--ci-demo", action="store_true", help="Demo CI/CD integration only")
    parser.add_argument("--vscode-demo", action="store_true", help="Demo VS Code integration only")
    parser.add_argument("--full-demo", action="store_true", help="Run complete demonstration")

    args = parser.parse_args()

    print("🔍 Performance Regression Prevention Framework")
    print("=" * 60)
    print("Protecting against reintroduction of:")
    print("• 134-1007ms database protocol startup penalties")
    print("• TYPE_CHECKING compliance violations")
    print("• Heavy dependency contamination during startup")
    print("• Architectural violations causing performance regressions")

    try:
        if args.install_hooks:
            await install_regression_prevention_system()
        elif args.check_compliance:
            await demo_architectural_compliance()
        elif args.test_startup:
            demo_startup_performance()
        elif args.ci_demo:
            demo_ci_integration()
        elif args.vscode_demo:
            await demo_vscode_integration()
        elif args.full_demo or not any([args.install_hooks, args.check_compliance, args.test_startup, args.ci_demo, args.vscode_demo]):
            # Run full demo by default
            await demo_architectural_compliance()
            demo_startup_performance()
            await demo_regression_prevention()
            demo_ci_integration()
            await demo_vscode_integration()

        print("\n" + "=" * 60)
        print("✅ FRAMEWORK DEMONSTRATION COMPLETED")
        print("=" * 60)
        print("The regression prevention framework is now ready to protect")
        print("against performance regressions in your development workflow!")

    except Exception as e:
        logger.exception(f"Demo failed: {e}")
        print(f"\n❌ Demo failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
