#!/usr/bin/env python3
"""
Week 3 Implementation Test Suite
Comprehensive testing of real training execution, performance monitoring, and graceful interruption.
"""

import asyncio
import subprocess
import sys
import time
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_cli_commands_enhanced_week3():
    """Test that CLI commands have Week 3 enhancements."""
    print("🧪 Testing Week 3 CLI command enhancements...")

    commands_to_test = [
        ("apes train --help", "interrupts"),  # Should mention user interruption
        ("apes status --help", "refresh"),  # Should have refresh option
        ("apes stop --help", "session")   # Should have session-specific stopping
    ]

    for command, expected_feature in commands_to_test:
        try:
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=30
            )

            if result.returncode == 0:
                if expected_feature.lower() in result.stdout.lower():
                    print(f"✅ {command.split()[1]} command has {expected_feature} feature")
                else:
                    print(f"❌ {command.split()[1]} command missing {expected_feature} feature")
                    return False
            else:
                print(f"❌ {command} failed")
                return False

        except Exception as e:
            print(f"❌ {command} test failed: {e}")
            return False

    return True

def test_signal_handlers():
    """Test signal handler implementation."""
    print("🧪 Testing signal handlers...")

    try:
        from prompt_improver.cli.clean_cli import setup_signal_handlers, graceful_shutdown

        # Test signal handler setup
        setup_signal_handlers()
        print("✅ Signal handlers setup successful")

        # Test graceful shutdown function exists
        if asyncio.iscoroutinefunction(graceful_shutdown):
            print("✅ Graceful shutdown function is async")
        else:
            print("❌ Graceful shutdown function not async")
            return False

        return True

    except Exception as e:
        print(f"❌ Signal handlers test failed: {e}")
        return False

def test_enhanced_performance_monitoring():
    """Test enhanced performance monitoring capabilities."""
    print("🧪 Testing enhanced performance monitoring...")

    try:
        from prompt_improver.cli.core.cli_orchestrator import CLIOrchestrator
        from rich.console import Console

        console = Console()
        orchestrator = CLIOrchestrator(console)

        # Test enhanced monitoring methods
        enhanced_methods = [
            '_get_training_metrics',
            '_calculate_performance_trend',
            '_should_stop_training'
        ]

        for method in enhanced_methods:
            if hasattr(orchestrator, method):
                print(f"✅ Enhanced method {method} exists")
            else:
                print(f"❌ Enhanced method {method} missing")
                return False

        # Test trend calculation
        trend_method = getattr(orchestrator, '_calculate_performance_trend')
        test_history = [
            {"performance": 0.1, "timestamp": time.time()},
            {"performance": 0.2, "timestamp": time.time()},
            {"performance": 0.3, "timestamp": time.time()}
        ]

        correlation = trend_method(test_history)
        if isinstance(correlation, float) and -1 <= correlation <= 1:
            print("✅ Performance trend calculation working")
        else:
            print(f"❌ Performance trend calculation failed: {correlation}")
            return False

        return True

    except Exception as e:
        print(f"❌ Enhanced performance monitoring test failed: {e}")
        return False

def test_intelligent_stopping_criteria():
    """Test intelligent stopping criteria implementation."""
    print("🧪 Testing intelligent stopping criteria...")

    try:
        from prompt_improver.cli.core.cli_orchestrator import CLIOrchestrator
        from rich.console import Console

        console = Console()
        orchestrator = CLIOrchestrator(console)

        # Test stopping decision method
        stopping_method = getattr(orchestrator, '_should_stop_training')

        # Test case 1: Should not stop (good performance)
        should_stop, reason = stopping_method(
            performance_history=[{"performance": 0.8}] * 5,
            consecutive_poor_iterations=0,
            max_poor_iterations=5,
            trend_correlation=0.8,
            min_correlation_threshold=0.7,
            time_since_improvement=10,
            plateau_threshold=300
        )

        if not should_stop:
            print("✅ Intelligent stopping correctly allows good performance")
        else:
            print(f"❌ Intelligent stopping incorrectly stopped: {reason}")
            return False

        # Test case 2: Should stop (poor consecutive iterations)
        should_stop, reason = stopping_method(
            performance_history=[{"performance": 0.1}] * 10,
            consecutive_poor_iterations=6,
            max_poor_iterations=5,
            trend_correlation=-0.5,
            min_correlation_threshold=0.7,
            time_since_improvement=100,
            plateau_threshold=300
        )

        if should_stop:
            print("✅ Intelligent stopping correctly detects poor performance")
        else:
            print("❌ Intelligent stopping failed to detect poor performance")
            return False

        return True

    except Exception as e:
        print(f"❌ Intelligent stopping criteria test failed: {e}")
        return False

def test_enhanced_status_command():
    """Test enhanced status command features."""
    print("🧪 Testing enhanced status command...")

    try:
        # Test that status command runs without error
        result = subprocess.run(
            ["python", "-c", """
import sys
sys.path.insert(0, 'src')
from prompt_improver.cli.clean_cli import app
import typer.testing
runner = typer.testing.CliRunner()
result = runner.invoke(app, ['status', '--help'])
print('Status help output length:', len(result.stdout))
print('Exit code:', result.exit_code)
"""],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=Path(__file__).parent
        )

        if result.returncode == 0:
            output = result.stdout
            if "Status help output length:" in output and "Exit code: 0" in output:
                print("✅ Enhanced status command help working")
            else:
                print(f"❌ Enhanced status command help failed: {output}")
                return False
        else:
            print(f"❌ Enhanced status command test failed: {result.stderr}")
            return False

        return True

    except Exception as e:
        print(f"❌ Enhanced status command test failed: {e}")
        return False

def test_workflow_integration():
    """Test workflow integration with orchestrator."""
    print("🧪 Testing workflow integration...")

    try:
        from prompt_improver.ml.orchestration.core.workflow_execution_engine import WorkflowExecutionEngine
        from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig
        from prompt_improver.ml.orchestration.config.workflow_templates import WorkflowTemplates

        # Test workflow engine initialization
        config = OrchestratorConfig()
        engine = WorkflowExecutionEngine(config)

        print("✅ Workflow execution engine instantiation successful")

        # Test workflow templates
        continuous_workflow = WorkflowTemplates.get_continuous_training_workflow()

        if continuous_workflow.workflow_type == "continuous_training":
            print("✅ Continuous training workflow template available")
        else:
            print("❌ Continuous training workflow template incorrect")
            return False

        # Test workflow has required fields
        required_fields = ['max_iterations', 'continuous', 'retry_policy', 'metadata']
        for field in required_fields:
            if hasattr(continuous_workflow, field):
                print(f"✅ Workflow has {field} field")
            else:
                print(f"❌ Workflow missing {field} field")
                return False

        return True

    except Exception as e:
        print(f"❌ Workflow integration test failed: {e}")
        return False

async def run_all_tests():
    """Run all Week 3 implementation tests."""
    print("🚀 Starting Week 3 Implementation Test Suite")
    print("=" * 60)

    tests = [
        ("Enhanced CLI Commands", test_cli_commands_enhanced_week3),
        ("Signal Handlers", test_signal_handlers),
        ("Enhanced Performance Monitoring", test_enhanced_performance_monitoring),
        ("Intelligent Stopping Criteria", test_intelligent_stopping_criteria),
        ("Enhanced Status Command", test_enhanced_status_command),
        ("Workflow Integration", test_workflow_integration),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)

        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()

            if result:
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")

        except Exception as e:
            print(f"💥 {test_name} ERROR: {e}")

    print("\n" + "=" * 60)
    print(f"🎯 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED - Week 3 implementation successful!")
        print("\n🚀 Ready to proceed to Week 4: Intelligence Features")
    else:
        print(f"⚠️  {total - passed} tests failed - review implementation")

    return passed == total

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
