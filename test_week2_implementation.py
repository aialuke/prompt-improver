#!/usr/bin/env python3
"""
Week 2 Implementation Test Suite
Comprehensive testing of the enhanced CLI-Orchestrator integration.
"""

import asyncio
import subprocess
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_cli_commands_enhanced():
    """Test that all CLI commands have enhanced functionality."""
    print("🧪 Testing enhanced CLI commands...")

    commands_to_test = [
        ("apes train --help", "Auto-i"),  # Matches "Auto-initialize" in truncated output
        ("apes status --help", "detailed"),
        ("apes stop --help", "session")
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
                if expected_feature in result.stdout:
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

def test_cli_orchestrator_methods():
    """Test that CLIOrchestrator has all required methods."""
    print("🧪 Testing CLIOrchestrator enhanced methods...")

    try:
        from prompt_improver.cli.core.cli_orchestrator import CLIOrchestrator
        from rich.console import Console

        console = Console()
        orchestrator = CLIOrchestrator(console)

        # Test original methods
        original_methods = [
            'start_continuous_training',
            'monitor_training_progress',
            'wait_for_completion',
            'stop_all_workflows',
            'get_workflow_status'
        ]

        # Test new methods
        new_methods = [
            'start_single_training',
            'stop_training_gracefully',
            'force_stop_training'
        ]

        all_methods = original_methods + new_methods

        for method in all_methods:
            if hasattr(orchestrator, method):
                print(f"✅ Method {method} exists")
            else:
                print(f"❌ Method {method} missing")
                return False

        return True

    except Exception as e:
        print(f"❌ CLIOrchestrator methods test failed: {e}")
        return False

def test_session_based_approach():
    """Test that the CLI uses session-based approach."""
    print("🧪 Testing session-based approach...")

    try:
        from prompt_improver.cli.core.training_system_manager import TrainingSystemManager
        from prompt_improver.database.models import TrainingSession, TrainingSessionCreate
        from rich.console import Console

        console = Console()
        manager = TrainingSystemManager(console)

        # Test session-related methods
        session_methods = [
            'smart_initialize',
            'validate_ready_for_training',
            'create_training_session',
            'get_system_status',
            'get_active_sessions'
        ]

        for method in session_methods:
            if hasattr(manager, method):
                print(f"✅ Session method {method} exists")
            else:
                print(f"❌ Session method {method} missing")
                return False

        # Test TrainingSession model
        session_data = TrainingSessionCreate(
            session_id="test_session_week2",
            continuous_mode=True,
            improvement_threshold=0.02
        )
        print("✅ TrainingSession model working")

        return True

    except Exception as e:
        print(f"❌ Session-based approach test failed: {e}")
        return False

def test_workflow_templates():
    """Test that workflow templates are available."""
    print("🧪 Testing workflow templates...")

    try:
        from prompt_improver.ml.orchestration.config.workflow_templates import WorkflowTemplates

        # Test that continuous training workflow exists
        continuous_workflow = WorkflowTemplates.get_continuous_training_workflow()

        if continuous_workflow.workflow_type == "continuous_training":
            print("✅ Continuous training workflow template exists")
        else:
            print("❌ Continuous training workflow template incorrect")
            return False

        # Test that standard training workflow exists
        training_workflow = WorkflowTemplates.get_training_workflow()

        if training_workflow.workflow_type == "training":
            print("✅ Standard training workflow template exists")
        else:
            print("❌ Standard training workflow template incorrect")
            return False

        # Test workflow features
        if hasattr(continuous_workflow, 'continuous') and continuous_workflow.continuous:
            print("✅ Continuous workflow has continuous mode")
        else:
            print("❌ Continuous workflow missing continuous mode")
            return False

        return True

    except Exception as e:
        print(f"❌ Workflow templates test failed: {e}")
        return False

async def test_orchestrator_integration():
    """Test basic orchestrator integration."""
    print("🧪 Testing orchestrator integration...")

    try:
        from prompt_improver.ml.orchestration.core.ml_pipeline_orchestrator import MLPipelineOrchestrator
        from prompt_improver.ml.orchestration.config.orchestrator_config import OrchestratorConfig

        # Test orchestrator instantiation
        config = OrchestratorConfig()
        orchestrator = MLPipelineOrchestrator(config)

        print("✅ ML Pipeline Orchestrator instantiation successful")

        # Test basic methods exist
        required_methods = [
            'initialize',
            'start_workflow',
            'stop_workflow',
            'get_workflow_status',
            'health_check'
        ]

        for method in required_methods:
            if hasattr(orchestrator, method):
                print(f"✅ Orchestrator method {method} exists")
            else:
                print(f"❌ Orchestrator method {method} missing")
                return False

        return True

    except Exception as e:
        print(f"❌ Orchestrator integration test failed: {e}")
        return False

async def test_clean_cli_integration():
    """Test that clean CLI integrates properly with orchestrator."""
    print("🧪 Testing clean CLI integration...")

    try:
        from prompt_improver.cli.clean_cli import training_manager, cli_orchestrator

        # Test that components are instantiated
        if training_manager is not None:
            print("✅ TrainingSystemManager instantiated in clean CLI")
        else:
            print("❌ TrainingSystemManager not instantiated")
            return False

        if cli_orchestrator is not None:
            print("✅ CLIOrchestrator instantiated in clean CLI")
        else:
            print("❌ CLIOrchestrator not instantiated")
            return False

        # Test that they have the required methods
        if hasattr(training_manager, 'smart_initialize'):
            print("✅ TrainingSystemManager has smart_initialize")
        else:
            print("❌ TrainingSystemManager missing smart_initialize")
            return False

        if hasattr(cli_orchestrator, 'start_continuous_training'):
            print("✅ CLIOrchestrator has start_continuous_training")
        else:
            print("❌ CLIOrchestrator missing start_continuous_training")
            return False

        return True

    except Exception as e:
        print(f"❌ Clean CLI integration test failed: {e}")
        return False

async def run_all_tests():
    """Run all Week 2 implementation tests."""
    print("🚀 Starting Week 2 Implementation Test Suite")
    print("=" * 60)

    tests = [
        ("Enhanced CLI Commands", test_cli_commands_enhanced),
        ("CLIOrchestrator Methods", test_cli_orchestrator_methods),
        ("Session-Based Approach", test_session_based_approach),
        ("Workflow Templates", test_workflow_templates),
        ("Orchestrator Integration", test_orchestrator_integration),
        ("Clean CLI Integration", test_clean_cli_integration),
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
        print("🎉 ALL TESTS PASSED - Week 2 implementation successful!")
        print("\n🚀 Ready to proceed to Week 3: Core Command Implementation")
    else:
        print(f"⚠️  {total - passed} tests failed - review implementation")

    return passed == total

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
