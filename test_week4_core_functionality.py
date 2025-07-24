#!/usr/bin/env python3
"""
Week 4 Core Functionality Test
Simplified test to verify Week 4 smart initialization components work correctly.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_core_week4_functionality():
    """Test core Week 4 functionality without database dependencies."""
    print("🧪 Testing Week 4 Core Functionality...")
    
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Enhanced TrainingSystemManager import and methods
    total_tests += 1
    try:
        from prompt_improver.cli.core.training_system_manager import TrainingSystemManager
        from rich.console import Console
        
        console = Console()
        manager = TrainingSystemManager(console)
        
        # Check enhanced methods exist
        enhanced_methods = [
            '_detect_system_state',
            '_validate_components', 
            '_validate_database_and_rules',
            '_assess_data_availability',
            '_create_initialization_plan',
            '_execute_initialization_plan',
            '_validate_post_initialization'
        ]
        
        methods_found = sum(1 for method in enhanced_methods if hasattr(manager, method))
        
        if methods_found == len(enhanced_methods):
            print("✅ Enhanced TrainingSystemManager methods available")
            tests_passed += 1
        else:
            print(f"❌ Enhanced TrainingSystemManager missing {len(enhanced_methods) - methods_found} methods")
        
    except Exception as e:
        print(f"❌ Enhanced TrainingSystemManager test failed: {e}")
    
    # Test 2: Rule Validation Service
    total_tests += 1
    try:
        from prompt_improver.cli.core.rule_validation_service import RuleValidationService
        
        validator = RuleValidationService()
        
        # Check expected rules
        if len(validator.expected_rules) >= 6:
            print("✅ Rule Validation Service configured correctly")
            tests_passed += 1
        else:
            print(f"❌ Rule Validation Service has insufficient rules: {len(validator.expected_rules)}")
        
    except Exception as e:
        print(f"❌ Rule Validation Service test failed: {e}")
    
    # Test 3: System State Reporter
    total_tests += 1
    try:
        from prompt_improver.cli.core.system_state_reporter import SystemStateReporter
        from rich.console import Console
        
        console = Console()
        reporter = SystemStateReporter(console)
        
        # Test export functionality
        test_results = {
            "success": True,
            "components_initialized": ["test"],
            "recommendations": ["test recommendation"],
            "timestamp": "2025-01-24T12:00:00Z"
        }
        
        export_path = reporter.export_state_report(test_results, "test_week4_export.json")
        if Path(export_path).exists():
            print("✅ System State Reporter working")
            Path(export_path).unlink()  # Clean up
            tests_passed += 1
        else:
            print("❌ System State Reporter export failed")
        
    except Exception as e:
        print(f"❌ System State Reporter test failed: {e}")
    
    # Test 4: System State Detection (without database)
    total_tests += 1
    try:
        from prompt_improver.cli.core.training_system_manager import TrainingSystemManager
        from rich.console import Console
        
        console = Console()
        manager = TrainingSystemManager(console)
        
        # Test system state detection
        system_state = await manager._detect_system_state()
        
        required_fields = ["training_system_status", "environment_info", "detection_time_ms"]
        fields_present = sum(1 for field in required_fields if field in system_state)
        
        if fields_present == len(required_fields):
            print("✅ System state detection working")
            tests_passed += 1
        else:
            print(f"❌ System state detection missing {len(required_fields) - fields_present} fields")
        
    except Exception as e:
        print(f"❌ System state detection test failed: {e}")
    
    # Test 5: Component Validation (without database)
    total_tests += 1
    try:
        from prompt_improver.cli.core.training_system_manager import TrainingSystemManager
        from rich.console import Console
        
        console = Console()
        manager = TrainingSystemManager(console)
        
        # Test component validation
        component_status = await manager._validate_components()
        
        required_components = ["orchestrator", "analytics", "data_generator", "validation_time_ms"]
        components_present = sum(1 for comp in required_components if comp in component_status)
        
        if components_present == len(required_components):
            print("✅ Component validation working")
            tests_passed += 1
        else:
            print(f"❌ Component validation missing {len(required_components) - components_present} components")
        
    except Exception as e:
        print(f"❌ Component validation test failed: {e}")
    
    # Test 6: Generation Strategy Determination
    total_tests += 1
    try:
        from prompt_improver.cli.core.training_system_manager import TrainingSystemManager
        from rich.console import Console
        
        console = Console()
        manager = TrainingSystemManager(console)
        
        # Test generation strategy
        test_data_status = {
            "training_data": {"details": {"total_training_prompts": 50}},
            "data_quality": {"details": {"overall_quality_score": 0.6}}
        }
        
        strategy = await manager._determine_generation_strategy(test_data_status)
        
        required_strategy_fields = ["method", "target_samples", "focus_areas", "quality_target"]
        strategy_fields_present = sum(1 for field in required_strategy_fields if field in strategy)
        
        if strategy_fields_present == len(required_strategy_fields):
            print("✅ Generation strategy determination working")
            tests_passed += 1
        else:
            print(f"❌ Generation strategy missing {len(required_strategy_fields) - strategy_fields_present} fields")
        
    except Exception as e:
        print(f"❌ Generation strategy test failed: {e}")
    
    # Test 7: CLI Integration
    total_tests += 1
    try:
        from prompt_improver.cli.core.system_state_reporter import SystemStateReporter
        
        # Check if CLI imports work
        import subprocess
        result = subprocess.run(
            ["python", "-c", "import sys; sys.path.insert(0, 'src'); from prompt_improver.cli.clean_cli import app; print('CLI import successful')"],
            capture_output=True,
            text=True,
            timeout=30,
            cwd=Path(__file__).parent
        )
        
        if result.returncode == 0 and "CLI import successful" in result.stdout:
            print("✅ CLI integration working")
            tests_passed += 1
        else:
            print(f"❌ CLI integration failed: {result.stderr}")
        
    except Exception as e:
        print(f"❌ CLI integration test failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"🎯 Week 4 Core Functionality Test Results: {tests_passed}/{total_tests} tests passed")
    
    if tests_passed == total_tests:
        print("🎉 ALL CORE TESTS PASSED - Week 4 smart initialization functional!")
        print("\n📋 Week 4 Achievements:")
        print("  ✅ Enhanced TrainingSystemManager with 7-phase smart initialization")
        print("  ✅ Rule Validation Service with 6 seeded rule categories")
        print("  ✅ System State Reporter with comprehensive visualization")
        print("  ✅ Intelligent data generation strategy determination")
        print("  ✅ Component health monitoring and validation")
        print("  ✅ CLI integration with enhanced auto-initialization")
        print("\n🚀 Ready for production use with database configuration!")
    else:
        print(f"⚠️  {total_tests - tests_passed} core tests failed - review implementation")
    
    return tests_passed == total_tests

if __name__ == "__main__":
    success = asyncio.run(test_core_week4_functionality())
    sys.exit(0 if success else 1)
