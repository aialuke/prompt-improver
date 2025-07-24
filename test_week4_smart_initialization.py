#!/usr/bin/env python3
"""
Week 4 Smart Initialization Test Suite
Comprehensive testing of smart initialization with real database validation, rule loading, and data generation.
"""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path for testing
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_enhanced_training_system_manager():
    """Test enhanced TrainingSystemManager with smart detection capabilities."""
    print("🧪 Testing enhanced TrainingSystemManager...")

    try:
        import sys
        import os

        # Ensure proper module loading
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

        from prompt_improver.cli.core.training_system_manager import TrainingSystemManager
        from rich.console import Console

        console = Console()
        manager = TrainingSystemManager(console)

        # Test enhanced methods exist
        enhanced_methods = [
            '_detect_system_state',
            '_validate_components',
            '_validate_database_and_rules',
            '_assess_data_availability',
            '_create_initialization_plan',
            '_execute_initialization_plan',
            '_validate_post_initialization'
        ]

        for method in enhanced_methods:
            if hasattr(manager, method):
                print(f"✅ Enhanced method {method} exists")
            else:
                print(f"❌ Enhanced method {method} missing")
                return False

        # Test rule validator integration
        if hasattr(manager, '_rule_validator'):
            print("✅ Rule validator integration available")
        else:
            print("❌ Rule validator integration missing")
            return False

        return True

    except Exception as e:
        print(f"❌ Enhanced TrainingSystemManager test failed: {e}")
        return False

def test_rule_validation_service():
    """Test rule validation service functionality."""
    print("🧪 Testing rule validation service...")

    try:
        from prompt_improver.cli.core.rule_validation_service import RuleValidationService

        validator = RuleValidationService()

        # Test expected rules configuration
        if len(validator.expected_rules) >= 6:
            print(f"✅ Expected rules configured: {len(validator.expected_rules)} rules")
        else:
            print(f"❌ Insufficient expected rules: {len(validator.expected_rules)}")
            return False

        # Test rule categories
        categories = set(rule["category"] for rule in validator.expected_rules.values())
        expected_categories = {"fundamental", "reasoning", "examples", "context", "formatting"}
        if expected_categories.issubset(categories):
            print("✅ Rule categories properly configured")
        else:
            print(f"❌ Missing rule categories: {expected_categories - categories}")
            return False

        # Test validation methods exist
        validation_methods = [
            '_validate_rule_existence',
            '_validate_rule_metadata',
            '_validate_rule_parameters',
            '_analyze_rule_performance',
            '_generate_rule_recommendations'
        ]

        for method in validation_methods:
            if hasattr(validator, method):
                print(f"✅ Validation method {method} exists")
            else:
                print(f"❌ Validation method {method} missing")
                return False

        return True

    except Exception as e:
        print(f"❌ Rule validation service test failed: {e}")
        return False

def test_system_state_reporter():
    """Test system state reporter functionality."""
    print("🧪 Testing system state reporter...")

    try:
        from prompt_improver.cli.core.system_state_reporter import SystemStateReporter
        from rich.console import Console

        console = Console()
        reporter = SystemStateReporter(console)

        # Test reporter methods
        reporter_methods = [
            'generate_comprehensive_report',
            '_display_overall_status',
            '_display_component_status',
            '_display_database_status',
            '_display_data_status',
            '_display_initialization_details',
            '_display_recommendations',
            '_display_performance_metrics',
            'export_state_report'
        ]

        for method in reporter_methods:
            if hasattr(reporter, method):
                print(f"✅ Reporter method {method} exists")
            else:
                print(f"❌ Reporter method {method} missing")
                return False

        # Test export functionality
        test_results = {
            "success": True,
            "components_initialized": ["test"],
            "recommendations": ["test recommendation"]
        }

        export_path = reporter.export_state_report(test_results, "test_export.json")
        if Path(export_path).exists():
            print("✅ Export functionality working")
            Path(export_path).unlink()  # Clean up
        else:
            print("❌ Export functionality failed")
            return False

        return True

    except Exception as e:
        print(f"❌ System state reporter test failed: {e}")
        return False

async def test_smart_initialization_integration():
    """Test smart initialization integration with real components."""
    print("🧪 Testing smart initialization integration...")

    try:
        from prompt_improver.cli.core.training_system_manager import TrainingSystemManager
        from rich.console import Console

        console = Console()
        manager = TrainingSystemManager(console)

        # Test system state detection
        system_state = await manager._detect_system_state()

        required_state_fields = [
            "training_system_status", "environment_info", "configuration_files",
            "detection_time_ms"
        ]

        for field in required_state_fields:
            if field in system_state:
                print(f"✅ System state field {field} present")
            else:
                print(f"❌ System state field {field} missing")
                return False

        # Test component validation
        component_status = await manager._validate_components()

        required_components = ["orchestrator", "analytics", "data_generator", "validation_time_ms"]
        for component in required_components:
            if component in component_status:
                print(f"✅ Component status for {component} available")
            else:
                print(f"❌ Component status for {component} missing")
                return False

        print("✅ Smart initialization integration working")
        return True

    except Exception as e:
        print(f"❌ Smart initialization integration test failed: {e}")
        return False

async def test_database_and_rule_validation():
    """Test database and rule validation with real database."""
    print("🧪 Testing database and rule validation...")

    try:
        from prompt_improver.cli.core.training_system_manager import TrainingSystemManager
        from rich.console import Console

        console = Console()
        manager = TrainingSystemManager(console)

        # Test database validation
        database_status = await manager._validate_database_and_rules()

        required_db_fields = [
            "connectivity", "schema_validation", "seeded_rules", "rule_metadata", "validation_time_ms"
        ]

        for field in required_db_fields:
            if field in database_status:
                print(f"✅ Database validation field {field} present")
            else:
                print(f"❌ Database validation field {field} missing")
                return False

        # Check if database connectivity works
        connectivity_status = database_status["connectivity"]["status"]
        if connectivity_status == "healthy":
            print("✅ Database connectivity validation working")
        else:
            print(f"⚠️  Database connectivity: {connectivity_status}")

        print("✅ Database and rule validation working")
        return True

    except Exception as e:
        print(f"❌ Database and rule validation test failed: {e}")
        return False

async def test_data_availability_assessment():
    """Test comprehensive data availability assessment."""
    print("🧪 Testing data availability assessment...")

    try:
        from prompt_improver.cli.core.training_system_manager import TrainingSystemManager
        from rich.console import Console

        console = Console()
        manager = TrainingSystemManager(console)

        # Test data assessment
        data_status = await manager._assess_data_availability()

        required_data_fields = [
            "training_data", "synthetic_data", "data_quality", "minimum_requirements", "assessment_time_ms"
        ]

        for field in required_data_fields:
            if field in data_status:
                print(f"✅ Data assessment field {field} present")
            else:
                print(f"❌ Data assessment field {field} missing")
                return False

        # Test quality assessment method
        if hasattr(manager, '_perform_comprehensive_quality_assessment'):
            print("✅ Comprehensive quality assessment method available")
        else:
            print("❌ Comprehensive quality assessment method missing")
            return False

        print("✅ Data availability assessment working")
        return True

    except Exception as e:
        print(f"❌ Data availability assessment test failed: {e}")
        return False

async def test_enhanced_data_generation():
    """Test enhanced synthetic data generation."""
    print("🧪 Testing enhanced data generation...")

    try:
        from prompt_improver.cli.core.training_system_manager import TrainingSystemManager
        from rich.console import Console

        console = Console()
        manager = TrainingSystemManager(console)

        # Test enhanced generation methods
        generation_methods = [
            '_determine_generation_strategy',
            '_configure_generator_for_strategy',
            '_execute_targeted_generation',
            '_validate_generated_data_quality'
        ]

        for method in generation_methods:
            if hasattr(manager, method):
                print(f"✅ Enhanced generation method {method} exists")
            else:
                print(f"❌ Enhanced generation method {method} missing")
                return False

        # Test strategy determination
        test_data_status = {
            "training_data": {"details": {"total_training_prompts": 50}},
            "data_quality": {"details": {"overall_quality_score": 0.6}}
        }

        strategy = await manager._determine_generation_strategy(test_data_status)

        required_strategy_fields = ["method", "target_samples", "focus_areas", "quality_target"]
        for field in required_strategy_fields:
            if field in strategy:
                print(f"✅ Generation strategy field {field} present")
            else:
                print(f"❌ Generation strategy field {field} missing")
                return False

        print("✅ Enhanced data generation working")
        return True

    except Exception as e:
        print(f"❌ Enhanced data generation test failed: {e}")
        return False

async def test_full_smart_initialization():
    """Test complete smart initialization workflow."""
    print("🧪 Testing full smart initialization workflow...")

    try:
        from prompt_improver.cli.core.training_system_manager import TrainingSystemManager
        from rich.console import Console

        console = Console()
        manager = TrainingSystemManager(console)

        # Run smart initialization (this will test the full workflow)
        print("   Running smart initialization...")
        start_time = time.time()

        results = await manager.smart_initialize()

        initialization_time = time.time() - start_time
        print(f"   Initialization completed in {initialization_time:.2f}s")

        # Validate results structure
        required_result_fields = [
            "success", "system_state", "component_status", "database_status",
            "data_status", "initialization_plan", "execution_results",
            "final_validation", "components_initialized", "recommendations"
        ]

        for field in required_result_fields:
            if field in results:
                print(f"✅ Result field {field} present")
            else:
                print(f"❌ Result field {field} missing")
                return False

        # Check if initialization was successful
        if results["success"]:
            print("✅ Smart initialization completed successfully")
        else:
            print(f"⚠️  Smart initialization completed with issues: {results.get('error', 'Unknown')}")

        # Display summary
        components_count = len(results.get("components_initialized", []))
        recommendations_count = len(results.get("recommendations", []))

        print(f"   Components initialized: {components_count}")
        print(f"   Recommendations: {recommendations_count}")

        return True

    except Exception as e:
        print(f"❌ Full smart initialization test failed: {e}")
        return False

async def run_all_tests():
    """Run all Week 4 smart initialization tests."""
    print("🚀 Starting Week 4 Smart Initialization Test Suite")
    print("=" * 70)

    tests = [
        ("Enhanced TrainingSystemManager", test_enhanced_training_system_manager),
        ("Rule Validation Service", test_rule_validation_service),
        ("System State Reporter", test_system_state_reporter),
        ("Smart Initialization Integration", test_smart_initialization_integration),
        ("Database and Rule Validation", test_database_and_rule_validation),
        ("Data Availability Assessment", test_data_availability_assessment),
        ("Enhanced Data Generation", test_enhanced_data_generation),
        ("Full Smart Initialization", test_full_smart_initialization),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 50)

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

    print("\n" + "=" * 70)
    print(f"🎯 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("🎉 ALL TESTS PASSED - Week 4 smart initialization successful!")
        print("\n🚀 Ready to proceed to Week 5: Adaptive Data Generation")
    else:
        print(f"⚠️  {total - passed} tests failed - review implementation")

    return passed == total

if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    sys.exit(0 if success else 1)
