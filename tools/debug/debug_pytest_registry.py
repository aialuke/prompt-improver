#!/usr/bin/env python3
"""
Debug script to identify pytest-specific SQLAlchemy registry conflicts.
"""
import sys
import os
sys.path.insert(0, 'src')

def debug_pytest_environment():
    """Debug the pytest environment specifically."""
    print("=== Pytest Environment Debug ===")
    
    # Import pytest and check for any pytest-specific behavior
    import pytest
    print(f"Pytest version: {pytest.__version__}")
    
    # Check for pytest plugins that might affect imports
    import pytest_asyncio
    print(f"Pytest-asyncio version: {pytest_asyncio.__version__}")
    
    # Check sys.modules for any pre-loaded modules that might conflict
    model_modules = [k for k in sys.modules.keys() if 'models' in k or 'database' in k]
    print(f"Pre-loaded database/model modules: {model_modules}")

def simulate_pytest_test_execution():
    """Simulate the exact test execution environment."""
    print("\n=== Simulating Pytest Test Execution ===")
    
    # Clear any existing imports
    modules_to_clear = [k for k in sys.modules.keys() if k.startswith('prompt_improver')]
    for module in modules_to_clear:
        del sys.modules[module]
    
    # Import registry first (as it would be in test)
    from prompt_improver.database.registry import get_registry_manager, clear_registry
    
    # Clear registry (as done in test setup)
    clear_registry()
    print("Registry cleared")
    
    # Import models the way the test does
    from prompt_improver.performance.testing.ab_testing_service import ABTestingService
    from prompt_improver.database.connection import get_session_context
    
    # Get registry manager
    rm = get_registry_manager()
    print(f"Registry state after imports: {len(rm.registry._class_registry)} classes")
    
    # Try to create an async session like the test does
    print("\nTrying to create async session...")
    try:
        import asyncio
        async def test_session():
            async with get_session_context() as session:
                return session
        
        # Run the async context
        session = asyncio.run(test_session())
        print(f"✓ Async session created: {type(session)}")
    except Exception as e:
        print(f"✗ Async session failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Try to create AB experiment like the test does
    print("\nTrying to create AB experiment like test...")
    try:
        ab_service = ABTestingService()
        
        # Import the session context again (as test does)
        async def test_ab_experiment():
            async with get_session_context() as session:
                control_rules = {
                    "rule_ids": ["control_rule_1", "control_rule_2"],
                    "name": "Control Configuration",
                    "parameters": {"threshold": 0.5, "weight": 1.0}
                }
                
                treatment_rules = {
                    "rule_ids": ["treatment_rule_1", "treatment_rule_2"],
                    "name": "Treatment Configuration",
                    "parameters": {"threshold": 0.7, "weight": 1.2}
                }
                
                result = await ab_service.create_experiment(
                    experiment_name="debug_test",
                    control_rules=control_rules,
                    treatment_rules=treatment_rules,
                    db_session=session,
                    target_metric="improvement_score",
                    sample_size_per_group=50
                )
                
                return result
        
        result = asyncio.run(test_ab_experiment())
        print(f"✓ AB experiment result: {result}")
    except Exception as e:
        print(f"✗ AB experiment failed: {e}")
        import traceback
        traceback.print_exc()

def check_registry_during_model_init():
    """Check registry state during model initialization."""
    print("\n=== Checking Registry During Model Init ===")
    
    from prompt_improver.database.registry import get_registry_manager
    rm = get_registry_manager()
    
    # Clear registry
    rm.clear_registry()
    
    # Hook into model creation to see what happens
    from prompt_improver.database.models import ABExperiment
    
    print(f"Registry before model init: {len(rm.registry._class_registry)} classes")
    print(f"Classes: {list(rm.registry._class_registry.keys())}")
    
    # Try to create model instance
    try:
        experiment_data = {
            'experiment_name': 'test',
            'control_rules': {},
            'treatment_rules': {},
            'target_metric': 'test',
            'sample_size_per_group': 100,
            'status': 'running'
        }
        
        print("Creating ABExperiment instance...")
        experiment = ABExperiment(**experiment_data)
        
        print(f"Registry after model init: {len(rm.registry._class_registry)} classes")
        print(f"Classes: {list(rm.registry._class_registry.keys())}")
        
        # Check for conflicts
        conflicts = rm.diagnose_registry_conflicts()
        print(f"Conflicts after init: {conflicts}")
        
        print("✓ ABExperiment created successfully")
        
    except Exception as e:
        print(f"✗ ABExperiment creation failed: {e}")
        
        # Check registry state after failure
        print(f"Registry after failure: {len(rm.registry._class_registry)} classes")
        print(f"Classes: {list(rm.registry._class_registry.keys())}")
        
        # Check for conflicts
        conflicts = rm.diagnose_registry_conflicts()
        print(f"Conflicts after failure: {conflicts}")
        
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Starting Pytest-Specific Registry Debug Session")
    print("=" * 60)
    
    debug_pytest_environment()
    check_registry_during_model_init()
    simulate_pytest_test_execution()
    
    print("\n" + "=" * 60)
    print("Pytest debug session completed")
