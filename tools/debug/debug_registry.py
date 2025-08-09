"""
Debug script to identify SQLAlchemy registry conflicts.
"""
import os
import sys
from prompt_improver.database.registry import get_registry_manager
sys.path.insert(0, 'src')

def debug_registry_state():
    """Debug the current registry state."""
    rm = get_registry_manager()
    print('=== Registry Debug Info ===')
    print(f'Registry ID: {id(rm.registry)}')
    print(f'Registry size: {len(rm.registry._class_registry)}')
    print(f'Classes in registry: {list(rm.registry._class_registry.keys())}')
    conflicts = rm.diagnose_registry_conflicts()
    print(f'Conflicts: {conflicts}')
    return rm

def simulate_test_imports():
    """Simulate the imports that happen during test execution."""
    print('\n=== Simulating Test Imports ===')
    rm = debug_registry_state()
    rm.clear_registry()
    print('Registry cleared')
    print('\nImporting models directly...')
    from prompt_improver.database.models import ABExperiment, RulePerformance
    debug_registry_state()
    print('\nImporting AB testing service...')
    from prompt_improver.performance.testing.ab_testing_service import ABTestingService
    debug_registry_state()
    print('\nImporting other services...')
    from prompt_improver.performance.analytics.real_time_analytics import RealTimeAnalyticsService
    debug_registry_state()
    print('\nImporting test framework components...')
    try:
        from prompt_improver.core.services.prompt_improvement import PromptImprovementService
        debug_registry_state()
    except ImportError as e:
        print(f'Import error: {e}')
    print('\nTrying to create ABExperiment...')
    try:
        experiment_data = {'experiment_name': 'test', 'control_rules': {}, 'treatment_rules': {}, 'target_metric': 'test', 'sample_size_per_group': 100, 'status': 'running'}
        experiment = ABExperiment(**experiment_data)
        print('✓ ABExperiment created successfully')
    except Exception as e:
        print(f'✗ ABExperiment creation failed: {e}')
        import traceback
        traceback.print_exc()

def check_multiple_registries():
    """Check if there are multiple registries in the system."""
    print('\n=== Checking for Multiple Registries ===')
    from sqlmodel import SQLModel
    from prompt_improver.database.registry import get_registry_manager
    rm = get_registry_manager()
    print(f'Our registry ID: {id(rm.registry)}')
    print(f'SQLModel registry ID: {id(SQLModel.registry)}')
    print(f'Are they the same? {rm.registry is SQLModel.registry}')
    from prompt_improver.database.models import ABExperiment, RulePerformance
    print(f'RulePerformance registry ID: {id(RulePerformance.registry)}')
    print(f'ABExperiment registry ID: {id(ABExperiment.registry)}')
    print(f'All registries are the same? {all([rm.registry is SQLModel.registry, rm.registry is RulePerformance.registry, rm.registry is ABExperiment.registry])}')

def check_class_registration():
    """Check how classes are registered."""
    print('\n=== Checking Class Registration ===')
    from prompt_improver.database.models import ABExperiment, RulePerformance
    print(f"RulePerformance has registry: {hasattr(RulePerformance, 'registry')}")
    print(f"ABExperiment has registry: {hasattr(ABExperiment, 'registry')}")
    print(f"RulePerformance in its registry: {'RulePerformance' in RulePerformance.registry._class_registry}")
    print(f"ABExperiment in its registry: {'ABExperiment' in ABExperiment.registry._class_registry}")
    registry_keys = list(RulePerformance.registry._class_registry.keys())
    print(f'All registry keys: {registry_keys}')
    rule_performance_entries = [k for k in registry_keys if 'RulePerformance' in k]
    print(f'RulePerformance related entries: {rule_performance_entries}')
if __name__ == '__main__':
    print('Starting SQLAlchemy Registry Debug Session')
    print('=' * 50)
    debug_registry_state()
    check_multiple_registries()
    check_class_registration()
    simulate_test_imports()
    print('\n' + '=' * 50)
    print('Debug session completed')
