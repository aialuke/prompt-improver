"""
Standalone test for ML orchestration health checkers.
"""
import asyncio
import sys
sys.path.insert(0, 'src')

async def test_standalone_health_checkers():
    """Test ML orchestration health checkers in isolation."""
    print('🔍 Testing ML orchestration health checkers (standalone)...')
    try:
        from prompt_improver.performance.monitoring.health.ml_orchestration_checkers import MLComponentRegistryHealthChecker, MLEventBusHealthChecker, MLOrchestratorHealthChecker, MLResourceManagerHealthChecker, MLWorkflowEngineHealthChecker
        print('  ✓ Successfully imported ML orchestration health checkers')
        orchestrator_checker = MLOrchestratorHealthChecker()
        registry_checker = MLComponentRegistryHealthChecker()
        resource_checker = MLResourceManagerHealthChecker()
        workflow_checker = MLWorkflowEngineHealthChecker()
        event_checker = MLEventBusHealthChecker()
        print('  ✓ Successfully instantiated all ML health checkers')
        orchestrator_result = await orchestrator_checker.check()
        print(f'  ✓ ML Orchestrator health check: {orchestrator_result.status.value}')
        registry_result = await registry_checker.check()
        print(f'  ✓ Component Registry health check: {registry_result.status.value}')
        resource_result = await resource_checker.check()
        print(f'  ✓ Resource Manager health check: {resource_result.status.value}')
        workflow_result = await workflow_checker.check()
        print(f'  ✓ Workflow Engine health check: {workflow_result.status.value}')
        event_result = await event_checker.check()
        print(f'  ✓ Event Bus health check: {event_result.status.value}')
        print('✅ ML orchestration health checkers work correctly')
        return True
    except Exception as e:
        print(f'❌ Health checkers test failed: {e}')
        import traceback
        traceback.print_exc()
        return False
if __name__ == '__main__':
    result = asyncio.run(test_standalone_health_checkers())
    if result:
        print('🎉 Health checkers test PASSED')
    else:
        print('⚠️ Health checkers test FAILED')
    sys.exit(0 if result else 1)
