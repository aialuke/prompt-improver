"""
Deep inspection of the 12 failing components to understand the REAL issues.
This will test them exactly as our orchestrator does.
"""
import asyncio
import importlib
import inspect
import logging
from pathlib import Path
import sys
import traceback
from typing import Any, Dict, Optional
from unittest.mock import Mock
sys.path.insert(0, str(Path(__file__).parent / 'src'))
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class DeepComponentInspector:
    """Deep inspector that mimics the actual orchestrator behavior"""

    def __init__(self):
        self.failing_components = {'enhanced_scorer': 'prompt_improver.ml.learning.quality.enhanced_scorer', 'monitoring': 'prompt_improver.performance.monitoring.monitoring', 'performance_validation': 'prompt_improver.performance.validation.performance_validation', 'multi_armed_bandit': 'prompt_improver.ml.optimization.algorithms.multi_armed_bandit', 'context_learner': 'prompt_improver.ml.learning.algorithms.context_learner', 'failure_analyzer': 'prompt_improver.ml.learning.algorithms.failure_analyzer', 'insight_engine': 'prompt_improver.ml.learning.algorithms.insight_engine', 'rule_analyzer': 'prompt_improver.ml.learning.algorithms.rule_analyzer', 'automl_orchestrator': 'prompt_improver.ml.automl.orchestrator', 'ner_extractor': 'prompt_improver.ml.analysis.ner_extractor', 'background_manager': 'prompt_improver.performance.monitoring.health.background_manager', 'automl_status': 'prompt_improver.tui.widgets.automl_status'}

    def _find_main_class(self, module: Any, component_name: str) -> Optional[type]:
        """Exact same logic as our orchestrator"""
        possible_names = [component_name.title().replace('_', ''), f"{component_name.title().replace('_', '')}Service", f"{component_name.title().replace('_', '')}Manager", f"{component_name.title().replace('_', '')}Analyzer", f"{component_name.title().replace('_', '')}Optimizer"]
        module_classes = [obj for name, obj in inspect.getmembers(module) if inspect.isclass(obj) and obj.__module__ == module.__name__]
        for class_name in possible_names:
            for cls in module_classes:
                if cls.__name__ == class_name:
                    return cls
        for cls in module_classes:
            if not cls.__name__.startswith('_') and len(cls.__name__) > 3 and (cls.__name__ not in ['ABC', 'BaseModel', 'Enum']):
                return cls
        return module_classes[0] if module_classes else None

    def _generate_mock_dependencies(self, component_class: type) -> Dict[str, Any]:
        """Exact same mock generation as orchestrator"""
        try:
            init_signature = inspect.signature(component_class.__init__)
            mock_kwargs = {}
            for param_name, param in init_signature.parameters.items():
                if param_name == 'self':
                    continue
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == str:
                        mock_kwargs[param_name] = f'mock_{param_name}'
                    elif param.annotation == int:
                        mock_kwargs[param_name] = 1
                    elif param.annotation == float:
                        mock_kwargs[param_name] = 1.0
                    elif param.annotation == bool:
                        mock_kwargs[param_name] = True
                    else:
                        mock_kwargs[param_name] = Mock()
                elif param.default != inspect.Parameter.empty:
                    continue
                else:
                    mock_kwargs[param_name] = Mock()
            return mock_kwargs
        except Exception as e:
            logger.error('Error generating mocks: %s', e)
            return {}

    async def deep_inspect_component(self, component_name: str, module_path: str) -> Dict[str, Any]:
        """Deep inspection that follows the exact orchestrator path"""
        print(f'\n🔍 DEEP INSPECTION: {component_name}')
        print('=' * 70)
        result = {'component_name': component_name, 'module_path': module_path, 'success': False, 'error': None, 'issues': [], 'warnings': []}
        try:
            print(f'📦 Phase 1: Importing {module_path}')
            module = importlib.import_module(module_path)
            print('✅ Module import successful')
            print('🔍 Phase 2: Finding main class')
            component_class = self._find_main_class(module, component_name)
            if not component_class:
                raise ImportError(f'No suitable class found in {module_path}')
            print(f'✅ Found main class: {component_class.__name__}')
            print('📋 Phase 3: Analyzing constructor')
            init_signature = inspect.signature(component_class.__init__)
            print(f'Constructor: {init_signature}')
            print('🎭 Phase 4: Generating mock dependencies')
            mock_kwargs = self._generate_mock_dependencies(component_class)
            if mock_kwargs:
                print(f'Generated mocks for: {list(mock_kwargs.keys())}')
            else:
                print('No mocks needed')
            print('🧪 Phase 5: Testing instantiation')
            try:
                if mock_kwargs:
                    print('Testing with mock arguments...')
                    component_instance = component_class(**mock_kwargs)
                else:
                    print('Testing with no arguments...')
                    component_instance = component_class()
                print('✅ Instantiation successful')
                result['success'] = True
                print('🔧 Phase 6: Analyzing methods')
                methods = [method for method in dir(component_instance) if not method.startswith('_') and callable(getattr(component_instance, method))]
                print(f"Found {len(methods)} public methods: {methods[:5]}{('...' if len(methods) > 5 else '')}")
                print('🧪 Phase 7: Testing basic method calls')
                working_methods = 0
                for method_name in methods[:3]:
                    try:
                        method = getattr(component_instance, method_name)
                        method_sig = inspect.signature(method)
                        if len(method_sig.parameters) == 0:
                            method()
                            working_methods += 1
                            print(f'  ✅ {method_name}() works')
                        else:
                            print(f'  ⚠️  {method_name}() requires arguments')
                    except Exception as method_error:
                        print(f'  ❌ {method_name}() failed: {method_error}')
                        result['warnings'].append(f'Method {method_name} failed: {method_error}')
                print(f'✅ {working_methods}/{min(3, len(methods))} methods tested successfully')
            except Exception as instantiation_error:
                print(f'❌ Instantiation failed: {instantiation_error}')
                result['error'] = str(instantiation_error)
                result['issues'].append(f'Instantiation failed: {instantiation_error}')
                print('🔍 Analyzing instantiation error...')
                if 'missing' in str(instantiation_error).lower():
                    missing_params = []
                    for param_name, param in init_signature.parameters.items():
                        if param_name != 'self' and param.default == inspect.Parameter.empty:
                            missing_params.append(f'{param_name}: {param.annotation}')
                    if missing_params:
                        print(f'Missing required parameters: {missing_params}')
                        result['issues'].append(f'Missing required parameters: {missing_params}')
                try:
                    print('🔍 Checking for import issues in class definition...')
                    source = inspect.getsource(component_class)
                    if 'import' in source:
                        result['warnings'].append('Class contains imports that might fail at runtime')
                except Exception:
                    pass
        except ImportError as import_error:
            print(f'❌ Import failed: {import_error}')
            result['error'] = str(import_error)
            result['issues'].append(f'Import failed: {import_error}')
            if 'No module named' in str(import_error):
                missing_module = str(import_error).split("'")[1] if "'" in str(import_error) else 'unknown'
                print(f'🔍 Missing module: {missing_module}')
                result['issues'].append(f'Missing dependency: {missing_module}')
        except Exception as general_error:
            print(f'❌ Unexpected error: {general_error}')
            result['error'] = str(general_error)
            result['issues'].append(f'Unexpected error: {general_error}')
            traceback.print_exc()
        return result

    async def run_deep_inspection(self):
        """Run deep inspection on all failing components"""
        print('🔬 DEEP COMPONENT INSPECTION')
        print('=' * 80)
        print('Testing components exactly as the orchestrator does...')
        print()
        results = []
        for component_name, module_path in self.failing_components.items():
            result = await self.deep_inspect_component(component_name, module_path)
            results.append(result)
        print('\n' + '=' * 80)
        print('📊 DEEP INSPECTION RESULTS')
        print('=' * 80)
        successful = [r for r in results if r['success']]
        failed = [r for r in results if not r['success']]
        print(f'✅ Successful: {len(successful)}/{len(results)} ({len(successful) / len(results):.1%})')
        print(f'❌ Failed: {len(failed)}/{len(results)} ({len(failed) / len(results):.1%})')
        if failed:
            print('\n🔥 FAILED COMPONENTS (need fixing):')
            for result in failed:
                print(f"\n❌ {result['component_name']}")
                print(f"   Module: {result['module_path']}")
                print(f"   Error: {result['error']}")
                for issue in result['issues']:
                    print(f'   Issue: {issue}')
                for warning in result['warnings']:
                    print(f'   Warning: {warning}')
        if successful:
            print('\n✅ SUCCESSFUL COMPONENTS:')
            for result in successful:
                print(f"✅ {result['component_name']}")
                if result['warnings']:
                    for warning in result['warnings']:
                        print(f'   Warning: {warning}')
        return results

async def main():
    inspector = DeepComponentInspector()
    results = await inspector.run_deep_inspection()
    failed_count = sum((1 for r in results if not r['success']))
    return 0 if failed_count == 0 else 1
if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
