"""
Comprehensive Test Failure Debug Script - AutoML Status Component

This script simulates exactly what happens in the comprehensive test when it evaluates
automl_status, comparing isolation testing (0.814 score) vs comprehensive testing (0.01 score).

The script will:
1. Extract the exact test logic from comprehensive_component_verification_test.py
2. Run that same logic in isolation to see what goes wrong
3. Test different mock dependency scenarios
4. Check if it's related to the kwargs filtering in the constructor
5. Look for Textual widget initialization issues
6. Provide step-by-step comparison between isolation and comprehensive testing
"""
import asyncio
import concurrent.futures
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import gc
import importlib
import inspect
import logging
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple
from unittest.mock import MagicMock, Mock, patch
import psutil
sys.path.insert(0, str(Path(__file__).parent / 'src'))
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', handlers=[logging.StreamHandler(), logging.FileHandler('debug_comprehensive_test_failure.log')])
logger = logging.getLogger(__name__)

class ComponentTestResult(Enum):
    """Component test result states following 2025 testing standards"""
    PASS = 'PASS'
    FAIL = 'FAIL'
    SKIP = 'SKIP'
    WARNING = 'WARNING'
    FALSE_POSITIVE = 'FALSE_POSITIVE'
    CRITICAL_FAIL = 'CRITICAL_FAIL'

class ComponentTestSeverity(Enum):
    """Test severity levels for prioritization"""
    CRITICAL = 'CRITICAL'
    HIGH = 'HIGH'
    MEDIUM = 'MEDIUM'
    LOW = 'LOW'
    INFO = 'INFO'

@dataclass
class ComponentTestMetrics:
    """Comprehensive test metrics following Quality Intelligence patterns"""
    component_name: str
    load_time_ms: float
    initialization_time_ms: float
    memory_usage_mb: float
    dependency_count: int
    method_count: int
    test_coverage_percent: float
    security_score: float
    performance_score: float
    reliability_score: float
    false_positive_risk: float
    business_impact_score: float

    def overall_quality_score(self) -> float:
        """Calculate overall quality score using 2025 weighting"""
        weights = {'security': 0.3, 'performance': 0.25, 'reliability': 0.25, 'coverage': 0.15, 'false_positive_risk': -0.05}
        score = self.security_score * weights['security'] + self.performance_score * weights['performance'] + self.reliability_score * weights['reliability'] + self.test_coverage_percent / 100 * weights['coverage'] - self.false_positive_risk * weights['false_positive_risk']
        return max(0.0, min(1.0, score))

@dataclass
class DebugTestComparison:
    """Comparison between isolation and comprehensive test results"""
    isolation_metrics: ComponentTestMetrics
    comprehensive_metrics: ComponentTestMetrics
    differences: Dict[str, Tuple[Any, Any]]
    root_cause_analysis: List[str]
    textual_specific_issues: List[str]
    mock_dependency_issues: List[str]
    kwargs_filtering_issues: List[str]

class ComprehensiveTestFailureDebugger:
    """Debug tool to analyze comprehensive test failures"""

    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.process = psutil.Process()
        self.component_name = 'automl_status'
        self.module_path = 'prompt_improver.tui.widgets.automl_status'
        self.target_class_name = 'AutoMLStatusWidget'
        self.isolation_environment = {}
        self.comprehensive_environment = {}

    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024

    def _find_main_class(self, module: Any, component_name: str) -> Optional[type]:
        """Find the main class in a module - EXACT copy from comprehensive test"""
        module_classes = [obj for name, obj in inspect.getmembers(module) if inspect.isclass(obj) and obj.__module__ == module.__name__]
        if not module_classes:
            return None
        specific_mappings = {'enhanced_scorer': 'EnhancedQualityScorer', 'monitoring': 'RealTimeMonitor', 'performance_validation': 'PerformanceValidator', 'multi_armed_bandit': 'MultiarmedBanditFramework', 'context_learner': 'ContextSpecificLearner', 'failure_analyzer': 'FailureModeAnalyzer', 'insight_engine': 'InsightGenerationEngine', 'rule_analyzer': 'RuleEffectivenessAnalyzer', 'automl_orchestrator': 'AutoMLOrchestrator', 'ner_extractor': 'NERExtractor', 'background_manager': 'BackgroundTaskManager', 'automl_status': 'AutoMLStatusWidget'}
        if component_name in specific_mappings:
            target_class_name = specific_mappings[component_name]
            for cls in module_classes:
                if cls.__name__ == target_class_name:
                    return cls
        possible_names = [component_name.title().replace('_', ''), component_name.replace('_', ' ').title().replace(' ', ''), f"{component_name.title().replace('_', '')}Service", f"{component_name.title().replace('_', '')}Manager", f"{component_name.title().replace('_', '')}Analyzer", f"{component_name.title().replace('_', '')}Optimizer", f"{component_name.title().replace('_', '')}Framework", f"{component_name.title().replace('_', '')}Engine", f"{component_name.title().replace('_', '')}Validator", f"{component_name.title().replace('_', '')}Monitor", f"{component_name.title().replace('_', '')}Extractor", f"{component_name.title().replace('_', '')}Widget"]
        for class_name in possible_names:
            for cls in module_classes:
                if cls.__name__ == class_name:
                    return cls

        def score_class(cls):
            score = 0
            class_name = cls.__name__
            if not hasattr(cls, '__dataclass_fields__'):
                score += 10
            methods = [m for m in dir(cls) if not m.startswith('_') and callable(getattr(cls, m, None))]
            score += len(methods)
            service_suffixes = ['Service', 'Manager', 'Analyzer', 'Optimizer', 'Framework', 'Engine', 'Validator', 'Monitor', 'Extractor', 'Widget', 'Orchestrator']
            for suffix in service_suffixes:
                if class_name.endswith(suffix):
                    score += 20
                    break
            avoid_patterns = ['Result', 'Config', 'Metrics', 'Alert', 'Task', 'Status']
            for pattern in avoid_patterns:
                if pattern in class_name:
                    score -= 5
            if len(class_name) < 4:
                score -= 5
            if class_name in ['ABC', 'BaseModel', 'Enum']:
                score -= 20
            return score
        scored_classes = [(score_class(cls), cls) for cls in module_classes]
        scored_classes.sort(key=lambda x: x[0], reverse=True)
        return scored_classes[0][1] if scored_classes else None

    def _generate_mock_dependencies(self, component_class: type) -> Dict[str, Any]:
        """Generate mock dependencies - EXACT copy from comprehensive test"""
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
        except Exception:
            return {}

    def _estimate_memory_usage(self, component_class: type) -> float:
        """Estimate memory usage - EXACT copy from comprehensive test"""
        methods = len([m for m in dir(component_class) if not m.startswith('_')])
        return methods * 0.1

    def _get_dependencies(self, component_class: type) -> List[str]:
        """Extract component dependencies - EXACT copy from comprehensive test"""
        dependencies = []
        try:
            init_signature = inspect.signature(component_class.__init__)
            for param_name, param in init_signature.parameters.items():
                if param_name != 'self' and param.annotation != inspect.Parameter.empty:
                    dep_name = getattr(param.annotation, '__name__', str(param.annotation))
                    if dep_name not in ['str', 'int', 'float', 'bool', 'dict', 'list']:
                        dependencies.append(dep_name)
        except Exception:
            pass
        return dependencies

    def _estimate_test_coverage(self, component_class: type) -> float:
        """Estimate test coverage - EXACT copy from comprehensive test"""
        methods = [m for m in dir(component_class) if not m.startswith('_') and callable(getattr(component_class, m))]
        return 70.0 if len(methods) > 0 else 0.0

    def calculate_security_score(self, component_class: type, module_path: str) -> float:
        """Calculate security score - EXACT copy from comprehensive test"""
        score = 0.8
        if 'security' in module_path:
            score += 0.2
        if hasattr(component_class, '__security_validated__'):
            score += 0.1
        if any((method.startswith('validate_') for method in dir(component_class))):
            score += 0.05
        if any((attr.startswith('_') and (not attr.startswith('__')) for attr in dir(component_class))):
            score -= 0.05
        return min(1.0, max(0.0, score))

    def calculate_performance_score(self, load_time_ms: float, init_time_ms: float, method_count: int) -> float:
        """Calculate performance score - EXACT copy from comprehensive test"""
        load_threshold_ms = 100
        init_threshold_ms = 50
        load_score = max(0, 1 - load_time_ms / load_threshold_ms)
        init_score = max(0, 1 - init_time_ms / init_threshold_ms)
        complexity_score = 1.0 if method_count < 20 else max(0.5, 1 - (method_count - 20) / 100)
        return load_score * 0.4 + init_score * 0.4 + complexity_score * 0.2

    def calculate_reliability_score(self, component_class: type, error: Optional[Exception]=None) -> float:
        """Calculate reliability score - EXACT copy from comprehensive test"""
        if error:
            return 0.0
        score = 0.8
        if hasattr(component_class, '__init__'):
            score += 0.1
        if hasattr(component_class, '__enter__') and hasattr(component_class, '__exit__'):
            score += 0.1
        methods = inspect.getmembers(component_class, inspect.isfunction)
        has_error_handling = any(('try' in inspect.getsource(method[1]) for method in methods if hasattr(method[1], '__code__')))
        if has_error_handling:
            score += 0.1
        return min(1.0, score)

    def _calculate_business_impact(self, component_name: str, severity: ComponentTestSeverity) -> float:
        """Calculate business impact score - EXACT copy from comprehensive test"""
        severity_scores = {ComponentTestSeverity.CRITICAL: 1.0, ComponentTestSeverity.HIGH: 0.8, ComponentTestSeverity.MEDIUM: 0.6, ComponentTestSeverity.LOW: 0.4, ComponentTestSeverity.INFO: 0.2}
        return severity_scores[severity]

    async def run_isolation_test(self) -> ComponentTestMetrics:
        """Run component test in isolation (should score ~0.814)"""
        print('🔬 Running ISOLATION test (expecting ~0.814 score)...')
        start_time = time.time()
        try:
            load_start = time.time()
            module = importlib.import_module(self.module_path)
            load_time_ms = (time.time() - load_start) * 1000
            component_class = self._find_main_class(module, self.component_name)
            if not component_class:
                raise ImportError(f'No suitable class found in {self.module_path}')
            print(f'   ✓ Found class: {component_class.__name__}')
            init_start = time.time()
            try:
                component_instance = component_class()
                init_time_ms = (time.time() - init_start) * 1000
                print(f'   ✓ Initialization successful in {init_time_ms:.2f}ms')
            except Exception as init_error:
                init_time_ms = (time.time() - init_start) * 1000
                print(f'   ⚠️ Initialization failed: {init_error}')
                component_instance = None
            try:
                methods = [method for method in dir(component_class) if not method.startswith('_')]
                callable_methods = []
                for method_name in methods:
                    try:
                        method_obj = getattr(component_class, method_name, None)
                        if callable(method_obj):
                            callable_methods.append(method_name)
                    except Exception:
                        pass
                methods = callable_methods
            except Exception as e:
                print(f'   ⚠️ Error collecting methods: {e}')
                methods = []
            metrics = ComponentTestMetrics(component_name=self.component_name, load_time_ms=load_time_ms, initialization_time_ms=init_time_ms, memory_usage_mb=self._estimate_memory_usage(component_class), dependency_count=len(self._get_dependencies(component_class)), method_count=len(methods), test_coverage_percent=self._estimate_test_coverage(component_class), security_score=self.calculate_security_score(component_class, self.module_path), performance_score=self.calculate_performance_score(load_time_ms, init_time_ms, len(methods)), reliability_score=self.calculate_reliability_score(component_class), false_positive_risk=0.1, business_impact_score=self._calculate_business_impact(self.component_name, ComponentTestSeverity.HIGH))
            self.isolation_environment = {'module': module, 'component_class': component_class, 'component_instance': component_instance, 'init_kwargs': {}, 'methods': methods, 'initialization_error': None if component_instance else init_error}
            print(f'   ✓ Isolation test completed - Quality Score: {metrics.overall_quality_score():.3f}')
            return metrics
        except Exception as e:
            print(f'   ❌ Isolation test failed: {e}')
            metrics = ComponentTestMetrics(component_name=self.component_name, load_time_ms=0.0, initialization_time_ms=0.0, memory_usage_mb=0.0, dependency_count=0, method_count=0, test_coverage_percent=0.0, security_score=0.0, performance_score=0.0, reliability_score=0.0, false_positive_risk=0.8, business_impact_score=0.0)
            return metrics

    async def run_exact_comprehensive_test(self) -> ComponentTestMetrics:
        """Run component test with EXACT conditions from comprehensive suite including concurrency"""
        print('🏗️ Running EXACT comprehensive test conditions...')
        start_time = time.time()
        semaphore = asyncio.Semaphore(5)

        async def comprehensive_component_test_exact():
            async with semaphore:
                return await self.run_comprehensive_test_internal()
        return await comprehensive_component_test_exact()

    async def run_comprehensive_test_internal(self) -> ComponentTestMetrics:
        """Internal method that matches the exact comprehensive test logic"""
        start_time = time.time()
        try:
            load_start = time.time()
            module = importlib.import_module(self.module_path)
            load_time_ms = (time.time() - load_start) * 1000
            component_class = self._find_main_class(module, self.component_name)
            if not component_class:
                raise ImportError(f'No suitable class found in {self.module_path}')
            print(f'   ✓ Found class: {component_class.__name__}')
            init_start = time.time()
            init_error = None
            try:
                init_kwargs = self._generate_mock_dependencies(component_class)
                print(f'   📦 Generated mock dependencies: {list(init_kwargs.keys())}')
                if init_kwargs:
                    component_instance = component_class(**init_kwargs)
                    print(f'   ✓ Initialized with kwargs: {init_kwargs}')
                else:
                    component_instance = component_class()
                    print(f'   ✓ Initialized without kwargs')
                init_time_ms = (time.time() - init_start) * 1000
                print(f'   ✓ Comprehensive initialization successful in {init_time_ms:.2f}ms')
            except Exception as e:
                init_time_ms = (time.time() - init_start) * 1000
                print(f'   ❌ Comprehensive initialization failed: {e}')
                print(f'   📋 Attempted kwargs: {init_kwargs}')
                component_instance = None
                init_error = e
            self.comprehensive_environment['initialization_error'] = init_error
            try:
                methods = [method for method in dir(component_class) if not method.startswith('_')]
                callable_methods = []
                for method_name in methods:
                    try:
                        method_obj = getattr(component_class, method_name, None)
                        if callable(method_obj):
                            callable_methods.append(method_name)
                    except Exception:
                        pass
                methods = callable_methods
            except Exception as e:
                print(f'   ⚠️ Error collecting methods: {e}')
                methods = []
            metrics = ComponentTestMetrics(component_name=self.component_name, load_time_ms=load_time_ms, initialization_time_ms=init_time_ms, memory_usage_mb=self._estimate_memory_usage(component_class), dependency_count=len(self._get_dependencies(component_class)), method_count=len(methods), test_coverage_percent=self._estimate_test_coverage(component_class), security_score=self.calculate_security_score(component_class, self.module_path), performance_score=self.calculate_performance_score(load_time_ms, init_time_ms, len(methods)), reliability_score=self.calculate_reliability_score(component_class, init_error), false_positive_risk=0.1, business_impact_score=self._calculate_business_impact(self.component_name, ComponentTestSeverity.HIGH))
            self.comprehensive_environment.update({'module': module, 'component_class': component_class, 'component_instance': component_instance, 'init_kwargs': init_kwargs, 'methods': methods})
            print(f'   ✓ Comprehensive test completed - Quality Score: {metrics.overall_quality_score():.3f}')
            return metrics
        except Exception as e:
            print(f'   ❌ Comprehensive test failed: {e}')
            traceback.print_exc()
            metrics = ComponentTestMetrics(component_name=self.component_name, load_time_ms=0.0, initialization_time_ms=0.0, memory_usage_mb=0.0, dependency_count=0, method_count=0, test_coverage_percent=0.0, security_score=0.0, performance_score=0.0, reliability_score=0.0, false_positive_risk=0.8, business_impact_score=0.0)
            return metrics

    async def run_comprehensive_test(self) -> ComponentTestMetrics:
        """Run component test exactly as in comprehensive suite (should score ~0.01)"""
        print('🏭 Running COMPREHENSIVE test (expecting ~0.01 score)...')
        start_time = time.time()
        try:
            load_start = time.time()
            module = importlib.import_module(self.module_path)
            load_time_ms = (time.time() - load_start) * 1000
            component_class = self._find_main_class(module, self.component_name)
            if not component_class:
                raise ImportError(f'No suitable class found in {self.module_path}')
            print(f'   ✓ Found class: {component_class.__name__}')
            init_start = time.time()
            try:
                init_kwargs = self._generate_mock_dependencies(component_class)
                print(f'   📦 Generated mock dependencies: {list(init_kwargs.keys())}')
                if init_kwargs:
                    component_instance = component_class(**init_kwargs)
                    print(f'   ✓ Initialized with kwargs: {init_kwargs}')
                else:
                    component_instance = component_class()
                    print(f'   ✓ Initialized without kwargs')
                init_time_ms = (time.time() - init_start) * 1000
                print(f'   ✓ Comprehensive initialization successful in {init_time_ms:.2f}ms')
            except Exception as init_error:
                init_time_ms = (time.time() - init_start) * 1000
                print(f'   ❌ Comprehensive initialization failed: {init_error}')
                print(f'   📋 Attempted kwargs: {init_kwargs}')
                component_instance = None
            try:
                methods = [method for method in dir(component_class) if not method.startswith('_')]
                callable_methods = []
                for method_name in methods:
                    try:
                        method_obj = getattr(component_class, method_name, None)
                        if callable(method_obj):
                            callable_methods.append(method_name)
                    except Exception:
                        pass
                methods = callable_methods
            except Exception as e:
                print(f'   ⚠️ Error collecting methods: {e}')
                methods = []
            metrics = ComponentTestMetrics(component_name=self.component_name, load_time_ms=load_time_ms, initialization_time_ms=init_time_ms, memory_usage_mb=self._estimate_memory_usage(component_class), dependency_count=len(self._get_dependencies(component_class)), method_count=len(methods), test_coverage_percent=self._estimate_test_coverage(component_class), security_score=self.calculate_security_score(component_class, self.module_path), performance_score=self.calculate_performance_score(load_time_ms, init_time_ms, len(methods)), reliability_score=self.calculate_reliability_score(component_class, self.comprehensive_environment.get('initialization_error')), false_positive_risk=0.1, business_impact_score=self._calculate_business_impact(self.component_name, ComponentTestSeverity.HIGH))
            self.comprehensive_environment = {'module': module, 'component_class': component_class, 'component_instance': component_instance, 'init_kwargs': init_kwargs, 'methods': methods, 'initialization_error': init_error if 'init_error' in locals() else None}
            print(f'   ✓ Comprehensive test completed - Quality Score: {metrics.overall_quality_score():.3f}')
            return metrics
        except Exception as e:
            print(f'   ❌ Comprehensive test failed: {e}')
            traceback.print_exc()
            metrics = ComponentTestMetrics(component_name=self.component_name, load_time_ms=0.0, initialization_time_ms=0.0, memory_usage_mb=0.0, dependency_count=0, method_count=0, test_coverage_percent=0.0, security_score=0.0, performance_score=0.0, reliability_score=0.0, false_positive_risk=0.8, business_impact_score=0.0)
            return metrics

    def analyze_kwargs_filtering(self) -> List[str]:
        """Analyze the kwargs filtering behavior in AutoMLStatusWidget"""
        issues = []
        try:
            module = importlib.import_module(self.module_path)
            component_class = getattr(module, self.target_class_name)
            init_signature = inspect.signature(component_class.__init__)
            init_source = inspect.getsource(component_class.__init__)
            print('🔍 Analyzing kwargs filtering in AutoMLStatusWidget.__init__:')
            print('=' * 60)
            print(init_source)
            print('=' * 60)
            if 'valid_static_params' in init_source:
                issues.append('Component uses kwargs filtering which may cause issues with mock dependencies')
                import ast
                tree = ast.parse(init_source)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Assign):
                        for target in node.targets:
                            if isinstance(target, ast.Name) and target.id == 'valid_static_params':
                                if isinstance(node.value, ast.List):
                                    valid_params = [elt.s for elt in node.value.elts if isinstance(elt, ast.Str)]
                                    print(f'   📋 Valid static params: {valid_params}')
                                    mock_kwargs = self._generate_mock_dependencies(component_class)
                                    conflicting_kwargs = [k for k in mock_kwargs.keys() if k not in valid_params]
                                    if conflicting_kwargs:
                                        issues.append(f'Mock kwargs {conflicting_kwargs} are filtered out by valid_static_params')
                                        print(f'   ⚠️ Conflicting kwargs: {conflicting_kwargs}')
                                    else:
                                        print(f'   ✓ No conflicts with mock kwargs')
            parent_classes = inspect.getmro(component_class)[1:]
            for parent_class in parent_classes:
                if hasattr(parent_class, '__init__'):
                    parent_signature = inspect.signature(parent_class.__init__)
                    parent_params = list(parent_signature.parameters.keys())
                    if len(parent_params) > 1:
                        issues.append(f'Parent class {parent_class.__name__} has parameters: {parent_params[1:]}')
                        print(f'   📋 Parent {parent_class.__name__} params: {parent_params[1:]}')
        except Exception as e:
            issues.append(f'Error analyzing kwargs filtering: {e}')
        return issues

    def test_textual_widget_initialization(self) -> List[str]:
        """Test various Textual widget initialization scenarios"""
        issues = []
        print('🎭 Testing Textual widget initialization scenarios...')
        try:
            from textual.reactive import reactive
            from textual.widgets import Static
            try:
                basic_static = Static()
                print('   ✓ Basic Static widget initializes successfully')
            except Exception as e:
                issues.append(f'Basic Static widget initialization fails: {e}')
                print(f'   ❌ Basic Static widget fails: {e}')
            try:
                param_static = Static(id='test-id', classes='test-class')
                print('   ✓ Parameterized Static widget initializes successfully')
            except Exception as e:
                issues.append(f'Parameterized Static widget initialization fails: {e}')
                print(f'   ❌ Parameterized Static widget fails: {e}')
            try:
                invalid_static = Static(invalid_param='test', another_invalid=123)
                print('   ⚠️ Static widget accepts invalid parameters - this might be the issue!')
                issues.append('Static widget unexpectedly accepts invalid parameters')
            except Exception as e:
                print(f'   ✓ Static widget properly rejects invalid parameters: {e}')
            try:
                module = importlib.import_module(self.module_path)
                component_class = getattr(module, self.target_class_name)
                widget_no_params = component_class()
                print('   ✓ AutoMLStatusWidget initializes with no parameters')
            except Exception as e:
                issues.append(f'AutoMLStatusWidget fails with no parameters: {e}')
                print(f'   ❌ AutoMLStatusWidget no params fails: {e}')
            try:
                widget_valid_params = component_class(id='automl-status', classes='status-widget')
                print('   ✓ AutoMLStatusWidget initializes with valid Textual parameters')
            except Exception as e:
                issues.append(f'AutoMLStatusWidget fails with valid Textual parameters: {e}')
                print(f'   ❌ AutoMLStatusWidget valid params fails: {e}')
            try:
                mock_kwargs = self._generate_mock_dependencies(component_class)
                if mock_kwargs:
                    widget_mock_params = component_class(**mock_kwargs)
                    print(f'   ❌ AutoMLStatusWidget unexpectedly accepts mock parameters: {mock_kwargs}')
                    issues.append('AutoMLStatusWidget accepts mock parameters when it should filter them')
                else:
                    print('   ✓ No mock parameters generated for AutoMLStatusWidget')
            except Exception as e:
                print(f'   ✓ AutoMLStatusWidget properly rejects mock parameters: {e}')
        except ImportError as e:
            issues.append(f'Cannot import Textual components: {e}')
            print(f'   ❌ Textual import error: {e}')
        return issues

    def test_mock_dependency_scenarios(self) -> List[str]:
        """Test different mock dependency scenarios"""
        issues = []
        print('🎭 Testing mock dependency scenarios...')
        try:
            module = importlib.import_module(self.module_path)
            component_class = getattr(module, self.target_class_name)
            try:
                instance_no_mocks = component_class()
                print('   ✓ Scenario 1: No mocks - SUCCESS')
            except Exception as e:
                issues.append(f'Scenario 1 (no mocks) fails: {e}')
                print(f'   ❌ Scenario 1: No mocks - FAIL: {e}')
            try:
                mock_kwargs = self._generate_mock_dependencies(component_class)
                if mock_kwargs:
                    instance_with_mocks = component_class(**mock_kwargs)
                    print(f'   ⚠️ Scenario 2: Generated mocks - UNEXPECTED SUCCESS: {mock_kwargs}')
                    issues.append('Component accepts generated mocks when it should filter them')
                else:
                    print('   ✓ Scenario 2: No mocks generated')
            except Exception as e:
                print(f'   ✓ Scenario 2: Generated mocks - EXPECTED FAIL: {e}')
            try:
                textual_kwargs = {'id': 'test-id', 'classes': 'test-class'}
                instance_textual = component_class(**textual_kwargs)
                print('   ✓ Scenario 3: Valid Textual params - SUCCESS')
            except Exception as e:
                issues.append(f'Scenario 3 (valid Textual params) fails: {e}')
                print(f'   ❌ Scenario 3: Valid Textual params - FAIL: {e}')
            try:
                mixed_kwargs = {'id': 'test-id', 'invalid_param': Mock(), 'classes': 'test'}
                instance_mixed = component_class(**mixed_kwargs)
                print(f'   ⚠️ Scenario 4: Mixed params - UNEXPECTED SUCCESS')
                issues.append('Component accepts mixed valid/invalid parameters')
            except Exception as e:
                print(f'   ✓ Scenario 4: Mixed params - EXPECTED FAIL: {e}')
        except Exception as e:
            issues.append(f'Error testing mock scenarios: {e}')
        return issues

    def compare_test_results(self, isolation_metrics: ComponentTestMetrics, comprehensive_metrics: ComponentTestMetrics) -> DebugTestComparison:
        """Compare isolation vs comprehensive test results"""
        differences = {}
        for field_name in isolation_metrics.__dataclass_fields__:
            isolation_value = getattr(isolation_metrics, field_name)
            comprehensive_value = getattr(comprehensive_metrics, field_name)
            if isolation_value != comprehensive_value:
                differences[field_name] = (isolation_value, comprehensive_value)
        root_causes = []
        if 'initialization_time_ms' in differences:
            iso_init, comp_init = differences['initialization_time_ms']
            if comp_init == 0 and iso_init > 0:
                root_causes.append('Comprehensive test fails initialization while isolation succeeds')
        if 'reliability_score' in differences:
            iso_rel, comp_rel = differences['reliability_score']
            if comp_rel == 0.0 and iso_rel > 0:
                root_causes.append('Comprehensive test sets reliability to 0.0 due to initialization failure')
        if 'performance_score' in differences:
            iso_perf, comp_perf = differences['performance_score']
            if comp_perf < iso_perf:
                root_causes.append('Comprehensive test has lower performance score')
        return DebugTestComparison(isolation_metrics=isolation_metrics, comprehensive_metrics=comprehensive_metrics, differences=differences, root_cause_analysis=root_causes, textual_specific_issues=self.test_textual_widget_initialization(), mock_dependency_issues=self.test_mock_dependency_scenarios(), kwargs_filtering_issues=self.analyze_kwargs_filtering())

    def print_detailed_comparison(self, comparison: DebugTestComparison):
        """Print detailed comparison results"""
        print('\n' + '=' * 80)
        print('📊 DETAILED TEST COMPARISON RESULTS')
        print('=' * 80)
        iso_score = comparison.isolation_metrics.overall_quality_score()
        comp_score = comparison.comprehensive_metrics.overall_quality_score()
        print(f'🎯 Quality Scores:')
        print(f'   Isolation Test:     {iso_score:.3f}')
        print(f'   Comprehensive Test: {comp_score:.3f}')
        print(f'   Difference:         {comp_score - iso_score:.3f}')
        print()
        if comparison.differences:
            print('📈 METRIC DIFFERENCES:')
            print('-' * 60)
            for field_name, (iso_val, comp_val) in comparison.differences.items():
                print(f'   {field_name:25} | {iso_val:>15} → {comp_val:<15}')
        print()
        print('🔍 ROOT CAUSE ANALYSIS:')
        print('-' * 60)
        for cause in comparison.root_cause_analysis:
            print(f'   • {cause}')
        if not comparison.root_cause_analysis:
            print('   • No specific root causes identified')
        print()
        print('🎭 TEXTUAL WIDGET ISSUES:')
        print('-' * 60)
        for issue in comparison.textual_specific_issues:
            print(f'   • {issue}')
        if not comparison.textual_specific_issues:
            print('   • No Textual-specific issues found')
        print()
        print('🎭 MOCK DEPENDENCY ISSUES:')
        print('-' * 60)
        for issue in comparison.mock_dependency_issues:
            print(f'   • {issue}')
        if not comparison.mock_dependency_issues:
            print('   • No mock dependency issues found')
        print()
        print('🔧 KWARGS FILTERING ISSUES:')
        print('-' * 60)
        for issue in comparison.kwargs_filtering_issues:
            print(f'   • {issue}')
        if not comparison.kwargs_filtering_issues:
            print('   • No kwargs filtering issues found')
        print()
        print('🌍 ENVIRONMENT COMPARISON:')
        print('-' * 60)
        if self.isolation_environment and self.comprehensive_environment:
            iso_init = self.isolation_environment.get('initialization_error')
            comp_init = self.comprehensive_environment.get('initialization_error')
            print(f'   Isolation init error:     {iso_init}')
            print(f'   Comprehensive init error: {comp_init}')
            iso_kwargs = self.isolation_environment.get('init_kwargs', {})
            comp_kwargs = self.comprehensive_environment.get('init_kwargs', {})
            print(f'   Isolation kwargs:         {iso_kwargs}')
            print(f'   Comprehensive kwargs:     {comp_kwargs}')
        print()
        print('💡 SUMMARY & RECOMMENDATIONS:')
        print('-' * 60)
        if comp_score < iso_score:
            print('   🔴 The comprehensive test produces a significantly lower score than isolation')
            if 'reliability_score' in comparison.differences and comparison.differences['reliability_score'][1] == 0.0:
                print('   🎯 PRIMARY ISSUE: Initialization failure in comprehensive test')
                print('   🔧 RECOMMENDATION: Fix mock dependency injection for Textual widgets')
            if comparison.kwargs_filtering_issues:
                print('   🎯 SECONDARY ISSUE: Kwargs filtering conflicts with mock generation')
                print('   🔧 RECOMMENDATION: Update mock generation to respect widget parameter filtering')
        else:
            print('   ✅ Test results are consistent between isolation and comprehensive modes')

    def demonstrate_exact_issue(self):
        """Demonstrate the exact issue that causes the 0.01 score"""
        print('\n🎯 DEMONSTRATING THE EXACT ISSUE')
        print('=' * 60)
        try:
            module = importlib.import_module(self.module_path)
            component_class = getattr(module, self.target_class_name)
            print('Testing the problematic _estimate_test_coverage method:')
            try:
                methods = [m for m in dir(component_class) if not m.startswith('_') and callable(getattr(component_class, m))]
                print(f'✅ Method collection succeeded: {len(methods)} methods')
            except Exception as e:
                print(f'❌ Method collection FAILED: {e}')
                print(f'   This is the root cause of the 0.01 score!')
                if "'NoneType' object has no attribute '_classes'" in str(e):
                    print('   🔍 TEXTUAL DESCRIPTOR ISSUE CONFIRMED')
                    print('   The Textual widget descriptor tries to access _classes on None')
                    print('   This happens when accessing class attributes via getattr()')
                return True
        except Exception as e:
            print(f'Error in demonstration: {e}')
        return False

    def show_solution(self):
        """Show the solution to fix the issue"""
        print('\n💡 SOLUTION')
        print('=' * 60)
        print('The fix is to update the _estimate_test_coverage method to handle')
        print('Textual widget descriptors properly:')
        print()
        print('BEFORE (causes exception):')
        print('    methods = [m for m in dir(component_class)')
        print("              if not m.startswith('_') and callable(getattr(component_class, m))]")
        print()
        print('AFTER (safe handling):')
        print('    methods = []')
        print('    for method_name in dir(component_class):')
        print("        if not method_name.startswith('_'):")
        print('            try:')
        print('                method_obj = getattr(component_class, method_name, None)')
        print('                if method_obj is not None and callable(method_obj):')
        print('                    methods.append(method_name)')
        print('            except Exception:')
        print('                pass  # Skip problematic descriptors')

async def main():
    """Main function to run the comprehensive debug analysis"""
    print('🚀 Comprehensive Test Failure Debug Tool')
    print('=' * 80)
    print('Analyzing why automl_status scores 0.814 in isolation but 0.01 in comprehensive testing')
    print()
    debugger = ComprehensiveTestFailureDebugger()
    try:
        issue_reproduced = debugger.demonstrate_exact_issue()
        if issue_reproduced:
            print('\n✅ ROOT CAUSE IDENTIFIED!')
            debugger.show_solution()
            print('\n' + '=' * 80)
            print('🎯 SUMMARY')
            print('=' * 80)
            print('❌ ISSUE: Textual widget descriptor access causes exception in _estimate_test_coverage')
            print('🔧 CAUSE: getattr(component_class, method_name) triggers _classes descriptor on None')
            print('💡 FIX: Add try/except when accessing class attributes via getattr')
            print('📊 IMPACT: Exception causes comprehensive test to return all-zero metrics (score: 0.01)')
            return 1
        else:
            print('⚠️ Could not reproduce the specific issue')
            return 0
    except Exception as e:
        print(f'❌ Debug analysis failed: {e}')
        traceback.print_exc()
        return 2
if __name__ == '__main__':
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
