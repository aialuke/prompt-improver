#!/usr/bin/env python3
"""
Focused AutoML Status Component Quality Diagnostic Script

This script provides detailed diagnostics on why the automl_status component
is getting such a poor quality score (0.01) in comprehensive testing.

Objectives:
1. Extract specific test logic that evaluates automl_status
2. Run component through each quality metric individually  
3. Capture detailed error logs and warnings
4. Test component both in isolation and in context
5. Check for Textual widget-specific issues
6. Analyze timing, memory usage, method calls, and exceptions
"""

import asyncio
import inspect
import importlib
import logging
import time
import traceback
import sys
import gc
import psutil
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
from unittest.mock import Mock, patch, MagicMock
import threading

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('automl_status_diagnostic.log')
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class DiagnosticMetrics:
    """Detailed diagnostic metrics for automl_status component"""
    component_name: str
    load_success: bool
    load_time_ms: float
    load_error: Optional[str]
    
    class_found: bool
    class_name: Optional[str]
    class_error: Optional[str]
    
    init_success: bool
    init_time_ms: float
    init_error: Optional[str]
    
    method_count: int
    method_names: List[str]
    
    memory_before_mb: float
    memory_after_mb: float
    memory_diff_mb: float
    
    security_issues: List[str]
    performance_warnings: List[str]
    textual_specific_issues: List[str]
    
    individual_scores: Dict[str, float]
    quality_calculation_details: Dict[str, Any]


class AutoMLStatusDiagnosticTool:
    """Comprehensive diagnostic tool for automl_status component"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.process = psutil.Process()
        self.component_path = "prompt_improver.tui.widgets.automl_status"
        self.component_name = "automl_status"
        
        # Textual-specific imports to test compatibility
        self.textual_imports = [
            'textual.reactive', 'textual.widgets', 'textual.app', 
            'textual.containers', 'textual.css'
        ]
        
        # Rich imports for UI components
        self.rich_imports = [
            'rich.bar', 'rich.console', 'rich.panel', 'rich.table'
        ]
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def test_environment_compatibility(self) -> Dict[str, Any]:
        """Test environment compatibility with Textual and Rich"""
        self.logger.info("🔍 Testing environment compatibility...")
        
        compatibility = {
            'textual_imports': {},
            'rich_imports': {},
            'python_version': sys.version,
            'platform': sys.platform,
        }
        
        # Test Textual imports
        for import_name in self.textual_imports:
            try:
                start_time = time.time()
                importlib.import_module(import_name)
                load_time = (time.time() - start_time) * 1000
                compatibility['textual_imports'][import_name] = {
                    'success': True, 
                    'load_time_ms': load_time
                }
                self.logger.debug(f"✅ {import_name}: {load_time:.2f}ms")
            except Exception as e:
                compatibility['textual_imports'][import_name] = {
                    'success': False, 
                    'error': str(e)
                }
                self.logger.error(f"❌ {import_name}: {e}")
        
        # Test Rich imports
        for import_name in self.rich_imports:
            try:
                start_time = time.time()
                importlib.import_module(import_name)
                load_time = (time.time() - start_time) * 1000
                compatibility['rich_imports'][import_name] = {
                    'success': True, 
                    'load_time_ms': load_time
                }
                self.logger.debug(f"✅ {import_name}: {load_time:.2f}ms")
            except Exception as e:
                compatibility['rich_imports'][import_name] = {
                    'success': False, 
                    'error': str(e)
                }
                self.logger.error(f"❌ {import_name}: {e}")
        
        return compatibility
    
    def load_component_with_diagnostics(self) -> DiagnosticMetrics:
        """Load component with comprehensive diagnostics"""
        self.logger.info(f"🚀 Starting comprehensive diagnostic for {self.component_name}")
        
        memory_before = self.get_memory_usage()
        
        metrics = DiagnosticMetrics(
            component_name=self.component_name,
            load_success=False,
            load_time_ms=0.0,
            load_error=None,
            class_found=False,
            class_name=None,
            class_error=None,
            init_success=False,
            init_time_ms=0.0,
            init_error=None,
            method_count=0,
            method_names=[],
            memory_before_mb=memory_before,
            memory_after_mb=0.0,
            memory_diff_mb=0.0,
            security_issues=[],
            performance_warnings=[],
            textual_specific_issues=[],
            individual_scores={},
            quality_calculation_details={}
        )
        
        # Phase 1: Module Loading
        self.logger.info("📦 Phase 1: Module Loading")
        load_start = time.time()
        try:
            module = importlib.import_module(self.component_path)
            metrics.load_time_ms = (time.time() - load_start) * 1000
            metrics.load_success = True
            self.logger.info(f"✅ Module loaded successfully in {metrics.load_time_ms:.2f}ms")
        except Exception as e:
            metrics.load_time_ms = (time.time() - load_start) * 1000
            metrics.load_error = str(e)
            self.logger.error(f"❌ Module loading failed: {e}")
            self.logger.error(f"📋 Traceback: {traceback.format_exc()}")
            return self._finalize_metrics(metrics)
        
        # Phase 2: Class Discovery
        self.logger.info("🔍 Phase 2: Class Discovery")
        try:
            component_class = self._find_automl_status_class(module)
            if component_class:
                metrics.class_found = True
                metrics.class_name = component_class.__name__
                self.logger.info(f"✅ Found class: {component_class.__name__}")
                
                # Get method information
                methods = [method for method in dir(component_class) 
                          if not method.startswith('_') and callable(getattr(component_class, method, None))]
                metrics.method_count = len(methods)
                metrics.method_names = methods
                self.logger.info(f"📊 Found {len(methods)} public methods: {methods[:5]}...")
                
            else:
                metrics.class_error = "No suitable class found"
                self.logger.error("❌ No AutoMLStatusWidget class found")
                return self._finalize_metrics(metrics)
        except Exception as e:
            metrics.class_error = str(e)
            self.logger.error(f"❌ Class discovery failed: {e}")
            return self._finalize_metrics(metrics)
        
        # Phase 3: Component Initialization Testing
        self.logger.info("🔧 Phase 3: Component Initialization Testing")
        init_start = time.time()
        try:
            # Test different initialization approaches
            init_attempts = [
                self._try_basic_init,
                self._try_init_with_kwargs,
                self._try_init_with_textual_context,
                self._try_init_with_mocked_deps
            ]
            
            component_instance = None
            for i, init_attempt in enumerate(init_attempts):
                try:
                    self.logger.debug(f"Trying initialization approach {i+1}")
                    component_instance = init_attempt(component_class)
                    metrics.init_success = True
                    metrics.init_time_ms = (time.time() - init_start) * 1000
                    self.logger.info(f"✅ Initialization successful (approach {i+1}) in {metrics.init_time_ms:.2f}ms")
                    break
                except Exception as init_e:
                    self.logger.debug(f"Initialization approach {i+1} failed: {init_e}")
                    continue
            
            if not component_instance:
                metrics.init_error = "All initialization approaches failed"
                self.logger.error("❌ All initialization approaches failed")
                
        except Exception as e:
            metrics.init_time_ms = (time.time() - init_start) * 1000
            metrics.init_error = str(e)
            self.logger.error(f"❌ Initialization testing failed: {e}")
        
        # Phase 4: Individual Quality Metric Analysis
        self.logger.info("📊 Phase 4: Individual Quality Metric Analysis")
        self._analyze_individual_quality_metrics(component_class, metrics)
        
        # Phase 5: Textual-Specific Analysis
        self.logger.info("🎨 Phase 5: Textual-Specific Analysis")
        self._analyze_textual_specific_issues(component_class, metrics)
        
        # Phase 6: Security Analysis
        self.logger.info("🔒 Phase 6: Security Analysis")
        self._analyze_security_issues(component_class, metrics)
        
        # Phase 7: Performance Analysis
        self.logger.info("⚡ Phase 7: Performance Analysis")
        self._analyze_performance_issues(metrics)
        
        return self._finalize_metrics(metrics)
    
    def _find_automl_status_class(self, module):
        """Find the AutoMLStatusWidget class"""
        module_classes = [obj for name, obj in inspect.getmembers(module) 
                         if inspect.isclass(obj) and obj.__module__ == module.__name__]
        
        # Look for AutoMLStatusWidget specifically
        for cls in module_classes:
            if cls.__name__ == 'AutoMLStatusWidget':
                return cls
        
        # Fallback to any widget-like class
        for cls in module_classes:
            if 'widget' in cls.__name__.lower() or 'status' in cls.__name__.lower():
                return cls
                
        return module_classes[0] if module_classes else None
    
    def _try_basic_init(self, component_class):
        """Try basic initialization with no arguments"""
        return component_class()
    
    def _try_init_with_kwargs(self, component_class):
        """Try initialization with common widget kwargs"""
        return component_class(id="test-automl-status", classes="test-class")
    
    def _try_init_with_textual_context(self, component_class):
        """Try initialization with Textual context mocking"""
        with patch('textual.widgets.Static.__init__', return_value=None):
            return component_class()
    
    def _try_init_with_mocked_deps(self, component_class):
        """Try initialization with all dependencies mocked"""
        with patch.multiple(
            'textual.widgets',
            Static=MagicMock,
            autospec=True
        ), patch.multiple(
            'rich.console',
            Console=MagicMock,
            autospec=True
        ):
            return component_class()
    
    def _analyze_individual_quality_metrics(self, component_class, metrics: DiagnosticMetrics):
        """Analyze each quality metric individually"""
        
        # Security Score Analysis
        self.logger.info("🔍 Analyzing Security Score...")
        try:
            security_score = self._calculate_security_score(component_class)
            metrics.individual_scores['security'] = security_score
            self.logger.info(f"🔒 Security Score: {security_score:.3f}")
        except Exception as e:
            metrics.individual_scores['security'] = 0.0
            metrics.security_issues.append(f"Security calculation failed: {e}")
            self.logger.error(f"❌ Security analysis failed: {e}")
        
        # Performance Score Analysis
        self.logger.info("🔍 Analyzing Performance Score...")
        try:
            performance_score = self._calculate_performance_score(
                metrics.load_time_ms, metrics.init_time_ms, metrics.method_count
            )
            metrics.individual_scores['performance'] = performance_score
            self.logger.info(f"⚡ Performance Score: {performance_score:.3f}")
        except Exception as e:
            metrics.individual_scores['performance'] = 0.0
            metrics.performance_warnings.append(f"Performance calculation failed: {e}")
            self.logger.error(f"❌ Performance analysis failed: {e}")
        
        # Reliability Score Analysis
        self.logger.info("🔍 Analyzing Reliability Score...")
        try:
            reliability_score = self._calculate_reliability_score(component_class, metrics.init_error)
            metrics.individual_scores['reliability'] = reliability_score
            self.logger.info(f"🎯 Reliability Score: {reliability_score:.3f}")
        except Exception as e:
            metrics.individual_scores['reliability'] = 0.0
            self.logger.error(f"❌ Reliability analysis failed: {e}")
        
        # Overall Quality Score Calculation
        self.logger.info("🔍 Calculating Overall Quality Score...")
        try:
            overall_score = self._calculate_overall_quality_score(metrics.individual_scores)
            metrics.individual_scores['overall'] = overall_score
            self.logger.info(f"🎯 Overall Quality Score: {overall_score:.3f}")
            
            # Store calculation details
            metrics.quality_calculation_details = {
                'weights': {
                    'security': 0.3,
                    'performance': 0.25,
                    'reliability': 0.25,
                    'coverage': 0.15,
                    'false_positive_risk': -0.05
                },
                'raw_scores': metrics.individual_scores,
                'calculation_formula': 'security*0.3 + performance*0.25 + reliability*0.25 + coverage*0.15 - false_positive_risk*0.05'
            }
            
        except Exception as e:
            metrics.individual_scores['overall'] = 0.0
            self.logger.error(f"❌ Overall quality calculation failed: {e}")
    
    def _calculate_security_score(self, component_class) -> float:
        """Calculate security score (mirrored from comprehensive test)"""
        score = 0.8  # Base score
        
        # Security indicators
        module_path = component_class.__module__
        if 'security' in module_path:
            score += 0.2
        if hasattr(component_class, '__security_validated__'):
            score += 0.1
        if any(method.startswith('validate_') for method in dir(component_class)):
            score += 0.05
            
        # Security penalties
        if any(attr.startswith('_') and not attr.startswith('__') for attr in dir(component_class)):
            score -= 0.05  # Protected attributes may indicate security considerations
            
        return min(1.0, max(0.0, score))
    
    def _calculate_performance_score(self, load_time_ms: float, init_time_ms: float, method_count: int) -> float:
        """Calculate performance score (mirrored from comprehensive test)"""
        load_threshold_ms = 100
        init_threshold_ms = 50
        
        load_score = max(0, 1 - (load_time_ms / load_threshold_ms))
        init_score = max(0, 1 - (init_time_ms / init_threshold_ms))
        
        # Complexity penalty for too many methods
        complexity_score = 1.0 if method_count < 20 else max(0.5, 1 - (method_count - 20) / 100)
        
        return (load_score * 0.4 + init_score * 0.4 + complexity_score * 0.2)
    
    def _calculate_reliability_score(self, component_class, init_error: Optional[str]) -> float:
        """Calculate reliability score (mirrored from comprehensive test)"""
        if init_error:
            return 0.0
            
        score = 0.8  # Base reliability
        
        # Reliability indicators
        if hasattr(component_class, '__init__'):
            score += 0.1
        if hasattr(component_class, '__enter__') and hasattr(component_class, '__exit__'):
            score += 0.1  # Context manager support
            
        # Check for error handling patterns
        try:
            methods = inspect.getmembers(component_class, inspect.isfunction)
            has_error_handling = False
            for method_name, method in methods:
                try:
                    source = inspect.getsource(method)
                    if 'try' in source or 'except' in source:
                        has_error_handling = True
                        break
                except (OSError, TypeError):
                    continue
            
            if has_error_handling:
                score += 0.1
        except Exception:
            pass
            
        return min(1.0, score)
    
    def _calculate_overall_quality_score(self, individual_scores: Dict[str, float]) -> float:
        """Calculate overall quality score using 2025 weighting"""
        weights = {
            'security': 0.3,
            'performance': 0.25,
            'reliability': 0.25,
            'coverage': 0.15,
            'false_positive_risk': -0.05
        }
        
        # Use test coverage estimate of 70% for components with methods
        coverage_score = 0.7 if individual_scores.get('performance', 0) > 0 else 0.0
        false_positive_risk = 0.1  # Low risk for successful analysis
        
        score = (
            individual_scores.get('security', 0) * weights['security'] +
            individual_scores.get('performance', 0) * weights['performance'] +
            individual_scores.get('reliability', 0) * weights['reliability'] +
            coverage_score * weights['coverage'] -
            false_positive_risk * weights['false_positive_risk']
        )
        return max(0.0, min(1.0, score))
    
    def _analyze_textual_specific_issues(self, component_class, metrics: DiagnosticMetrics):
        """Analyze Textual-specific issues"""
        
        # Check if it's properly inheriting from Textual widgets
        base_classes = [cls.__name__ for cls in component_class.__mro__]
        self.logger.info(f"🔍 Class hierarchy: {base_classes}")
        
        if 'Static' not in base_classes and 'Widget' not in base_classes:
            metrics.textual_specific_issues.append("Does not inherit from Textual Static or Widget")
        
        # Check for reactive attributes
        if hasattr(component_class, 'automl_data'):
            self.logger.info("✅ Has reactive automl_data attribute")
        else:
            metrics.textual_specific_issues.append("Missing automl_data reactive attribute")
        
        # Check for required Textual methods
        required_methods = ['compose', 'on_mount']
        for method in required_methods:
            if hasattr(component_class, method):
                self.logger.info(f"✅ Has {method} method")
            else:
                metrics.textual_specific_issues.append(f"Missing {method} method")
        
        # Check for Rich integration
        try:
            source = inspect.getsource(component_class)
            if 'rich.' in source:
                self.logger.info("✅ Uses Rich components")
            else:
                metrics.textual_specific_issues.append("No Rich component usage detected")
        except Exception as e:
            metrics.textual_specific_issues.append(f"Could not analyze source code: {e}")
    
    def _analyze_security_issues(self, component_class, metrics: DiagnosticMetrics):
        """Analyze security-specific issues"""
        
        try:
            source = inspect.getsource(component_class)
            
            # Check for potential security issues
            security_keywords = ['password', 'secret', 'token', 'key', 'auth']
            for keyword in security_keywords:
                if keyword in source.lower():
                    metrics.security_issues.append(f"Potential security keyword detected: {keyword}")
            
            # Check for hardcoded values
            if '"' in source and len([line for line in source.split('\n') if '"' in line and '=' in line]) > 5:
                metrics.security_issues.append("Multiple hardcoded string values detected")
                
        except Exception as e:
            metrics.security_issues.append(f"Could not analyze source for security: {e}")
    
    def _analyze_performance_issues(self, metrics: DiagnosticMetrics):
        """Analyze performance-specific issues"""
        
        if metrics.load_time_ms > 100:
            metrics.performance_warnings.append(f"Load time {metrics.load_time_ms:.1f}ms exceeds 100ms threshold")
        
        if metrics.init_time_ms > 50:
            metrics.performance_warnings.append(f"Initialization time {metrics.init_time_ms:.1f}ms exceeds 50ms threshold")
        
        if metrics.method_count > 50:
            metrics.performance_warnings.append(f"High method count ({metrics.method_count}) may indicate poor separation of concerns")
        
        if metrics.memory_diff_mb > 10:
            metrics.performance_warnings.append(f"High memory usage increase: {metrics.memory_diff_mb:.1f}MB")
    
    def _finalize_metrics(self, metrics: DiagnosticMetrics) -> DiagnosticMetrics:
        """Finalize metrics with memory calculations"""
        metrics.memory_after_mb = self.get_memory_usage()
        metrics.memory_diff_mb = metrics.memory_after_mb - metrics.memory_before_mb
        return metrics
    
    def test_component_in_isolation(self) -> Dict[str, Any]:
        """Test component in complete isolation"""
        self.logger.info("🧪 Testing component in complete isolation...")
        
        isolation_results = {
            'import_test': {},
            'class_instantiation': {},
            'method_calls': {},
            'mock_data_test': {},
            'comprehensive_test_simulation': {}
        }
        
        try:
            # Test 1: Import in clean environment
            start_time = time.time()
            module = importlib.import_module(self.component_path)
            import_time = (time.time() - start_time) * 1000
            
            isolation_results['import_test'] = {
                'success': True,
                'time_ms': import_time,
                'module_attrs': len(dir(module))
            }
            
            # Test 2: Class instantiation with minimal dependencies
            component_class = self._find_automl_status_class(module)
            if component_class:
                try:
                    with patch.multiple(
                        'textual.widgets.Static',
                        __init__=lambda self, **kwargs: None,
                        autospec=True
                    ):
                        instance = component_class()
                        isolation_results['class_instantiation'] = {
                            'success': True,
                            'class_name': component_class.__name__
                        }
                        
                        # Test 3: Method calls with mock data
                        if hasattr(instance, 'update_display'):
                            instance.automl_data = {
                                "status": "running",
                                "current_trial": 5,
                                "total_trials": 10,
                                "best_score": 0.85
                            }
                            instance.update_display()
                            isolation_results['method_calls']['update_display'] = 'success'
                        
                        if hasattr(instance, 'update_data'):
                            mock_provider = Mock()
                            mock_provider.get_automl_status = Mock(return_value={'status': 'idle'})
                            asyncio.run(instance.update_data(mock_provider))
                            isolation_results['method_calls']['update_data'] = 'success'
                            
                except Exception as e:
                    isolation_results['class_instantiation'] = {
                        'success': False,
                        'error': str(e)
                    }
            
            # Test 4: Simulate comprehensive test failure condition
            self.logger.info("🔍 Simulating comprehensive test failure condition...")
            try:
                from comprehensive_component_verification_test import EnhancedComponentLoader, ComponentTestSeverity
                
                loader = EnhancedComponentLoader()
                async_result = asyncio.run(loader.comprehensive_component_test(
                    self.component_name, 
                    self.component_path, 
                    ComponentTestSeverity.MEDIUM
                ))
                
                isolation_results['comprehensive_test_simulation'] = {
                    'success': async_result.test_result.value == 'PASS',
                    'test_result': async_result.test_result.value,
                    'quality_score': async_result.metrics.overall_quality_score(),
                    'error_details': async_result.error_details,
                    'load_time': async_result.metrics.load_time_ms,
                    'init_time': async_result.metrics.initialization_time_ms,
                    'warnings': async_result.warnings,
                    'false_positive_indicators': async_result.false_positive_indicators
                }
                
                self.logger.info(f"Comprehensive test result: {async_result.test_result.value}")
                self.logger.info(f"Comprehensive quality score: {async_result.metrics.overall_quality_score():.3f}")
                
            except Exception as e:
                isolation_results['comprehensive_test_simulation'] = {
                    'success': False,
                    'error': str(e),
                    'traceback': traceback.format_exc()
                }
                self.logger.error(f"Comprehensive test simulation failed: {e}")
            
        except Exception as e:
            isolation_results['import_test'] = {
                'success': False,
                'error': str(e)
            }
        
        return isolation_results
    
    def generate_comprehensive_report(self, metrics: DiagnosticMetrics, 
                                    compatibility: Dict[str, Any], 
                                    isolation_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive diagnostic report"""
        
        report = {
            'component_info': {
                'name': metrics.component_name,
                'path': self.component_path,
                'timestamp': datetime.now().isoformat(),
                'python_version': sys.version,
                'platform': sys.platform
            },
            'load_analysis': {
                'success': metrics.load_success,
                'time_ms': metrics.load_time_ms,
                'error': metrics.load_error
            },
            'class_analysis': {
                'found': metrics.class_found,
                'name': metrics.class_name,
                'method_count': metrics.method_count,
                'methods': metrics.method_names,
                'error': metrics.class_error
            },
            'initialization_analysis': {
                'success': metrics.init_success,
                'time_ms': metrics.init_time_ms,
                'error': metrics.init_error
            },
            'quality_scores': {
                'individual': metrics.individual_scores,
                'calculation_details': metrics.quality_calculation_details
            },
            'performance_analysis': {
                'memory_before_mb': metrics.memory_before_mb,
                'memory_after_mb': metrics.memory_after_mb,
                'memory_diff_mb': metrics.memory_diff_mb,
                'warnings': metrics.performance_warnings
            },
            'security_analysis': {
                'issues': metrics.security_issues
            },
            'textual_analysis': {
                'issues': metrics.textual_specific_issues
            },
            'environment_compatibility': compatibility,
            'isolation_test_results': isolation_results,
            'recommendations': self._generate_recommendations(metrics, compatibility, isolation_results)
        }
        
        return report
    
    def _generate_recommendations(self, metrics: DiagnosticMetrics, 
                                 compatibility: Dict[str, Any], 
                                 isolation_results: Dict[str, Any]) -> List[str]:
        """Generate specific recommendations for fixing the quality score"""
        recommendations = []
        
        # Analyze comprehensive test simulation results
        comp_sim = isolation_results.get('comprehensive_test_simulation', {})
        if not comp_sim.get('success', True) and comp_sim.get('error'):
            error_msg = comp_sim.get('error', '')
            if '_classes' in error_msg:
                recommendations.append("CRITICAL: Fix Textual widget _classes attribute initialization")
                recommendations.append("  - Add proper super().__init__() call in AutoMLStatusWidget.__init__")
                recommendations.append("  - Ensure all parent class initialization parameters are passed correctly")
                recommendations.append("  - The '_classes' error indicates widget DOM node is not properly initialized")
            elif 'NoneType' in error_msg:
                recommendations.append("CRITICAL: Fix NoneType attribute access")
                recommendations.append("  - Check for missing widget initialization in parent classes")
                recommendations.append("  - Ensure all required Textual widget attributes are set")
        
        # Load issues
        if not metrics.load_success:
            recommendations.append("Fix module import issues - check dependencies and import paths")
            if "No module named" in str(metrics.load_error):
                recommendations.append("Install missing dependencies or verify module structure")
        
        # Class discovery issues
        if not metrics.class_found:
            recommendations.append("Verify class name and structure - expecting AutoMLStatusWidget")
        
        # Initialization issues
        if not metrics.init_success:
            recommendations.append("Fix component initialization - likely Textual widget inheritance issue")
            recommendations.append("Review parent class __init__ call and parameter passing")
        
        # Performance issues
        if metrics.performance_warnings:
            recommendations.append("Address performance warnings to improve performance score")
        
        # Textual-specific issues
        if metrics.textual_specific_issues:
            recommendations.append("Fix Textual widget implementation issues")
            for issue in metrics.textual_specific_issues:
                recommendations.append(f"  - {issue}")
        
        # Environment issues
        failed_imports = [name for name, result in compatibility.get('textual_imports', {}).items() 
                         if not result.get('success', False)]
        if failed_imports:
            recommendations.append(f"Fix environment dependencies: {failed_imports}")
        
        # Specific quality score analysis
        overall_score = metrics.individual_scores.get('overall', 0)
        comp_quality_score = comp_sim.get('quality_score', overall_score)
        
        if comp_quality_score < 0.01 and overall_score > 0.5:
            recommendations.append("DISCREPANCY DETECTED: Component works in isolation but fails in comprehensive test")
            recommendations.append("  - This indicates environment or initialization context issues")
            recommendations.append("  - Focus on fixing the comprehensive test initialization rather than component logic")
        
        if overall_score < 0.1:
            recommendations.append("Critical: Component completely non-functional - requires major fixes")
        elif overall_score < 0.5:
            recommendations.append("Major issues detected - component needs significant work")
        elif overall_score < 0.8:
            recommendations.append("Moderate issues detected - component needs optimization")
        
        return recommendations
    
    def print_detailed_report(self, report: Dict[str, Any]):
        """Print detailed diagnostic report"""
        print("\n" + "="*80)
        print("🔍 AUTOML STATUS COMPONENT DIAGNOSTIC REPORT")
        print("="*80)
        
        # Component Info
        info = report['component_info']
        print(f"📦 Component: {info['name']}")
        print(f"📁 Path: {info['path']}")
        print(f"⏰ Timestamp: {info['timestamp']}")
        print(f"🐍 Python: {info['python_version'].split()[0]}")
        
        # Load Analysis
        print(f"\n📥 LOAD ANALYSIS")
        print("-" * 40)
        load = report['load_analysis']
        status = "✅ SUCCESS" if load['success'] else "❌ FAILED"
        print(f"Status: {status}")
        print(f"Time: {load['time_ms']:.2f}ms")
        if load['error']:
            print(f"Error: {load['error']}")
        
        # Class Analysis
        print(f"\n🏗️  CLASS ANALYSIS")
        print("-" * 40)
        class_info = report['class_analysis']
        status = "✅ FOUND" if class_info['found'] else "❌ NOT FOUND"
        print(f"Status: {status}")
        if class_info['name']:
            print(f"Class Name: {class_info['name']}")
            print(f"Method Count: {class_info['method_count']}")
            print(f"Methods: {', '.join(class_info['methods'][:5])}...")
        if class_info['error']:
            print(f"Error: {class_info['error']}")
        
        # Initialization Analysis
        print(f"\n⚙️  INITIALIZATION ANALYSIS")
        print("-" * 40)
        init = report['initialization_analysis']
        status = "✅ SUCCESS" if init['success'] else "❌ FAILED"
        print(f"Status: {status}")
        print(f"Time: {init['time_ms']:.2f}ms")
        if init['error']:
            print(f"Error: {init['error']}")
        
        # Quality Scores
        print(f"\n📊 QUALITY SCORES")
        print("-" * 40)
        scores = report['quality_scores']['individual']
        for metric, score in scores.items():
            indicator = "🟢" if score > 0.8 else "🟡" if score > 0.5 else "🔴"
            print(f"{indicator} {metric.capitalize()}: {score:.3f}")
        
        # Quality Calculation Details
        if report['quality_scores']['calculation_details']:
            print(f"\n📋 QUALITY CALCULATION DETAILS")
            print("-" * 40)
            details = report['quality_scores']['calculation_details']
            print(f"Formula: {details.get('calculation_formula', 'N/A')}")
            print("Weights:")
            for weight_name, weight_val in details.get('weights', {}).items():
                print(f"  {weight_name}: {weight_val}")
        
        # Performance Analysis
        print(f"\n⚡ PERFORMANCE ANALYSIS")
        print("-" * 40)
        perf = report['performance_analysis']
        print(f"Memory Before: {perf['memory_before_mb']:.1f}MB")
        print(f"Memory After: {perf['memory_after_mb']:.1f}MB")
        print(f"Memory Delta: {perf['memory_diff_mb']:.1f}MB")
        if perf['warnings']:
            print("Warnings:")
            for warning in perf['warnings']:
                print(f"  ⚠️  {warning}")
        
        # Security Analysis
        print(f"\n🔒 SECURITY ANALYSIS")
        print("-" * 40)
        sec_issues = report['security_analysis']['issues']
        if sec_issues:
            for issue in sec_issues:
                print(f"  ⚠️  {issue}")
        else:
            print("✅ No security issues detected")
        
        # Textual Analysis
        print(f"\n🎨 TEXTUAL WIDGET ANALYSIS")
        print("-" * 40)
        textual_issues = report['textual_analysis']['issues']
        if textual_issues:
            for issue in textual_issues:
                print(f"  ❌ {issue}")
        else:
            print("✅ No Textual-specific issues detected")
        
        # Environment Compatibility
        print(f"\n🌍 ENVIRONMENT COMPATIBILITY")
        print("-" * 40)
        compat = report['environment_compatibility']
        
        print("Textual Imports:")
        for name, result in compat['textual_imports'].items():
            status = "✅" if result.get('success', False) else "❌"
            if result.get('success', False):
                print(f"  {status} {name} ({result.get('load_time_ms', 0):.1f}ms)")
            else:
                print(f"  {status} {name}: {result.get('error', 'Unknown error')}")
        
        print("Rich Imports:")
        for name, result in compat['rich_imports'].items():
            status = "✅" if result.get('success', False) else "❌"
            if result.get('success', False):
                print(f"  {status} {name} ({result.get('load_time_ms', 0):.1f}ms)")
            else:
                print(f"  {status} {name}: {result.get('error', 'Unknown error')}")
        
        # Isolation Test Results
        print(f"\n🧪 ISOLATION TEST RESULTS")
        print("-" * 40)
        isolation = report['isolation_test_results']
        for test_name, result in isolation.items():
            if isinstance(result, dict):
                success = result.get('success', False)
                status = "✅" if success else "❌"
                print(f"{status} {test_name}: {result}")
            else:
                print(f"🔍 {test_name}: {result}")
        
        # Recommendations
        print(f"\n💡 RECOMMENDATIONS")
        print("-" * 40)
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"{i}. {rec}")
        
        # Final Assessment
        print(f"\n🎯 FINAL ASSESSMENT")
        print("=" * 40)
        overall_score = scores.get('overall', 0)
        if overall_score < 0.1:
            print("🔴 CRITICAL: Component is completely non-functional")
            print("   - This explains the 0.01 quality score")
            print("   - All subsystems (load, init, etc.) are failing")
        elif overall_score < 0.3:
            print("🔴 SEVERE: Major functionality issues")
        elif overall_score < 0.6:
            print("🟡 MODERATE: Significant issues need addressing")
        else:
            print("🟢 GOOD: Minor issues to optimize")
        
        print(f"\n🏁 Quality Score Diagnosis Complete!")
        print(f"📈 Current Score: {overall_score:.3f} (Target: >0.80)")
        print(f"🎯 Score Improvement Potential: {0.80 - overall_score:.3f}")


async def main():
    """Main diagnostic execution"""
    print("🚀 AutoML Status Component Quality Diagnostic Tool")
    print("="*80)
    
    diagnostic_tool = AutoMLStatusDiagnosticTool()
    
    try:
        # Step 1: Test environment compatibility
        compatibility = diagnostic_tool.test_environment_compatibility()
        
        # Step 2: Load component with comprehensive diagnostics
        metrics = diagnostic_tool.load_component_with_diagnostics()
        
        # Step 3: Test component in isolation
        isolation_results = diagnostic_tool.test_component_in_isolation()
        
        # Step 4: Generate and display comprehensive report
        report = diagnostic_tool.generate_comprehensive_report(metrics, compatibility, isolation_results)
        diagnostic_tool.print_detailed_report(report)
        
        # Step 5: Save detailed logs
        print(f"\n💾 Detailed logs saved to: automl_status_diagnostic.log")
        
        return 0
        
    except Exception as e:
        print(f"\n❌ Diagnostic tool failed: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)