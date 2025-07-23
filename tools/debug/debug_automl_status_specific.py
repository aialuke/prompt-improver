#!/usr/bin/env python3
"""
Specific AutoML Status Debug Script

This script replicates the EXACT comprehensive test logic for automl_status
to identify why it gets 0.01 score when other components get higher scores.
"""

import asyncio
import inspect
import importlib
import logging
import time
import traceback
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from unittest.mock import Mock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Copy exact classes from comprehensive test
from enum import Enum

class ComponentTestResult(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    SKIP = "SKIP"
    WARNING = "WARNING"
    FALSE_POSITIVE = "FALSE_POSITIVE"
    CRITICAL_FAIL = "CRITICAL_FAIL"

class ComponentTestSeverity(Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH" 
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"

@dataclass
class ComponentTestMetrics:
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
        weights = {
            'security': 0.3,
            'performance': 0.25,
            'reliability': 0.25,
            'coverage': 0.15,
            'false_positive_risk': -0.05  # Penalty for false positive risk
        }
        
        score = (
            self.security_score * weights['security'] +
            self.performance_score * weights['performance'] +
            self.reliability_score * weights['reliability'] +
            self.test_coverage_percent/100 * weights['coverage'] -
            self.false_positive_risk * weights['false_positive_risk']
        )
        return max(0.0, min(1.0, score))

class AutoMLStatusDebugger:
    """Debug automl_status specifically"""
    
    def __init__(self):
        self.component_name = "automl_status"
        self.module_path = "prompt_improver.tui.widgets.automl_status"
        
    def _find_main_class(self, module: Any, component_name: str) -> Optional[type]:
        """Find the main class - EXACT copy from comprehensive test"""
        module_classes = [obj for name, obj in inspect.getmembers(module) 
                         if inspect.isclass(obj) and obj.__module__ == module.__name__]
        
        if not module_classes:
            return None
        
        # Specific mappings for problematic components
        specific_mappings = {
            "automl_status": "AutoMLStatusWidget",
        }
        
        # Check specific mappings first
        if component_name in specific_mappings:
            target_class_name = specific_mappings[component_name]
            for cls in module_classes:
                if cls.__name__ == target_class_name:
                    return cls
        
        return None
    
    def _generate_mock_dependencies(self, component_class: type) -> Dict[str, Any]:
        """Generate mock dependencies - EXACT copy from comprehensive test"""
        try:
            init_signature = inspect.signature(component_class.__init__)
            mock_kwargs = {}
            
            for param_name, param in init_signature.parameters.items():
                if param_name == 'self':
                    continue
                    
                # Generate appropriate mocks based on type hints
                if param.annotation != inspect.Parameter.empty:
                    if param.annotation == str:
                        mock_kwargs[param_name] = f"mock_{param_name}"
                    elif param.annotation == int:
                        mock_kwargs[param_name] = 1
                    elif param.annotation == float:
                        mock_kwargs[param_name] = 1.0
                    elif param.annotation == bool:
                        mock_kwargs[param_name] = True
                    else:
                        mock_kwargs[param_name] = Mock()
                elif param.default != inspect.Parameter.empty:
                    # Parameter has default, don't provide mock
                    continue
                else:
                    # Unknown type, provide generic mock
                    mock_kwargs[param_name] = Mock()
            
            return mock_kwargs
        except Exception:
            return {}

    def _estimate_memory_usage(self, component_class: type) -> float:
        """Estimate memory usage - EXACT copy"""
        methods = len([m for m in dir(component_class) if not m.startswith('_')])
        return methods * 0.1

    def _get_dependencies(self, component_class: type) -> List[str]:
        """Extract component dependencies - EXACT copy"""
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
        """Estimate test coverage - EXACT copy"""
        methods = [m for m in dir(component_class) if not m.startswith('_') and callable(getattr(component_class, m))]
        return 70.0 if len(methods) > 0 else 0.0

    def calculate_security_score(self, component_class: type, module_path: str) -> float:
        """Calculate security score - EXACT copy"""
        score = 0.8  # Base score
        
        # Security indicators
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

    def calculate_performance_score(self, load_time_ms: float, init_time_ms: float, 
                                   method_count: int) -> float:
        """Calculate performance score - EXACT copy"""
        # Performance thresholds (2025 standards)
        load_threshold_ms = 100  # Components should load under 100ms
        init_threshold_ms = 50   # Initialization under 50ms
        
        load_score = max(0, 1 - (load_time_ms / load_threshold_ms))
        init_score = max(0, 1 - (init_time_ms / init_threshold_ms))
        
        # Complexity penalty for too many methods
        complexity_score = 1.0 if method_count < 20 else max(0.5, 1 - (method_count - 20) / 100)
        
        return (load_score * 0.4 + init_score * 0.4 + complexity_score * 0.2)

    def calculate_reliability_score(self, component_class: type, error: Optional[Exception] = None) -> float:
        """Calculate reliability score - EXACT copy"""
        if error:
            return 0.0
            
        score = 0.8  # Base reliability
        
        # Reliability indicators
        if hasattr(component_class, '__init__'):
            score += 0.1
        if hasattr(component_class, '__enter__') and hasattr(component_class, '__exit__'):
            score += 0.1  # Context manager support
            
        # Check for error handling patterns - THIS MIGHT CAUSE ISSUES
        try:
            methods = inspect.getmembers(component_class, inspect.isfunction)
            has_error_handling = False
            for method_name, method in methods:
                if hasattr(method, '__code__'):
                    try:
                        source = inspect.getsource(method)
                        if 'try' in source:
                            has_error_handling = True
                            break
                    except Exception:
                        pass
            if has_error_handling:
                score += 0.1
        except Exception as e:
            print(f"   ⚠️ Error in reliability calculation: {e}")
            
        return min(1.0, score)

    def _calculate_business_impact(self, component_name: str, severity: ComponentTestSeverity) -> float:
        """Calculate business impact score - EXACT copy"""
        severity_scores = {
            ComponentTestSeverity.CRITICAL: 1.0,
            ComponentTestSeverity.HIGH: 0.8,
            ComponentTestSeverity.MEDIUM: 0.6,
            ComponentTestSeverity.LOW: 0.4,
            ComponentTestSeverity.INFO: 0.2
        }
        return severity_scores[severity]

    async def comprehensive_component_test_exact(self) -> ComponentTestMetrics:
        """EXACT replica of comprehensive_component_test from the suite"""
        print(f"🔍 Testing {self.component_name} with EXACT comprehensive test logic")
        start_time = time.time()
        
        try:
            # Phase 1: Load Testing with Performance Monitoring
            load_start = time.time()
            module = importlib.import_module(self.module_path)
            load_time_ms = (time.time() - load_start) * 1000
            print(f"   ✅ Module loaded in {load_time_ms:.2f}ms")
            
            # Phase 2: Component Class Discovery with AI Pattern Recognition
            component_class = self._find_main_class(module, self.component_name)
            if not component_class:
                raise ImportError(f"No suitable class found in {self.module_path}")
            print(f"   ✅ Found class: {component_class.__name__}")
            
            # Phase 3: Initialization Testing with Dependency Injection
            init_start = time.time()
            init_error = None
            try:
                # Smart initialization with mock dependencies
                init_kwargs = self._generate_mock_dependencies(component_class)
                print(f"   📦 Mock kwargs: {init_kwargs}")
                if init_kwargs:
                    component_instance = component_class(**init_kwargs)
                    print(f"   ✅ Initialized WITH mocks")
                else:
                    component_instance = component_class()
                    print(f"   ✅ Initialized WITHOUT mocks")
                init_time_ms = (time.time() - init_start) * 1000
                print(f"   ✅ Initialization time: {init_time_ms:.2f}ms")
            except Exception as e:
                init_time_ms = (time.time() - init_start) * 1000
                init_error = e
                component_instance = None
                print(f"   ❌ Initialization failed: {e}")
            
            # Phase 4: Comprehensive Metrics Collection
            methods = []
            try:
                all_methods = [method for method in dir(component_class) 
                             if not method.startswith('_')]
                
                # Filter callable methods - this might trigger the Textual descriptor issue
                for method_name in all_methods:
                    try:
                        method_obj = getattr(component_class, method_name, None)
                        if callable(method_obj):
                            methods.append(method_name)
                    except Exception as method_error:
                        print(f"   ⚠️ Error accessing method {method_name}: {method_error}")
                        # In comprehensive test, this error might be silently caught
                
                print(f"   📊 Found {len(methods)} callable methods")
                
            except Exception as e:
                print(f"   ❌ Error collecting methods: {e}")
                methods = []
            
            # Calculate individual scores
            security_score = self.calculate_security_score(component_class, self.module_path)
            performance_score = self.calculate_performance_score(load_time_ms, init_time_ms, len(methods))
            reliability_score = self.calculate_reliability_score(component_class, init_error)
            
            print(f"   📊 Security score: {security_score:.3f}")
            print(f"   📊 Performance score: {performance_score:.3f}")
            print(f"   📊 Reliability score: {reliability_score:.3f}")
            
            metrics = ComponentTestMetrics(
                component_name=self.component_name,
                load_time_ms=load_time_ms,
                initialization_time_ms=init_time_ms,
                memory_usage_mb=self._estimate_memory_usage(component_class),
                dependency_count=len(self._get_dependencies(component_class)),
                method_count=len(methods),
                test_coverage_percent=self._estimate_test_coverage(component_class),
                security_score=security_score,
                performance_score=performance_score,
                reliability_score=reliability_score,
                false_positive_risk=0.1,  # Low risk for successful load
                business_impact_score=self._calculate_business_impact(self.component_name, ComponentTestSeverity.HIGH)
            )
            
            overall_score = metrics.overall_quality_score()
            print(f"   🎯 OVERALL QUALITY SCORE: {overall_score:.3f}")
            
            execution_time_ms = (time.time() - start_time) * 1000
            print(f"   ⏱️ Total execution time: {execution_time_ms:.2f}ms")
            
            return metrics
            
        except Exception as e:
            print(f"   ❌ Comprehensive test FAILED: {e}")
            traceback.print_exc()
            
            # Create failure metrics
            metrics = ComponentTestMetrics(
                component_name=self.component_name,
                load_time_ms=0.0,
                initialization_time_ms=0.0,
                memory_usage_mb=0.0,
                dependency_count=0,
                method_count=0,
                test_coverage_percent=0.0,
                security_score=0.0,
                performance_score=0.0,
                reliability_score=0.0,
                false_positive_risk=0.8,
                business_impact_score=0.0
            )
            
            overall_score = metrics.overall_quality_score()
            print(f"   🎯 FAILURE QUALITY SCORE: {overall_score:.3f}")
            
            return metrics

    def analyze_score_components(self, metrics: ComponentTestMetrics):
        """Analyze why the score is low"""
        print("\n" + "="*60)
        print("📊 DETAILED SCORE ANALYSIS")
        print("="*60)
        
        weights = {
            'security': 0.3,
            'performance': 0.25,
            'reliability': 0.25,
            'coverage': 0.15,
            'false_positive_risk': -0.05
        }
        
        # Calculate individual contributions
        security_contribution = metrics.security_score * weights['security']
        performance_contribution = metrics.performance_score * weights['performance']
        reliability_contribution = metrics.reliability_score * weights['reliability']
        coverage_contribution = (metrics.test_coverage_percent/100) * weights['coverage']
        false_positive_penalty = metrics.false_positive_risk * weights['false_positive_risk']
        
        print(f"Security score:     {metrics.security_score:.3f} × {weights['security']} = {security_contribution:.3f}")
        print(f"Performance score:  {metrics.performance_score:.3f} × {weights['performance']} = {performance_contribution:.3f}")
        print(f"Reliability score:  {metrics.reliability_score:.3f} × {weights['reliability']} = {reliability_contribution:.3f}")
        print(f"Coverage score:     {metrics.test_coverage_percent/100:.3f} × {weights['coverage']} = {coverage_contribution:.3f}")
        print(f"False positive pen: {metrics.false_positive_risk:.3f} × {weights['false_positive_risk']} = {false_positive_penalty:.3f}")
        
        total = security_contribution + performance_contribution + reliability_contribution + coverage_contribution - false_positive_penalty
        print(f"\nTotal before clamp: {total:.3f}")
        print(f"Final score:        {max(0.0, min(1.0, total)):.3f}")
        
        # Identify the main issue
        if metrics.security_score == 0.0:
            print("\n🚨 MAJOR ISSUE: Security score is 0.0")
        if metrics.performance_score == 0.0:
            print("🚨 MAJOR ISSUE: Performance score is 0.0")
        if metrics.reliability_score == 0.0:
            print("🚨 MAJOR ISSUE: Reliability score is 0.0")

async def main():
    """Run the specific automl_status debug"""
    print("🚀 AutoML Status Specific Debug Tool")
    print("="*60)
    
    debugger = AutoMLStatusDebugger()
    
    try:
        metrics = await debugger.comprehensive_component_test_exact()
        debugger.analyze_score_components(metrics)
        
        print(f"\n🎯 FINAL RESULT:")
        print(f"Component: {metrics.component_name}")
        print(f"Quality Score: {metrics.overall_quality_score():.3f}")
        
        if metrics.overall_quality_score() < 0.1:
            print("❌ CONFIRMED: Low quality score reproduced")
            return 1
        else:
            print("✅ Score is acceptable")
            return 0
        
    except Exception as e:
        print(f"❌ Debug failed: {e}")
        traceback.print_exc()
        return 2

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)