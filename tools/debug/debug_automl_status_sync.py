#!/usr/bin/env python3
"""
Synchronous debug test for automl_status comprehensive test failure
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from comprehensive_component_verification_test import EnhancedComponentLoader, ComponentTestSeverity
import asyncio

def debug_automl_status_comprehensive():
    """Debug the specific automl_status component test failure"""
    print("🔍 Debugging automl_status comprehensive test failure...")
    
    loader = EnhancedComponentLoader()
    component_name = "automl_status"
    module_path = "prompt_improver.tui.widgets.automl_status"
    severity = ComponentTestSeverity.MEDIUM
    
    try:
        # Run the test
        result = asyncio.run(loader.comprehensive_component_test(component_name, module_path, severity))
        
        print(f"\n📊 COMPREHENSIVE TEST RESULTS:")
        print(f"Result: {result.test_result.value}")
        print(f"Quality Score: {result.metrics.overall_quality_score():.6f}")
        print(f"Security Score: {result.metrics.security_score:.3f}")
        print(f"Performance Score: {result.metrics.performance_score:.3f}")
        print(f"Reliability Score: {result.metrics.reliability_score:.3f}")
        
        if result.error_details:
            print(f"\n❌ ERROR DETAILS:")
            print(f"  {result.error_details}")
            
            # Analyze the specific error
            if '_classes' in result.error_details:
                print(f"\n🔍 ROOT CAUSE ANALYSIS:")
                print(f"  - The error '_classes' indicates a Textual widget initialization issue")
                print(f"  - This happens when the widget's DOM node is not properly initialized")
                print(f"  - Likely cause: Missing or incorrect super().__init__() call")
        
        if result.warnings:
            print(f"\n⚠️  WARNINGS:")
            for warning in result.warnings:
                print(f"  - {warning}")
        
        # Try to get more details about the initialization
        print(f"\n🔧 DETAILED INITIALIZATION ANALYSIS:")
        try:
            import importlib
            module = importlib.import_module(module_path)
            component_class = getattr(module, 'AutoMLStatusWidget')
            
            print(f"  - Class: {component_class.__name__}")
            print(f"  - MRO: {[cls.__name__ for cls in component_class.__mro__]}")
            
            # Try different initialization approaches
            print(f"  - Testing different initialization approaches...")
            
            # Approach 1: Basic init
            try:
                instance1 = component_class()
                print(f"    ✅ Basic init works")
            except Exception as e:
                print(f"    ❌ Basic init failed: {e}")
            
            # Approach 2: With kwargs
            try:
                instance2 = component_class(id="test")
                print(f"    ✅ Init with id works")
            except Exception as e:
                print(f"    ❌ Init with id failed: {e}")
            
            # Approach 3: Mock dependencies approach (like comprehensive test)
            try:
                from unittest.mock import Mock
                import inspect
                
                # Simulate the mock generation from comprehensive test
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
                
                print(f"    - Mock kwargs: {mock_kwargs}")
                
                if mock_kwargs:
                    instance3 = component_class(**mock_kwargs)
                    print(f"    ✅ Init with mocks works: {list(mock_kwargs.keys())}")
                else:
                    instance3 = component_class()
                    print(f"    ✅ Init with no mocks needed works")
                    
            except Exception as e:
                print(f"    ❌ Mock init failed: {e}")
                import traceback
                print(f"    📋 Traceback: {traceback.format_exc()}")
        
        except Exception as e:
            print(f"  ❌ Detailed analysis failed: {e}")
        
        return result
        
    except Exception as e:
        print(f"❌ Comprehensive test completely failed: {e}")
        import traceback
        print(f"📋 Full traceback:")
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = debug_automl_status_comprehensive()
    
    print(f"\n🎯 DIAGNOSIS SUMMARY:")
    if result and result.error_details and '_classes' in result.error_details:
        print(f"🔴 CONFIRMED: Textual widget initialization issue causing 0.01 quality score")
        print(f"🔧 SOLUTION: Fix the __init__ method in AutoMLStatusWidget")
        print(f"📝 SPECIFIC ACTION: Ensure super().__init__() is called with correct parameters")
    elif result:
        print(f"🟡 Unexpected result - quality score: {result.metrics.overall_quality_score():.6f}")
    else:
        print(f"🔴 Could not complete diagnosis")