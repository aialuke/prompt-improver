#!/usr/bin/env python3
"""
Debug script to investigate the 12 failing components and identify root causes.
"""

import sys
import traceback
import importlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

# The 12 failing components from our test
FAILING_COMPONENTS = {
    "enhanced_scorer": "prompt_improver.ml.learning.quality.enhanced_scorer",
    "monitoring": "prompt_improver.performance.monitoring.monitoring", 
    "performance_validation": "prompt_improver.performance.validation.performance_validation",
    "multi_armed_bandit": "prompt_improver.ml.optimization.algorithms.multi_armed_bandit",
    "context_learner": "prompt_improver.ml.learning.algorithms.context_learner",
    "failure_analyzer": "prompt_improver.ml.learning.algorithms.failure_analyzer",
    "insight_engine": "prompt_improver.ml.learning.algorithms.insight_engine",
    "rule_analyzer": "prompt_improver.ml.learning.algorithms.rule_analyzer",
    "automl_orchestrator": "prompt_improver.ml.automl.orchestrator",
    "ner_extractor": "prompt_improver.ml.analysis.ner_extractor",
    "background_manager": "prompt_improver.performance.monitoring.health.background_manager",
    "automl_status": "prompt_improver.tui.widgets.automl_status",
}

def debug_component(component_name: str, module_path: str):
    """Debug a specific component to understand the failure"""
    print(f"\n🔍 Debugging {component_name}")
    print("=" * 60)
    
    try:
        # Try to import the module
        print(f"📦 Importing module: {module_path}")
        module = importlib.import_module(module_path)
        print("✅ Module import successful")
        
        # Find classes in the module
        import inspect
        classes = [obj for name, obj in inspect.getmembers(module) 
                  if inspect.isclass(obj) and obj.__module__ == module.__name__]
        
        if not classes:
            print("❌ No classes found in module")
            return
        
        print(f"📋 Found {len(classes)} classes: {[cls.__name__ for cls in classes]}")
        
        # Try to find the main class
        main_class = None
        possible_names = [
            component_name.title().replace("_", ""),
            f"{component_name.title().replace('_', '')}Service",
            f"{component_name.title().replace('_', '')}Manager",
            f"{component_name.title().replace('_', '')}Analyzer",
        ]
        
        for name in possible_names:
            for cls in classes:
                if cls.__name__ == name:
                    main_class = cls
                    break
            if main_class:
                break
        
        if not main_class:
            main_class = classes[0]
        
        print(f"🎯 Using main class: {main_class.__name__}")
        
        # Analyze the class constructor
        try:
            sig = inspect.signature(main_class.__init__)
            print(f"📝 Constructor signature: {sig}")
            
            # Try to instantiate with no arguments
            print("🧪 Testing no-argument instantiation...")
            try:
                instance = main_class()
                print("✅ No-argument instantiation successful")
            except Exception as e:
                print(f"❌ No-argument instantiation failed: {e}")
                
                # Try with basic mock arguments
                print("🧪 Testing with mock arguments...")
                try:
                    from unittest.mock import Mock
                    kwargs = {}
                    for param_name, param in sig.parameters.items():
                        if param_name != 'self' and param.default == inspect.Parameter.empty:
                            kwargs[param_name] = Mock()
                    
                    if kwargs:
                        print(f"📋 Using mock arguments: {list(kwargs.keys())}")
                        instance = main_class(**kwargs)
                        print("✅ Mock argument instantiation successful")
                    else:
                        print("❌ No required parameters but still failed")
                        
                except Exception as mock_e:
                    print(f"❌ Mock argument instantiation failed: {mock_e}")
                    # Print the full traceback for this error
                    print("📚 Full error traceback:")
                    traceback.print_exc()
        
        except Exception as sig_e:
            print(f"❌ Could not analyze constructor: {sig_e}")
            
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("📚 Full error traceback:")
        traceback.print_exc()
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        print("📚 Full error traceback:")
        traceback.print_exc()

def main():
    print("🔧 Debugging 12 Failing Components")
    print("=" * 80)
    
    for component_name, module_path in FAILING_COMPONENTS.items():
        debug_component(component_name, module_path)
    
    print("\n" + "=" * 80)
    print("🎯 Investigation Complete")

if __name__ == "__main__":
    main()