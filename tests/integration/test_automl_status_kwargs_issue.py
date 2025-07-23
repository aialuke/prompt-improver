#!/usr/bin/env python3
"""
Test script to verify the kwargs issue with AutoMLStatusWidget

This script tests the specific scenario where the mock dependency generation
creates a 'kwargs' parameter for the **kwargs signature, causing issues.
"""

import sys
from pathlib import Path
import inspect
from unittest.mock import Mock

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

def test_mock_generation_issue():
    """Test the specific mock generation issue"""
    print("🧪 Testing Mock Generation Issue with AutoMLStatusWidget")
    print("=" * 60)
    
    from prompt_improver.tui.widgets.automl_status import AutoMLStatusWidget
    
    # Test the signature
    signature = inspect.signature(AutoMLStatusWidget.__init__)
    print(f"📋 AutoMLStatusWidget.__init__ signature: {signature}")
    
    # Replicate the mock generation logic from comprehensive test
    def generate_mock_dependencies(component_class):
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
    
    # Generate mocks
    mock_kwargs = generate_mock_dependencies(AutoMLStatusWidget)
    print(f"🎭 Generated mock kwargs: {mock_kwargs}")
    
    # Test scenarios
    scenarios = [
        ("No parameters", {}),
        ("Valid Textual parameters", {"id": "test-widget", "classes": "test-class"}),
        ("Mock parameters (comprehensive test)", mock_kwargs),
        ("Mixed parameters", {"id": "test", "invalid": Mock()}),
    ]
    
    for scenario_name, test_kwargs in scenarios:
        print(f"\n🧪 Testing: {scenario_name}")
        print(f"   Parameters: {test_kwargs}")
        
        try:
            widget = AutoMLStatusWidget(**test_kwargs)
            print(f"   ✅ SUCCESS - Widget created successfully")
            
            # Test if the widget actually filters kwargs properly
            if hasattr(widget, '__dict__'):
                widget_attrs = {k: v for k, v in widget.__dict__.items() if not k.startswith('_')}
                print(f"   📦 Widget attributes: {list(widget_attrs.keys())}")
        
        except Exception as e:
            print(f"   ❌ FAILED - {e}")

    # Test the actual filtering mechanism
    print(f"\n🔍 Testing kwargs filtering mechanism:")
    print("-" * 40)
    
    try:
        # Create instance and check if it properly filters
        widget = AutoMLStatusWidget(id="test", invalid_param="should_be_filtered", classes="valid")
        print("✅ Widget creation with mixed valid/invalid params succeeded")
        print("   (This suggests kwargs filtering is working)")
        
        # Check if the invalid param made it through
        if hasattr(widget, 'invalid_param'):
            print("❌ Invalid parameter was not filtered!")
        else:
            print("✅ Invalid parameter was properly filtered")
            
    except Exception as e:
        print(f"❌ Widget creation failed: {e}")

    # Test specifically what the comprehensive test does
    print(f"\n🏭 Simulating exact comprehensive test scenario:")
    print("-" * 50)
    
    # Generate mock exactly as comprehensive test does
    comprehensive_mocks = generate_mock_dependencies(AutoMLStatusWidget)
    print(f"Mock kwargs generated: {comprehensive_mocks}")
    
    if comprehensive_mocks:
        try:
            widget = AutoMLStatusWidget(**comprehensive_mocks)
            print("✅ Comprehensive test mock scenario SUCCEEDS")
            print("   (Widget accepts mock dependencies)")
        except Exception as e:
            print(f"❌ Comprehensive test mock scenario FAILS: {e}")
            print("   (Widget rejects mock dependencies)")
    else:
        print("ℹ️  No mock dependencies generated")

if __name__ == "__main__":
    test_mock_generation_issue()