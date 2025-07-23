#!/usr/bin/env python3
"""
Debug the specific automl_status test from the comprehensive suite
to see exactly what error is causing the 0.01 quality score.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from comprehensive_component_verification_test import EnhancedComponentLoader, ComponentTestSeverity

async def debug_automl_status():
    """Debug the specific automl_status component test"""
    print("🔍 Debugging automl_status from comprehensive test suite...")
    
    loader = EnhancedComponentLoader()
    component_name = "automl_status"
    module_path = "prompt_improver.tui.widgets.automl_status"
    severity = ComponentTestSeverity.MEDIUM
    
    print(f"Testing: {component_name} at {module_path}")
    
    try:
        report = await loader.comprehensive_component_test(component_name, module_path, severity)
        
        print(f"\n📊 RESULTS:")
        print(f"Test Result: {report.test_result}")
        print(f"Quality Score: {report.metrics.overall_quality_score():.3f}")
        print(f"Security Score: {report.metrics.security_score:.3f}")
        print(f"Performance Score: {report.metrics.performance_score:.3f}")
        print(f"Reliability Score: {report.metrics.reliability_score:.3f}")
        print(f"Load Time: {report.metrics.load_time_ms:.2f}ms")
        print(f"Init Time: {report.metrics.initialization_time_ms:.2f}ms")
        
        if report.error_details:
            print(f"\n❌ ERROR DETAILS:")
            print(report.error_details)
        
        if report.warnings:
            print(f"\n⚠️  WARNINGS:")
            for warning in report.warnings:
                print(f"  - {warning}")
        
        if report.false_positive_indicators:
            print(f"\n🎭 FALSE POSITIVE INDICATORS:")
            for indicator in report.false_positive_indicators:
                print(f"  - {indicator}")
        
        if report.recommended_actions:
            print(f"\n💡 RECOMMENDATIONS:")
            for action in report.recommended_actions:
                print(f"  - {action}")
                
        return report
        
    except Exception as e:
        print(f"❌ Debug test failed: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    result = asyncio.run(debug_automl_status())
    if result:
        print(f"\n🎯 Final Quality Score: {result.metrics.overall_quality_score():.3f}")
    else:
        print("🔴 Test completely failed")