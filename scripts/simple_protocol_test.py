#!/usr/bin/env python3
"""
Simple Cache Protocol Compliance Test

Basic test to validate protocol inheritance without triggering
complex import chains.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_protocol_inheritance():
    """Test that UnifiedConnectionManager inherits from cache protocols."""
    print("🔍 Testing Cache Protocol Inheritance...")
    print("")
    
    try:
        # Import protocols
        from prompt_improver.core.protocols.cache_protocol import (
            BasicCacheProtocol, AdvancedCacheProtocol, CacheHealthProtocol,
            CacheSubscriptionProtocol, CacheLockProtocol, RedisCacheProtocol,
            MultiLevelCacheProtocol
        )
        
        print("✅ Successfully imported cache protocols")
        
        # Import UnifiedConnectionManager directly
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "unified_connection_manager", 
            Path(__file__).parent.parent / "src" / "prompt_improver" / "database" / "unified_connection_manager.py"
        )
        module = importlib.util.module_from_spec(spec)
        
        # Mock dependencies to avoid circular imports
        sys.modules['prompt_improver.performance.monitoring.health.background_manager'] = type(sys)('mock')
        sys.modules['prompt_improver.performance.monitoring.health.background_manager'].get_background_task_manager = lambda: None
        sys.modules['prompt_improver.performance.monitoring.health.background_manager'].TaskPriority = type('TaskPriority', (), {
            'CRITICAL': 'critical',
            'HIGH': 'high', 
            'NORMAL': 'normal',
            'LOW': 'low',
            'BACKGROUND': 'background'
        })
        
        spec.loader.exec_module(module)
        UnifiedConnectionManager = module.UnifiedConnectionManager
        
        print("✅ Successfully imported UnifiedConnectionManager")
        print("")
        
        # Test protocol inheritance
        protocols_to_test = [
            ("BasicCacheProtocol", BasicCacheProtocol),
            ("AdvancedCacheProtocol", AdvancedCacheProtocol),
            ("CacheHealthProtocol", CacheHealthProtocol),
            ("CacheSubscriptionProtocol", CacheSubscriptionProtocol),
            ("CacheLockProtocol", CacheLockProtocol),
            ("RedisCacheProtocol", RedisCacheProtocol),
            ("MultiLevelCacheProtocol", MultiLevelCacheProtocol),
        ]
        
        compliance_results = {}
        
        print("📊 Testing Protocol Inheritance Compliance:")
        for protocol_name, protocol_class in protocols_to_test:
            try:
                is_compliant = issubclass(UnifiedConnectionManager, protocol_class)
                compliance_results[protocol_name] = is_compliant
                status = "✅" if is_compliant else "❌"
                print(f"   {status} {protocol_name}: {'COMPLIANT' if is_compliant else 'NON-COMPLIANT'}")
            except Exception as e:
                print(f"   ❌ {protocol_name}: ERROR - {e}")
                compliance_results[protocol_name] = False
        
        print("")
        
        # Test method existence
        print("📝 Testing Method Existence:")
        
        required_methods = [
            # BasicCacheProtocol
            'get', 'set', 'delete', 'exists', 'clear',
            # AdvancedCacheProtocol
            'get_many', 'set_many', 'delete_many', 'get_or_set', 'increment', 'expire',
            # CacheHealthProtocol
            'ping', 'get_info', 'get_stats', 'get_memory_usage',
            # CacheSubscriptionProtocol
            'publish', 'subscribe', 'unsubscribe',
            # CacheLockProtocol
            'acquire_lock', 'release_lock', 'extend_lock',
            # MultiLevelCacheProtocol
            'get_from_level', 'set_to_level', 'invalidate_levels', 'get_cache_hierarchy'
        ]
        
        method_compliance = {}
        for method_name in required_methods:
            has_method = hasattr(UnifiedConnectionManager, method_name)
            method_compliance[method_name] = has_method
            status = "✅" if has_method else "❌"
            print(f"   {status} {method_name}: {'EXISTS' if has_method else 'MISSING'}")
        
        print("")
        
        # Summary
        total_protocols = len(compliance_results)
        compliant_protocols = sum(compliance_results.values())
        protocol_compliance_rate = (compliant_protocols / total_protocols) * 100
        
        total_methods = len(method_compliance)
        compliant_methods = sum(method_compliance.values())
        method_compliance_rate = (compliant_methods / total_methods) * 100
        
        print(f"📈 COMPLIANCE SUMMARY:")
        print(f"   • Protocol Inheritance: {compliant_protocols}/{total_protocols} ({protocol_compliance_rate:.1f}%)")
        print(f"   • Method Implementation: {compliant_methods}/{total_methods} ({method_compliance_rate:.1f}%)")
        print(f"   • Overall Compliance: {(protocol_compliance_rate + method_compliance_rate) / 2:.1f}%")
        print("")
        
        if protocol_compliance_rate >= 95.0 and method_compliance_rate >= 95.0:
            print("🎉 VALIDATION PASSED: Cache protocol compliance meets requirements!")
            return 0
        else:
            print("❌ VALIDATION FAILED: Cache protocol compliance issues found.")
            return 1
            
    except Exception as e:
        print(f"💥 VALIDATION ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = test_protocol_inheritance()
    sys.exit(exit_code)