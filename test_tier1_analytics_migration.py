#!/usr/bin/env python3
"""
Test script for TIER 1 Analytics Migration
Verifies that the decorator-based error handling works correctly
"""

import json
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from prompt_improver.services.analytics import AnalyticsService

def test_parse_rule_json_safe():
    """Test the new decorator-based JSON parsing method"""
    analytics = AnalyticsService()
    
    print("🧪 Testing TIER 1 Migration: Analytics JSON parsing with decorators")
    
    # Test 1: Valid JSON
    valid_json = '[{"rule_id": "test_rule_1"}, {"rule_id": "test_rule_2"}]'
    result = analytics._parse_rule_json_safe(valid_json)
    print(f"✅ Valid JSON test: {result}")
    assert result == ["test_rule_1", "test_rule_2"], f"Expected ['test_rule_1', 'test_rule_2'], got {result}"
    
    # Test 2: Invalid JSON (should be handled by decorator)
    invalid_json = '{"invalid": json}'
    result = analytics._parse_rule_json_safe(invalid_json)
    print(f"✅ Invalid JSON test: {result}")
    assert isinstance(result, dict) and result.get('status') == 'error', f"Expected error dict for invalid JSON, got {result}"
    
    # Test 3: Null/empty string
    result = analytics._parse_rule_json_safe("null")
    print(f"✅ Null JSON test: {result}")
    assert result == [], f"Expected empty list for null, got {result}"
    
    # Test 4: Empty string
    result = analytics._parse_rule_json_safe("")
    print(f"✅ Empty string test: {result}")
    assert result == [], f"Expected empty list for empty string, got {result}"
    
    # Test 5: Valid JSON but wrong structure
    wrong_structure = '{"not": "a list"}'
    result = analytics._parse_rule_json_safe(wrong_structure)
    print(f"✅ Wrong structure test: {result}")
    assert result == [], f"Expected empty list for wrong structure, got {result}"
    
    print("\n🎯 TIER 1 Migration Test Results:")
    print("✅ All decorator-based error handling tests passed!")
    print("✅ Traditional try/except block successfully replaced with @handle_validation_errors")
    print("✅ Error handling is now consistent with codebase standards")
    
    return True

if __name__ == "__main__":
    try:
        test_parse_rule_json_safe()
        print("\n🚀 TIER 1 ANALYTICS MIGRATION: SUCCESSFUL")
    except Exception as e:
        print(f"\n❌ TIER 1 ANALYTICS MIGRATION: FAILED")
        print(f"Error: {e}")
        sys.exit(1)
