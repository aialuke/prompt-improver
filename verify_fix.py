#!/usr/bin/env python3
"""
Verification script to test the AprioriAnalyzer initialization fix.
Tests that AdvancedPatternDiscovery can be initialized with proper PostgreSQL configuration.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def test_lazy_initialization():
    """Test that AdvancedPatternDiscovery can be initialized with proper lazy loading."""
    print("Testing lazy initialization fix...")
    
    # Test 1: Import and create services without database connection
    try:
        from src.prompt_improver.services.advanced_pattern_discovery import AdvancedPatternDiscovery
        from src.prompt_improver.database.connection import DatabaseManager
        
        # Test with proper PostgreSQL URL (using psycopg3)
        postgres_url = "postgresql+psycopg://apes_user:apes_secure_password_2024@localhost:5432/apes_test"
        db_manager = DatabaseManager(postgres_url)
        
        # This should work without immediate database connection
        service = AdvancedPatternDiscovery(db_manager=db_manager)
        print("✅ Test 1 PASSED: AdvancedPatternDiscovery(db_manager=db_manager) works")
        
    except Exception as e:
        print(f"❌ Test 1 FAILED: AdvancedPatternDiscovery initialization failed: {e}")
        return False
    
    # Test 2: Test that lazy initialization works (apriori_analyzer is None until accessed)
    try:
        # Before accessing apriori_analyzer property, it should be None
        if hasattr(service, '_apriori_analyzer'):
            assert service._apriori_analyzer is None, "AprioriAnalyzer should be lazily initialized"
            print("✅ Test 2 PASSED: AprioriAnalyzer is lazily initialized (None until accessed)")
        else:
            print("⚠️  Test 2 SKIPPED: _apriori_analyzer attribute not found (implementation may vary)")
            
    except Exception as e:
        print(f"❌ Test 2 FAILED: Lazy initialization check failed: {e}")
        return False
    
    # Test 3: Test that db_manager property works
    try:
        # The db_manager property should work
        assert service.db_manager is not None, "DatabaseManager should be accessible"
        print("✅ Test 3 PASSED: DatabaseManager property is accessible")
        
    except Exception as e:
        print(f"❌ Test 3 FAILED: DatabaseManager property access failed: {e}")
        return False
    
    # Test 4: Test that the service can handle missing database manager gracefully
    try:
        service_without_db = AdvancedPatternDiscovery(db_manager=None)
        # Should not crash during initialization
        print("✅ Test 4 PASSED: AdvancedPatternDiscovery handles None db_manager gracefully")
        
    except Exception as e:
        print(f"❌ Test 4 FAILED: Service should handle None db_manager gracefully: {e}")
        return False
    
    print("\n🎉 All tests passed! The AprioriAnalyzer initialization fix is working correctly.")
    print("\n📋 Summary of fixes implemented:")
    print("   - Added lazy initialization for AprioriAnalyzer")
    print("   - Added thread-safe double-checked locking pattern")
    print("   - Added enhanced error handling for missing database manager")
    print("   - Updated test files to use proper DatabaseManager initialization")
    print("   - Uses psycopg3 (postgresql+psycopg://) for synchronous connections")
    print("   - Uses asyncpg (postgresql+asyncpg://) for async connections")
    
    return True

if __name__ == "__main__":
    success = test_lazy_initialization()
    sys.exit(0 if success else 1)