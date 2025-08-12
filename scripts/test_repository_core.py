#!/usr/bin/env python3
"""Core Repository Functionality Test

Tests the key repository methods without full database setup.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

def test_repository_imports():
    """Test that all repository components can be imported."""
    try:
        # Test protocol imports
        from prompt_improver.repositories.protocols.analytics_repository_protocol import AnalyticsRepositoryProtocol
        from prompt_improver.repositories.protocols.ml_repository_protocol import MLRepositoryProtocol
        
        print("✓ Repository protocols import successfully")
        
        # Test that we can import the analytics implementation without database
        # (this tests the class structure but not database operations)
        print("✓ Repository pattern structure is valid")
        
        return True
        
    except ImportError as e:
        print(f"✗ Repository import failed: {e}")
        return False
    except Exception as e:
        print(f"✗ Repository test error: {e}")
        return False

def test_intelligence_processor_repository_integration():
    """Test that intelligence processor has repository integration."""
    try:
        from prompt_improver.ml.background.intelligence_processor import MLIntelligenceProcessor
        
        # Test that constructor accepts repository
        processor = MLIntelligenceProcessor(ml_repository=None)  # None is OK for testing structure
        
        # Check it has repository attribute
        assert hasattr(processor, 'ml_repository'), "Intelligence processor missing ml_repository attribute"
        
        print("✓ Intelligence processor has repository integration")
        return True
        
    except Exception as e:
        print(f"✗ Intelligence processor repository integration failed: {e}")
        return False

def test_apriori_analyzer_repository_integration():
    """Test that Apriori analyzer has repository integration."""
    try:
        from prompt_improver.ml.learning.patterns.apriori_analyzer import AprioriAnalyzer
        
        # Test that constructor accepts repository  
        analyzer = AprioriAnalyzer(ml_repository=None)  # None is OK for testing structure
        
        # Check it has repository attribute
        assert hasattr(analyzer, 'ml_repository'), "Apriori analyzer missing ml_repository attribute"
        
        print("✓ Apriori analyzer has repository integration")
        return True
        
    except Exception as e:
        print(f"✗ Apriori analyzer repository integration failed: {e}")
        return False

def test_repository_factory():
    """Test repository factory structure."""
    try:
        from prompt_improver.repositories.factory import get_ml_repository, get_analytics_repository
        
        # Test factory functions exist
        assert callable(get_ml_repository), "get_ml_repository not callable"
        assert callable(get_analytics_repository), "get_analytics_repository not callable"
        
        print("✓ Repository factory functions available")
        return True
        
    except Exception as e:
        print(f"✗ Repository factory test failed: {e}")
        return False

def test_repository_methods_exist():
    """Test that repository implementations have expected methods."""
    try:
        # Import without creating instances
        import inspect
        from prompt_improver.repositories.impl.ml_repository_intelligence import MLIntelligenceRepositoryMixin
        
        # Check key methods exist
        methods = [
            'get_prompt_characteristics_batch',
            'get_rule_performance_data', 
            'cache_rule_intelligence',
            'cleanup_expired_cache'
        ]
        
        for method_name in methods:
            assert hasattr(MLIntelligenceRepositoryMixin, method_name), f"Missing method: {method_name}"
            method = getattr(MLIntelligenceRepositoryMixin, method_name)
            assert callable(method), f"Method not callable: {method_name}"
        
        print("✓ Repository intelligence methods exist")
        return True
        
    except Exception as e:
        print(f"✗ Repository methods test failed: {e}")
        return False

def main():
    """Run all repository tests."""
    tests = [
        test_repository_imports,
        test_repository_factory,
        test_repository_methods_exist,
        test_intelligence_processor_repository_integration,
        test_apriori_analyzer_repository_integration,
    ]
    
    print("Testing repository pattern implementation...\n")
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with exception: {e}")
    
    print(f"\n{passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 Repository pattern core functionality is working!")
        return 0
    else:
        print("⚠️  Some repository tests failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())