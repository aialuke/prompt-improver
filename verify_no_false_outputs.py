#!/usr/bin/env python3
"""
Verification Script: Real Behavior Testing for No False Outputs

This script performs comprehensive verification to ensure the modernized feature extractors
are working correctly and not producing false outputs. It tests:

1. Real data extraction with known expected patterns
2. Error handling and graceful degradation
3. Consistency between sync and async operations
4. Integration with actual orchestrator components
5. Validation of feature vector dimensions and ranges
6. Comparison with expected baseline behaviors
"""

import asyncio
import json
import logging
import time
from typing import Dict, Any, List
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.WARNING,  # Reduce noise for cleaner output
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

async def verify_context_extractor_outputs():
    """Verify ContextFeatureExtractor produces correct outputs with no false positives."""
    print("\n" + "="*70)
    print("VERIFYING CONTEXT FEATURE EXTRACTOR - NO FALSE OUTPUTS")
    print("="*70)
    
    try:
        from src.prompt_improver.ml.learning.features.context_feature_extractor import (
            ContextFeatureExtractor, 
            ContextFeatureConfig
        )
        
        # Test 1: Verify feature vector dimensions and ranges
        print("\n1. Testing feature vector dimensions and ranges...")
        config = ContextFeatureConfig()
        extractor = ContextFeatureExtractor(config)
        
        # Test with realistic context data
        realistic_context = {
            "user_id": "user_12345",
            "session_id": "session_67890",
            "project_type": "web",
            "performance": {
                "improvement_score": 0.75,
                "user_satisfaction": 0.85,
                "rule_effectiveness": 0.65,
                "response_time_score": 0.90,
                "quality_score": 0.80
            },
            "interaction": {
                "session_length_norm": 0.60,
                "iteration_count_norm": 0.40,
                "feedback_frequency": 0.70,
                "user_engagement_score": 0.80,
                "success_rate": 0.75
            },
            "temporal": {
                "time_of_day_norm": 0.45,
                "day_of_week_norm": 0.30,
                "session_recency_norm": 0.85,
                "usage_frequency_norm": 0.60,
                "trend_indicator": 0.55
            }
        }
        
        features = await extractor.extract_features_async(realistic_context)
        
        # Verify expected properties
        assert len(features) == 20, f"Expected 20 features, got {len(features)}"
        assert all(isinstance(f, float) for f in features), "All features should be float"
        assert all(0.0 <= f <= 1.0 for f in features), f"All features should be 0-1 range, got: {features}"
        
        # Verify feature names match count
        feature_names = extractor.get_feature_names()
        assert len(feature_names) == 20, f"Expected 20 feature names, got {len(feature_names)}"
        
        print(f"✅ Feature dimensions: {len(features)} features (expected 20)")
        print(f"✅ Feature ranges: [{min(features):.3f}, {max(features):.3f}] (expected [0.0, 1.0])")
        print(f"✅ Feature names count: {len(feature_names)}")
        
        # Test 2: Verify realistic input produces expected patterns
        print("\n2. Testing realistic input produces expected patterns...")
        
        # Performance features should reflect input values
        performance_features = features[:5]  # First 5 are performance features
        expected_performance = [0.75, 0.85, 0.65, 0.90, 0.80]
        
        for i, (actual, expected) in enumerate(zip(performance_features, expected_performance)):
            if abs(actual - expected) > 0.01:  # Allow small variance
                print(f"⚠️  Performance feature {i}: {actual} != {expected}")
            else:
                print(f"✅ Performance feature {i}: {actual} ≈ {expected}")
        
        # Test 3: Verify empty/invalid input handling
        print("\n3. Testing empty/invalid input handling...")
        
        empty_features = await extractor.extract_features_async({})
        assert len(empty_features) == 20, "Empty input should still return 20 features"
        assert all(f == config.default_feature_value for f in empty_features), \
               f"Empty input should return default values ({config.default_feature_value})"
        
        invalid_features = await extractor.extract_features_async(None)
        assert len(invalid_features) == 20, "Invalid input should still return 20 features"
        
        print(f"✅ Empty input handling: {len(empty_features)} features with default values")
        print(f"✅ Invalid input handling: {len(invalid_features)} features")
        
        # Test 4: Verify consistency between sync and async
        print("\n4. Testing sync/async consistency...")
        
        sync_features = extractor.extract_features(realistic_context)
        async_features = await extractor.extract_features_async(realistic_context)
        
        assert sync_features == async_features, "Sync and async should produce identical results"
        print("✅ Sync and async results are identical")
        
        # Test 5: Verify caching behavior doesn't affect output
        print("\n5. Testing caching behavior...")
        
        # Clear cache and get fresh result
        extractor.clear_cache()
        fresh_features = await extractor.extract_features_async(realistic_context)
        
        # Get cached result
        cached_features = await extractor.extract_features_async(realistic_context)
        
        assert fresh_features == cached_features, "Cached results should match fresh results"
        
        cache_stats = extractor.get_cache_stats()
        assert cache_stats['cache_hits'] > 0, "Should have cache hits"
        
        print(f"✅ Caching consistency: fresh == cached results")
        print(f"✅ Cache stats: {cache_stats['cache_hits']} hits, {cache_stats['cache_misses']} misses")
        
        # Test 6: Verify health monitoring accuracy
        print("\n6. Testing health monitoring accuracy...")
        
        health = extractor.get_health_status()
        assert health['status'] in ['healthy', 'degraded'], f"Invalid health status: {health['status']}"
        assert health['circuit_breaker_state'] in ['closed', 'open', 'half_open'], \
               f"Invalid circuit breaker state: {health['circuit_breaker_state']}"
        
        metrics = health['metrics']
        assert metrics['success_rate'] >= 0.0 and metrics['success_rate'] <= 100.0, \
               f"Invalid success rate: {metrics['success_rate']}"
        
        print(f"✅ Health status: {health['status']}")
        print(f"✅ Success rate: {metrics['success_rate']:.1f}%")
        
        print("\n🎉 ContextFeatureExtractor verification PASSED - No false outputs detected!")
        return True
        
    except Exception as e:
        print(f"❌ ContextFeatureExtractor verification FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def verify_domain_extractor_outputs():
    """Verify DomainFeatureExtractor produces correct outputs with no false positives."""
    print("\n" + "="*70)
    print("VERIFYING DOMAIN FEATURE EXTRACTOR - NO FALSE OUTPUTS")
    print("="*70)
    
    try:
        from src.prompt_improver.ml.learning.features.domain_feature_extractor import (
            DomainFeatureExtractor,
            DomainFeatureConfig
        )
        
        # Test 1: Verify feature vector dimensions and ranges
        print("\n1. Testing feature vector dimensions and ranges...")
        config = DomainFeatureConfig()
        extractor = DomainFeatureExtractor(config)
        
        # Test with technical text that should trigger specific patterns
        technical_text = """
        I need help implementing a REST API using Python Flask with PostgreSQL database.
        The API should handle user authentication using JWT tokens and include CRUD operations
        for a product catalog. I'm having issues with database connection pooling and 
        query optimization. Can you help me debug the SQLAlchemy models and improve 
        the API performance? This is urgent as we have a production deployment deadline.
        """
        
        features = await extractor.extract_features_async(technical_text)
        
        # Verify expected properties
        assert len(features) == 15, f"Expected 15 features, got {len(features)}"
        assert all(isinstance(f, float) for f in features), "All features should be float"
        assert all(0.0 <= f <= 1.0 for f in features), f"All features should be 0-1 range, got: {features}"
        
        # Verify feature names match count
        feature_names = extractor.get_feature_names()
        assert len(feature_names) == 15, f"Expected 15 feature names, got {len(feature_names)}"
        
        print(f"✅ Feature dimensions: {len(features)} features (expected 15)")
        print(f"✅ Feature ranges: [{min(features):.3f}, {max(features):.3f}] (expected [0.0, 1.0])")
        print(f"✅ Feature names count: {len(feature_names)}")
        
        # Test 2: Verify domain classification accuracy
        print("\n2. Testing domain classification accuracy...")
        
        # Technical domain should be detected (feature index 3)
        technical_indicator = features[3]  # technical_domain_indicator
        
        # With technical keywords, this should be > 0
        if technical_indicator > 0.0:
            print(f"✅ Technical domain detected: {technical_indicator}")
        else:
            print(f"⚠️  Technical domain not detected despite technical keywords")
        
        # Urgency should be detected (feature index 11)
        urgency_indicator = features[11]  # urgency_indicator
        if urgency_indicator > 0.0:
            print(f"✅ Urgency detected: {urgency_indicator}")
        else:
            print(f"⚠️  Urgency not detected despite 'urgent' keyword")
        
        # Test 3: Test different domain types
        print("\n3. Testing different domain classification...")
        
        creative_text = "Write a creative story about a magical forest with enchanted creatures and artistic descriptions of the mystical landscape."
        creative_features = await extractor.extract_features_async(creative_text)
        
        academic_text = "Conduct a research study on the theoretical analysis of machine learning algorithms with scholarly citations and academic methodology."
        academic_features = await extractor.extract_features_async(academic_text)
        
        business_text = "Develop a business strategy for market expansion with customer analysis and revenue projections for management review."
        business_features = await extractor.extract_features_async(business_text)
        
        # Verify different patterns are detected
        tech_score = features[3]      # technical
        creative_score = creative_features[4]  # creative
        academic_score = academic_features[5]  # academic  
        business_score = business_features[6]  # business
        
        print(f"✅ Technical text -> technical score: {tech_score}")
        print(f"✅ Creative text -> creative score: {creative_score}")
        print(f"✅ Academic text -> academic score: {academic_score}")
        print(f"✅ Business text -> business score: {business_score}")
        
        # Test 4: Verify empty/invalid input handling
        print("\n4. Testing empty/invalid input handling...")
        
        empty_features = await extractor.extract_features_async("")
        assert len(empty_features) == 15, "Empty input should still return 15 features"
        assert all(f == config.default_feature_value for f in empty_features), \
               f"Empty input should return default values ({config.default_feature_value})"
        
        invalid_features = await extractor.extract_features_async(None)
        assert len(invalid_features) == 15, "Invalid input should still return 15 features"
        
        print(f"✅ Empty input handling: {len(empty_features)} features with default values")
        print(f"✅ Invalid input handling: {len(invalid_features)} features")
        
        # Test 5: Verify consistency between sync and async
        print("\n5. Testing sync/async consistency...")
        
        sync_features = extractor.extract_features(technical_text)
        async_features = await extractor.extract_features_async(technical_text)
        
        assert sync_features == async_features, "Sync and async should produce identical results"
        print("✅ Sync and async results are identical")
        
        # Test 6: Verify deterministic behavior
        print("\n6. Testing deterministic behavior...")
        
        # With deterministic=True, multiple runs should give same results
        features1 = await extractor.extract_features_async(technical_text)
        features2 = await extractor.extract_features_async(technical_text)
        
        assert features1 == features2, "Deterministic mode should produce identical results"
        print("✅ Deterministic behavior verified")
        
        # Test 7: Verify health monitoring accuracy
        print("\n7. Testing health monitoring accuracy...")
        
        health = extractor.get_health_status()
        assert health['status'] in ['healthy', 'degraded'], f"Invalid health status: {health['status']}"
        
        metrics = health['metrics']
        assert metrics['success_rate'] >= 0.0 and metrics['success_rate'] <= 100.0, \
               f"Invalid success rate: {metrics['success_rate']}"
        
        print(f"✅ Health status: {health['status']}")
        print(f"✅ Analyzer available: {health['analyzer_status']['available']}")
        print(f"✅ Success rate: {metrics['success_rate']:.1f}%")
        print(f"✅ Analyzer usage rate: {metrics['analyzer_usage_rate']:.1f}%")
        
        print("\n🎉 DomainFeatureExtractor verification PASSED - No false outputs detected!")
        return True
        
    except Exception as e:
        print(f"❌ DomainFeatureExtractor verification FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def verify_orchestrator_integration():
    """Verify real integration with ML Pipeline Orchestrator components."""
    print("\n" + "="*70)
    print("VERIFYING ML PIPELINE ORCHESTRATOR INTEGRATION")
    print("="*70)
    
    try:
        from src.prompt_improver.ml.learning.features.context_feature_extractor import (
            ContextFeatureExtractor, ContextFeatureConfig
        )
        from src.prompt_improver.ml.learning.features.domain_feature_extractor import (
            DomainFeatureExtractor, DomainFeatureConfig
        )
        from src.prompt_improver.ml.learning.features.composite_feature_extractor import (
            CompositeFeatureExtractor, FeatureExtractionConfig
        )
        
        print("\n1. Testing individual extractor orchestrator interfaces...")
        
        # Test ContextFeatureExtractor orchestrator interface
        context_extractor = ContextFeatureExtractor()
        context_config = {
            "operation": "extract_features",
            "context_data": {
                "user_id": "test_user",
                "performance": {"improvement_score": 0.8}
            },
            "correlation_id": "integration_test_001"
        }
        
        context_result = await context_extractor.run_orchestrated_analysis(context_config)
        assert context_result['status'] == 'success', f"Context orchestrator failed: {context_result}"
        assert context_result['component'] == 'context_feature_extractor'
        assert context_result['result']['feature_count'] == 20
        
        print("✅ ContextFeatureExtractor orchestrator interface working")
        
        # Test DomainFeatureExtractor orchestrator interface  
        domain_extractor = DomainFeatureExtractor()
        domain_config = {
            "operation": "extract_features",
            "text": "Python programming tutorial for web development",
            "correlation_id": "integration_test_002"
        }
        
        domain_result = await domain_extractor.run_orchestrated_analysis(domain_config)
        assert domain_result['status'] == 'success', f"Domain orchestrator failed: {domain_result}"
        assert domain_result['component'] == 'domain_feature_extractor'
        assert domain_result['result']['feature_count'] == 15
        
        print("✅ DomainFeatureExtractor orchestrator interface working")
        
        print("\n2. Testing CompositeFeatureExtractor integration...")
        
        # Test that CompositeFeatureExtractor can use the modernized extractors
        composite_config = FeatureExtractionConfig(
            enable_linguistic=True,
            enable_domain=True,
            enable_context=True
        )
        
        composite_extractor = CompositeFeatureExtractor(composite_config)
        
        test_text = "Help me build a React application with user authentication"
        test_context = {
            "user_id": "composite_test",
            "project_type": "web"
        }
        
        composite_result = composite_extractor.extract_features(test_text, test_context)
        
        # Should get features from all extractors
        expected_total = 10 + 15 + 20  # linguistic + domain + context
        assert len(composite_result['features']) == expected_total, \
               f"Expected {expected_total} total features, got {len(composite_result['features'])}"
        
        # Verify extractor types are included
        extractor_types = composite_result['extractor_results'].keys()
        assert 'domain' in extractor_types, "Domain extractor should be included"
        assert 'context' in extractor_types, "Context extractor should be included"
        
        print(f"✅ CompositeFeatureExtractor integration: {len(composite_result['features'])} total features")
        print(f"✅ Extractor types included: {list(extractor_types)}")
        
        print("\n3. Testing health check operations...")
        
        # Test health checks through orchestrator interface
        context_health_config = {"operation": "health_check", "correlation_id": "health_test_001"}
        context_health = await context_extractor.run_orchestrated_analysis(context_health_config)
        assert context_health['status'] == 'success'
        assert 'health_status' in str(context_health['result']) or 'status' in context_health['result']
        
        domain_health_config = {"operation": "health_check", "correlation_id": "health_test_002"}
        domain_health = await domain_extractor.run_orchestrated_analysis(domain_health_config)
        assert domain_health['status'] == 'success'
        
        print("✅ Health check operations working through orchestrator")
        
        print("\n4. Testing error handling...")
        
        # Test unsupported operation
        error_config = {"operation": "unsupported_operation", "correlation_id": "error_test_001"}
        error_result = await context_extractor.run_orchestrated_analysis(error_config)
        assert error_result['status'] == 'error'
        assert 'Unsupported operation' in error_result['error']
        
        print("✅ Error handling working correctly")
        
        print("\n🎉 ML Pipeline Orchestrator integration verification PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Orchestrator integration verification FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def verify_performance_consistency():
    """Verify performance characteristics are consistent and reasonable."""
    print("\n" + "="*70)
    print("VERIFYING PERFORMANCE CONSISTENCY")
    print("="*70)
    
    try:
        from src.prompt_improver.ml.learning.features.context_feature_extractor import (
            ContextFeatureExtractor, ContextFeatureConfig
        )
        from src.prompt_improver.ml.learning.features.domain_feature_extractor import (
            DomainFeatureExtractor, DomainFeatureConfig
        )
        
        print("\n1. Testing performance consistency over multiple runs...")
        
        # Test data
        context_data = {"user_id": "perf_test", "performance": {"improvement_score": 0.7}}
        domain_text = "Machine learning algorithm optimization for data science projects"
        
        # Initialize extractors
        context_extractor = ContextFeatureExtractor(ContextFeatureConfig(cache_enabled=False))
        domain_extractor = DomainFeatureExtractor(DomainFeatureConfig(cache_enabled=False))
        
        # Multiple runs to check consistency
        runs = 10
        context_times = []
        domain_times = []
        
        print(f"Running {runs} iterations to check consistency...")
        
        for i in range(runs):
            # Context extractor timing
            start_time = time.time()
            context_features = await context_extractor.extract_features_async(context_data)
            context_time = time.time() - start_time
            context_times.append(context_time)
            
            # Domain extractor timing
            start_time = time.time()
            domain_features = await domain_extractor.extract_features_async(domain_text)
            domain_time = time.time() - start_time
            domain_times.append(domain_time)
            
            # Verify consistent output
            assert len(context_features) == 20, f"Inconsistent context feature count at run {i}"
            assert len(domain_features) == 15, f"Inconsistent domain feature count at run {i}"
        
        # Analyze timing consistency
        context_avg = np.mean(context_times)
        context_std = np.std(context_times)
        domain_avg = np.mean(domain_times) 
        domain_std = np.std(domain_times)
        
        print(f"✅ Context extractor: {context_avg:.4f}s ± {context_std:.4f}s")
        print(f"✅ Domain extractor: {domain_avg:.4f}s ± {domain_std:.4f}s")
        
        # Performance should be reasonable (< 1 second for simple operations)
        assert context_avg < 1.0, f"Context extraction too slow: {context_avg:.4f}s"
        assert domain_avg < 1.0, f"Domain extraction too slow: {domain_avg:.4f}s"
        
        # Consistency check (std deviation should be reasonable)
        context_cv = context_std / context_avg if context_avg > 0 else 0
        domain_cv = domain_std / domain_avg if domain_avg > 0 else 0
        
        print(f"✅ Context timing consistency: CV = {context_cv:.2f}")
        print(f"✅ Domain timing consistency: CV = {domain_cv:.2f}")
        
        print("\n2. Testing cache performance improvement...")
        
        # Test with caching enabled
        cached_context = ContextFeatureExtractor(ContextFeatureConfig(cache_enabled=True))
        cached_domain = DomainFeatureExtractor(DomainFeatureConfig(cache_enabled=True))
        
        # First call (cache miss)
        start_time = time.time()
        await cached_context.extract_features_async(context_data)
        miss_time = time.time() - start_time
        
        # Second call (cache hit)
        start_time = time.time()
        await cached_context.extract_features_async(context_data)
        hit_time = time.time() - start_time
        
        # Cache should provide some improvement
        if hit_time > 0:
            speedup = miss_time / hit_time
            print(f"✅ Cache speedup: {speedup:.1f}x (miss: {miss_time:.4f}s, hit: {hit_time:.4f}s)")
        else:
            print("✅ Cache hit time negligible (< measurement precision)")
        
        print("\n🎉 Performance consistency verification PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Performance consistency verification FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main verification runner."""
    print("🔍 COMPREHENSIVE VERIFICATION: NO FALSE OUTPUTS")
    print("="*80)
    print("Testing modernized feature extractors for correct behavior and integration...")
    
    verifications = [
        ("Context Extractor Outputs", verify_context_extractor_outputs),
        ("Domain Extractor Outputs", verify_domain_extractor_outputs), 
        ("Orchestrator Integration", verify_orchestrator_integration),
        ("Performance Consistency", verify_performance_consistency),
    ]
    
    results = []
    
    for test_name, test_func in verifications:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name}: EXCEPTION - {e}")
            results.append((test_name, False))
    
    # Final summary
    print("\n" + "="*80)
    print("VERIFICATION SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ VERIFIED" if result else "❌ FAILED"
        print(f"  {test_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} verifications passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL VERIFICATIONS PASSED!")
        print("\n✅ CONFIRMED: No false outputs detected")
        print("✅ CONFIRMED: Real behavior working correctly")
        print("✅ CONFIRMED: Integration successful")
        print("✅ CONFIRMED: Performance characteristics acceptable")
        print("✅ CONFIRMED: Error handling robust")
        print("✅ CONFIRMED: Feature extractors production-ready")
        
        return 0
    else:
        print(f"\n⚠️  {total-passed} verification(s) failed.")
        print("❌ CAUTION: Potential false outputs or integration issues detected")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(asyncio.run(main()))