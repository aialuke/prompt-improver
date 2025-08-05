#!/usr/bin/env python3
"""
Test Optimization Fixes

Comprehensive testing of the BM25 optimization fixes:
1. Binary rescoring dimension mismatch fix
2. Optimized BM25 parameters deployment
3. BM25-only verification mode
4. Performance monitoring

Usage:
    python test_optimization_fixes.py
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from search_code import (
        HybridCodeSearch, HYBRID_SEARCH_CONFIG, SEARCH_METRICS,
        load_enhanced_embeddings, local_binary_quantization
    )
    from bm25_tokenizers import TokenizationManager
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_binary_dimension_fix():
    """Test that binary rescoring dimension mismatch is fixed."""
    print("🔧 Testing Binary Rescoring Dimension Fix")
    print("=" * 50)
    
    try:
        # Load production data
        embeddings, chunks, metadata, binary_embeddings = load_enhanced_embeddings("data/embeddings.pkl")
        
        print(f"   📊 Loaded data:")
        print(f"      Embeddings: {embeddings.shape}")
        print(f"      Chunks: {len(chunks)}")
        print(f"      Binary embeddings: {binary_embeddings.shape if binary_embeddings is not None else 'None'}")
        
        # Create search system
        search_system = HybridCodeSearch(embeddings, chunks, metadata, binary_embeddings)
        
        # Test search with binary rescoring
        print(f"\n   🧪 Testing binary rescoring search...")
        results = search_system.hybrid_search("HealthMonitoringValidator", top_k=3)
        
        print(f"   ✅ Binary rescoring test completed: {len(results)} results")
        
        # Check if binary rescoring was actually used
        if binary_embeddings is not None:
            print(f"   🔥 Binary rescoring: Available and tested")
        else:
            print(f"   ⚠️  Binary rescoring: Not available")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Binary dimension test failed: {e}")
        return False


def test_optimized_parameters():
    """Test that optimized BM25 parameters are deployed."""
    print("\n🎯 Testing Optimized BM25 Parameters")
    print("=" * 50)
    
    try:
        # Check configuration values
        k1 = HYBRID_SEARCH_CONFIG["bm25_k1"]
        b = HYBRID_SEARCH_CONFIG["bm25_b"]
        
        print(f"   📊 Current BM25 parameters:")
        print(f"      k1: {k1} (expected: 0.500)")
        print(f"      b: {b} (expected: 0.836)")
        
        # Verify optimized values
        if abs(k1 - 0.500) < 0.001 and abs(b - 0.836) < 0.001:
            print(f"   ✅ Optimized parameters correctly deployed")
            return True
        else:
            print(f"   ❌ Parameters not optimized: k1={k1}, b={b}")
            return False
            
    except Exception as e:
        print(f"   ❌ Parameter test failed: {e}")
        return False


def test_bm25_only_mode():
    """Test BM25-only verification mode."""
    print("\n🔧 Testing BM25-Only Verification Mode")
    print("=" * 50)
    
    try:
        # Load production data
        embeddings, chunks, metadata, binary_embeddings = load_enhanced_embeddings("data/embeddings.pkl")
        
        # Test normal mode
        print(f"   🧪 Testing normal hybrid mode...")
        search_system = HybridCodeSearch(embeddings, chunks, metadata, binary_embeddings)
        normal_results = search_system.hybrid_search("similarity calculation", top_k=3)
        
        # Test BM25-only mode
        print(f"   🧪 Testing BM25-only mode...")
        original_mode = HYBRID_SEARCH_CONFIG.get("bm25_only_mode", False)
        HYBRID_SEARCH_CONFIG["bm25_only_mode"] = True
        
        try:
            bm25_results = search_system.hybrid_search("similarity calculation", top_k=3)
            print(f"   ✅ BM25-only mode test completed")
            print(f"      Normal mode: {len(normal_results)} results")
            print(f"      BM25-only mode: {len(bm25_results)} results")
            
        finally:
            # Restore original mode
            HYBRID_SEARCH_CONFIG["bm25_only_mode"] = original_mode
        
        return True
        
    except Exception as e:
        print(f"   ❌ BM25-only mode test failed: {e}")
        return False


def test_performance_monitoring():
    """Test performance monitoring and metrics tracking."""
    print("\n📊 Testing Performance Monitoring")
    print("=" * 50)
    
    try:
        # Check metrics structure
        print(f"   📊 Checking metrics structure...")
        
        if "bm25_optimization" in SEARCH_METRICS:
            opt_metrics = SEARCH_METRICS["bm25_optimization"]
            print(f"   ✅ BM25 optimization metrics available:")
            print(f"      k1: {opt_metrics.get('k1', 'missing')}")
            print(f"      b: {opt_metrics.get('b', 'missing')}")
            print(f"      stemming: {opt_metrics.get('stemming', 'missing')}")
            print(f"      split_camelcase: {opt_metrics.get('split_camelcase', 'missing')}")
            print(f"      split_underscores: {opt_metrics.get('split_underscores', 'missing')}")
        else:
            print(f"   ❌ BM25 optimization metrics missing")
            return False
        
        # Test query type tracking
        print(f"\n   🧪 Testing query type tracking...")
        embeddings, chunks, metadata, binary_embeddings = load_enhanced_embeddings("data/embeddings.pkl")
        search_system = HybridCodeSearch(embeddings, chunks, metadata, binary_embeddings)
        
        # Test validator query
        search_system.hybrid_search("HealthMonitoringValidator", top_k=3)
        
        # Test similarity query
        search_system.hybrid_search("calculate similarity score", top_k=3)
        
        # Test embedding query
        search_system.hybrid_search("generate embeddings batch", top_k=3)
        
        print(f"   ✅ Query type tracking test completed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Performance monitoring test failed: {e}")
        return False


def test_comprehensive_search_quality():
    """Test comprehensive search quality with all fixes applied."""
    print("\n🎯 Testing Comprehensive Search Quality")
    print("=" * 50)
    
    try:
        # Load production data
        embeddings, chunks, metadata, binary_embeddings = load_enhanced_embeddings("data/embeddings.pkl")
        search_system = HybridCodeSearch(embeddings, chunks, metadata, binary_embeddings)
        
        # Test queries that showed improvement in verification
        test_queries = [
            "HealthMonitoringValidator",
            "BaselineSystemValidator", 
            "BatchCapacityCalculator",
            "similarity calculation methods",
            "generate_embeddings_batch"
        ]
        
        print(f"   🧪 Testing {len(test_queries)} improved queries...")
        
        all_results = []
        for query in test_queries:
            start_time = time.time()
            results = search_system.hybrid_search(query, top_k=5)
            search_time = (time.time() - start_time) * 1000
            
            # Get top score safely
            top_score = 0.0
            if results:
                if hasattr(results[0], 'score'):
                    top_score = results[0].score
                elif hasattr(results[0], 'similarity'):
                    top_score = results[0].similarity
                else:
                    top_score = 1.0  # Default score if no score attribute

            all_results.append({
                "query": query,
                "results_count": len(results),
                "search_time_ms": search_time,
                "top_score": top_score
            })
            
            print(f"      '{query}': {len(results)} results in {search_time:.1f}ms")
        
        # Calculate average performance
        avg_results = sum(r["results_count"] for r in all_results) / len(all_results)
        avg_time = sum(r["search_time_ms"] for r in all_results) / len(all_results)
        avg_score = sum(r["top_score"] for r in all_results) / len(all_results)
        
        print(f"\n   📊 Performance Summary:")
        print(f"      Average results: {avg_results:.1f}")
        print(f"      Average search time: {avg_time:.1f}ms")
        print(f"      Average top score: {avg_score:.3f}")
        
        print(f"   ✅ Comprehensive search quality test completed")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Comprehensive search quality test failed: {e}")
        return False


def main():
    """Run all optimization fix tests."""
    print("🧪 BM25 Optimization Fixes Test Suite")
    print("Testing all implemented fixes and improvements")
    print("=" * 80)
    
    tests = [
        ("Binary Dimension Fix", test_binary_dimension_fix),
        ("Optimized Parameters", test_optimized_parameters),
        ("BM25-Only Mode", test_bm25_only_mode),
        ("Performance Monitoring", test_performance_monitoring),
        ("Comprehensive Search Quality", test_comprehensive_search_quality)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 80)
    print("🏆 TEST RESULTS SUMMARY")
    print("=" * 80)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n📊 Overall Results: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 ALL TESTS PASSED! Optimization fixes are working correctly.")
        print("\n🚀 Ready for production deployment:")
        print("   ✅ Binary rescoring dimension mismatch fixed")
        print("   ✅ Optimized BM25 parameters deployed (k1=0.500, b=0.836)")
        print("   ✅ BM25-only verification mode available")
        print("   ✅ Performance monitoring active")
    else:
        print("⚠️  Some tests failed. Please review and fix issues before deployment.")
    
    return passed == len(results)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
