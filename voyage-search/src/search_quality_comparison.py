#!/usr/bin/env python3
"""
Search Quality Comparison: Voyage AI vs Simple Tokenization
Tests whether Voyage AI tokenization actually improves search results.
"""

import os
import time
import pickle
import numpy as np
from typing import List, Dict, Any, Tuple
from dotenv import load_dotenv
from pathlib import Path

# Load environment variables
load_dotenv('../.env')
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Import search components
from search_code import HybridCodeSearch, HYBRID_SEARCH_CONFIG

def test_search_quality_comparison():
    """Compare search quality with Voyage AI vs simple tokenization."""
    
    print("🔍 SEARCH QUALITY COMPARISON: Voyage AI vs Simple Tokenization")
    print("=" * 70)
    
    # Load embeddings
    embeddings_file = "embeddings.pkl"
    if not Path(embeddings_file).exists():
        print("❌ Embeddings file not found. Please run generate_embeddings.py first.")
        return
    
    with open(embeddings_file, 'rb') as f:
        embeddings_data = pickle.load(f)
    
    chunks = embeddings_data['chunks']
    embeddings = embeddings_data['embeddings']
    binary_embeddings = embeddings_data.get('binary_embeddings')
    
    print(f"✅ Loaded {len(chunks)} chunks for testing")
    
    # Test queries that should find relevant results in the codebase
    test_queries = [
        "search code implementation",
        "embedding generation",
        "binary quantization",
        "voyage AI client",
        "BM25 scoring",
        "hybrid search",
        "tokenization methods",
        "similarity calculation"
    ]
    
    results_comparison = []
    
    for query in test_queries:
        print(f"\n🔍 Testing query: '{query}'")
        
        # Test with Voyage AI tokenization
        print("   Testing with Voyage AI tokenization...")
        start_time = time.time()
        
        # Temporarily enable Voyage tokenization
        original_setting = HYBRID_SEARCH_CONFIG["use_voyage_tokenization"]
        HYBRID_SEARCH_CONFIG["use_voyage_tokenization"] = True
        
        try:
            search_voyage = HybridCodeSearch(chunks, embeddings, binary_embeddings)
            results_voyage = search_voyage.hybrid_search(query, top_k=5, min_similarity=0.1)
            voyage_time = (time.time() - start_time) * 1000
            voyage_api_calls = search_voyage.get_search_metrics()["api_calls"]
        except Exception as e:
            print(f"      ❌ Voyage search failed: {e}")
            results_voyage = []
            voyage_time = 0
            voyage_api_calls = 0
        
        # Test with simple tokenization
        print("   Testing with simple tokenization...")
        start_time = time.time()
        
        # Disable Voyage tokenization
        HYBRID_SEARCH_CONFIG["use_voyage_tokenization"] = False
        
        try:
            search_simple = HybridCodeSearch(chunks, embeddings, binary_embeddings)
            results_simple = search_simple.hybrid_search(query, top_k=5, min_similarity=0.1)
            simple_time = (time.time() - start_time) * 1000
            simple_api_calls = search_simple.get_search_metrics()["api_calls"]
        except Exception as e:
            print(f"      ❌ Simple search failed: {e}")
            results_simple = []
            simple_time = 0
            simple_api_calls = 0
        
        # Restore original setting
        HYBRID_SEARCH_CONFIG["use_voyage_tokenization"] = original_setting
        
        # Compare results
        voyage_count = len(results_voyage)
        simple_count = len(results_simple)
        
        # Calculate result overlap
        voyage_files = set(r.file_path for r in results_voyage)
        simple_files = set(r.file_path for r in results_simple)
        common_files = voyage_files & simple_files
        overlap_ratio = len(common_files) / max(len(voyage_files | simple_files), 1)
        
        # Calculate average similarities
        voyage_avg_sim = np.mean([r.similarity_score for r in results_voyage]) if results_voyage else 0
        simple_avg_sim = np.mean([r.similarity_score for r in results_simple]) if results_simple else 0
        
        print(f"      Voyage AI: {voyage_count} results, avg sim: {voyage_avg_sim:.3f}, {voyage_time:.1f}ms, {voyage_api_calls} API calls")
        print(f"      Simple:    {simple_count} results, avg sim: {simple_avg_sim:.3f}, {simple_time:.1f}ms, {simple_api_calls} API calls")
        print(f"      Overlap:   {len(common_files)}/{len(voyage_files | simple_files)} files ({overlap_ratio:.1%})")
        
        # Analyze result quality
        quality_difference = "Similar"
        if voyage_avg_sim > simple_avg_sim + 0.05:
            quality_difference = "Voyage Better"
        elif simple_avg_sim > voyage_avg_sim + 0.05:
            quality_difference = "Simple Better"
        
        results_comparison.append({
            'query': query,
            'voyage_count': voyage_count,
            'simple_count': simple_count,
            'voyage_avg_sim': voyage_avg_sim,
            'simple_avg_sim': simple_avg_sim,
            'voyage_time': voyage_time,
            'simple_time': simple_time,
            'voyage_api_calls': voyage_api_calls,
            'simple_api_calls': simple_api_calls,
            'overlap_ratio': overlap_ratio,
            'quality_difference': quality_difference
        })
    
    # Summary analysis
    print(f"\n📊 SUMMARY ANALYSIS")
    print("=" * 70)
    
    total_voyage_time = sum(r['voyage_time'] for r in results_comparison)
    total_simple_time = sum(r['simple_time'] for r in results_comparison)
    total_voyage_api_calls = sum(r['voyage_api_calls'] for r in results_comparison)
    total_simple_api_calls = sum(r['simple_api_calls'] for r in results_comparison)
    
    avg_overlap = np.mean([r['overlap_ratio'] for r in results_comparison])
    
    quality_better_voyage = sum(1 for r in results_comparison if r['quality_difference'] == 'Voyage Better')
    quality_better_simple = sum(1 for r in results_comparison if r['quality_difference'] == 'Simple Better')
    quality_similar = sum(1 for r in results_comparison if r['quality_difference'] == 'Similar')
    
    print(f"Total queries tested: {len(test_queries)}")
    print(f"Average result overlap: {avg_overlap:.1%}")
    print(f"Quality comparison:")
    print(f"  - Voyage AI better: {quality_better_voyage}/{len(test_queries)}")
    print(f"  - Simple better: {quality_better_simple}/{len(test_queries)}")
    print(f"  - Similar quality: {quality_similar}/{len(test_queries)}")
    
    print(f"\nPerformance comparison:")
    print(f"  - Total Voyage time: {total_voyage_time:.1f}ms")
    print(f"  - Total Simple time: {total_simple_time:.1f}ms")
    print(f"  - Speed difference: {total_voyage_time/max(total_simple_time, 1):.1f}x slower")
    
    print(f"\nAPI call comparison:")
    print(f"  - Total Voyage API calls: {total_voyage_api_calls}")
    print(f"  - Total Simple API calls: {total_simple_api_calls}")
    print(f"  - API call difference: +{total_voyage_api_calls - total_simple_api_calls}")
    
    # Final recommendation
    print(f"\n🏆 FINAL RECOMMENDATION")
    print("=" * 70)
    
    if avg_overlap > 0.8 and quality_similar >= quality_better_voyage:
        print("❌ Voyage AI tokenization provides MINIMAL search quality benefit")
        print(f"   - {avg_overlap:.1%} result overlap (very similar results)")
        print(f"   - {total_voyage_time/max(total_simple_time, 1):.1f}x slower")
        print(f"   - +{total_voyage_api_calls - total_simple_api_calls} extra API calls")
        print("   - Recommendation: Use simple tokenization for BM25")
    elif quality_better_voyage > quality_better_simple:
        print("✅ Voyage AI tokenization provides SIGNIFICANT search quality benefit")
        print(f"   - Better quality in {quality_better_voyage}/{len(test_queries)} queries")
        print(f"   - Worth the performance cost for better results")
    else:
        print("⚠️  Voyage AI tokenization provides MIXED results")
        print(f"   - Consider use case specific testing")
    
    return results_comparison

if __name__ == "__main__":
    try:
        results = test_search_quality_comparison()
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        import traceback
        traceback.print_exc()
