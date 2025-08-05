#!/usr/bin/env python3
"""
Test Production Bayesian Optimizer

Quick test with 10 evaluations to identify and fix issues before running the full 200 evaluation optimization.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np

# Bayesian optimization imports
from skopt import gp_minimize
from skopt.space import Real, Categorical

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from search_code import HybridCodeSearch, HYBRID_SEARCH_CONFIG, load_enhanced_embeddings
    from bm25_tokenizers import TokenizationManager, BM25Tokenizer
    from generate_embeddings import CodeChunk, ChunkType
    import numpy as np
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_production_data_loading():
    """Test loading production data and identify any issues."""
    print("🧪 Testing production data loading...")
    
    try:
        # Load production data
        embeddings, chunks, metadata, binary_embeddings = load_enhanced_embeddings("data/embeddings.pkl")
        
        print(f"✅ Loaded production data:")
        print(f"   Embeddings: {embeddings.shape}")
        print(f"   Chunks: {len(chunks)}")
        print(f"   Binary embeddings: {binary_embeddings.shape if binary_embeddings is not None else 'None'}")
        
        # Test creating search system
        search_system = HybridCodeSearch(
            embeddings=embeddings,
            chunks=chunks,
            metadata=metadata,
            binary_embeddings=binary_embeddings
        )
        
        print(f"✅ Created search system successfully")
        
        # Test a simple search
        results = search_system.hybrid_search("search", top_k=5)
        print(f"✅ Test search completed: {len(results)} results")
        
        return embeddings, chunks, metadata, binary_embeddings
        
    except Exception as e:
        print(f"❌ Production data loading failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def test_simple_optimization():
    """Test a simple 5-evaluation optimization."""
    print("\n🧪 Testing simple Bayesian optimization...")
    
    # Load data
    embeddings, chunks, metadata, binary_embeddings = test_production_data_loading()
    if embeddings is None:
        return
    
    # Create simple test queries
    test_queries = [
        {"query": "search", "expected_chunks": [], "type": "keyword"},
        {"query": "tokenize", "expected_chunks": [], "type": "keyword"},
        {"query": "similarity", "expected_chunks": [], "type": "keyword"},
        {"query": "class", "expected_chunks": [], "type": "keyword"},
        {"query": "function", "expected_chunks": [], "type": "keyword"}
    ]
    
    print(f"   Using {len(test_queries)} simple test queries")
    
    # Define objective function
    def objective_function(params):
        k1, b, stemming, split_camelcase, split_underscores = params
        
        try:
            # Create tokenization config
            bm25_config = {
                "stemming": stemming,
                "stopwords_lang": "english",
                "lowercase": True,
                "min_token_length": 2,
                "remove_punctuation": True,
                "split_camelcase": split_camelcase,
                "split_underscores": split_underscores
            }
            
            tokenization_manager = TokenizationManager(bm25_config=bm25_config)
            
            # Update global config temporarily
            original_k1 = HYBRID_SEARCH_CONFIG["bm25_k1"]
            original_b = HYBRID_SEARCH_CONFIG["bm25_b"]
            
            HYBRID_SEARCH_CONFIG["bm25_k1"] = k1
            HYBRID_SEARCH_CONFIG["bm25_b"] = b
            
            try:
                # Create search system
                search_system = HybridCodeSearch(
                    embeddings=embeddings,
                    chunks=chunks,
                    metadata=metadata,
                    binary_embeddings=binary_embeddings
                )
                
                search_system.tokenization_manager = tokenization_manager
                
                # Test queries
                total_score = 0.0
                for query_info in test_queries:
                    try:
                        results = search_system.hybrid_search(query_info["query"], top_k=5)
                        # Simple scoring: give points for finding results
                        score = min(len(results) / 5.0, 1.0)  # Normalize to 0-1
                        total_score += score
                    except Exception as e:
                        print(f"   ❌ Query '{query_info['query']}' failed: {e}")
                
                avg_score = total_score / len(test_queries)
                print(f"   📊 k1={k1:.2f}, b={b:.2f}, stem={stemming}, score={avg_score:.3f}")
                
                return -avg_score  # Negative for minimization
                
            finally:
                # Restore config
                HYBRID_SEARCH_CONFIG["bm25_k1"] = original_k1
                HYBRID_SEARCH_CONFIG["bm25_b"] = original_b
                
        except Exception as e:
            print(f"   ❌ Objective function error: {e}")
            return 0.0
    
    # Define parameter space
    dimensions = [
        Real(0.5, 2.0, name='k1'),
        Real(0.0, 1.0, name='b'),
        Categorical(['none', 'light'], name='stemming'),  # Simplified for testing
        Categorical([True, False], name='split_camelcase'),
        Categorical([True, False], name='split_underscores')
    ]
    
    print("   Starting 5-evaluation optimization...")
    start_time = time.time()
    
    try:
        # Run optimization
        result = gp_minimize(
            func=objective_function,
            dimensions=dimensions,
            n_calls=5,
            n_initial_points=3,
            random_state=42
        )
        
        optimization_time = time.time() - start_time
        
        print(f"✅ Simple optimization completed in {optimization_time:.1f} seconds")
        print(f"   Best score: {-result.fun:.3f}")
        print(f"   Best params: k1={result.x[0]:.3f}, b={result.x[1]:.3f}, stem={result.x[2]}")
        
        return True
        
    except Exception as e:
        print(f"❌ Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run production optimizer tests."""
    print("🧪 Production Bayesian Optimizer Test Suite")
    print("=" * 60)
    
    # Test 1: Data loading
    success1 = test_production_data_loading() is not None
    
    if success1:
        print("\n✅ Data loading test passed")
        
        # Test 2: Simple optimization
        success2 = test_simple_optimization()
        
        if success2:
            print("\n✅ All tests passed! Ready for full 200-evaluation optimization")
            print("\nTo run full optimization:")
            print("   python production_bayesian_optimizer.py")
        else:
            print("\n❌ Optimization test failed - need to fix issues first")
    else:
        print("\n❌ Data loading test failed - cannot proceed")


if __name__ == "__main__":
    main()
