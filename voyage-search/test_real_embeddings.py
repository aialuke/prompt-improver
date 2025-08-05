#!/usr/bin/env python3
"""
Quick test with real embeddings to verify BM25 optimizations.
"""

import sys
import pickle
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from search_code import HybridCodeSearch
    from bm25_tokenizers import TokenizationManager
    import numpy as np
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


def test_real_embeddings():
    """Test BM25 optimizations with real codebase embeddings."""
    print("🔍 Testing BM25 Optimizations with Real Codebase Data")
    print("=" * 60)
    
    # Load real embeddings
    embeddings_path = Path("src/embeddings.pkl")
    if not embeddings_path.exists():
        print(f"❌ Embeddings not found: {embeddings_path}")
        return
    
    print(f"📂 Loading embeddings from {embeddings_path}")
    
    try:
        # Load without pickle class dependencies
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
        
        embeddings = data['embeddings']
        chunks = data['chunks']
        
        print(f"   ✅ Loaded {len(chunks)} code chunks")
        print(f"   📊 Embeddings shape: {embeddings.shape}")
        
        # Initialize search system
        search_system = HybridCodeSearch(
            embeddings=embeddings,
            chunks=chunks,
            metadata={"test": True}
        )
        
        # Test queries
        test_queries = [
            "calculate similarity",
            "BM25Tokenizer", 
            "TokenizationManager",
            "HybridCodeSearch",
            "generateEmbeddings"  # camelCase test
        ]
        
        print(f"\n🧪 Testing {len(test_queries)} queries:")
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n📝 Query {i}: '{query}'")
            
            try:
                results = search_system.hybrid_search(query, top_k=3)
                print(f"   ✅ Found {len(results)} results")
                
                # Show top result
                if results:
                    top_result = results[0]
                    print(f"   🥇 Top result: {top_result.chunk_name}")
                    print(f"      File: {top_result.file_path}")
                    print(f"      Type: {top_result.chunk_type}")
                
            except Exception as e:
                print(f"   ❌ Search failed: {e}")
        
        # Test tokenization comparison
        print(f"\n🔧 Tokenization Comparison:")
        tokenization_manager = TokenizationManager()
        
        test_tokens = [
            "calculateSimilarity",
            "BM25Tokenizer", 
            "remove_stopwords",
            "the best algorithm for search"
        ]
        
        for token_text in test_tokens:
            try:
                bm25_tokens = tokenization_manager.tokenize_for_bm25(token_text)
                print(f"   '{token_text}' → {bm25_tokens}")
            except Exception as e:
                print(f"   ❌ Tokenization failed for '{token_text}': {e}")
        
        print(f"\n✅ Real embeddings test completed successfully!")
        
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        print(f"   Error type: {type(e).__name__}")


if __name__ == "__main__":
    test_real_embeddings()
