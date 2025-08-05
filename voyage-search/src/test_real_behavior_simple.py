#!/usr/bin/env python3
"""
Simple Real Behavior Test

Quick test to verify our real behavior testing framework works correctly.
Tests actual components without mocks.
"""

import os
import sys
from pathlib import Path

def test_binary_quantization():
    """Test real binary quantization implementation."""
    print("🔢 Testing real binary quantization...")
    
    try:
        from search_code import local_binary_quantization
        
        # Test with known values from Voyage AI documentation
        test_embedding = [-0.03955078, 0.006214142, -0.07446289, -0.039001465,
                         0.0046463013, 0.00030612946, -0.08496094, 0.03994751]
        
        result = local_binary_quantization(test_embedding)
        expected = -51
        
        if result[0] == expected:
            print(f"   ✅ Binary quantization works: {result[0]} == {expected}")
            return True
        else:
            print(f"   ❌ Binary quantization failed: {result[0]} != {expected}")
            return False
            
    except Exception as e:
        print(f"   ❌ Binary quantization error: {e}")
        return False


def test_embeddings_loading():
    """Test real embeddings loading."""
    print("📊 Testing real embeddings loading...")
    
    try:
        from search_code import load_enhanced_embeddings
        
        # Find embeddings file
        possible_paths = ["embeddings.pkl", "../data/embeddings.pkl", "data/embeddings.pkl"]
        embeddings_file = None
        for path in possible_paths:
            if Path(path).exists():
                embeddings_file = path
                break
        
        if not embeddings_file:
            print("   ❌ No embeddings file found")
            return False
        
        embeddings, chunks, metadata, binary_embeddings = load_enhanced_embeddings(embeddings_file)
        
        print(f"   ✅ Loaded {len(chunks)} real code chunks")
        print(f"   ✅ Embeddings shape: {embeddings.shape}")
        if binary_embeddings is not None:
            print(f"   ✅ Binary embeddings shape: {binary_embeddings.shape}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Embeddings loading error: {e}")
        return False


def test_environment():
    """Test environment setup."""
    print("🔍 Testing environment setup...")
    
    checks = []
    
    # Check API key
    if os.getenv('VOYAGE_API_KEY'):
        print("   ✅ VOYAGE_API_KEY is set")
        checks.append(True)
    else:
        print("   ⚠️  VOYAGE_API_KEY not set (needed for API tests)")
        checks.append(False)
    
    # Check packages
    try:
        import voyageai
        print("   ✅ voyageai package available")
        checks.append(True)
    except ImportError:
        print("   ❌ voyageai package not available")
        checks.append(False)
    
    try:
        import rank_bm25
        print("   ✅ rank_bm25 package available")
        checks.append(True)
    except ImportError:
        print("   ❌ rank_bm25 package not available")
        checks.append(False)
    
    return all(checks)


def test_search_instance_creation():
    """Test creating real search instance."""
    print("🔍 Testing real search instance creation...")
    
    try:
        from search_code import HybridCodeSearch, load_enhanced_embeddings
        
        # Find embeddings file
        possible_paths = ["embeddings.pkl", "../data/embeddings.pkl", "data/embeddings.pkl"]
        embeddings_file = None
        for path in possible_paths:
            if Path(path).exists():
                embeddings_file = path
                break
        
        if not embeddings_file:
            print("   ❌ No embeddings file found")
            return False
        
        # Load real data
        embeddings, chunks, metadata, binary_embeddings = load_enhanced_embeddings(embeddings_file)
        
        # Create real search instance
        search = HybridCodeSearch(embeddings, chunks, metadata, binary_embeddings)
        
        print(f"   ✅ Search instance created successfully")
        print(f"   ✅ BM25 available: {search.bm25 is not None}")
        print(f"   ✅ Cross-encoder available: {search.cross_encoder is not None}")
        print(f"   ✅ Binary embeddings available: {search.binary_embeddings is not None}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Search instance creation error: {e}")
        return False


def main():
    """Run simple real behavior tests."""
    print("🧪 SIMPLE REAL BEHAVIOR TESTING")
    print("=" * 50)
    print("🎯 Testing actual components without mocks")
    print("=" * 50)
    
    tests = [
        ("Environment Setup", test_environment),
        ("Binary Quantization", test_binary_quantization),
        ("Embeddings Loading", test_embeddings_loading),
        ("Search Instance Creation", test_search_instance_creation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"   ❌ Test failed with exception: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    passed = sum(results)
    total = len(results)
    
    if passed == total:
        print(f"✅ ALL TESTS PASSED ({passed}/{total})")
        print("🎉 Real behavior testing framework is working!")
        return True
    else:
        print(f"❌ SOME TESTS FAILED ({passed}/{total})")
        print("⚠️  Check the failed tests above")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
