#!/usr/bin/env python3
"""
Tokenization Analysis: Voyage AI vs Simple Tokenization
Analyzes the actual benefits and costs of using Voyage AI tokenization for BM25.
"""

import os
import time
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv('../.env')

# Configure tokenizers to prevent warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def simple_tokenize(text: str) -> List[str]:
    """Simple tokenization: lowercase and split on whitespace."""
    return text.lower().split()

def voyage_tokenize(text: str) -> List[str]:
    """Voyage AI tokenization."""
    try:
        import voyageai
        api_key = os.getenv('VOYAGE_API_KEY')
        if not api_key:
            raise ValueError("VOYAGE_API_KEY not set")
        
        vo = voyageai.Client(api_key=api_key)
        result = vo.tokenize([text], model="voyage-context-3")
        return list(result[0].tokens)
    except Exception as e:
        print(f"Voyage tokenization failed: {e}")
        return simple_tokenize(text)

def analyze_tokenization_differences():
    """Analyze differences between Voyage AI and simple tokenization."""
    
    # Test cases representing typical code content
    test_cases = [
        # Python code
        "def binary_search(arr, target): return arr.index(target)",
        
        # Function with underscores
        "async def process_user_data(user_id, session_token):",
        
        # Camel case
        "class DatabaseConnectionManager:",
        
        # Mixed case with symbols
        "HTTP_STATUS_CODES = {'200': 'OK', '404': 'Not Found'}",
        
        # Comments and docstrings
        '"""This function implements a quicksort algorithm for sorting arrays."""',
        
        # Complex code with punctuation
        "result = await db.execute(query='SELECT * FROM users WHERE id = ?', params=[user_id])",
        
        # Natural language query
        "sorting algorithm implementation",
        
        # Technical terms
        "machine learning model training optimization",
    ]
    
    print("🔍 TOKENIZATION ANALYSIS: Voyage AI vs Simple")
    print("=" * 60)
    
    total_voyage_time = 0
    total_simple_time = 0
    api_calls = 0
    
    differences = []
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n📝 Test Case {i}: {text[:50]}...")
        
        # Simple tokenization
        start_time = time.time()
        simple_tokens = simple_tokenize(text)
        simple_time = (time.time() - start_time) * 1000
        total_simple_time += simple_time
        
        # Voyage tokenization
        start_time = time.time()
        voyage_tokens = voyage_tokenize(text)
        voyage_time = (time.time() - start_time) * 1000
        total_voyage_time += voyage_time
        api_calls += 1
        
        # Compare results
        simple_count = len(simple_tokens)
        voyage_count = len(voyage_tokens)
        
        # Find differences
        simple_set = set(simple_tokens)
        voyage_set = set(voyage_tokens)
        
        only_simple = simple_set - voyage_set
        only_voyage = voyage_set - simple_set
        common = simple_set & voyage_set
        
        print(f"   Simple:  {simple_count} tokens in {simple_time:.1f}ms")
        print(f"   Voyage:  {voyage_count} tokens in {voyage_time:.1f}ms")
        print(f"   Common:  {len(common)} tokens")
        
        if only_simple:
            print(f"   Only Simple: {list(only_simple)[:5]}")
        if only_voyage:
            print(f"   Only Voyage: {list(only_voyage)[:5]}")
        
        # Calculate similarity
        if simple_count > 0 and voyage_count > 0:
            jaccard_similarity = len(common) / len(simple_set | voyage_set)
            print(f"   Similarity: {jaccard_similarity:.2%}")
        else:
            jaccard_similarity = 0.0
        
        differences.append({
            'text': text,
            'simple_tokens': simple_count,
            'voyage_tokens': voyage_count,
            'simple_time_ms': simple_time,
            'voyage_time_ms': voyage_time,
            'similarity': jaccard_similarity,
            'only_simple': list(only_simple),
            'only_voyage': list(only_voyage)
        })
    
    # Summary analysis
    print(f"\n📊 SUMMARY ANALYSIS")
    print("=" * 60)
    print(f"Total test cases: {len(test_cases)}")
    print(f"API calls made: {api_calls}")
    print(f"Total simple time: {total_simple_time:.1f}ms")
    print(f"Total voyage time: {total_voyage_time:.1f}ms")
    print(f"Speed difference: {(total_voyage_time/total_simple_time):.1f}x slower")
    
    # Calculate average similarity
    avg_similarity = sum(d['similarity'] for d in differences) / len(differences)
    print(f"Average token similarity: {avg_similarity:.2%}")
    
    # Analyze token count differences
    token_diffs = [d['voyage_tokens'] - d['simple_tokens'] for d in differences]
    avg_token_diff = sum(token_diffs) / len(token_diffs)
    print(f"Average token count difference: {avg_token_diff:+.1f}")
    
    # Cost analysis
    print(f"\n💰 COST ANALYSIS")
    print("=" * 60)
    print(f"API calls per BM25 initialization: ~{len(test_cases)} (for {len(test_cases)} chunks)")
    print(f"API calls per query: 1 (for query tokenization)")
    print(f"Time overhead per API call: ~{total_voyage_time/api_calls:.1f}ms")
    
    # Benefits analysis
    print(f"\n🎯 BENEFITS ANALYSIS")
    print("=" * 60)
    
    # Check for meaningful differences
    meaningful_diffs = [d for d in differences if d['similarity'] < 0.9]
    print(f"Cases with <90% similarity: {len(meaningful_diffs)}/{len(differences)}")
    
    if meaningful_diffs:
        print("Significant differences found in:")
        for diff in meaningful_diffs:
            print(f"  - {diff['text'][:40]}... (similarity: {diff['similarity']:.2%})")
    
    # Recommendation
    print(f"\n🏆 RECOMMENDATION")
    print("=" * 60)
    
    if avg_similarity > 0.95:
        print("❌ Voyage AI tokenization provides MINIMAL benefit")
        print(f"   - {avg_similarity:.1%} token similarity with simple tokenization")
        print(f"   - {total_voyage_time/total_simple_time:.1f}x slower")
        print(f"   - Adds {api_calls} API calls per search")
        print("   - Recommendation: Use simple tokenization")
    elif avg_similarity > 0.85:
        print("⚠️  Voyage AI tokenization provides MODERATE benefit")
        print(f"   - {avg_similarity:.1%} token similarity")
        print(f"   - Consider cost vs benefit for your use case")
    else:
        print("✅ Voyage AI tokenization provides SIGNIFICANT benefit")
        print(f"   - Only {avg_similarity:.1%} token similarity")
        print(f"   - Meaningful tokenization differences found")
    
    return differences

if __name__ == "__main__":
    try:
        differences = analyze_tokenization_differences()
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
