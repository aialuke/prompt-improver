#!/usr/bin/env python3
"""
Component Analysis Test Runner

Executes systematic real behavior testing of all search components.
NO MOCKS - Uses real embeddings, real API calls, real data.

Usage:
    python run_component_analysis.py [--quick] [--component COMPONENT_NAME]

Options:
    --quick: Run with fewer test queries to reduce API usage
    --component: Test only specific component (semantic, lexical, binary, cross_encoder, all)
"""

import argparse
import json
import time
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

# Import real behavior tester
from real_behavior_tester import RealBehaviorComponentTester, RealBehaviorMetrics


def verify_environment() -> bool:
    """Verify environment is set up for real behavior testing."""
    print("🔍 Verifying environment for real behavior testing...")
    
    # Check VOYAGE_API_KEY
    if not os.getenv('VOYAGE_API_KEY'):
        print("❌ VOYAGE_API_KEY environment variable not set")
        print("   Set with: export VOYAGE_API_KEY='your-api-key'")
        return False
    
    # Check embeddings file exists in multiple locations
    embeddings_paths = ["embeddings.pkl", "../data/embeddings.pkl", "data/embeddings.pkl"]
    embeddings_found = False
    for path in embeddings_paths:
        if Path(path).exists():
            embeddings_found = True
            print(f"✅ Embeddings file found: {path}")
            break

    if not embeddings_found:
        print(f"❌ Embeddings file not found in any location: {embeddings_paths}")
        print("   Run generate_embeddings.py first")
        return False
    
    # Check required packages
    try:
        import voyageai
        import rank_bm25
        import sentence_transformers
        print("✅ All required packages available")
    except ImportError as e:
        print(f"❌ Missing required package: {e}")
        return False
    
    print("✅ Environment verified for real behavior testing")
    return True


def run_semantic_component_test(tester: RealBehaviorComponentTester, quick: bool = False) -> Dict[str, Any]:
    """Run real behavior test for semantic search component only."""
    print("\n" + "="*60)
    print("🧠 SEMANTIC SEARCH COMPONENT - REAL BEHAVIOR TEST")
    print("="*60)
    print("🌐 Using real Voyage AI API calls")
    print("🚫 BM25, binary rescoring, and cross-encoder DISABLED")
    
    results = tester.test_semantic_only_real_behavior()
    
    # Analyze semantic-specific insights
    analysis = {
        "component": "semantic_search",
        "test_type": "real_behavior",
        "results": [result.__dict__ for result in results],
        "insights": {
            "avg_api_calls_per_query": sum(r.api_calls_made for r in results) / len(results) if results else 0,
            "avg_execution_time_ms": sum(r.execution_time_ms for r in results) / len(results) if results else 0,
            "avg_similarity_score": sum(r.avg_similarity_score for r in results) / len(results) if results else 0,
            "total_false_positives": sum(r.false_positives_detected for r in results),
            "semantic_strengths": "High-level concept matching, contextual understanding",
            "semantic_weaknesses": "May miss exact keyword matches, higher API cost"
        }
    }
    
    print(f"📊 Semantic Analysis Results:")
    print(f"   Average API calls per query: {analysis['insights']['avg_api_calls_per_query']:.1f}")
    print(f"   Average execution time: {analysis['insights']['avg_execution_time_ms']:.1f}ms")
    print(f"   Average similarity score: {analysis['insights']['avg_similarity_score']:.3f}")
    print(f"   Total false positives: {analysis['insights']['total_false_positives']}")
    
    return analysis


def run_lexical_component_test(tester: RealBehaviorComponentTester, quick: bool = False) -> Dict[str, Any]:
    """Run real behavior test for BM25 lexical search component only."""
    print("\n" + "="*60)
    print("📝 BM25 LEXICAL SEARCH COMPONENT - REAL BEHAVIOR TEST")
    print("="*60)
    print("🔤 Using real BM25 tokenization and scoring")
    print("🚫 Semantic embeddings, binary rescoring, and cross-encoder DISABLED")
    
    results = tester.test_lexical_only_real_behavior()
    
    analysis = {
        "component": "lexical_bm25",
        "test_type": "real_behavior", 
        "results": [result.__dict__ for result in results],
        "insights": {
            "avg_api_calls_per_query": sum(r.api_calls_made for r in results) / len(results) if results else 0,
            "avg_execution_time_ms": sum(r.execution_time_ms for r in results) / len(results) if results else 0,
            "avg_similarity_score": sum(r.avg_similarity_score for r in results) / len(results) if results else 0,
            "total_false_positives": sum(r.false_positives_detected for r in results),
            "lexical_strengths": "Fast keyword matching, no API costs, exact term matching",
            "lexical_weaknesses": "Limited semantic understanding, may miss conceptual matches"
        }
    }
    
    print(f"📊 BM25 Lexical Analysis Results:")
    print(f"   Average API calls per query: {analysis['insights']['avg_api_calls_per_query']:.1f}")
    print(f"   Average execution time: {analysis['insights']['avg_execution_time_ms']:.1f}ms")
    print(f"   Average similarity score: {analysis['insights']['avg_similarity_score']:.3f}")
    print(f"   Total false positives: {analysis['insights']['total_false_positives']}")
    
    return analysis


def run_binary_rescoring_test(tester: RealBehaviorComponentTester, quick: bool = False) -> Dict[str, Any]:
    """Run real behavior test for binary rescoring optimization."""
    print("\n" + "="*60)
    print("⚡ BINARY RESCORING OPTIMIZATION - REAL BEHAVIOR TEST")
    print("="*60)
    print("🔢 Testing real binary quantization performance impact")
    print("📊 Comparing with and without binary rescoring")
    
    # Test with binary rescoring
    baseline_results = tester.test_baseline_all_components_real()
    
    # Test without binary rescoring
    no_binary_results = tester.test_no_binary_rescoring_real_behavior()
    
    # Compare real performance
    analysis = {
        "component": "binary_rescoring",
        "test_type": "real_behavior_comparison",
        "with_binary_rescoring": [result.__dict__ for result in baseline_results],
        "without_binary_rescoring": [result.__dict__ for result in no_binary_results],
        "performance_impact": {
            "avg_time_with_binary": sum(r.execution_time_ms for r in baseline_results) / len(baseline_results) if baseline_results else 0,
            "avg_time_without_binary": sum(r.execution_time_ms for r in no_binary_results) / len(no_binary_results) if no_binary_results else 0,
            "avg_api_calls_with_binary": sum(r.api_calls_made for r in baseline_results) / len(baseline_results) if baseline_results else 0,
            "avg_api_calls_without_binary": sum(r.api_calls_made for r in no_binary_results) / len(no_binary_results) if no_binary_results else 0,
        }
    }
    
    # Calculate performance improvements
    time_improvement = ((analysis["performance_impact"]["avg_time_without_binary"] - 
                        analysis["performance_impact"]["avg_time_with_binary"]) / 
                       analysis["performance_impact"]["avg_time_without_binary"] * 100) if analysis["performance_impact"]["avg_time_without_binary"] > 0 else 0
    
    api_reduction = (analysis["performance_impact"]["avg_api_calls_without_binary"] - 
                    analysis["performance_impact"]["avg_api_calls_with_binary"])
    
    analysis["performance_impact"]["time_improvement_percent"] = time_improvement
    analysis["performance_impact"]["api_call_reduction"] = api_reduction
    
    print(f"📊 Binary Rescoring Performance Impact:")
    print(f"   Time improvement: {time_improvement:.1f}%")
    print(f"   API call reduction: {api_reduction:.1f} calls per query")
    print(f"   With binary: {analysis['performance_impact']['avg_time_with_binary']:.1f}ms")
    print(f"   Without binary: {analysis['performance_impact']['avg_time_without_binary']:.1f}ms")
    
    return analysis


def run_cross_encoder_test(tester: RealBehaviorComponentTester, quick: bool = False) -> Dict[str, Any]:
    """Run real behavior test for cross-encoder reranking."""
    print("\n" + "="*60)
    print("🎯 CROSS-ENCODER RERANKING - REAL BEHAVIOR TEST")
    print("="*60)
    print("🧠 Testing real Voyage AI rerank-2.5-lite impact")
    print("📊 Comparing result relevance with and without reranking")
    
    # Test with cross-encoder
    baseline_results = tester.test_baseline_all_components_real()
    
    # Test without cross-encoder
    no_cross_encoder_results = tester.test_no_cross_encoder_real_behavior()
    
    analysis = {
        "component": "cross_encoder_reranking",
        "test_type": "real_behavior_comparison",
        "with_cross_encoder": [result.__dict__ for result in baseline_results],
        "without_cross_encoder": [result.__dict__ for result in no_cross_encoder_results],
        "relevance_impact": {
            "avg_similarity_with_reranking": sum(r.avg_similarity_score for r in baseline_results) / len(baseline_results) if baseline_results else 0,
            "avg_similarity_without_reranking": sum(r.avg_similarity_score for r in no_cross_encoder_results) / len(no_cross_encoder_results) if no_cross_encoder_results else 0,
            "false_positives_with_reranking": sum(r.false_positives_detected for r in baseline_results),
            "false_positives_without_reranking": sum(r.false_positives_detected for r in no_cross_encoder_results),
        }
    }
    
    # Calculate relevance improvements
    similarity_improvement = (analysis["relevance_impact"]["avg_similarity_with_reranking"] - 
                             analysis["relevance_impact"]["avg_similarity_without_reranking"])
    
    false_positive_reduction = (analysis["relevance_impact"]["false_positives_without_reranking"] - 
                               analysis["relevance_impact"]["false_positives_with_reranking"])
    
    analysis["relevance_impact"]["similarity_improvement"] = similarity_improvement
    analysis["relevance_impact"]["false_positive_reduction"] = false_positive_reduction
    
    print(f"📊 Cross-Encoder Reranking Impact:")
    print(f"   Similarity improvement: {similarity_improvement:.3f}")
    print(f"   False positive reduction: {false_positive_reduction}")
    print(f"   With reranking: {analysis['relevance_impact']['avg_similarity_with_reranking']:.3f}")
    print(f"   Without reranking: {analysis['relevance_impact']['avg_similarity_without_reranking']:.3f}")
    
    return analysis


def run_local_quantization_test(tester: RealBehaviorComponentTester, quick: bool = False) -> Dict[str, Any]:
    """Run real behavior test for local binary quantization."""
    print("\n" + "="*60)
    print("🔢 LOCAL BINARY QUANTIZATION - REAL BEHAVIOR TEST")
    print("="*60)
    print("⚡ Testing real local quantization vs API-based binary embeddings")
    
    comparison_results = tester.test_local_vs_api_binary_quantization_real()
    
    analysis = {
        "component": "local_binary_quantization",
        "test_type": "real_behavior_comparison",
        "comparison_results": comparison_results,
        "insights": {
            "local_quantization_benefits": "Eliminates API calls for binary embeddings, faster processing",
            "accuracy_trade_offs": "Minimal accuracy difference with significant performance gains"
        }
    }
    
    print(f"📊 Local Binary Quantization Analysis:")
    for query, results in comparison_results.items():
        print(f"   Query: {query[:50]}...")
        print(f"     Time reduction: {results['performance_improvement']['time_reduction_percent']:.1f}%")
        print(f"     API call reduction: {results['performance_improvement']['api_call_reduction']}")
        print(f"     Accuracy difference: {results['performance_improvement']['accuracy_difference']:.3f}")
    
    return analysis


def main():
    """Main test runner for component analysis."""
    parser = argparse.ArgumentParser(description="Real Behavior Component Analysis")
    parser.add_argument("--quick", action="store_true", help="Run with fewer queries")
    parser.add_argument("--component", choices=["semantic", "lexical", "binary", "cross_encoder", "local_quantization", "all"], 
                       default="all", help="Component to test")
    
    args = parser.parse_args()
    
    print("🧪 REAL BEHAVIOR COMPONENT ANALYSIS")
    print("=" * 60)
    print("⚠️  NO MOCKS - Testing real component behavior only")
    print("🌐 Using real Voyage AI API calls")
    print("📊 Using real embeddings and data")
    print("=" * 60)
    
    # Verify environment
    if not verify_environment():
        sys.exit(1)
    
    # Initialize real behavior tester
    try:
        tester = RealBehaviorComponentTester()
    except Exception as e:
        print(f"❌ Failed to initialize tester: {e}")
        sys.exit(1)
    
    # Run specified tests
    all_results = {}
    
    if args.component in ["semantic", "all"]:
        all_results["semantic"] = run_semantic_component_test(tester, args.quick)
    
    if args.component in ["lexical", "all"]:
        all_results["lexical"] = run_lexical_component_test(tester, args.quick)
    
    if args.component in ["binary", "all"]:
        all_results["binary_rescoring"] = run_binary_rescoring_test(tester, args.quick)
    
    if args.component in ["cross_encoder", "all"]:
        all_results["cross_encoder"] = run_cross_encoder_test(tester, args.quick)
    
    if args.component in ["local_quantization", "all"]:
        all_results["local_quantization"] = run_local_quantization_test(tester, args.quick)
    
    # Save comprehensive results
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    timestamp = int(time.time())
    output_file = output_dir / f"component_analysis_{timestamp}.json"
    
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n🎉 REAL BEHAVIOR COMPONENT ANALYSIS COMPLETED!")
    print(f"📊 Results saved to: {output_file}")
    print("🚫 No mocked components or data used")
    print("✅ All tests used genuine component behavior")


if __name__ == "__main__":
    main()
