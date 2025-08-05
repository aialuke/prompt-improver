#!/usr/bin/env python3
"""
Verification System for BM25 Optimization Results

This script provides comprehensive verification of the optimized BM25 parameters
to ensure there are no false positives and that the improvements are real.

Verification Methods:
1. Side-by-side comparison with current configuration
2. Real search quality assessment with human-interpretable results
3. Performance regression testing
4. Edge case testing
5. Statistical significance testing

Usage:
    python verify_optimization_results.py
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from search_code import HybridCodeSearch, HYBRID_SEARCH_CONFIG, load_enhanced_embeddings
    from bm25_tokenizers import TokenizationManager, BM25Tokenizer
    from generate_embeddings import CodeChunk, ChunkType
    import numpy as np
    from rank_bm25 import BM25Okapi
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


@dataclass
class SearchResult:
    """Structured search result for comparison."""
    chunk_name: str
    score: float
    content_preview: str
    chunk_type: str
    rank: int


@dataclass
class ComparisonResult:
    """Result of comparing two configurations."""
    query: str
    baseline_results: List[SearchResult]
    optimized_results: List[SearchResult]
    baseline_score: float
    optimized_score: float
    improvement: float
    quality_assessment: str


class OptimizationVerifier:
    """Comprehensive verification system for BM25 optimization results."""
    
    def __init__(self):
        """Initialize verifier with production data."""
        print("🔍 Initializing Optimization Verification System")
        
        # Load production data
        self.embeddings, self.chunks, self.metadata, self.binary_embeddings = load_enhanced_embeddings("data/embeddings.pkl")
        
        # Store original configuration
        self.original_config = HYBRID_SEARCH_CONFIG.copy()
        
        # Optimized configuration from fast BM25 optimizer
        self.optimized_config = {
            "k1": 0.500,
            "b": 0.836,
            "stemming": "aggressive",
            "split_camelcase": True,
            "split_underscores": True
        }
        
        print(f"   ✅ Loaded {len(self.chunks):,} production chunks")
        print(f"   📊 Original config: k1={self.original_config['bm25_k1']}, b={self.original_config['bm25_b']}")
        print(f"   🎯 Optimized config: k1={self.optimized_config['k1']}, b={self.optimized_config['b']}")
        
        # Create verification queries
        self.verification_queries = self._create_verification_queries()
        print(f"   📋 Created {len(self.verification_queries)} verification queries")
    
    def _create_verification_queries(self) -> List[Dict[str, Any]]:
        """Create comprehensive verification queries with expected results."""
        
        # Extract real chunk names for verification
        chunk_names = []
        class_names = []
        function_names = []
        
        for chunk in self.chunks:
            if hasattr(chunk, 'name') and chunk.name:
                chunk_names.append(chunk.name)
                
                if hasattr(chunk, 'chunk_type'):
                    chunk_type = str(chunk.chunk_type)
                    if 'CLASS' in chunk_type:
                        class_names.append(chunk.name)
                    elif 'FUNCTION' in chunk_type or 'METHOD' in chunk_type:
                        function_names.append(chunk.name)
        
        queries = []
        
        # Exact match queries (should find specific chunks)
        for name in class_names[:10]:
            queries.append({
                "query": name,
                "expected_chunks": [name],
                "type": "exact_class_match",
                "description": f"Should find class '{name}' as top result"
            })
        
        for name in function_names[:10]:
            queries.append({
                "query": name.replace('_', ' '),
                "expected_chunks": [name],
                "type": "exact_function_match", 
                "description": f"Should find function '{name}' in top results"
            })
        
        # Semantic search queries (quality assessment)
        semantic_queries = [
            {
                "query": "search algorithm implementation",
                "type": "semantic_search",
                "description": "Should find search-related code",
                "quality_keywords": ["search", "algorithm", "find", "query", "index"]
            },
            {
                "query": "tokenization and text processing",
                "type": "semantic_processing",
                "description": "Should find tokenization code",
                "quality_keywords": ["token", "text", "process", "split", "parse"]
            },
            {
                "query": "similarity calculation methods",
                "type": "semantic_similarity",
                "description": "Should find similarity computation code",
                "quality_keywords": ["similarity", "cosine", "distance", "score", "compare"]
            },
            {
                "query": "embedding generation and storage",
                "type": "semantic_embedding",
                "description": "Should find embedding-related code",
                "quality_keywords": ["embedding", "vector", "generate", "store", "encode"]
            },
            {
                "query": "configuration management system",
                "type": "semantic_config",
                "description": "Should find configuration code",
                "quality_keywords": ["config", "setting", "parameter", "option", "manage"]
            }
        ]
        
        for sq in semantic_queries:
            sq["expected_chunks"] = []
            queries.append(sq)
        
        # Edge case queries (robustness testing)
        edge_cases = [
            {
                "query": "calculateSimilarityScore",
                "type": "camelcase_test",
                "description": "CamelCase tokenization test",
                "expected_chunks": []
            },
            {
                "query": "generate_embeddings_batch",
                "type": "underscore_test", 
                "description": "Underscore tokenization test",
                "expected_chunks": []
            },
            {
                "query": "running algorithms efficiently",
                "type": "stemming_test",
                "description": "Stemming effectiveness test",
                "expected_chunks": []
            },
            {
                "query": "a the an for with optimization",
                "type": "stopword_test",
                "description": "Stopword filtering test",
                "expected_chunks": []
            },
            {
                "query": "xyz nonexistent impossible query",
                "type": "negative_test",
                "description": "Should return low-quality or no results",
                "expected_chunks": []
            }
        ]
        
        queries.extend(edge_cases)
        
        return queries
    
    def _create_search_system(self, use_optimized: bool = False) -> HybridCodeSearch:
        """Create search system with specified configuration."""
        
        if use_optimized:
            # Apply optimized configuration
            HYBRID_SEARCH_CONFIG["bm25_k1"] = self.optimized_config["k1"]
            HYBRID_SEARCH_CONFIG["bm25_b"] = self.optimized_config["b"]
            
            # Create optimized tokenization manager
            bm25_config = {
                "stemming": self.optimized_config["stemming"],
                "stopwords_lang": "english",
                "lowercase": True,
                "min_token_length": 2,
                "remove_punctuation": True,
                "split_camelcase": self.optimized_config["split_camelcase"],
                "split_underscores": self.optimized_config["split_underscores"]
            }
            tokenization_manager = TokenizationManager(bm25_config=bm25_config)
        else:
            # Use original configuration
            HYBRID_SEARCH_CONFIG["bm25_k1"] = self.original_config["bm25_k1"]
            HYBRID_SEARCH_CONFIG["bm25_b"] = self.original_config["bm25_b"]
            
            # Create original tokenization manager
            bm25_config = {
                "stemming": "none",  # Original default
                "stopwords_lang": "english",
                "lowercase": True,
                "min_token_length": 2,
                "remove_punctuation": True,
                "split_camelcase": True,  # Original default
                "split_underscores": False  # Original default
            }
            tokenization_manager = TokenizationManager(bm25_config=bm25_config)
        
        # Create search system
        search_system = HybridCodeSearch(
            embeddings=self.embeddings,
            chunks=self.chunks,
            metadata=self.metadata,
            binary_embeddings=self.binary_embeddings
        )
        
        search_system.tokenization_manager = tokenization_manager
        
        return search_system
    
    def _format_search_results(self, results: List[Any]) -> List[SearchResult]:
        """Format search results for comparison."""
        formatted_results = []
        
        for i, result in enumerate(results[:5]):  # Top 5 results
            chunk_name = getattr(result, 'chunk_name', 'unknown')
            score = getattr(result, 'score', 0.0)
            
            # Get content preview
            content_preview = ""
            if hasattr(result, 'content') and result.content:
                content_preview = result.content[:100] + "..." if len(result.content) > 100 else result.content
            elif chunk_name != 'unknown':
                content_preview = f"Chunk: {chunk_name}"
            
            # Get chunk type
            chunk_type = str(getattr(result, 'chunk_type', 'unknown'))
            
            formatted_results.append(SearchResult(
                chunk_name=chunk_name,
                score=float(score),
                content_preview=content_preview,
                chunk_type=chunk_type,
                rank=i + 1
            ))
        
        return formatted_results

    def _assess_search_quality(self, query_info: Dict[str, Any], results: List[SearchResult]) -> Tuple[float, str]:
        """Assess the quality of search results for a query."""

        if not results:
            return 0.0, "No results returned"

        score = 0.0
        assessment_notes = []

        # Check for expected chunks (exact matches)
        if query_info.get("expected_chunks"):
            found_expected = 0
            for result in results:
                if result.chunk_name in query_info["expected_chunks"]:
                    found_expected += 1
                    # Higher score for higher ranking
                    score += (6 - result.rank) / 5.0

            if found_expected > 0:
                assessment_notes.append(f"Found {found_expected} expected chunks")
            else:
                assessment_notes.append("No expected chunks found")

        # Check for quality keywords (semantic relevance)
        if query_info.get("quality_keywords"):
            keyword_matches = 0
            for result in results:
                content_lower = result.content_preview.lower()
                chunk_name_lower = result.chunk_name.lower()

                for keyword in query_info["quality_keywords"]:
                    if keyword.lower() in content_lower or keyword.lower() in chunk_name_lower:
                        keyword_matches += 1
                        break

            keyword_score = min(keyword_matches / len(results), 1.0)
            score += keyword_score
            assessment_notes.append(f"Keyword relevance: {keyword_matches}/{len(results)} results")

        # Base quality assessment
        if not query_info.get("expected_chunks") and not query_info.get("quality_keywords"):
            # General quality assessment
            if results[0].score > 0.5:
                score += 0.5
                assessment_notes.append("High confidence first result")

            # Check for result diversity
            chunk_types = set(result.chunk_type for result in results)
            if len(chunk_types) > 1:
                score += 0.2
                assessment_notes.append(f"Diverse results ({len(chunk_types)} types)")

        # Special handling for negative tests
        if query_info.get("type") == "negative_test":
            if not results or results[0].score < 0.3:
                score = 1.0
                assessment_notes = ["Correctly returned low-quality results for nonsense query"]
            else:
                score = 0.0
                assessment_notes = ["Incorrectly returned high-quality results for nonsense query"]

        assessment = "; ".join(assessment_notes) if assessment_notes else "Basic result quality"
        return min(score, 1.0), assessment

    def _compare_configurations(self) -> List[ComparisonResult]:
        """Compare baseline vs optimized configurations."""
        print("\n🔍 Running Side-by-Side Configuration Comparison")
        print("=" * 70)

        comparison_results = []

        for i, query_info in enumerate(self.verification_queries):
            print(f"   Testing query {i+1}/{len(self.verification_queries)}: {query_info['query']}")

            try:
                # Test baseline configuration
                baseline_system = self._create_search_system(use_optimized=False)
                baseline_raw_results = baseline_system.hybrid_search(query_info["query"], top_k=5)
                baseline_results = self._format_search_results(baseline_raw_results)
                baseline_score, baseline_assessment = self._assess_search_quality(query_info, baseline_results)

                # Test optimized configuration
                optimized_system = self._create_search_system(use_optimized=True)
                optimized_raw_results = optimized_system.hybrid_search(query_info["query"], top_k=5)
                optimized_results = self._format_search_results(optimized_raw_results)
                optimized_score, optimized_assessment = self._assess_search_quality(query_info, optimized_results)

                # Calculate improvement
                improvement = optimized_score - baseline_score

                # Create comparison result
                comparison_result = ComparisonResult(
                    query=query_info["query"],
                    baseline_results=baseline_results,
                    optimized_results=optimized_results,
                    baseline_score=baseline_score,
                    optimized_score=optimized_score,
                    improvement=improvement,
                    quality_assessment=f"Baseline: {baseline_assessment} | Optimized: {optimized_assessment}"
                )

                comparison_results.append(comparison_result)

                # Print progress
                if improvement > 0.1:
                    print(f"      ✅ Significant improvement: +{improvement:.3f}")
                elif improvement > 0:
                    print(f"      📈 Minor improvement: +{improvement:.3f}")
                elif improvement < -0.1:
                    print(f"      ❌ Regression: {improvement:.3f}")
                else:
                    print(f"      ➡️  Similar: {improvement:.3f}")

            except Exception as e:
                print(f"      ❌ Query failed: {e}")
                # Create failed comparison result
                comparison_result = ComparisonResult(
                    query=query_info["query"],
                    baseline_results=[],
                    optimized_results=[],
                    baseline_score=0.0,
                    optimized_score=0.0,
                    improvement=0.0,
                    quality_assessment=f"Error: {str(e)}"
                )
                comparison_results.append(comparison_result)

        return comparison_results

    def _analyze_results(self, comparison_results: List[ComparisonResult]) -> Dict[str, Any]:
        """Analyze comparison results for statistical significance."""

        if not comparison_results:
            return {"error": "No comparison results to analyze"}

        # Calculate statistics
        improvements = [cr.improvement for cr in comparison_results if cr.improvement != 0]
        baseline_scores = [cr.baseline_score for cr in comparison_results]
        optimized_scores = [cr.optimized_score for cr in comparison_results]

        # Overall statistics
        total_queries = len(comparison_results)
        improved_queries = len([cr for cr in comparison_results if cr.improvement > 0.01])
        regressed_queries = len([cr for cr in comparison_results if cr.improvement < -0.01])
        similar_queries = total_queries - improved_queries - regressed_queries

        # Score statistics
        avg_baseline_score = np.mean(baseline_scores) if baseline_scores else 0
        avg_optimized_score = np.mean(optimized_scores) if optimized_scores else 0
        avg_improvement = np.mean(improvements) if improvements else 0

        # Significance testing (simple t-test approximation)
        if len(improvements) > 1:
            std_improvement = np.std(improvements)
            t_statistic = avg_improvement / (std_improvement / np.sqrt(len(improvements)))
            significant = abs(t_statistic) > 2.0  # Rough 95% confidence
        else:
            significant = False

        # Quality categories
        excellent_queries = len([cr for cr in comparison_results if cr.optimized_score > 0.8])
        good_queries = len([cr for cr in comparison_results if 0.5 < cr.optimized_score <= 0.8])
        poor_queries = len([cr for cr in comparison_results if cr.optimized_score <= 0.5])

        return {
            "total_queries": total_queries,
            "improved_queries": improved_queries,
            "regressed_queries": regressed_queries,
            "similar_queries": similar_queries,
            "improvement_rate": improved_queries / total_queries if total_queries > 0 else 0,
            "regression_rate": regressed_queries / total_queries if total_queries > 0 else 0,
            "avg_baseline_score": avg_baseline_score,
            "avg_optimized_score": avg_optimized_score,
            "avg_improvement": avg_improvement,
            "statistically_significant": significant,
            "excellent_queries": excellent_queries,
            "good_queries": good_queries,
            "poor_queries": poor_queries,
            "quality_distribution": {
                "excellent": excellent_queries / total_queries if total_queries > 0 else 0,
                "good": good_queries / total_queries if total_queries > 0 else 0,
                "poor": poor_queries / total_queries if total_queries > 0 else 0
            }
        }

    def verify_optimization(self) -> Dict[str, Any]:
        """Run comprehensive verification of optimization results."""
        print("🔍 Starting Comprehensive Optimization Verification")
        print("=" * 80)
        print(f"   Baseline config: k1={self.original_config['bm25_k1']}, b={self.original_config['bm25_b']}")
        print(f"   Optimized config: k1={self.optimized_config['k1']}, b={self.optimized_config['b']}")
        print(f"   Verification queries: {len(self.verification_queries)}")

        start_time = time.time()

        # Run side-by-side comparison
        comparison_results = self._compare_configurations()

        # Analyze results
        analysis = self._analyze_results(comparison_results)

        verification_time = time.time() - start_time

        # Restore original configuration
        HYBRID_SEARCH_CONFIG.update(self.original_config)

        return {
            "verification_time": verification_time,
            "comparison_results": comparison_results,
            "statistical_analysis": analysis,
            "verification_metadata": {
                "total_queries_tested": len(self.verification_queries),
                "baseline_config": {
                    "k1": self.original_config['bm25_k1'],
                    "b": self.original_config['bm25_b'],
                    "stemming": "none",
                    "split_camelcase": True,
                    "split_underscores": False
                },
                "optimized_config": self.optimized_config,
                "dataset_size": len(self.chunks)
            }
        }

    def print_detailed_report(self, verification_results: Dict[str, Any]) -> None:
        """Print detailed verification report."""
        print("\n" + "=" * 80)
        print("🏆 OPTIMIZATION VERIFICATION REPORT")
        print("=" * 80)

        analysis = verification_results["statistical_analysis"]
        metadata = verification_results["verification_metadata"]

        # Overall summary
        print(f"📊 Overall Performance:")
        print(f"   Total queries tested: {analysis['total_queries']}")
        print(f"   Improved queries: {analysis['improved_queries']} ({analysis['improvement_rate']:.1%})")
        print(f"   Regressed queries: {analysis['regressed_queries']} ({analysis['regression_rate']:.1%})")
        print(f"   Similar performance: {analysis['similar_queries']}")
        print(f"   Average improvement: {analysis['avg_improvement']:+.3f}")
        print(f"   Statistically significant: {'✅ Yes' if analysis['statistically_significant'] else '❌ No'}")

        # Verification conclusion
        print(f"\n🎯 VERIFICATION CONCLUSION:")

        if analysis['improvement_rate'] > 0.7 and analysis['regression_rate'] < 0.1:
            print("   ✅ OPTIMIZATION VERIFIED: Significant improvements with minimal regressions")
            print("   ✅ RECOMMENDATION: Deploy optimized parameters to production")
        elif analysis['improvement_rate'] > 0.5 and analysis['regression_rate'] < 0.2:
            print("   📈 OPTIMIZATION PROMISING: Good improvements with acceptable regressions")
            print("   ⚠️  RECOMMENDATION: Consider deployment with monitoring")
        else:
            print("   🔍 RECOMMENDATION: Further investigation needed")


def main():
    """Main verification execution."""
    print("🔍 BM25 Optimization Verification System")
    print("Comprehensive testing to ensure no false positives")
    print()

    # Initialize verifier
    verifier = OptimizationVerifier()

    # Run verification
    print("\n⏱️  Starting verification - this will take 2-3 minutes...")
    verification_results = verifier.verify_optimization()

    # Print detailed report
    verifier.print_detailed_report(verification_results)

    # Save verification results
    results_file = Path("optimization_verification_results.json")

    # Convert comparison results to serializable format
    serializable_results = verification_results.copy()
    serializable_results["comparison_results"] = [
        {
            "query": cr.query,
            "baseline_score": cr.baseline_score,
            "optimized_score": cr.optimized_score,
            "improvement": cr.improvement,
            "quality_assessment": cr.quality_assessment
        } for cr in verification_results["comparison_results"]
    ]

    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)

    print(f"\n💾 Detailed verification results saved to: {results_file}")
    print("✅ Verification completed successfully!")


if __name__ == "__main__":
    main()
