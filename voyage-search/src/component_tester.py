"""
Component Testing Framework for Search System Analysis

This module provides systematic testing of individual search components to measure
their effectiveness, performance impact, and contribution to overall search quality.

Features:
- Configurable component enabling/disabling
- Standardized test queries and validation
- Performance measurement and comparison
- False positive detection and analysis
- Real behavior testing with actual embeddings and data
"""

import time
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path
import sys

# Import the search system components
from search_code import (
    HybridCodeSearch, SearchResult, load_enhanced_embeddings,
    HYBRID_SEARCH_CONFIG, SEARCH_METRICS, get_search_metrics
)


@dataclass
class ComponentTestConfig:
    """Configuration for component testing."""
    enable_semantic: bool = True
    enable_lexical: bool = True
    enable_binary_rescoring: bool = True
    enable_cross_encoder: bool = True
    enable_local_quantization: bool = True
    test_name: str = "default"


@dataclass
class TestMetrics:
    """Metrics collected during component testing."""
    search_time_ms: float
    api_calls_count: int
    results_count: int
    avg_similarity_score: float
    top_result_score: float
    false_positive_count: int
    memory_usage_mb: float
    component_config: str


@dataclass
class TestQuery:
    """Test query with expected results for validation."""
    query: str
    expected_file_patterns: List[str]  # File patterns that should be found
    expected_function_patterns: List[str]  # Function patterns that should be found
    max_false_positives: int = 2  # Maximum acceptable false positives
    min_expected_results: int = 1  # Minimum expected relevant results
    category: str = "general"  # Query category for analysis


class ComponentTester:
    """Framework for systematic component testing."""
    
    def __init__(self, embeddings_path: str = "embeddings.pkl"):
        """Initialize the component tester."""
        self.embeddings_path = embeddings_path
        self.embeddings = None
        self.chunks = None
        self.metadata = None
        self.binary_embeddings = None
        self.test_results: List[Dict[str, Any]] = []
        
        # Load embeddings once for all tests
        self._load_embeddings()
        
        # Define comprehensive test queries
        self.test_queries = self._create_test_queries()
        
    def _load_embeddings(self) -> None:
        """Load embeddings and data for testing."""
        try:
            print("📊 Loading embeddings for component testing...")
            self.embeddings, self.chunks, self.metadata, self.binary_embeddings = load_enhanced_embeddings(
                self.embeddings_path
            )
            print(f"✅ Loaded {len(self.chunks)} chunks for testing")
        except Exception as e:
            print(f"❌ Failed to load embeddings: {e}")
            raise
    
    def _create_test_queries(self) -> List[TestQuery]:
        """Create comprehensive test query dataset."""
        return [
            # Algorithm queries
            TestQuery(
                query="sorting algorithm implementation",
                expected_file_patterns=["sort", "algorithm"],
                expected_function_patterns=["sort", "quicksort", "mergesort", "bubblesort"],
                category="algorithms"
            ),
            TestQuery(
                query="binary search function",
                expected_file_patterns=["search", "binary"],
                expected_function_patterns=["binary_search", "bsearch", "search"],
                category="algorithms"
            ),
            
            # Data structure queries
            TestQuery(
                query="linked list implementation",
                expected_file_patterns=["list", "linked", "node"],
                expected_function_patterns=["LinkedList", "Node", "insert", "delete"],
                category="data_structures"
            ),
            TestQuery(
                query="hash table or dictionary",
                expected_file_patterns=["hash", "dict", "map"],
                expected_function_patterns=["hash", "put", "get", "dict"],
                category="data_structures"
            ),
            
            # API and web queries
            TestQuery(
                query="HTTP request handling",
                expected_file_patterns=["http", "request", "api", "server"],
                expected_function_patterns=["request", "get", "post", "handler"],
                category="web_api"
            ),
            TestQuery(
                query="database connection and queries",
                expected_file_patterns=["db", "database", "sql", "connection"],
                expected_function_patterns=["connect", "query", "execute", "cursor"],
                category="database"
            ),
            
            # Utility and helper queries
            TestQuery(
                query="file reading and writing operations",
                expected_file_patterns=["file", "io", "read", "write"],
                expected_function_patterns=["read", "write", "open", "close"],
                category="file_operations"
            ),
            TestQuery(
                query="string manipulation and parsing",
                expected_file_patterns=["string", "text", "parse"],
                expected_function_patterns=["parse", "split", "join", "replace"],
                category="string_processing"
            ),
            
            # Error handling queries
            TestQuery(
                query="exception handling and error management",
                expected_file_patterns=["error", "exception", "handle"],
                expected_function_patterns=["try", "catch", "except", "handle_error"],
                category="error_handling"
            ),
            
            # Async and concurrency queries
            TestQuery(
                query="asynchronous programming and async functions",
                expected_file_patterns=["async", "await", "concurrent"],
                expected_function_patterns=["async", "await", "asyncio", "concurrent"],
                category="async_programming"
            )
        ]
    
    def create_search_instance(self, config: ComponentTestConfig) -> HybridCodeSearch:
        """Create a search instance with specified component configuration."""
        
        # Temporarily modify global configuration
        original_config = HYBRID_SEARCH_CONFIG.copy()
        
        try:
            # Configure components based on test config
            HYBRID_SEARCH_CONFIG["enable_binary_rescoring"] = config.enable_binary_rescoring
            HYBRID_SEARCH_CONFIG["enable_cross_encoder"] = config.enable_cross_encoder

            # CRITICAL FIX: Only use Voyage tokenization when semantic search is enabled
            # This prevents unnecessary API calls for pure lexical tests
            if config.enable_semantic:
                HYBRID_SEARCH_CONFIG["use_voyage_tokenization"] = True
            else:
                # Pure lexical search should use simple tokenization (no API calls)
                HYBRID_SEARCH_CONFIG["use_voyage_tokenization"] = False
            
            # Create search instance
            search_instance = HybridCodeSearch(
                embeddings=self.embeddings,
                chunks=self.chunks,
                metadata=self.metadata,
                binary_embeddings=self.binary_embeddings if config.enable_binary_rescoring else None
            )
            
            # Manually disable components if needed
            if not config.enable_lexical:
                search_instance.bm25 = None
                
            if not config.enable_cross_encoder:
                search_instance.cross_encoder = None
                
            return search_instance
            
        finally:
            # Restore original configuration
            HYBRID_SEARCH_CONFIG.update(original_config)
    
    def run_component_test(self, config: ComponentTestConfig, 
                          max_queries: int = 5) -> Dict[str, Any]:
        """Run a complete test with specified component configuration."""
        
        print(f"\n🧪 Testing configuration: {config.test_name}")
        print(f"   Semantic: {'✅' if config.enable_semantic else '❌'}")
        print(f"   Lexical (BM25): {'✅' if config.enable_lexical else '❌'}")
        print(f"   Binary Rescoring: {'✅' if config.enable_binary_rescoring else '❌'}")
        print(f"   Cross-encoder: {'✅' if config.enable_cross_encoder else '❌'}")
        print(f"   Local Quantization: {'✅' if config.enable_local_quantization else '❌'}")
        
        # Reset metrics
        SEARCH_METRICS["total_searches"] = 0
        SEARCH_METRICS["api_calls"] = 0
        SEARCH_METRICS["cache_hits"] = 0
        
        # Create search instance with configuration
        search_instance = self.create_search_instance(config)
        
        test_results = []
        total_time = 0
        total_false_positives = 0
        
        # Run tests on subset of queries
        test_queries = self.test_queries[:max_queries]
        
        for query in test_queries:
            print(f"\n  🔍 Testing: '{query.query}'")
            
            start_time = time.time()
            
            try:
                # Execute search
                results = search_instance.hybrid_search(
                    query=query.query,
                    top_k=5,
                    min_similarity=0.2
                )
                
                search_time = (time.time() - start_time) * 1000
                total_time += search_time
                
                # Analyze results
                false_positives = self._count_false_positives(query, results)
                total_false_positives += false_positives
                
                # Collect metrics
                query_metrics = {
                    "query": query.query,
                    "category": query.category,
                    "search_time_ms": search_time,
                    "results_count": len(results),
                    "false_positives": false_positives,
                    "avg_similarity": np.mean([r.similarity_score for r in results]) if results else 0.0,
                    "top_similarity": results[0].similarity_score if results else 0.0,
                    "relevant_results": len(results) - false_positives
                }
                
                test_results.append(query_metrics)
                
                print(f"    ⏱️  {search_time:.1f}ms, {len(results)} results, {false_positives} false positives")
                
            except Exception as e:
                print(f"    ❌ Error: {e}")
                test_results.append({
                    "query": query.query,
                    "category": query.category,
                    "error": str(e),
                    "search_time_ms": 0,
                    "results_count": 0,
                    "false_positives": 0,
                    "avg_similarity": 0.0,
                    "top_similarity": 0.0,
                    "relevant_results": 0
                })
        
        # Compile overall metrics
        final_metrics = get_search_metrics()
        
        overall_results = {
            "config": asdict(config),
            "total_queries": len(test_queries),
            "total_time_ms": total_time,
            "avg_time_per_query_ms": total_time / len(test_queries) if test_queries else 0,
            "total_api_calls": final_metrics["api_calls"],
            "total_false_positives": total_false_positives,
            "avg_false_positives_per_query": total_false_positives / len(test_queries) if test_queries else 0,
            "query_results": test_results,
            "timestamp": time.time()
        }
        
        return overall_results
    
    def _count_false_positives(self, query: TestQuery, results: List[SearchResult]) -> int:
        """Count false positive results based on expected patterns."""
        false_positives = 0
        
        for result in results:
            is_relevant = False
            
            # Check if result matches expected file patterns
            for pattern in query.expected_file_patterns:
                if pattern.lower() in result.file_path.lower() or pattern.lower() in result.content.lower():
                    is_relevant = True
                    break
            
            # Check if result matches expected function patterns
            if not is_relevant:
                for pattern in query.expected_function_patterns:
                    if (pattern.lower() in result.chunk_name.lower() or 
                        pattern.lower() in result.content.lower()):
                        is_relevant = True
                        break
            
            if not is_relevant:
                false_positives += 1
        
        return false_positives
    
    def save_test_results(self, results: Dict[str, Any], output_file: str) -> None:
        """Save test results to JSON file."""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Test results saved to {output_path}")


def create_test_configurations() -> List[ComponentTestConfig]:
    """Create all test configurations for component analysis."""
    return [
        # Baseline - all components enabled
        ComponentTestConfig(
            enable_semantic=True,
            enable_lexical=True,
            enable_binary_rescoring=True,
            enable_cross_encoder=True,
            enable_local_quantization=True,
            test_name="baseline_all_components"
        ),
        
        # Semantic only
        ComponentTestConfig(
            enable_semantic=True,
            enable_lexical=False,
            enable_binary_rescoring=False,
            enable_cross_encoder=False,
            enable_local_quantization=True,
            test_name="semantic_only"
        ),
        
        # Lexical (BM25) only
        ComponentTestConfig(
            enable_semantic=False,
            enable_lexical=True,
            enable_binary_rescoring=False,
            enable_cross_encoder=False,
            enable_local_quantization=False,
            test_name="lexical_bm25_only"
        ),
        
        # No binary rescoring
        ComponentTestConfig(
            enable_semantic=True,
            enable_lexical=True,
            enable_binary_rescoring=False,
            enable_cross_encoder=True,
            enable_local_quantization=True,
            test_name="no_binary_rescoring"
        ),
        
        # No cross-encoder reranking
        ComponentTestConfig(
            enable_semantic=True,
            enable_lexical=True,
            enable_binary_rescoring=True,
            enable_cross_encoder=False,
            enable_local_quantization=True,
            test_name="no_cross_encoder"
        ),
        
        # Semantic + Lexical (no optimizations)
        ComponentTestConfig(
            enable_semantic=True,
            enable_lexical=True,
            enable_binary_rescoring=False,
            enable_cross_encoder=False,
            enable_local_quantization=True,
            test_name="semantic_plus_lexical"
        )
    ]


if __name__ == "__main__":
    print("🧪 Component Testing Framework")
    print("=" * 50)
    
    # Initialize tester
    tester = ComponentTester()
    
    # Get test configurations
    configs = create_test_configurations()
    
    # Run tests for each configuration
    all_results = {}
    
    for config in configs:
        try:
            results = tester.run_component_test(config, max_queries=3)  # Limit for initial testing
            all_results[config.test_name] = results
            
            # Save individual results
            tester.save_test_results(results, f"test_results/{config.test_name}.json")
            
        except Exception as e:
            print(f"❌ Failed to test {config.test_name}: {e}")
    
    # Save combined results
    tester.save_test_results(all_results, "test_results/component_analysis_results.json")
    
    print("\n🎉 Component testing completed!")
    print("📊 Results saved to test_results/ directory")
