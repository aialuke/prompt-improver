#!/usr/bin/env python3
"""
BM25 Parameter Optimization for Code Search

Systematically tests different BM25 parameters (k1, b) and tokenization
configurations to find optimal settings for code search quality.

Tests:
1. Different k1 values (term frequency saturation)
2. Different b values (length normalization)
3. Different stemming levels (none, light, aggressive)
4. CamelCase and underscore splitting impact

Usage:
    python optimize_bm25_parameters.py
"""

import os
import sys
import time
import json
import itertools
from pathlib import Path
from typing import List, Dict, Any, Tuple
import numpy as np

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

try:
    from search_code import HybridCodeSearch, HYBRID_SEARCH_CONFIG
    from bm25_tokenizers import TokenizationManager, BM25Tokenizer
    from generate_embeddings import CodeChunk, ChunkType
    import numpy as np
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class BM25ParameterOptimizer:
    """Optimize BM25 parameters for code search quality."""
    
    def __init__(self):
        """Initialize optimizer with test data and configurations."""
        self.test_chunks = self._create_test_chunks()
        self.test_embeddings = self._create_test_embeddings()
        self.test_queries = self._create_test_queries()
        self.optimization_results = []
        
        print("🔧 BM25 Parameter Optimizer initialized")
        print(f"   Test chunks: {len(self.test_chunks)}")
        print(f"   Test queries: {len(self.test_queries)}")
    
    def _create_test_chunks(self) -> List[CodeChunk]:
        """Create comprehensive test chunks for optimization with expanded coverage."""
        return [
            # Function chunks for function_name queries
            CodeChunk(
                content="def calculate_similarity(vector1, vector2):\n    return cosine_similarity(vector1, vector2)",
                chunk_type=ChunkType.FUNCTION,
                name="calculate_similarity",
                file_path="similarity.py",
                start_line=1,
                end_line=2,
                docstring="Calculate cosine similarity between vectors"
            ),
            CodeChunk(
                content="def tokenize_text(text, language='english'):\n    tokens = word_tokenize(text)\n    return [token.lower() for token in tokens]",
                chunk_type=ChunkType.FUNCTION,
                name="tokenize_text",
                file_path="text_processing.py",
                start_line=5,
                end_line=7,
                docstring="Tokenize text into individual words"
            ),
            CodeChunk(
                content="def generate_embeddings(texts, model='voyage-context-3'):\n    embeddings = model.encode(texts)\n    return embeddings",
                chunk_type=ChunkType.FUNCTION,
                name="generate_embeddings",
                file_path="embeddings.py",
                start_line=10,
                end_line=12,
                docstring="Generate embeddings for text using specified model"
            ),

            # Class chunks for class_name queries
            CodeChunk(
                content="class BM25Tokenizer:\n    def __init__(self, stemming=True):\n        self.stemming = stemming",
                chunk_type=ChunkType.CLASS,
                name="BM25Tokenizer",
                file_path="tokenizers.py",
                start_line=10,
                end_line=12,
                docstring="Optimized tokenizer for BM25 lexical search"
            ),
            CodeChunk(
                content="class HybridCodeSearch:\n    def __init__(self, embeddings, chunks):\n        self.embeddings = embeddings\n        self.chunks = chunks",
                chunk_type=ChunkType.CLASS,
                name="HybridCodeSearch",
                file_path="search_code.py",
                start_line=20,
                end_line=23,
                docstring="Hybrid search combining semantic and lexical search"
            ),
            CodeChunk(
                content="class TokenizationManager:\n    def __init__(self):\n        self.bm25_tokenizer = BM25Tokenizer()",
                chunk_type=ChunkType.CLASS,
                name="TokenizationManager",
                file_path="tokenizers.py",
                start_line=50,
                end_line=52,
                docstring="Manages independent tokenization strategies"
            ),

            # Method chunks for method_call queries
            CodeChunk(
                content="def cosine_similarity(vector1, vector2):\n    dot_product = np.dot(vector1, vector2)\n    norms = np.linalg.norm(vector1) * np.linalg.norm(vector2)\n    return dot_product / norms",
                chunk_type=ChunkType.FUNCTION,
                name="cosine_similarity",
                file_path="metrics.py",
                start_line=15,
                end_line=18,
                docstring="Calculate cosine similarity between two vectors"
            ),
            CodeChunk(
                content="def stem_words(words, language='english'):\n    stemmer = SnowballStemmer(language)\n    return [stemmer.stem(word) for word in words]",
                chunk_type=ChunkType.FUNCTION,
                name="stem_words",
                file_path="text_processing.py",
                start_line=25,
                end_line=27,
                docstring="Apply stemming to a list of words"
            ),
            CodeChunk(
                content="def removeStopwords(tokens, stopwords):\n    return [token for token in tokens if token not in stopwords]",
                chunk_type=ChunkType.FUNCTION,
                name="removeStopwords",
                file_path="preprocessing.py",
                start_line=15,
                end_line=16,
                docstring="Remove stopwords from token list"
            ),

            # Code pattern chunks
            CodeChunk(
                content="for i, item in enumerate(items):\n    if item.score > threshold:\n        results.append(item)",
                chunk_type=ChunkType.FUNCTION,
                name="for_loop_iteration",
                file_path="processing.py",
                start_line=40,
                end_line=42,
                docstring="Example of for loop iteration pattern"
            ),
            CodeChunk(
                content="try:\n    result = risky_operation()\nexcept Exception as e:\n    logger.error(f'Operation failed: {e}')\n    result = None",
                chunk_type=ChunkType.FUNCTION,
                name="try_except_error",
                file_path="error_handling.py",
                start_line=5,
                end_line=9,
                docstring="Error handling with try-except pattern"
            ),
            CodeChunk(
                content="import numpy as np\nfrom sklearn.metrics.pairwise import cosine_similarity",
                chunk_type=ChunkType.IMPORT_BLOCK,
                name="imports",
                file_path="search.py",
                start_line=1,
                end_line=2,
                docstring="Required imports for search functionality"
            ),

            # CamelCase method chunks
            CodeChunk(
                content="def calculateSimilarity(self, vector1, vector2):\n    return self.cosine_similarity(vector1, vector2)",
                chunk_type=ChunkType.METHOD,
                name="calculateSimilarity",
                file_path="similarity_class.py",
                start_line=15,
                end_line=16,
                docstring="CamelCase method for similarity calculation"
            ),
            CodeChunk(
                content="def tokenizeText(self, text):\n    return self.tokenizer.tokenize(text)",
                chunk_type=ChunkType.METHOD,
                name="tokenizeText",
                file_path="text_processor.py",
                start_line=20,
                end_line=21,
                docstring="CamelCase method for text tokenization"
            ),

            # Stemming test chunks (different forms of same concept)
            CodeChunk(
                content="def run_algorithm(self):\n    self.is_running = True\n    return self.execute()",
                chunk_type=ChunkType.METHOD,
                name="run_algorithm",
                file_path="algorithm.py",
                start_line=25,
                end_line=27,
                docstring="Run the algorithm execution"
            ),
            CodeChunk(
                content="def running_status(self):\n    return self.is_running",
                chunk_type=ChunkType.METHOD,
                name="running_status",
                file_path="algorithm.py",
                start_line=30,
                end_line=31,
                docstring="Check if algorithm is currently running"
            ),
            CodeChunk(
                content="def tokenization_process(self, text):\n    return self.tokenize(text)",
                chunk_type=ChunkType.METHOD,
                name="tokenization_process",
                file_path="tokenizers.py",
                start_line=60,
                end_line=61,
                docstring="Process text through tokenization"
            ),
            CodeChunk(
                content="def similarity_calculation(self, vectors):\n    similarities = []\n    for i, vec in enumerate(vectors):\n        similarities.append(self.calculate_similarity(vec))\n    return similarities",
                chunk_type=ChunkType.METHOD,
                name="similarity_calculation",
                file_path="metrics.py",
                start_line=35,
                end_line=39,
                docstring="Calculate similarities for multiple vectors"
            ),

            # Algorithm and search related chunks for stopword tests
            CodeChunk(
                content="def best_algorithm_for_search(self, query_type):\n    if query_type == 'semantic':\n        return 'voyage_search'\n    return 'bm25_search'",
                chunk_type=ChunkType.METHOD,
                name="best_algorithm_for_search",
                file_path="search_optimizer.py",
                start_line=45,
                end_line=48,
                docstring="Select the best search algorithm for given query type"
            ),
            CodeChunk(
                content="def function_to_calculate(self, operation, values):\n    if operation == 'mean':\n        return np.mean(values)\n    elif operation == 'sum':\n        return np.sum(values)\n    return values",
                chunk_type=ChunkType.METHOD,
                name="function_to_calculate",
                file_path="calculator.py",
                start_line=10,
                end_line=15,
                docstring="Generic function to calculate various operations on values"
            ),

            # Additional method chunk for tokenizeForBM25 compatibility
            CodeChunk(
                content="def tokenizeForBM25(self, text):\n    tokens = text.lower().split()\n    return [self.stemmer.stem(token) for token in tokens]",
                chunk_type=ChunkType.METHOD,
                name="tokenizeForBM25",
                file_path="tokenizers.py",
                start_line=70,
                end_line=72,
                docstring="Tokenize text for BM25 with stemming"
            )
        ]
    
    def _create_test_embeddings(self) -> np.ndarray:
        """Create test embeddings matching voyage-context-3 dimensions."""
        np.random.seed(42)
        embeddings = np.random.rand(len(self.test_chunks), 1024).astype(np.float32)
        return embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    def _create_test_queries(self) -> List[Dict[str, Any]]:
        """Create comprehensive test queries with expected results for optimization (expanded from 6 to 20 queries)."""
        return [
            # Function name queries (3 queries)
            {
                "query": "calculate similarity",
                "expected_chunks": ["calculate_similarity"],
                "type": "function_name"
            },
            {
                "query": "tokenize text",
                "expected_chunks": ["tokenize_text"],
                "type": "function_name"
            },
            {
                "query": "generate embeddings",
                "expected_chunks": ["generate_embeddings"],
                "type": "function_name"
            },

            # Class name queries (3 queries)
            {
                "query": "BM25Tokenizer",
                "expected_chunks": ["BM25Tokenizer"],
                "type": "class_name"
            },
            {
                "query": "HybridCodeSearch",
                "expected_chunks": ["HybridCodeSearch"],
                "type": "class_name"
            },
            {
                "query": "TokenizationManager",
                "expected_chunks": ["TokenizationManager"],
                "type": "class_name"
            },

            # Method/API queries (3 queries)
            {
                "query": "cosine similarity",
                "expected_chunks": ["cosine_similarity"],
                "type": "method_call"
            },
            {
                "query": "stem words",
                "expected_chunks": ["stem_words"],
                "type": "method_call"
            },
            {
                "query": "remove stopwords",
                "expected_chunks": ["removeStopwords"],
                "type": "method_call"
            },

            # Code pattern queries (3 queries)
            {
                "query": "for loop iteration",
                "expected_chunks": ["for_loop_iteration"],
                "type": "code_pattern"
            },
            {
                "query": "try except error",
                "expected_chunks": ["try_except_error"],
                "type": "code_pattern"
            },
            {
                "query": "import numpy",
                "expected_chunks": ["imports"],
                "type": "code_pattern"
            },

            # Camel case variations (2 queries)
            {
                "query": "calculateSimilarity",
                "expected_chunks": ["calculateSimilarity"],
                "type": "camel_case"
            },
            {
                "query": "tokenizeText",
                "expected_chunks": ["tokenizeText"],
                "type": "camel_case"
            },

            # Partial matching (stemming test) (3 queries)
            {
                "query": "running",
                "expected_chunks": ["run_algorithm", "running_status"],
                "type": "stemming_test"
            },
            {
                "query": "tokenization",
                "expected_chunks": ["tokenization_process", "TokenizationManager"],
                "type": "stemming_test"
            },
            {
                "query": "similarities",
                "expected_chunks": ["similarity_calculation", "calculate_similarity"],
                "type": "stemming_test"
            },

            # Stopword filtering test (2 queries)
            {
                "query": "the best algorithm for search",
                "expected_chunks": ["best_algorithm_for_search"],
                "type": "stopword_test"
            },
            {
                "query": "a function to calculate",
                "expected_chunks": ["function_to_calculate"],
                "type": "stopword_test"
            },

            # Legacy compatibility query (1 query)
            {
                "query": "tokenizeForBM25",
                "expected_chunks": ["tokenizeForBM25"],
                "type": "camelcase_search"
            }
        ]
    
    def test_configuration(self, k1: float, b: float, 
                          stemming: str, split_camelcase: bool, 
                          split_underscores: bool) -> Dict[str, Any]:
        """Test a specific BM25 configuration."""
        try:
            # Create tokenization manager with specific config
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
            
            # Create search system with custom BM25 parameters
            search_system = HybridCodeSearch(
                embeddings=self.test_embeddings,
                chunks=self.test_chunks,
                metadata={"test": True}
            )
            
            # Override BM25 parameters
            if search_system.bm25:
                search_system.bm25.k1 = k1
                search_system.bm25.b = b
            
            # Test all queries
            total_score = 0.0
            query_results = []
            
            for query_info in self.test_queries:
                try:
                    results = search_system.hybrid_search(query_info["query"], top_k=5)
                    
                    # Calculate relevance score
                    score = 0.0
                    for i, result in enumerate(results[:3]):  # Top 3 results
                        if result.chunk_name in query_info["expected_chunks"]:
                            # Higher score for higher ranking
                            score += (3 - i) / 3.0
                    
                    total_score += score
                    query_results.append({
                        "query": query_info["query"],
                        "score": score,
                        "results_count": len(results)
                    })
                    
                except Exception as e:
                    print(f"   ❌ Query failed: {query_info['query']} - {e}")
                    query_results.append({
                        "query": query_info["query"],
                        "score": 0.0,
                        "error": str(e)
                    })
            
            avg_score = total_score / len(self.test_queries)
            
            return {
                "k1": k1,
                "b": b,
                "stemming": stemming,
                "split_camelcase": split_camelcase,
                "split_underscores": split_underscores,
                "avg_score": avg_score,
                "total_score": total_score,
                "query_results": query_results
            }
            
        except Exception as e:
            return {
                "k1": k1,
                "b": b,
                "stemming": stemming,
                "split_camelcase": split_camelcase,
                "split_underscores": split_underscores,
                "avg_score": 0.0,
                "error": str(e)
            }
    
    def optimize_parameters(self) -> Dict[str, Any]:
        """Run comprehensive parameter optimization."""
        print("\n🔍 Starting BM25 Parameter Optimization")
        print("=" * 60)
        
        # Parameter ranges to test
        k1_values = [0.8, 1.0, 1.2, 1.5, 2.0]
        b_values = [0.0, 0.25, 0.5, 0.75, 1.0]
        stemming_options = ["none", "light", "aggressive"]
        camelcase_options = [True, False]
        underscore_options = [True, False]
        
        total_configs = (len(k1_values) * len(b_values) * len(stemming_options) * 
                        len(camelcase_options) * len(underscore_options))
        
        print(f"Testing {total_configs} configurations...")
        
        best_config = None
        best_score = 0.0
        config_count = 0
        
        # Test all combinations
        for k1, b, stemming, camelcase, underscore in itertools.product(
            k1_values, b_values, stemming_options, camelcase_options, underscore_options
        ):
            config_count += 1
            
            print(f"\n📊 Config {config_count}/{total_configs}: k1={k1}, b={b}, "
                  f"stem={stemming}, camel={camelcase}, under={underscore}")
            
            result = self.test_configuration(k1, b, stemming, camelcase, underscore)
            self.optimization_results.append(result)
            
            print(f"   Score: {result['avg_score']:.3f}")
            
            if result['avg_score'] > best_score:
                best_score = result['avg_score']
                best_config = result
                print(f"   🎉 New best configuration! Score: {best_score:.3f}")
        
        # Analyze results
        optimization_summary = {
            "best_config": best_config,
            "best_score": best_score,
            "total_configs_tested": total_configs,
            "all_results": self.optimization_results,
            "analysis": self._analyze_results()
        }
        
        return optimization_summary
    
    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze optimization results to identify patterns."""
        if not self.optimization_results:
            return {}
        
        # Group by parameter
        k1_scores = {}
        b_scores = {}
        stemming_scores = {}
        camelcase_scores = {True: [], False: []}
        underscore_scores = {True: [], False: []}
        
        for result in self.optimization_results:
            if 'error' in result:
                continue
                
            k1 = result['k1']
            b = result['b']
            stemming = result['stemming']
            score = result['avg_score']
            
            if k1 not in k1_scores:
                k1_scores[k1] = []
            k1_scores[k1].append(score)
            
            if b not in b_scores:
                b_scores[b] = []
            b_scores[b].append(score)
            
            if stemming not in stemming_scores:
                stemming_scores[stemming] = []
            stemming_scores[stemming].append(score)
            
            camelcase_scores[result['split_camelcase']].append(score)
            underscore_scores[result['split_underscores']].append(score)
        
        # Calculate averages
        analysis = {
            "k1_analysis": {k: np.mean(scores) for k, scores in k1_scores.items()},
            "b_analysis": {b: np.mean(scores) for b, scores in b_scores.items()},
            "stemming_analysis": {s: np.mean(scores) for s, scores in stemming_scores.items()},
            "camelcase_analysis": {k: np.mean(scores) for k, scores in camelcase_scores.items()},
            "underscore_analysis": {k: np.mean(scores) for k, scores in underscore_scores.items()}
        }
        
        return analysis


def main():
    """Main optimization execution."""
    print("🔧 BM25 Parameter Optimization for Code Search")
    print("Systematically testing configurations for optimal performance")
    print()
    
    optimizer = BM25ParameterOptimizer()
    results = optimizer.optimize_parameters()
    
    # Print summary
    print("\n" + "=" * 60)
    print("🏆 OPTIMIZATION RESULTS")
    print("=" * 60)
    
    best = results["best_config"]
    print(f"Best Configuration:")
    print(f"   k1: {best['k1']}")
    print(f"   b: {best['b']}")
    print(f"   Stemming: {best['stemming']}")
    print(f"   CamelCase splitting: {best['split_camelcase']}")
    print(f"   Underscore splitting: {best['split_underscores']}")
    print(f"   Score: {best['avg_score']:.3f}")
    
    # Print analysis
    analysis = results["analysis"]
    print(f"\n📊 Parameter Analysis:")
    print(f"   Best k1: {max(analysis['k1_analysis'], key=analysis['k1_analysis'].get)}")
    print(f"   Best b: {max(analysis['b_analysis'], key=analysis['b_analysis'].get)}")
    print(f"   Best stemming: {max(analysis['stemming_analysis'], key=analysis['stemming_analysis'].get)}")
    print(f"   CamelCase helps: {analysis['camelcase_analysis'][True] > analysis['camelcase_analysis'][False]}")
    print(f"   Underscore helps: {analysis['underscore_analysis'][True] > analysis['underscore_analysis'][False]}")
    
    # Save results
    results_file = Path("bm25_optimization_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: {results_file}")


if __name__ == "__main__":
    main()
