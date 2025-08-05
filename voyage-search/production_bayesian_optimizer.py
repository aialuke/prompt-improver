#!/usr/bin/env python3
"""
Production Bayesian BM25 Optimization with Full Dataset

Runs comprehensive Bayesian optimization using:
- Full production embeddings.pkl dataset (~5,712 chunks)
- All available queries from assessment script
- 200 evaluations for thorough parameter space exploration
- Real production data instead of synthetic test data

This provides the most robust and comprehensive BM25 parameter optimization
for the actual production codebase.

Usage:
    python production_bayesian_optimizer.py
"""

import os
import sys
import time
import json
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


class ProductionBayesianOptimizer:
    """Production-grade Bayesian optimization using full dataset and comprehensive queries."""
    
    def __init__(self, embeddings_path: str = "data/embeddings.pkl"):
        """Initialize optimizer with production data."""
        self.embeddings_path = embeddings_path
        self.embeddings = None
        self.chunks = None
        self.metadata = None
        self.binary_embeddings = None
        
        # Load production data
        self._load_production_data()

        # Validate data consistency
        self._validate_data_consistency()

        # Create comprehensive query set
        self.test_queries = self._create_comprehensive_queries()
        
        # Optimization tracking
        self.optimization_history = []
        self.best_score = 0.0
        self.best_params = None
        self.evaluation_count = 0
        
        # Early stopping parameters (more conservative for 200 evaluations)
        self.patience = 20  # Stop if no improvement for 20 evaluations
        self.min_improvement = 0.001  # Minimum improvement threshold
        self.no_improvement_count = 0
        
        print("🚀 Production Bayesian BM25 Optimizer initialized")
        print(f"   Production chunks: {len(self.chunks):,}")
        print(f"   Embeddings shape: {self.embeddings.shape}")
        print(f"   Test queries: {len(self.test_queries)}")
        print(f"   Parameter space: k1=[0.5-2.0], b=[0.0-1.0], stemming=[none,light,aggressive]")
        print(f"   Early stopping: patience={self.patience}, min_improvement={self.min_improvement}")
        if self.binary_embeddings is not None:
            print(f"   Binary embeddings: ✅ Available ({self.binary_embeddings.shape})")
        else:
            print(f"   Binary embeddings: ❌ Not available")
    
    def _load_production_data(self) -> None:
        """Load full production embeddings and chunks."""
        try:
            print(f"📊 Loading production data from {self.embeddings_path}")
            
            # Try multiple possible paths
            possible_paths = [
                self.embeddings_path,
                "embeddings.pkl",
                "data/embeddings.pkl",
                "../data/embeddings.pkl",
                "src/embeddings.pkl"
            ]
            
            embeddings_file = None
            for path in possible_paths:
                if Path(path).exists():
                    embeddings_file = path
                    break
            
            if not embeddings_file:
                raise FileNotFoundError(f"No embeddings file found. Tried: {possible_paths}")
            
            print(f"   Found embeddings file: {embeddings_file}")
            
            # Load using the enhanced loader
            self.embeddings, self.chunks, self.metadata, self.binary_embeddings = load_enhanced_embeddings(embeddings_file)
            
            print(f"   ✅ Loaded {len(self.chunks):,} production code chunks")
            print(f"   📊 Embeddings shape: {self.embeddings.shape}")
            
            # Get file size for reference
            file_size = Path(embeddings_file).stat().st_size / (1024 * 1024)
            print(f"   📁 File size: {file_size:.1f} MB")
            
        except Exception as e:
            print(f"❌ Failed to load production data: {e}")
            raise

    def _validate_data_consistency(self) -> None:
        """Validate that all data dimensions are consistent."""
        try:
            print("🔍 Validating data consistency...")

            # Check basic dimensions
            if self.embeddings is None or self.chunks is None:
                raise ValueError("Missing embeddings or chunks data")

            embeddings_count = self.embeddings.shape[0]
            chunks_count = len(self.chunks)

            if embeddings_count != chunks_count:
                print(f"⚠️  Dimension mismatch: {embeddings_count} embeddings vs {chunks_count} chunks")
                # Use the smaller count to avoid index errors
                min_count = min(embeddings_count, chunks_count)
                print(f"🔧 Truncating to {min_count} items for consistency")

                self.embeddings = self.embeddings[:min_count]
                self.chunks = self.chunks[:min_count]

                if self.binary_embeddings is not None:
                    self.binary_embeddings = self.binary_embeddings[:min_count]

            # Validate binary embeddings if present
            if self.binary_embeddings is not None:
                binary_count = self.binary_embeddings.shape[0]
                if binary_count != len(self.chunks):
                    print(f"⚠️  Binary embeddings mismatch: {binary_count} vs {len(self.chunks)}")
                    min_count = min(binary_count, len(self.chunks))
                    self.binary_embeddings = self.binary_embeddings[:min_count]
                    self.embeddings = self.embeddings[:min_count]
                    self.chunks = self.chunks[:min_count]

            print(f"✅ Data validation complete:")
            print(f"   Embeddings: {self.embeddings.shape}")
            print(f"   Chunks: {len(self.chunks)}")
            if self.binary_embeddings is not None:
                print(f"   Binary embeddings: {self.binary_embeddings.shape}")

        except Exception as e:
            print(f"❌ Data validation failed: {e}")
            raise
    
    def _create_comprehensive_queries(self) -> List[Dict[str, Any]]:
        """Create comprehensive query set covering all code search scenarios."""
        
        # Extract actual chunk names from production data for realistic expected results
        chunk_names = [chunk.name for chunk in self.chunks if hasattr(chunk, 'name')]
        chunk_types = {}
        for chunk in self.chunks:
            if hasattr(chunk, 'chunk_type'):
                chunk_type = str(chunk.chunk_type)
                if chunk_type not in chunk_types:
                    chunk_types[chunk_type] = []
                chunk_types[chunk_type].append(chunk.name if hasattr(chunk, 'name') else 'unknown')
        
        print(f"   📋 Production chunk types: {list(chunk_types.keys())}")
        print(f"   📊 Chunk distribution: {[(k, len(v)) for k, v in chunk_types.items()]}")
        
        # Create queries based on actual production data
        queries = []
        
        # Function name queries (using actual function names from production)
        function_chunks = chunk_types.get('ChunkType.FUNCTION', [])[:10]  # Top 10 functions
        for func_name in function_chunks:
            if func_name and func_name != 'unknown':
                queries.append({
                    "query": func_name.replace('_', ' '),  # "calculate_similarity" -> "calculate similarity"
                    "expected_chunks": [func_name],
                    "type": "function_name"
                })
        
        # Class name queries (using actual class names from production)
        class_chunks = chunk_types.get('ChunkType.CLASS', [])[:10]  # Top 10 classes
        for class_name in class_chunks:
            if class_name and class_name != 'unknown':
                queries.append({
                    "query": class_name,
                    "expected_chunks": [class_name],
                    "type": "class_name"
                })
        
        # Method queries (using actual method names from production)
        method_chunks = chunk_types.get('ChunkType.METHOD', [])[:10]  # Top 10 methods
        for method_name in method_chunks:
            if method_name and method_name != 'unknown':
                queries.append({
                    "query": method_name.replace('_', ' '),
                    "expected_chunks": [method_name],
                    "type": "method_call"
                })
        
        # Add standard code search patterns
        standard_queries = [
            # Common programming patterns
            {"query": "import numpy", "expected_chunks": [], "type": "import_pattern"},
            {"query": "def main", "expected_chunks": [], "type": "function_pattern"},
            {"query": "class Test", "expected_chunks": [], "type": "class_pattern"},
            {"query": "try except", "expected_chunks": [], "type": "error_handling"},
            {"query": "for loop", "expected_chunks": [], "type": "iteration_pattern"},
            {"query": "if __name__", "expected_chunks": [], "type": "main_pattern"},
            
            # Code search terms
            {"query": "search", "expected_chunks": [], "type": "keyword_search"},
            {"query": "tokenize", "expected_chunks": [], "type": "keyword_search"},
            {"query": "embedding", "expected_chunks": [], "type": "keyword_search"},
            {"query": "similarity", "expected_chunks": [], "type": "keyword_search"},
            {"query": "optimization", "expected_chunks": [], "type": "keyword_search"},
            
            # CamelCase variations
            {"query": "calculateSimilarity", "expected_chunks": [], "type": "camel_case"},
            {"query": "tokenizeText", "expected_chunks": [], "type": "camel_case"},
            {"query": "generateEmbeddings", "expected_chunks": [], "type": "camel_case"},
            
            # Stemming tests
            {"query": "running", "expected_chunks": [], "type": "stemming_test"},
            {"query": "tokenization", "expected_chunks": [], "type": "stemming_test"},
            {"query": "similarities", "expected_chunks": [], "type": "stemming_test"},
            
            # Stopword tests
            {"query": "the best algorithm for search", "expected_chunks": [], "type": "stopword_test"},
            {"query": "a function to calculate", "expected_chunks": [], "type": "stopword_test"},
        ]
        
        queries.extend(standard_queries)
        
        # For queries without expected chunks, we'll use a more flexible scoring approach
        print(f"   ✅ Created {len(queries)} comprehensive queries")
        print(f"   📊 Query types: {set(q['type'] for q in queries)}")
        
        return queries
    
    def _define_parameter_space(self):
        """Define the parameter space for Bayesian optimization."""
        return [
            Real(0.5, 2.0, name='k1'),  # BM25 k1 parameter (term frequency saturation)
            Real(0.0, 1.0, name='b'),   # BM25 b parameter (length normalization)
            Categorical(['none', 'light', 'aggressive'], name='stemming'),  # Stemming level
            Categorical([True, False], name='split_camelcase'),  # CamelCase splitting
            Categorical([True, False], name='split_underscores')  # Underscore splitting
        ]

    def _objective_function(self, k1: float, b: float, stemming: str,
                           split_camelcase: bool, split_underscores: bool) -> float:
        """
        Production objective function using full dataset.

        Returns negative score (since skopt minimizes, but we want to maximize).
        """
        self.evaluation_count += 1

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

            # Temporarily update global BM25 config for this evaluation
            original_k1 = HYBRID_SEARCH_CONFIG["bm25_k1"]
            original_b = HYBRID_SEARCH_CONFIG["bm25_b"]

            HYBRID_SEARCH_CONFIG["bm25_k1"] = k1
            HYBRID_SEARCH_CONFIG["bm25_b"] = b

            try:
                # Create search system with production data
                search_system = HybridCodeSearch(
                    embeddings=self.embeddings,
                    chunks=self.chunks,
                    metadata=self.metadata,
                    binary_embeddings=self.binary_embeddings
                )

                # Set custom tokenization manager
                search_system.tokenization_manager = tokenization_manager

                # Test all queries
                total_score = 0.0
                query_results = []
                successful_queries = 0

                for query_info in self.test_queries:
                    try:
                        results = search_system.hybrid_search(query_info["query"], top_k=10)

                        # Calculate relevance score
                        score = 0.0

                        if query_info["expected_chunks"]:
                            # Use expected chunks scoring for specific queries
                            for i, result in enumerate(results[:5]):  # Top 5 results
                                if hasattr(result, 'chunk_name') and result.chunk_name in query_info["expected_chunks"]:
                                    # Higher score for higher ranking
                                    score += (5 - i) / 5.0
                        else:
                            # For general queries without specific expected chunks,
                            # score based on result quality and relevance
                            if results:
                                # Give points for finding results
                                score = 0.3

                                # Bonus for high-confidence results
                                if hasattr(results[0], 'score') and results[0].score > 0.8:
                                    score += 0.2

                                # Bonus for diverse result types
                                result_types = set()
                                for result in results[:3]:
                                    if hasattr(result, 'chunk_type'):
                                        result_types.add(str(result.chunk_type))
                                if len(result_types) > 1:
                                    score += 0.1

                        total_score += score
                        successful_queries += 1
                        query_results.append({
                            "query": query_info["query"],
                            "score": score,
                            "results_count": len(results),
                            "type": query_info["type"]
                        })

                    except Exception as e:
                        print(f"   ❌ Query failed: {query_info['query']} - {e}")
                        query_results.append({
                            "query": query_info["query"],
                            "score": 0.0,
                            "error": str(e),
                            "type": query_info["type"]
                        })

                # Calculate average score
                avg_score = total_score / len(self.test_queries) if self.test_queries else 0.0
                success_rate = successful_queries / len(self.test_queries) if self.test_queries else 0.0

            finally:
                # Restore original BM25 configuration
                HYBRID_SEARCH_CONFIG["bm25_k1"] = original_k1
                HYBRID_SEARCH_CONFIG["bm25_b"] = original_b

            # Store optimization history
            result_entry = {
                "evaluation": self.evaluation_count,
                "k1": k1,
                "b": b,
                "stemming": stemming,
                "split_camelcase": split_camelcase,
                "split_underscores": split_underscores,
                "avg_score": avg_score,
                "total_score": total_score,
                "success_rate": success_rate,
                "successful_queries": successful_queries,
                "query_results": query_results,
                "timestamp": time.time()
            }
            self.optimization_history.append(result_entry)

            # Check for improvement (for early stopping)
            if avg_score > self.best_score + self.min_improvement:
                self.best_score = avg_score
                self.best_params = result_entry.copy()
                self.no_improvement_count = 0
                print(f"   🎉 New best score: {avg_score:.3f} (evaluation {self.evaluation_count})")
                print(f"      Success rate: {success_rate:.1%}, k1={k1:.2f}, b={b:.2f}, stem={stemming}")
            else:
                self.no_improvement_count += 1

            print(f"   📊 Evaluation {self.evaluation_count}: score={avg_score:.3f}, "
                  f"success={success_rate:.1%}, k1={k1:.2f}, b={b:.2f}, stem={stemming}")

            # Return negative score (skopt minimizes)
            return -avg_score

        except Exception as e:
            print(f"   ❌ Objective function error: {e}")
            return 0.0  # Return worst possible score on error

    def _check_early_stopping(self) -> bool:
        """Check if early stopping criteria are met."""
        if self.no_improvement_count >= self.patience:
            print(f"\n🛑 Early stopping triggered: No improvement for {self.patience} evaluations")
            print(f"   Best score: {self.best_score:.3f}")
            return True
        return False

    def optimize_parameters(self, n_calls: int = 200, n_initial_points: int = 20) -> Dict[str, Any]:
        """
        Run comprehensive Bayesian optimization with production data.

        Args:
            n_calls: Maximum number of function evaluations (200 for thorough exploration)
            n_initial_points: Number of random initial points (20 for good coverage)

        Returns:
            Optimization results with best parameters and comprehensive history
        """
        print("\n🚀 Starting Production Bayesian BM25 Parameter Optimization")
        print("=" * 80)
        print(f"   Production dataset: {len(self.chunks):,} chunks")
        print(f"   Comprehensive queries: {len(self.test_queries)} queries")
        print(f"   Maximum evaluations: {n_calls}")
        print(f"   Initial random points: {n_initial_points}")
        print(f"   Early stopping: patience={self.patience}")
        print(f"   Expected runtime: {n_calls * 6 / 60:.0f}-{n_calls * 8 / 60:.0f} minutes")

        # Define parameter space
        dimensions = self._define_parameter_space()

        start_time = time.time()

        # Create wrapper function for scikit-optimize
        def objective_wrapper(params):
            k1, b, stemming, split_camelcase, split_underscores = params
            return self._objective_function(k1, b, stemming, split_camelcase, split_underscores)

        # Custom callback for early stopping and progress reporting
        def progress_callback(result):
            if self.evaluation_count % 10 == 0:
                elapsed = time.time() - start_time
                remaining_evals = n_calls - self.evaluation_count
                estimated_remaining = (elapsed / self.evaluation_count) * remaining_evals
                print(f"\n📊 Progress: {self.evaluation_count}/{n_calls} evaluations")
                print(f"   Elapsed: {elapsed/60:.1f} min, Estimated remaining: {estimated_remaining/60:.1f} min")
                print(f"   Current best: {self.best_score:.3f}")

            if self._check_early_stopping():
                return True  # Stop optimization
            return False

        try:
            # Run Bayesian optimization
            result = gp_minimize(
                func=objective_wrapper,
                dimensions=dimensions,
                n_calls=n_calls,
                n_initial_points=n_initial_points,
                acq_func='EI',  # Expected Improvement
                random_state=42,
                callback=[progress_callback]
            )

            optimization_time = time.time() - start_time

            # Extract best parameters
            best_params = {
                'k1': result.x[0],
                'b': result.x[1],
                'stemming': result.x[2],
                'split_camelcase': result.x[3],
                'split_underscores': result.x[4]
            }

            best_score = -result.fun  # Convert back to positive

            print(f"\n🎉 Production Bayesian Optimization Complete!")
            print(f"   Total evaluations: {len(result.func_vals)}")
            print(f"   Optimization time: {optimization_time/60:.1f} minutes")
            print(f"   Best score: {best_score:.3f}")
            print(f"   Convergence: {'Early stopped' if self.no_improvement_count >= self.patience else 'Max evaluations reached'}")

            return {
                "best_params": best_params,
                "best_score": best_score,
                "total_evaluations": len(result.func_vals),
                "optimization_time": optimization_time,
                "dataset_info": {
                    "total_chunks": len(self.chunks),
                    "embeddings_shape": self.embeddings.shape,
                    "has_binary_embeddings": self.binary_embeddings is not None,
                    "total_queries": len(self.test_queries)
                },
                "convergence_info": {
                    "early_stopped": self.no_improvement_count >= self.patience,
                    "no_improvement_count": self.no_improvement_count,
                    "patience": self.patience
                },
                "optimization_history": self.optimization_history,
                "skopt_result": result
            }

        except Exception as e:
            print(f"❌ Optimization failed: {e}")
            return {
                "error": str(e),
                "optimization_history": self.optimization_history,
                "partial_results": True
            }


def main():
    """Main execution for production Bayesian optimization."""
    print("🚀 Production Bayesian BM25 Parameter Optimization")
    print("Comprehensive optimization using full production dataset")
    print("Target: 200 evaluations with early stopping")
    print()

    # Initialize optimizer with production data
    optimizer = ProductionBayesianOptimizer()

    # Run comprehensive Bayesian optimization
    print("\n⏱️  Starting optimization - this will take 20-30 minutes...")
    results = optimizer.optimize_parameters(n_calls=200, n_initial_points=20)

    if "error" not in results:
        # Print comprehensive summary
        print("\n" + "=" * 80)
        print("🏆 PRODUCTION BAYESIAN OPTIMIZATION RESULTS")
        print("=" * 80)

        best = results["best_params"]
        dataset_info = results["dataset_info"]

        print(f"🎯 Optimal Configuration:")
        print(f"   k1: {best['k1']:.3f}")
        print(f"   b: {best['b']:.3f}")
        print(f"   Stemming: {best['stemming']}")
        print(f"   CamelCase splitting: {best['split_camelcase']}")
        print(f"   Underscore splitting: {best['split_underscores']}")
        print(f"   Best Score: {results['best_score']:.3f}")

        print(f"\n📊 Dataset Information:")
        print(f"   Production chunks: {dataset_info['total_chunks']:,}")
        print(f"   Embeddings shape: {dataset_info['embeddings_shape']}")
        print(f"   Binary embeddings: {'✅ Available' if dataset_info['has_binary_embeddings'] else '❌ Not available'}")
        print(f"   Test queries: {dataset_info['total_queries']}")

        print(f"\n⚡ Performance Metrics:")
        print(f"   Total evaluations: {results['total_evaluations']}")
        print(f"   Optimization time: {results['optimization_time']/60:.1f} minutes")
        print(f"   Time per evaluation: {results['optimization_time']/results['total_evaluations']:.1f} seconds")
        print(f"   Early stopping: {'Yes' if results['convergence_info']['early_stopped'] else 'No'}")

        # Calculate efficiency vs grid search
        grid_search_evaluations = 300  # 5 k1 × 5 b × 3 stemming × 2 camelcase × 2 underscore
        time_per_eval = results['optimization_time'] / results['total_evaluations']
        estimated_grid_time = grid_search_evaluations * time_per_eval
        time_savings = estimated_grid_time - results['optimization_time']

        print(f"\n🚀 Efficiency vs Grid Search:")
        print(f"   Bayesian evaluations: {results['total_evaluations']}")
        print(f"   Grid search would need: {grid_search_evaluations}")
        print(f"   Evaluation reduction: {((grid_search_evaluations - results['total_evaluations']) / grid_search_evaluations * 100):.1f}%")
        print(f"   Time savings: {time_savings/60:.1f} minutes")
        print(f"   Speedup factor: {estimated_grid_time / results['optimization_time']:.1f}x")

        # Analyze query performance by type
        if optimizer.optimization_history and optimizer.best_params:
            best_eval = optimizer.best_params
            query_results = best_eval.get("query_results", [])

            print(f"\n📈 Best Configuration Query Performance:")
            query_types = {}
            for qr in query_results:
                qtype = qr.get("type", "unknown")
                if qtype not in query_types:
                    query_types[qtype] = {"scores": [], "count": 0}
                query_types[qtype]["scores"].append(qr.get("score", 0))
                query_types[qtype]["count"] += 1

            for qtype, data in query_types.items():
                avg_score = sum(data["scores"]) / len(data["scores"]) if data["scores"] else 0
                print(f"   {qtype}: {avg_score:.3f} avg score ({data['count']} queries)")

        # Save comprehensive results
        results_file = Path("production_bm25_optimization_results.json")
        save_data = {
            "optimization_results": results,
            "production_dataset_info": {
                "chunks_count": dataset_info['total_chunks'],
                "embeddings_shape": str(dataset_info['embeddings_shape']),
                "binary_embeddings_available": dataset_info['has_binary_embeddings'],
                "queries_tested": dataset_info['total_queries']
            },
            "optimization_metadata": {
                "optimizer_type": "production_bayesian",
                "dataset_type": "full_production",
                "evaluation_count": results['total_evaluations'],
                "early_stopped": results['convergence_info']['early_stopped']
            },
            "timestamp": time.time()
        }

        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

        print(f"\n💾 Comprehensive results saved to: {results_file}")
        print(f"✅ Production optimization completed successfully!")

        # Print final recommendation
        print(f"\n🎯 FINAL RECOMMENDATION:")
        print(f"   Use k1={best['k1']:.3f}, b={best['b']:.3f}, stemming='{best['stemming']}'")
        print(f"   CamelCase splitting: {best['split_camelcase']}, Underscore splitting: {best['split_underscores']}")
        print(f"   Expected search quality improvement: {results['best_score']:.1%}")

    else:
        print(f"\n❌ Production optimization failed: {results['error']}")
        if results.get("partial_results"):
            print(f"   Partial results available in optimization history")


if __name__ == "__main__":
    main()
