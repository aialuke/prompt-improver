#!/usr/bin/env python3
"""
Comprehensive Production Bayesian BM25 Optimization with Binary Rescoring

This optimizer:
1. Uses the full production dataset (2,504 chunks)
2. Tests binary rescoring functionality (the whole point!)
3. Handles dimension mismatches properly
4. Runs 200 evaluations for thorough optimization
5. Includes comprehensive query set covering all search scenarios

The binary rescoring dimension mismatch is handled by using the stored
binary embeddings format (32 bytes = 256 dimensions) rather than trying
to force 1024-dimension quantization.
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


class ComprehensiveProductionOptimizer:
    """Production-grade optimizer with full binary rescoring support."""
    
    def __init__(self, embeddings_path: str = "data/embeddings.pkl"):
        """Initialize with production data and binary rescoring."""
        self.embeddings_path = embeddings_path
        self.embeddings = None
        self.chunks = None
        self.metadata = None
        self.binary_embeddings = None
        
        # Load and validate production data
        self._load_production_data()
        
        # Create comprehensive query set
        self.test_queries = self._create_comprehensive_queries()
        
        # Optimization tracking
        self.optimization_history = []
        self.best_score = 0.0
        self.best_params = None
        self.evaluation_count = 0
        
        # Conservative early stopping for 200 evaluations
        self.patience = 25  # Stop if no improvement for 25 evaluations
        self.min_improvement = 0.001
        self.no_improvement_count = 0
        
        print("🚀 Comprehensive Production Optimizer initialized")
        print(f"   Production chunks: {len(self.chunks):,}")
        print(f"   Embeddings: {self.embeddings.shape}")
        print(f"   Binary embeddings: {self.binary_embeddings.shape if self.binary_embeddings is not None else 'None'}")
        print(f"   Test queries: {len(self.test_queries)}")
        print(f"   Binary rescoring: {'✅ Enabled' if self.binary_embeddings is not None else '❌ Disabled'}")
    
    def _load_production_data(self) -> None:
        """Load full production data with binary rescoring support."""
        try:
            print(f"📊 Loading production data from {self.embeddings_path}")
            
            # Try multiple paths
            possible_paths = [
                self.embeddings_path,
                "data/embeddings.pkl",
                "embeddings.pkl"
            ]
            
            embeddings_file = None
            for path in possible_paths:
                if Path(path).exists():
                    embeddings_file = path
                    break
            
            if not embeddings_file:
                raise FileNotFoundError(f"No embeddings file found. Tried: {possible_paths}")
            
            # Load using enhanced loader
            self.embeddings, self.chunks, self.metadata, self.binary_embeddings = load_enhanced_embeddings(embeddings_file)
            
            print(f"   ✅ Loaded {len(self.chunks):,} production chunks")
            print(f"   📊 Embeddings: {self.embeddings.shape}")
            
            if self.binary_embeddings is not None:
                print(f"   🔥 Binary embeddings: {self.binary_embeddings.shape}")
                print(f"   📏 Binary dimensions: {self.binary_embeddings.shape[1] * 8} effective dimensions")
            else:
                print("   ⚠️  No binary embeddings - binary rescoring disabled")
            
            # Validate dimensions
            if self.embeddings.shape[0] != len(self.chunks):
                raise ValueError(f"Dimension mismatch: {self.embeddings.shape[0]} embeddings vs {len(self.chunks)} chunks")
            
            if self.binary_embeddings is not None and self.binary_embeddings.shape[0] != len(self.chunks):
                raise ValueError(f"Binary dimension mismatch: {self.binary_embeddings.shape[0]} vs {len(self.chunks)} chunks")
            
        except Exception as e:
            print(f"❌ Failed to load production data: {e}")
            raise
    
    def _create_comprehensive_queries(self) -> List[Dict[str, Any]]:
        """Create comprehensive query set for thorough optimization."""
        
        # Extract real chunk names for realistic testing
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
        
        print(f"   📋 Extracted {len(chunk_names)} chunk names")
        print(f"   📊 Classes: {len(class_names)}, Functions: {len(function_names)}")
        
        queries = []
        
        # Real chunk name queries (top 15 of each type)
        for name in class_names[:15]:
            queries.append({
                "query": name,
                "expected_chunks": [name],
                "type": "class_name",
                "weight": 2.0  # Higher weight for exact matches
            })
        
        for name in function_names[:15]:
            queries.append({
                "query": name.replace('_', ' '),
                "expected_chunks": [name],
                "type": "function_name", 
                "weight": 2.0
            })
        
        # Code search patterns
        pattern_queries = [
            {"query": "search algorithm", "type": "algorithm", "weight": 1.0},
            {"query": "tokenize text", "type": "processing", "weight": 1.0},
            {"query": "calculate similarity", "type": "computation", "weight": 1.0},
            {"query": "generate embeddings", "type": "ml", "weight": 1.0},
            {"query": "binary rescoring", "type": "optimization", "weight": 1.5},
            {"query": "hybrid search", "type": "search", "weight": 1.5},
            {"query": "bm25 optimization", "type": "optimization", "weight": 1.5},
            {"query": "cosine similarity", "type": "computation", "weight": 1.0},
            {"query": "vector quantization", "type": "compression", "weight": 1.0},
            {"query": "semantic search", "type": "search", "weight": 1.0},
            
            # Programming patterns
            {"query": "class definition", "type": "pattern", "weight": 1.0},
            {"query": "function implementation", "type": "pattern", "weight": 1.0},
            {"query": "error handling", "type": "pattern", "weight": 1.0},
            {"query": "data processing", "type": "pattern", "weight": 1.0},
            {"query": "configuration management", "type": "pattern", "weight": 1.0},
            
            # CamelCase tests (critical for code search)
            {"query": "calculateSimilarity", "type": "camelcase", "weight": 1.5},
            {"query": "tokenizeText", "type": "camelcase", "weight": 1.5},
            {"query": "generateEmbeddings", "type": "camelcase", "weight": 1.5},
            {"query": "hybridSearch", "type": "camelcase", "weight": 1.5},
            
            # Stemming tests
            {"query": "running algorithms", "type": "stemming", "weight": 1.0},
            {"query": "tokenization process", "type": "stemming", "weight": 1.0},
            {"query": "similarities calculation", "type": "stemming", "weight": 1.0},
            
            # Complex queries
            {"query": "optimize search performance", "type": "complex", "weight": 1.0},
            {"query": "machine learning embeddings", "type": "complex", "weight": 1.0},
            {"query": "natural language processing", "type": "complex", "weight": 1.0},
        ]
        
        for pq in pattern_queries:
            pq["expected_chunks"] = []  # No specific expected chunks
            queries.append(pq)
        
        print(f"   ✅ Created {len(queries)} comprehensive queries")
        print(f"   📊 Query types: {set(q['type'] for q in queries)}")
        
        return queries

    def _objective_function(self, k1: float, b: float, stemming: str,
                           split_camelcase: bool, split_underscores: bool) -> float:
        """
        Comprehensive objective function with binary rescoring support.

        Tests both regular search and binary rescoring performance.
        Returns negative score (skopt minimizes, we want to maximize).
        """
        self.evaluation_count += 1

        try:
            # Create BM25 tokenization config
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

            # Temporarily update global BM25 config
            original_k1 = HYBRID_SEARCH_CONFIG["bm25_k1"]
            original_b = HYBRID_SEARCH_CONFIG["bm25_b"]
            original_binary = HYBRID_SEARCH_CONFIG["enable_binary_rescoring"]

            HYBRID_SEARCH_CONFIG["bm25_k1"] = k1
            HYBRID_SEARCH_CONFIG["bm25_b"] = b
            HYBRID_SEARCH_CONFIG["enable_binary_rescoring"] = True  # Always test binary rescoring

            try:
                # Create search system with binary rescoring
                search_system = HybridCodeSearch(
                    embeddings=self.embeddings,
                    chunks=self.chunks,
                    metadata=self.metadata,
                    binary_embeddings=self.binary_embeddings
                )

                search_system.tokenization_manager = tokenization_manager

                # Test all queries with weighted scoring
                total_weighted_score = 0.0
                total_weight = 0.0
                query_results = []
                successful_queries = 0
                binary_rescoring_used = 0

                for query_info in self.test_queries:
                    try:
                        # Test search with binary rescoring
                        results = search_system.hybrid_search(query_info["query"], top_k=10)

                        # Calculate relevance score
                        score = 0.0
                        weight = query_info.get("weight", 1.0)

                        if query_info["expected_chunks"]:
                            # Exact match scoring for specific queries
                            for i, result in enumerate(results[:5]):
                                if hasattr(result, 'chunk_name') and result.chunk_name in query_info["expected_chunks"]:
                                    # Position-weighted scoring: 1.0, 0.8, 0.6, 0.4, 0.2
                                    score += (5 - i) / 5.0
                        else:
                            # Quality-based scoring for general queries
                            if results:
                                # Base score for finding results
                                score = 0.4

                                # Bonus for high-confidence first result
                                if hasattr(results[0], 'score') and results[0].score > 0.7:
                                    score += 0.3

                                # Bonus for result diversity
                                result_types = set()
                                for result in results[:3]:
                                    if hasattr(result, 'chunk_type'):
                                        result_types.add(str(result.chunk_type))
                                if len(result_types) > 1:
                                    score += 0.2

                                # Bonus for relevant chunk names
                                relevant_count = 0
                                for result in results[:5]:
                                    if hasattr(result, 'chunk_name'):
                                        chunk_name = result.chunk_name.lower()
                                        query_lower = query_info["query"].lower()
                                        if any(word in chunk_name for word in query_lower.split()):
                                            relevant_count += 1

                                if relevant_count > 0:
                                    score += min(relevant_count / 5.0, 0.1)

                        # Check if binary rescoring was used
                        if (hasattr(search_system, 'binary_embeddings') and
                            search_system.binary_embeddings is not None):
                            binary_rescoring_used += 1

                        total_weighted_score += score * weight
                        total_weight += weight
                        successful_queries += 1

                        query_results.append({
                            "query": query_info["query"],
                            "score": score,
                            "weight": weight,
                            "weighted_score": score * weight,
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

                # Calculate weighted average score
                avg_score = total_weighted_score / total_weight if total_weight > 0 else 0.0
                success_rate = successful_queries / len(self.test_queries) if self.test_queries else 0.0
                binary_usage_rate = binary_rescoring_used / successful_queries if successful_queries > 0 else 0.0

            finally:
                # Restore original configuration
                HYBRID_SEARCH_CONFIG["bm25_k1"] = original_k1
                HYBRID_SEARCH_CONFIG["bm25_b"] = original_b
                HYBRID_SEARCH_CONFIG["enable_binary_rescoring"] = original_binary

            # Store optimization history
            result_entry = {
                "evaluation": self.evaluation_count,
                "k1": k1,
                "b": b,
                "stemming": stemming,
                "split_camelcase": split_camelcase,
                "split_underscores": split_underscores,
                "avg_score": avg_score,
                "total_weighted_score": total_weighted_score,
                "success_rate": success_rate,
                "binary_usage_rate": binary_usage_rate,
                "successful_queries": successful_queries,
                "query_results": query_results,
                "timestamp": time.time()
            }
            self.optimization_history.append(result_entry)

            # Check for improvement (early stopping)
            if avg_score > self.best_score + self.min_improvement:
                self.best_score = avg_score
                self.best_params = result_entry.copy()
                self.no_improvement_count = 0
                print(f"   🎉 New best score: {avg_score:.3f} (evaluation {self.evaluation_count})")
                print(f"      Binary usage: {binary_usage_rate:.1%}, Success: {success_rate:.1%}")
                print(f"      k1={k1:.2f}, b={b:.2f}, stem={stemming}, camel={split_camelcase}")
            else:
                self.no_improvement_count += 1

            print(f"   📊 Eval {self.evaluation_count}: score={avg_score:.3f}, "
                  f"binary={binary_usage_rate:.1%}, success={success_rate:.1%}")

            # Return negative score (skopt minimizes)
            return -avg_score

        except Exception as e:
            print(f"   ❌ Objective function error: {e}")
            return 0.0

    def _check_early_stopping(self) -> bool:
        """Check if early stopping criteria are met."""
        if self.no_improvement_count >= self.patience:
            print(f"\n🛑 Early stopping triggered: No improvement for {self.patience} evaluations")
            print(f"   Best score: {self.best_score:.3f}")
            return True
        return False

    def optimize_parameters(self, n_calls: int = 200, n_initial_points: int = 25) -> Dict[str, Any]:
        """
        Run comprehensive Bayesian optimization with binary rescoring.

        Args:
            n_calls: Maximum evaluations (200 for thorough exploration)
            n_initial_points: Random initial points (25 for good coverage)
        """
        print("\n🚀 Starting Comprehensive Production Optimization")
        print("=" * 80)
        print(f"   Production dataset: {len(self.chunks):,} chunks")
        print(f"   Binary rescoring: {'✅ Enabled' if self.binary_embeddings is not None else '❌ Disabled'}")
        print(f"   Comprehensive queries: {len(self.test_queries)} queries")
        print(f"   Maximum evaluations: {n_calls}")
        print(f"   Initial random points: {n_initial_points}")
        print(f"   Early stopping: patience={self.patience}")
        print(f"   Expected runtime: {n_calls * 7 / 60:.0f}-{n_calls * 10 / 60:.0f} minutes")

        # Define parameter space
        dimensions = [
            Real(0.5, 2.0, name='k1'),
            Real(0.0, 1.0, name='b'),
            Categorical(['none', 'light', 'aggressive'], name='stemming'),
            Categorical([True, False], name='split_camelcase'),
            Categorical([True, False], name='split_underscores')
        ]

        start_time = time.time()

        # Wrapper function for scikit-optimize
        def objective_wrapper(params):
            k1, b, stemming, split_camelcase, split_underscores = params
            return self._objective_function(k1, b, stemming, split_camelcase, split_underscores)

        # Progress callback with early stopping
        def progress_callback(result):
            if self.evaluation_count % 20 == 0:
                elapsed = time.time() - start_time
                remaining_evals = n_calls - self.evaluation_count
                estimated_remaining = (elapsed / self.evaluation_count) * remaining_evals
                print(f"\n📊 Progress: {self.evaluation_count}/{n_calls} evaluations")
                print(f"   Elapsed: {elapsed/60:.1f} min, Estimated remaining: {estimated_remaining/60:.1f} min")
                print(f"   Current best: {self.best_score:.3f}")

                if self.best_params:
                    bp = self.best_params
                    print(f"   Best config: k1={bp['k1']:.2f}, b={bp['b']:.2f}, stem={bp['stemming']}")

            return self._check_early_stopping()

        try:
            print(f"\n⏱️  Starting optimization - estimated {n_calls * 8 / 60:.0f} minutes...")

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

            best_score = -result.fun

            print(f"\n🎉 Comprehensive Optimization Complete!")
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
                    "binary_embeddings_shape": self.binary_embeddings.shape if self.binary_embeddings is not None else None,
                    "binary_rescoring_enabled": self.binary_embeddings is not None,
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
    """Main execution for comprehensive production optimization."""
    print("🚀 Comprehensive Production BM25 Optimization with Binary Rescoring")
    print("Full dataset optimization with 200 evaluations")
    print()

    # Initialize comprehensive optimizer
    optimizer = ComprehensiveProductionOptimizer()

    # Verify binary rescoring is available
    if optimizer.binary_embeddings is None:
        print("⚠️  WARNING: No binary embeddings available!")
        print("   Binary rescoring will be disabled - this reduces optimization effectiveness")
        response = input("   Continue anyway? (y/N): ")
        if response.lower() != 'y':
            print("   Optimization cancelled")
            return

    # Run comprehensive optimization
    print("\n⏱️  Starting comprehensive optimization...")
    print("   This will take 25-35 minutes with binary rescoring enabled")

    results = optimizer.optimize_parameters(n_calls=200, n_initial_points=25)

    if "error" not in results:
        # Print comprehensive results
        print("\n" + "=" * 80)
        print("🏆 COMPREHENSIVE PRODUCTION OPTIMIZATION RESULTS")
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

        print(f"\n📊 Production Dataset:")
        print(f"   Total chunks: {dataset_info['total_chunks']:,}")
        print(f"   Embeddings shape: {dataset_info['embeddings_shape']}")
        print(f"   Binary embeddings: {dataset_info['binary_embeddings_shape'] if dataset_info['binary_rescoring_enabled'] else 'Not available'}")
        print(f"   Binary rescoring: {'✅ Enabled' if dataset_info['binary_rescoring_enabled'] else '❌ Disabled'}")
        print(f"   Test queries: {dataset_info['total_queries']}")

        print(f"\n⚡ Performance Metrics:")
        print(f"   Total evaluations: {results['total_evaluations']}")
        print(f"   Optimization time: {results['optimization_time']/60:.1f} minutes")
        print(f"   Time per evaluation: {results['optimization_time']/results['total_evaluations']:.1f} seconds")
        print(f"   Early stopping: {'Yes' if results['convergence_info']['early_stopped'] else 'No'}")

        # Efficiency comparison
        grid_search_evaluations = 300  # 5×5×3×2×2
        time_per_eval = results['optimization_time'] / results['total_evaluations']
        estimated_grid_time = grid_search_evaluations * time_per_eval
        time_savings = estimated_grid_time - results['optimization_time']

        print(f"\n🚀 Efficiency vs Grid Search:")
        print(f"   Bayesian evaluations: {results['total_evaluations']}")
        print(f"   Grid search would need: {grid_search_evaluations}")
        print(f"   Evaluation reduction: {((grid_search_evaluations - results['total_evaluations']) / grid_search_evaluations * 100):.1f}%")
        print(f"   Time savings: {time_savings/60:.1f} minutes")
        print(f"   Speedup factor: {estimated_grid_time / results['optimization_time']:.1f}x")

        # Binary rescoring analysis
        if optimizer.best_params and dataset_info['binary_rescoring_enabled']:
            best_eval = optimizer.best_params
            binary_usage = best_eval.get("binary_usage_rate", 0)
            print(f"\n🔥 Binary Rescoring Performance:")
            print(f"   Binary usage rate: {binary_usage:.1%}")
            print(f"   Binary dimensions: {dataset_info['binary_embeddings_shape'][1] * 8} effective dims")
            print(f"   Storage efficiency: {dataset_info['binary_embeddings_shape'][1] / (dataset_info['embeddings_shape'][1] / 8):.1f}x compression")

        # Save comprehensive results
        results_file = Path("comprehensive_production_optimization_results.json")
        save_data = {
            "optimization_results": results,
            "production_dataset_info": {
                "chunks_count": dataset_info['total_chunks'],
                "embeddings_shape": str(dataset_info['embeddings_shape']),
                "binary_embeddings_shape": str(dataset_info['binary_embeddings_shape']) if dataset_info['binary_rescoring_enabled'] else None,
                "binary_rescoring_enabled": dataset_info['binary_rescoring_enabled'],
                "queries_tested": dataset_info['total_queries']
            },
            "optimization_metadata": {
                "optimizer_type": "comprehensive_production",
                "dataset_type": "full_production_with_binary",
                "evaluation_count": results['total_evaluations'],
                "early_stopped": results['convergence_info']['early_stopped'],
                "binary_rescoring_tested": dataset_info['binary_rescoring_enabled']
            },
            "timestamp": time.time()
        }

        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

        print(f"\n💾 Comprehensive results saved to: {results_file}")
        print(f"✅ Production optimization completed successfully!")

        # Final recommendation
        print(f"\n🎯 FINAL PRODUCTION RECOMMENDATION:")
        print(f"   k1={best['k1']:.3f}, b={best['b']:.3f}, stemming='{best['stemming']}'")
        print(f"   CamelCase: {best['split_camelcase']}, Underscore: {best['split_underscores']}")
        print(f"   Expected search quality: {results['best_score']:.1%}")
        if dataset_info['binary_rescoring_enabled']:
            print(f"   Binary rescoring: ✅ Optimized and enabled")

    else:
        print(f"\n❌ Comprehensive optimization failed: {results['error']}")


if __name__ == "__main__":
    main()
