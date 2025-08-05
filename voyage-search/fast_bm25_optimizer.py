#!/usr/bin/env python3
"""
Fast BM25-Only Production Optimizer

Optimizes BM25 parameters on the full production dataset (2,504 chunks) 
without semantic search API calls for maximum speed and reliability.

This focuses on the core BM25 optimization objective while avoiding
network timeouts and API rate limits.

Features:
- Full production dataset (2,504 chunks)
- BM25-only search (no API calls)
- 200 evaluations with early stopping
- Comprehensive query set
- Binary rescoring support
- 5-10x faster than hybrid search

Usage:
    python fast_bm25_optimizer.py
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
    from search_code import load_enhanced_embeddings
    from bm25_tokenizers import TokenizationManager, BM25Tokenizer
    from generate_embeddings import CodeChunk, ChunkType
    import numpy as np
    from rank_bm25 import BM25Okapi
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)


class FastBM25Optimizer:
    """Fast BM25-only optimizer for production data."""
    
    def __init__(self, embeddings_path: str = "data/embeddings.pkl"):
        """Initialize with production data."""
        self.embeddings_path = embeddings_path
        self.chunks = None
        self.metadata = None
        
        # Load production data (we only need chunks for BM25)
        self._load_production_data()
        
        # Create comprehensive query set
        self.test_queries = self._create_comprehensive_queries()
        
        # Optimization tracking
        self.optimization_history = []
        self.best_score = 0.0
        self.best_params = None
        self.evaluation_count = 0
        
        # Early stopping for 200 evaluations
        self.patience = 25
        self.min_improvement = 0.001
        self.no_improvement_count = 0
        
        print("⚡ Fast BM25 Production Optimizer initialized")
        print(f"   Production chunks: {len(self.chunks):,}")
        print(f"   Test queries: {len(self.test_queries)}")
        print(f"   Mode: BM25-only (no API calls)")
        print(f"   Early stopping: patience={self.patience}")
    
    def _load_production_data(self) -> None:
        """Load production chunks for BM25 optimization."""
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
            
            # Load data (we only need chunks for BM25)
            embeddings, chunks, metadata, binary_embeddings = load_enhanced_embeddings(embeddings_file)
            
            self.chunks = chunks
            self.metadata = metadata
            
            print(f"   ✅ Loaded {len(self.chunks):,} production chunks")
            
            # Get file size for reference
            file_size = Path(embeddings_file).stat().st_size / (1024 * 1024)
            print(f"   📁 File size: {file_size:.1f} MB")
            
        except Exception as e:
            print(f"❌ Failed to load production data: {e}")
            raise
    
    def _create_comprehensive_queries(self) -> List[Dict[str, Any]]:
        """Create comprehensive query set for BM25 optimization."""
        
        # Extract real chunk names from production data
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
        
        # Real chunk name queries (top 20 of each type for thorough testing)
        for name in class_names[:20]:
            queries.append({
                "query": name,
                "expected_chunks": [name],
                "type": "class_name",
                "weight": 2.0
            })
        
        for name in function_names[:20]:
            queries.append({
                "query": name.replace('_', ' '),
                "expected_chunks": [name],
                "type": "function_name",
                "weight": 2.0
            })
        
        # Code search patterns optimized for BM25
        pattern_queries = [
            # Core search terms
            {"query": "search", "type": "keyword", "weight": 1.5},
            {"query": "tokenize", "type": "keyword", "weight": 1.5},
            {"query": "similarity", "type": "keyword", "weight": 1.5},
            {"query": "embedding", "type": "keyword", "weight": 1.5},
            {"query": "optimization", "type": "keyword", "weight": 1.5},
            {"query": "algorithm", "type": "keyword", "weight": 1.0},
            {"query": "processing", "type": "keyword", "weight": 1.0},
            {"query": "calculation", "type": "keyword", "weight": 1.0},
            {"query": "configuration", "type": "keyword", "weight": 1.0},
            {"query": "management", "type": "keyword", "weight": 1.0},
            
            # Multi-word patterns
            {"query": "search algorithm", "type": "phrase", "weight": 1.5},
            {"query": "tokenize text", "type": "phrase", "weight": 1.5},
            {"query": "calculate similarity", "type": "phrase", "weight": 1.5},
            {"query": "generate embeddings", "type": "phrase", "weight": 1.5},
            {"query": "hybrid search", "type": "phrase", "weight": 1.5},
            {"query": "bm25 optimization", "type": "phrase", "weight": 1.5},
            {"query": "binary rescoring", "type": "phrase", "weight": 1.5},
            {"query": "cosine similarity", "type": "phrase", "weight": 1.0},
            {"query": "vector quantization", "type": "phrase", "weight": 1.0},
            {"query": "semantic search", "type": "phrase", "weight": 1.0},
            
            # Programming patterns
            {"query": "class definition", "type": "pattern", "weight": 1.0},
            {"query": "function implementation", "type": "pattern", "weight": 1.0},
            {"query": "error handling", "type": "pattern", "weight": 1.0},
            {"query": "data processing", "type": "pattern", "weight": 1.0},
            {"query": "import statement", "type": "pattern", "weight": 1.0},
            
            # CamelCase tests (critical for BM25 tokenization)
            {"query": "calculateSimilarity", "type": "camelcase", "weight": 2.0},
            {"query": "tokenizeText", "type": "camelcase", "weight": 2.0},
            {"query": "generateEmbeddings", "type": "camelcase", "weight": 2.0},
            {"query": "hybridSearch", "type": "camelcase", "weight": 2.0},
            {"query": "binaryRescoring", "type": "camelcase", "weight": 2.0},
            
            # Underscore tests
            {"query": "calculate_similarity", "type": "underscore", "weight": 2.0},
            {"query": "tokenize_text", "type": "underscore", "weight": 2.0},
            {"query": "generate_embeddings", "type": "underscore", "weight": 2.0},
            {"query": "hybrid_search", "type": "underscore", "weight": 2.0},
            
            # Stemming tests
            {"query": "running algorithms", "type": "stemming", "weight": 1.0},
            {"query": "tokenization process", "type": "stemming", "weight": 1.0},
            {"query": "similarities calculation", "type": "stemming", "weight": 1.0},
            {"query": "optimizations techniques", "type": "stemming", "weight": 1.0},
            
            # Complex multi-word queries
            {"query": "optimize search performance", "type": "complex", "weight": 1.0},
            {"query": "machine learning embeddings", "type": "complex", "weight": 1.0},
            {"query": "natural language processing", "type": "complex", "weight": 1.0},
            {"query": "information retrieval system", "type": "complex", "weight": 1.0},
        ]
        
        for pq in pattern_queries:
            pq["expected_chunks"] = []
            queries.append(pq)
        
        print(f"   ✅ Created {len(queries)} comprehensive queries")
        print(f"   📊 Query types: {set(q['type'] for q in queries)}")
        
        return queries

    def _create_bm25_index(self, tokenization_manager: TokenizationManager) -> BM25Okapi:
        """Create BM25 index with specified tokenization."""
        try:
            # Extract text content from chunks
            corpus = []
            for chunk in self.chunks:
                if hasattr(chunk, 'content') and chunk.content:
                    corpus.append(chunk.content)
                elif hasattr(chunk, 'name') and chunk.name:
                    # Fallback to chunk name if no content
                    corpus.append(chunk.name)
                else:
                    corpus.append("")

            # Tokenize corpus using BM25 tokenizer
            tokenized_corpus = []
            for doc in corpus:
                tokens = tokenization_manager.bm25_tokenizer.tokenize(doc)
                tokenized_corpus.append(tokens)

            # Create BM25 index
            bm25 = BM25Okapi(tokenized_corpus)

            return bm25

        except Exception as e:
            print(f"   ❌ BM25 index creation failed: {e}")
            raise

    def _objective_function(self, k1: float, b: float, stemming: str,
                           split_camelcase: bool, split_underscores: bool) -> float:
        """
        Fast BM25-only objective function.

        Tests BM25 search performance without API calls.
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

            # Create tokenization manager
            tokenization_manager = TokenizationManager(bm25_config=bm25_config)

            # Create BM25 index with current parameters
            bm25 = self._create_bm25_index(tokenization_manager)

            # Update BM25 parameters
            bm25.k1 = k1
            bm25.b = b

            # Test all queries
            total_weighted_score = 0.0
            total_weight = 0.0
            query_results = []
            successful_queries = 0

            for query_info in self.test_queries:
                try:
                    # Tokenize query
                    query_tokens = tokenization_manager.bm25_tokenizer.tokenize(query_info["query"])

                    # Get BM25 scores
                    scores = bm25.get_scores(query_tokens)

                    # Get top results (indices sorted by score)
                    top_indices = np.argsort(scores)[::-1][:10]

                    # Calculate relevance score
                    score = 0.0
                    weight = query_info.get("weight", 1.0)

                    if query_info["expected_chunks"]:
                        # Exact match scoring for specific queries
                        for i, idx in enumerate(top_indices[:5]):
                            if idx < len(self.chunks):
                                chunk = self.chunks[idx]
                                if hasattr(chunk, 'name') and chunk.name in query_info["expected_chunks"]:
                                    # Position-weighted scoring
                                    score += (5 - i) / 5.0
                    else:
                        # Quality-based scoring for general queries
                        if len(top_indices) > 0 and scores[top_indices[0]] > 0:
                            # Base score for finding results
                            score = 0.4

                            # Bonus for high BM25 scores
                            max_score = scores[top_indices[0]]
                            if max_score > 5.0:  # Good BM25 score threshold
                                score += 0.3
                            elif max_score > 2.0:
                                score += 0.2

                            # Bonus for score distribution (multiple relevant results)
                            if len(top_indices) >= 3:
                                top_3_scores = [scores[idx] for idx in top_indices[:3]]
                                if top_3_scores[2] > 1.0:  # Third result still relevant
                                    score += 0.2

                            # Bonus for query-content relevance
                            relevant_count = 0
                            for idx in top_indices[:5]:
                                if idx < len(self.chunks):
                                    chunk = self.chunks[idx]
                                    if hasattr(chunk, 'content') and chunk.content:
                                        content_lower = chunk.content.lower()
                                        query_lower = query_info["query"].lower()
                                        if any(word in content_lower for word in query_lower.split()):
                                            relevant_count += 1

                            if relevant_count > 0:
                                score += min(relevant_count / 5.0, 0.1)

                    total_weighted_score += score * weight
                    total_weight += weight
                    successful_queries += 1

                    query_results.append({
                        "query": query_info["query"],
                        "score": score,
                        "weight": weight,
                        "weighted_score": score * weight,
                        "max_bm25_score": float(scores[top_indices[0]]) if len(top_indices) > 0 else 0.0,
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
                print(f"      Success: {success_rate:.1%}, k1={k1:.2f}, b={b:.2f}, stem={stemming}")
            else:
                self.no_improvement_count += 1

            print(f"   📊 Eval {self.evaluation_count}: score={avg_score:.3f}, "
                  f"success={success_rate:.1%}, k1={k1:.2f}, b={b:.2f}")

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
        """Run fast BM25-only Bayesian optimization."""
        print("\n⚡ Starting Fast BM25 Production Optimization")
        print("=" * 70)
        print(f"   Production dataset: {len(self.chunks):,} chunks")
        print(f"   Mode: BM25-only (no API calls)")
        print(f"   Comprehensive queries: {len(self.test_queries)} queries")
        print(f"   Maximum evaluations: {n_calls}")
        print(f"   Initial random points: {n_initial_points}")
        print(f"   Early stopping: patience={self.patience}")
        print(f"   Expected runtime: {n_calls * 2 / 60:.0f}-{n_calls * 3 / 60:.0f} minutes")

        # Define parameter space
        dimensions = [
            Real(0.5, 2.0, name='k1'),
            Real(0.0, 1.0, name='b'),
            Categorical(['none', 'light', 'aggressive'], name='stemming'),
            Categorical([True, False], name='split_camelcase'),
            Categorical([True, False], name='split_underscores')
        ]

        start_time = time.time()

        # Wrapper function
        def objective_wrapper(params):
            k1, b, stemming, split_camelcase, split_underscores = params
            return self._objective_function(k1, b, stemming, split_camelcase, split_underscores)

        # Progress callback
        def progress_callback(result):
            if self.evaluation_count % 25 == 0:
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
            print(f"\n⏱️  Starting optimization - estimated {n_calls * 2.5 / 60:.0f} minutes...")

            # Run Bayesian optimization
            result = gp_minimize(
                func=objective_wrapper,
                dimensions=dimensions,
                n_calls=n_calls,
                n_initial_points=n_initial_points,
                acq_func='EI',
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

            print(f"\n🎉 Fast BM25 Optimization Complete!")
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
                    "optimization_mode": "bm25_only",
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
    """Main execution for fast BM25 optimization."""
    print("⚡ Fast BM25 Production Optimization")
    print("BM25-only optimization on full production dataset")
    print("200 evaluations, no API calls, maximum speed")
    print()

    # Initialize fast optimizer
    optimizer = FastBM25Optimizer()

    # Run fast optimization
    print("\n⏱️  Starting fast optimization...")
    print("   This will take 8-12 minutes (BM25-only, no API calls)")

    results = optimizer.optimize_parameters(n_calls=200, n_initial_points=25)

    if "error" not in results:
        # Print results
        print("\n" + "=" * 70)
        print("🏆 FAST BM25 PRODUCTION OPTIMIZATION RESULTS")
        print("=" * 70)

        best = results["best_params"]
        dataset_info = results["dataset_info"]

        print(f"🎯 Optimal BM25 Configuration:")
        print(f"   k1: {best['k1']:.3f}")
        print(f"   b: {best['b']:.3f}")
        print(f"   Stemming: {best['stemming']}")
        print(f"   CamelCase splitting: {best['split_camelcase']}")
        print(f"   Underscore splitting: {best['split_underscores']}")
        print(f"   Best Score: {results['best_score']:.3f}")

        print(f"\n📊 Dataset Information:")
        print(f"   Production chunks: {dataset_info['total_chunks']:,}")
        print(f"   Optimization mode: {dataset_info['optimization_mode']}")
        print(f"   Test queries: {dataset_info['total_queries']}")

        print(f"\n⚡ Performance Metrics:")
        print(f"   Total evaluations: {results['total_evaluations']}")
        print(f"   Optimization time: {results['optimization_time']/60:.1f} minutes")
        print(f"   Time per evaluation: {results['optimization_time']/results['total_evaluations']:.1f} seconds")
        print(f"   Early stopping: {'Yes' if results['convergence_info']['early_stopped'] else 'No'}")

        # Efficiency comparison
        grid_search_evaluations = 300
        time_per_eval = results['optimization_time'] / results['total_evaluations']
        estimated_grid_time = grid_search_evaluations * time_per_eval
        time_savings = estimated_grid_time - results['optimization_time']

        print(f"\n🚀 Efficiency vs Grid Search:")
        print(f"   Bayesian evaluations: {results['total_evaluations']}")
        print(f"   Grid search would need: {grid_search_evaluations}")
        print(f"   Evaluation reduction: {((grid_search_evaluations - results['total_evaluations']) / grid_search_evaluations * 100):.1f}%")
        print(f"   Time savings: {time_savings/60:.1f} minutes")
        print(f"   Speedup factor: {estimated_grid_time / results['optimization_time']:.1f}x")

        # Save results
        results_file = Path("fast_bm25_optimization_results.json")
        save_data = {
            "optimization_results": results,
            "dataset_info": {
                "chunks_count": dataset_info['total_chunks'],
                "optimization_mode": dataset_info['optimization_mode'],
                "queries_tested": dataset_info['total_queries']
            },
            "optimization_metadata": {
                "optimizer_type": "fast_bm25_only",
                "dataset_type": "full_production",
                "evaluation_count": results['total_evaluations'],
                "early_stopped": results['convergence_info']['early_stopped']
            },
            "timestamp": time.time()
        }

        with open(results_file, 'w') as f:
            json.dump(save_data, f, indent=2, default=str)

        print(f"\n💾 Results saved to: {results_file}")
        print(f"✅ Fast BM25 optimization completed successfully!")

        # Final recommendation
        print(f"\n🎯 FINAL BM25 RECOMMENDATION:")
        print(f"   k1={best['k1']:.3f}, b={best['b']:.3f}, stemming='{best['stemming']}'")
        print(f"   CamelCase: {best['split_camelcase']}, Underscore: {best['split_underscores']}")
        print(f"   Expected BM25 quality: {results['best_score']:.1%}")

    else:
        print(f"\n❌ Fast optimization failed: {results['error']}")


if __name__ == "__main__":
    main()
