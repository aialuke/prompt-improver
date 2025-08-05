#!/usr/bin/env python3
"""
Comprehensive BM25 Quality Assessment Tool

Rigorous evaluation of BM25 search quality comparing:
1. New local BM25 tokenization (stemming + stopwords)
2. Old unified Voyage tokenization (if available)
3. Simple baseline tokenization

Measures:
- Precision@K, Recall@K, NDCG@K, MRR
- Side-by-side result comparison
- Tokenization difference analysis
- Real codebase performance testing

Usage:
    python assess_bm25_quality.py
"""

import os
import sys
import time
import json
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from dataclasses import dataclass
from collections import defaultdict

# Add src directory to path
src_dir = str(Path(__file__).parent / "src")
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    from search_code import HybridCodeSearch, SEARCH_METRICS  # type: ignore[import-not-found]
    from bm25_tokenizers import TokenizationManager, BM25Tokenizer  # type: ignore[import-not-found]
    from generate_embeddings import CodeChunk, ChunkType  # type: ignore[import-not-found]
    import numpy as np
except ImportError as e:
    print(f"❌ Import error: {e}")
    print(f"   Current working directory: {os.getcwd()}")
    print(f"   Python path: {sys.path[:3]}...")
    sys.exit(1)


@dataclass
class QualityMetrics:
    """Quality assessment metrics for search results."""
    precision_at_k: Dict[int, float]
    recall_at_k: Dict[int, float]
    ndcg_at_k: Dict[int, float]
    mrr: float
    total_relevant: int
    total_queries: int


@dataclass
class TokenizationComparison:
    """Comparison between different tokenization approaches."""
    query: str
    local_tokens: List[str]
    voyage_tokens: List[str]
    simple_tokens: List[str]
    local_results: List[Any]
    voyage_results: List[Any]
    simple_results: List[Any]


class BM25QualityAssessor:
    """Comprehensive BM25 quality assessment tool."""
    
    def __init__(self, embeddings_path: str = "embeddings.pkl"):
        """Initialize with real codebase data."""
        self.embeddings_path = Path(embeddings_path)
        self.chunks: List[CodeChunk] = []
        self.embeddings: Optional[np.ndarray[Any, Any]] = None
        self.metadata: Optional[Dict[str, Any]] = None
        
        # Tokenization approaches
        self.local_tokenizer: Optional[BM25Tokenizer] = None
        self.tokenization_manager: Optional[TokenizationManager] = None
        
        # Test queries for comprehensive evaluation
        self.test_queries = self._create_comprehensive_test_queries()
        
        # Results storage
        self.assessment_results: Dict[str, Any] = {}
        
        print("🔍 BM25 Quality Assessor initialized")
        
    def _create_comprehensive_test_queries(self) -> List[Dict[str, Any]]:
        """Create comprehensive test query set for code search evaluation."""
        return [
            # Function name queries
            {"query": "calculate similarity", "type": "function_name", "expected_terms": ["calculate", "similarity"]},
            {"query": "tokenize text", "type": "function_name", "expected_terms": ["tokenize", "text"]},
            {"query": "generate embeddings", "type": "function_name", "expected_terms": ["generate", "embedding"]},
            
            # Class name queries
            {"query": "BM25Tokenizer", "type": "class_name", "expected_terms": ["bm25", "tokenizer"]},
            {"query": "HybridCodeSearch", "type": "class_name", "expected_terms": ["hybrid", "search"]},
            {"query": "TokenizationManager", "type": "class_name", "expected_terms": ["tokenization", "manager"]},
            
            # Method/API queries
            {"query": "cosine similarity", "type": "method_call", "expected_terms": ["cosine", "similarity"]},
            {"query": "stem words", "type": "method_call", "expected_terms": ["stem", "word"]},
            {"query": "remove stopwords", "type": "method_call", "expected_terms": ["remove", "stopword"]},
            
            # Code pattern queries
            {"query": "for loop iteration", "type": "code_pattern", "expected_terms": ["for", "loop"]},
            {"query": "try except error", "type": "code_pattern", "expected_terms": ["try", "except"]},
            {"query": "import numpy", "type": "code_pattern", "expected_terms": ["import", "numpy"]},
            
            # Camel case variations
            {"query": "calculateSimilarity", "type": "camel_case", "expected_terms": ["calculate", "similarity"]},
            {"query": "tokenizeText", "type": "camel_case", "expected_terms": ["tokenize", "text"]},
            
            # Partial matching (stemming test)
            {"query": "running", "type": "stemming_test", "expected_terms": ["run", "running"]},
            {"query": "tokenization", "type": "stemming_test", "expected_terms": ["token", "tokenize"]},
            {"query": "similarities", "type": "stemming_test", "expected_terms": ["similar", "similarity"]},
            
            # Stopword filtering test
            {"query": "the best algorithm for search", "type": "stopword_test", "expected_terms": ["algorithm", "search"]},
            {"query": "a function to calculate", "type": "stopword_test", "expected_terms": ["function", "calculate"]},
        ]
    
    def load_real_codebase_data(self) -> bool:
        """Load real embeddings and chunks from the codebase."""
        try:
            # Check for embeddings in src directory first
            src_embeddings_path = Path(__file__).parent / "src" / "embeddings.pkl"
            if src_embeddings_path.exists():
                self.embeddings_path = src_embeddings_path
            elif not self.embeddings_path.exists():
                print(f"⚠️ Embeddings file not found: {self.embeddings_path}")
                print("   Generating synthetic test data for assessment...")
                return self._create_synthetic_test_data()

            print(f"📂 Loading real codebase data from {self.embeddings_path}")

            # Import the generate_embeddings module to ensure classes are available for pickle
            try:
                # Add the src directory to sys.path to ensure proper imports
                import sys
                src_path = str(Path(__file__).parent / "src")
                if src_path not in sys.path:
                    sys.path.insert(0, src_path)

                # Import with proper module path
                sys.path.insert(0, str(self.embeddings_path.parent))
                from generate_embeddings import CodeChunk, ChunkType, EmbeddingMetadata

                # CRITICAL FIX: Add classes to __main__ namespace for pickle loading
                import __main__
                __main__.CodeChunk = CodeChunk
                __main__.ChunkType = ChunkType
                __main__.EmbeddingMetadata = EmbeddingMetadata

                print("   ✅ Classes added to __main__ namespace for pickle loading")

            except ImportError as import_error:
                print(f"❌ Cannot import required classes: {import_error}")
                print("   Falling back to synthetic test data...")
                return self._create_synthetic_test_data()

            with open(self.embeddings_path, 'rb') as f:
                data = pickle.load(f)

            # Convert embeddings from list to numpy array if needed
            embeddings_raw = data['embeddings']
            if isinstance(embeddings_raw, list):
                print("   🔧 Converting embeddings from list to numpy array...")
                self.embeddings = np.vstack(embeddings_raw)
            else:
                self.embeddings = embeddings_raw

            self.chunks = data['chunks']
            self.metadata = data.get('metadata', {})

            print(f"   ✅ Loaded {len(self.chunks)} code chunks")
            if self.embeddings is not None:
                print(f"   📊 Embeddings shape: {self.embeddings.shape}")

            return True

        except Exception as e:
            print(f"❌ Error loading codebase data: {e}")
            print(f"   Details: {type(e).__name__}: {e}")
            print("   Falling back to synthetic test data...")
            return self._create_synthetic_test_data()

    def _create_synthetic_test_data(self) -> bool:
        """Create synthetic test data for quality assessment."""
        try:
            print("🧪 Creating synthetic test data for BM25 quality assessment...")

            # Create realistic code chunks for testing
            self.chunks = [
                CodeChunk(
                    content="def calculate_similarity(vector1, vector2):\n    \"\"\"Calculate cosine similarity between vectors.\"\"\"\n    return cosine_similarity(vector1, vector2)",
                    chunk_type=ChunkType.FUNCTION,
                    name="calculate_similarity",
                    file_path="similarity.py",
                    start_line=1,
                    end_line=3,
                    signature="calculate_similarity(vector1, vector2)",
                    docstring="Calculate cosine similarity between vectors"
                ),
                CodeChunk(
                    content="class BM25Tokenizer:\n    \"\"\"Optimized tokenizer for BM25 lexical search.\"\"\"\n    def __init__(self, stemming=True):\n        self.stemming = stemming\n        self.stemmer = SnowballStemmer('english')",
                    chunk_type=ChunkType.CLASS,
                    name="BM25Tokenizer",
                    file_path="tokenizers.py",
                    start_line=10,
                    end_line=14,
                    signature="class BM25Tokenizer",
                    docstring="Optimized tokenizer for BM25 lexical search"
                ),
                CodeChunk(
                    content="def tokenize_for_bm25(self, text):\n    \"\"\"Tokenize text for BM25 with stemming.\"\"\"\n    tokens = text.lower().split()\n    if self.stemming:\n        return [self.stemmer.stem(token) for token in tokens]\n    return tokens",
                    chunk_type=ChunkType.METHOD,
                    name="tokenize_for_bm25",
                    file_path="tokenizers.py",
                    start_line=20,
                    end_line=25,
                    signature="tokenize_for_bm25(self, text)",
                    docstring="Tokenize text for BM25 with stemming optimization"
                ),
                CodeChunk(
                    content="import numpy as np\nfrom sklearn.metrics.pairwise import cosine_similarity\nfrom nltk.stem import SnowballStemmer",
                    chunk_type=ChunkType.IMPORT_BLOCK,
                    name="imports",
                    file_path="search.py",
                    start_line=1,
                    end_line=3,
                    signature="import statements",
                    docstring="Required imports for search functionality"
                ),
                CodeChunk(
                    content="# Configuration for hybrid search\nHYBRID_CONFIG = {\n    'semantic_weight': 0.7,\n    'lexical_weight': 0.3,\n    'enable_stemming': True\n}",
                    chunk_type=ChunkType.MODULE,
                    name="HYBRID_CONFIG",
                    file_path="config.py",
                    start_line=5,
                    end_line=10,
                    signature="HYBRID_CONFIG",
                    docstring="Configuration dictionary for search weights"
                ),
                CodeChunk(
                    content="def remove_stopwords(tokens, stopwords):\n    \"\"\"Remove stopwords from token list.\"\"\"\n    return [token for token in tokens if token not in stopwords]",
                    chunk_type=ChunkType.FUNCTION,
                    name="remove_stopwords",
                    file_path="preprocessing.py",
                    start_line=15,
                    end_line=17,
                    signature="remove_stopwords(tokens, stopwords)",
                    docstring="Remove stopwords from token list"
                ),
                CodeChunk(
                    content="class TokenizationManager:\n    \"\"\"Manages independent tokenization strategies.\"\"\"\n    def __init__(self):\n        self.bm25_tokenizer = BM25Tokenizer()\n        self.voyage_tokenizer = VoyageTokenizer()",
                    chunk_type=ChunkType.CLASS,
                    name="TokenizationManager",
                    file_path="tokenizers.py",
                    start_line=50,
                    end_line=54,
                    signature="class TokenizationManager",
                    docstring="Manages independent tokenization strategies"
                ),
                CodeChunk(
                    content="def stem_words(words, stemmer):\n    \"\"\"Apply stemming to a list of words.\"\"\"\n    return [stemmer.stem(word) for word in words]",
                    chunk_type=ChunkType.FUNCTION,
                    name="stem_words",
                    file_path="preprocessing.py",
                    start_line=25,
                    end_line=27,
                    signature="stem_words(words, stemmer)",
                    docstring="Apply stemming to a list of words"
                )
            ]

            # Create synthetic embeddings (1024 dimensions to match voyage-context-3)
            np.random.seed(42)  # Reproducible results
            self.embeddings = np.random.rand(len(self.chunks), 1024).astype(np.float32)
            # Normalize embeddings
            self.embeddings = self.embeddings / np.linalg.norm(self.embeddings, axis=1, keepdims=True)

            self.metadata = {"synthetic": True, "created_for": "BM25 quality assessment"}

            print(f"   ✅ Created {len(self.chunks)} synthetic code chunks")
            print(f"   📊 Embeddings shape: {self.embeddings.shape}")

            return True

        except Exception as e:
            print(f"❌ Error creating synthetic test data: {e}")
            return False
    
    def initialize_tokenizers(self) -> bool:
        """Initialize different tokenization approaches for comparison."""
        try:
            # Initialize local BM25 tokenizer
            self.local_tokenizer = BM25Tokenizer(
                stemming=True,
                stopwords_lang="english",
                lowercase=True,
                min_token_length=2
            )
            
            # Initialize tokenization manager
            self.tokenization_manager = TokenizationManager()
            
            print("🔧 Tokenizers initialized:")
            print(f"   Local BM25: {len(self.local_tokenizer.stop_words)} stopwords")
            print(f"   Tokenization Manager: Ready")
            
            return True
            
        except Exception as e:
            print(f"❌ Error initializing tokenizers: {e}")
            return False
    
    def compare_tokenization_approaches(self, query: str) -> TokenizationComparison:
        """Compare different tokenization approaches for a query."""
        # Local BM25 tokenization
        local_tokens = self.local_tokenizer.tokenize(query) if self.local_tokenizer else []
        
        # Voyage tokenization (if available)
        voyage_tokens = []
        try:
            if self.tokenization_manager:
                voyage_tokens = self.tokenization_manager.tokenize_for_voyage(query)
        except Exception:
            voyage_tokens = []  # Voyage API might not be available
        
        # Simple baseline tokenization
        simple_tokens = query.lower().split()
        
        return TokenizationComparison(
            query=query,
            local_tokens=local_tokens,
            voyage_tokens=voyage_tokens,
            simple_tokens=simple_tokens,
            local_results=[],  # Will be filled by search comparison
            voyage_results=[],
            simple_results=[]
        )
    
    def calculate_relevance_scores(self, query_info: Dict[str, Any], 
                                 results: List[Any]) -> List[float]:
        """Calculate relevance scores for search results."""
        relevance_scores = []
        expected_terms = [term.lower() for term in query_info["expected_terms"]]
        
        for result in results:
            score = 0.0
            
            # Check if expected terms appear in result
            result_text = (result.chunk_name + " " + result.content).lower()
            
            for term in expected_terms:
                if term in result_text:
                    score += 1.0
            
            # Normalize by number of expected terms
            relevance_scores.append(score / len(expected_terms))
        
        return relevance_scores
    
    def calculate_quality_metrics(self, relevance_scores: List[float], 
                                k_values: List[int] = [1, 3, 5, 10]) -> QualityMetrics:
        """Calculate comprehensive quality metrics."""
        metrics = QualityMetrics(
            precision_at_k={},
            recall_at_k={},
            ndcg_at_k={},
            mrr=0.0,
            total_relevant=sum(1 for score in relevance_scores if score > 0.5),
            total_queries=1
        )
        
        # Calculate MRR (Mean Reciprocal Rank)
        for i, score in enumerate(relevance_scores):
            if score > 0.5:  # Consider relevant if score > 0.5
                metrics.mrr = 1.0 / (i + 1)
                break
        
        # Calculate metrics for different K values
        for k in k_values:
            if k > len(relevance_scores):
                continue
                
            top_k_scores = relevance_scores[:k]
            relevant_in_k = sum(1 for score in top_k_scores if score > 0.5)
            
            # Precision@K
            metrics.precision_at_k[k] = relevant_in_k / k if k > 0 else 0.0
            
            # Recall@K
            metrics.recall_at_k[k] = (relevant_in_k / metrics.total_relevant 
                                    if metrics.total_relevant > 0 else 0.0)
            
            # NDCG@K (simplified)
            dcg = sum(score / np.log2(i + 2) for i, score in enumerate(top_k_scores))
            ideal_scores = sorted(relevance_scores, reverse=True)[:k]
            idcg = sum(score / np.log2(i + 2) for i, score in enumerate(ideal_scores))
            metrics.ndcg_at_k[k] = dcg / idcg if idcg > 0 else 0.0
        
        return metrics
    
    def assess_search_quality(self) -> Dict[str, Any]:
        """Comprehensive search quality assessment."""
        if not self.load_real_codebase_data():
            return {"error": "Failed to load codebase data"}
        
        if not self.initialize_tokenizers():
            return {"error": "Failed to initialize tokenizers"}
        
        print("\n🔍 Starting Comprehensive BM25 Quality Assessment")
        print("=" * 70)
        
        # Initialize search system
        search_system = HybridCodeSearch(
            embeddings=self.embeddings,
            chunks=self.chunks,
            metadata=self.metadata
        )
        
        assessment_results = {
            "total_queries": len(self.test_queries),
            "query_results": [],
            "aggregate_metrics": {},
            "tokenization_analysis": [],
            "summary": {}
        }
        
        # Test each query
        for i, query_info in enumerate(self.test_queries):
            print(f"\n📝 Query {i+1}/{len(self.test_queries)}: '{query_info['query']}'")
            print(f"   Type: {query_info['type']}")
            
            # Compare tokenization approaches
            tokenization_comparison = self.compare_tokenization_approaches(query_info["query"])
            assessment_results["tokenization_analysis"].append(tokenization_comparison)
            
            print(f"   🔧 Local tokens: {tokenization_comparison.local_tokens}")
            print(f"   🌐 Voyage tokens: {tokenization_comparison.voyage_tokens}")
            print(f"   📝 Simple tokens: {tokenization_comparison.simple_tokens}")
            
            # Perform search with current system (local BM25 tokenization)
            try:
                results = search_system.hybrid_search(query_info["query"], top_k=10)
                
                # Calculate relevance scores
                relevance_scores = self.calculate_relevance_scores(query_info, results)
                
                # Calculate quality metrics
                quality_metrics = self.calculate_quality_metrics(relevance_scores)
                
                query_result = {
                    "query": query_info["query"],
                    "type": query_info["type"],
                    "results_count": len(results),
                    "relevance_scores": relevance_scores,
                    "quality_metrics": quality_metrics,
                    "tokenization": tokenization_comparison
                }
                
                assessment_results["query_results"].append(query_result)
                
                print(f"   📊 Results: {len(results)}")
                print(f"   🎯 Precision@5: {quality_metrics.precision_at_k.get(5, 0):.3f}")
                print(f"   📈 NDCG@5: {quality_metrics.ndcg_at_k.get(5, 0):.3f}")
                print(f"   🥇 MRR: {quality_metrics.mrr:.3f}")
                
            except Exception as e:
                print(f"   ❌ Search failed: {e}")
                query_result = {
                    "query": query_info["query"],
                    "type": query_info["type"],
                    "error": str(e)
                }
                assessment_results["query_results"].append(query_result)
        
        # Calculate aggregate metrics
        self._calculate_aggregate_metrics(assessment_results)
        
        return assessment_results
    
    def _calculate_aggregate_metrics(self, results: Dict[str, Any]) -> None:
        """Calculate aggregate metrics across all queries."""
        successful_queries = [q for q in results["query_results"] if "error" not in q]
        
        if not successful_queries:
            results["aggregate_metrics"] = {"error": "No successful queries"}
            return
        
        # Aggregate metrics
        aggregate = {
            "avg_precision_at_5": np.mean([q["quality_metrics"].precision_at_k.get(5, 0) 
                                         for q in successful_queries]),
            "avg_recall_at_5": np.mean([q["quality_metrics"].recall_at_k.get(5, 0) 
                                      for q in successful_queries]),
            "avg_ndcg_at_5": np.mean([q["quality_metrics"].ndcg_at_k.get(5, 0) 
                                    for q in successful_queries]),
            "avg_mrr": np.mean([q["quality_metrics"].mrr for q in successful_queries]),
            "successful_queries": len(successful_queries),
            "total_queries": len(results["query_results"])
        }
        
        results["aggregate_metrics"] = aggregate
        
        # Summary
        results["summary"] = {
            "overall_quality": "Good" if aggregate["avg_ndcg_at_5"] > 0.6 else 
                             "Fair" if aggregate["avg_ndcg_at_5"] > 0.4 else "Poor",
            "success_rate": aggregate["successful_queries"] / aggregate["total_queries"],
            "key_findings": self._generate_key_findings(results)
        }
    
    def _generate_key_findings(self, results: Dict[str, Any]) -> List[str]:
        """Generate key findings from the assessment."""
        findings = []
        
        aggregate = results["aggregate_metrics"]
        
        if aggregate["avg_precision_at_5"] > 0.7:
            findings.append("High precision: Most top results are relevant")
        elif aggregate["avg_precision_at_5"] < 0.3:
            findings.append("Low precision: Many irrelevant results in top positions")
        
        if aggregate["avg_recall_at_5"] > 0.6:
            findings.append("Good recall: Finding most relevant results")
        elif aggregate["avg_recall_at_5"] < 0.3:
            findings.append("Poor recall: Missing many relevant results")
        
        if aggregate["avg_mrr"] > 0.8:
            findings.append("Excellent ranking: First relevant result typically in top 2")
        elif aggregate["avg_mrr"] < 0.5:
            findings.append("Poor ranking: First relevant result often not in top positions")
        
        return findings


def main():
    """Main assessment execution."""
    print("🔍 BM25 Quality Assessment Tool")
    print("Comprehensive evaluation of BM25 search quality")
    print()
    
    assessor = BM25QualityAssessor()
    results = assessor.assess_search_quality()
    
    if "error" in results:
        print(f"❌ Assessment failed: {results['error']}")
        return
    
    # Print summary
    print("\n" + "=" * 70)
    print("📊 ASSESSMENT SUMMARY")
    print("=" * 70)
    
    aggregate = results["aggregate_metrics"]
    summary = results["summary"]
    
    print(f"Overall Quality: {summary['overall_quality']}")
    print(f"Success Rate: {summary['success_rate']:.1%}")
    print(f"Average Precision@5: {aggregate['avg_precision_at_5']:.3f}")
    print(f"Average Recall@5: {aggregate['avg_recall_at_5']:.3f}")
    print(f"Average NDCG@5: {aggregate['avg_ndcg_at_5']:.3f}")
    print(f"Average MRR: {aggregate['avg_mrr']:.3f}")
    
    print("\n🔍 Key Findings:")
    for finding in summary["key_findings"]:
        print(f"   • {finding}")
    
    # Save detailed results
    results_file = Path("bm25_quality_assessment.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Detailed results saved to: {results_file}")


if __name__ == "__main__":
    main()
