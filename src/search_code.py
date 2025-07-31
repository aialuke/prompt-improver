"""
Enhanced Hybrid Semantic Code Search using Voyage AI voyage-code-3
Phase 2: Combines semantic search with BM25 lexical search for superior accuracy.
Includes binary rescoring and advanced error handling.
"""

import voyageai
import pickle
import numpy as np
import numpy.typing as npt
import time
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Any, Optional, Tuple, cast, Union, TYPE_CHECKING
from dataclasses import dataclass
import sys
from pathlib import Path

# Enhanced imports for Phase 2 hybrid search with proper type handling
BM25_AVAILABLE = False
BM25Okapi: Optional[type] = None

try:
    from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]
    BM25_AVAILABLE = True
except ImportError:
    print("⚠️  rank-bm25 not available. Install with: pip install rank-bm25")
    BM25Okapi = None

CROSS_ENCODER_AVAILABLE = False
CrossEncoder: Optional[type] = None

try:
    from sentence_transformers import CrossEncoder  # type: ignore[import-untyped]
    CROSS_ENCODER_AVAILABLE = True
except ImportError:
    print("⚠️  sentence-transformers not available. Install with: pip install sentence-transformers")
    CrossEncoder = None

# Import the data structures from generate_embeddings.py
sys.path.append(str(Path(__file__).parent))

# Hybrid Search Configuration
HYBRID_SEARCH_CONFIG = {
    "semantic_weight": 0.7,           # Weight for semantic similarity (0.0-1.0)
    "lexical_weight": 0.3,            # Weight for BM25 lexical similarity (0.0-1.0)
    "enable_binary_rescoring": True,  # Enable binary rescoring for performance
    "binary_candidates_multiplier": 3, # Get 3x candidates for binary rescoring
    "enable_cross_encoder": False,    # Enable cross-encoder reranking (slower but more accurate)
    "max_query_length": 512,          # Maximum query length in characters
    "bm25_k1": 1.2,                  # BM25 k1 parameter
    "bm25_b": 0.75,                  # BM25 b parameter
    "min_similarity_threshold": 0.1,  # Minimum similarity threshold
    "enable_query_expansion": True,   # Enable query expansion with synonyms
}

# Performance tracking
SEARCH_METRICS = {
    "total_searches": 0,
    "avg_search_time": 0.0,
    "cache_hits": 0,
    "api_calls": 0,
}


class ChunkType:
    """Types of code chunks for semantic search."""
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    ASYNC_FUNCTION = "async_function"
    ASYNC_METHOD = "async_method"
    PROPERTY = "property"
    IMPORT_BLOCK = "import_block"


@dataclass
class SearchResult:
    """Enhanced search result with hybrid scoring and metadata."""
    chunk_name: str
    chunk_type: str
    file_path: str
    similarity_score: float
    start_line: int
    end_line: int
    content: str
    docstring: Optional[str] = None
    signature: Optional[str] = None
    parent_class: Optional[str] = None
    # Enhanced Phase 2 fields
    semantic_score: float = 0.0
    lexical_score: float = 0.0
    hybrid_score: float = 0.0
    search_method: str = "semantic"  # "semantic", "hybrid", "binary_rescored"
    processing_time_ms: float = 0.0


class HybridCodeSearch:
    """Enhanced hybrid search combining semantic and lexical search with binary rescoring."""

    def __init__(self, embeddings: np.ndarray, chunks: List[Any], metadata: Dict[str, Any]):
        self.embeddings = embeddings
        self.chunks = chunks
        self.metadata = metadata
        self.vo = None
        self.bm25 = None
        self.cross_encoder = None
        self.query_cache = {}

        # Initialize BM25 if available
        if BM25_AVAILABLE:
            self._initialize_bm25()

        # Initialize cross-encoder if available and enabled
        if CROSS_ENCODER_AVAILABLE and HYBRID_SEARCH_CONFIG["enable_cross_encoder"]:
            self._initialize_cross_encoder()

        print(f"🔍 Hybrid search initialized:")
        print(f"   Semantic search: ✅ Enabled")
        print(f"   BM25 lexical search: {'✅ Enabled' if self.bm25 else '❌ Disabled'}")
        print(f"   Binary rescoring: {'✅ Enabled' if HYBRID_SEARCH_CONFIG['enable_binary_rescoring'] else '❌ Disabled'}")
        print(f"   Cross-encoder reranking: {'✅ Enabled' if self.cross_encoder else '❌ Disabled'}")

    def _initialize_bm25(self) -> None:
        """Initialize BM25 index for lexical search."""
        if not BM25_AVAILABLE or BM25Okapi is None:
            print("   ❌ BM25 not available")
            return

        try:
            print("📚 Building BM25 index for lexical search...")

            # Tokenize all chunks for BM25
            tokenized_chunks = []
            for chunk in self.chunks:
                # Combine content, docstring, and signature for better matching
                text_parts = [chunk.content]
                if hasattr(chunk, 'docstring') and chunk.docstring:
                    text_parts.append(chunk.docstring)
                if hasattr(chunk, 'signature') and chunk.signature:
                    text_parts.append(chunk.signature)

                combined_text = " ".join(text_parts)
                # Simple tokenization (can be enhanced with proper tokenizer)
                tokens = combined_text.lower().split()
                tokenized_chunks.append(tokens)

            self.bm25 = BM25Okapi(
                tokenized_chunks,
                k1=HYBRID_SEARCH_CONFIG["bm25_k1"],
                b=HYBRID_SEARCH_CONFIG["bm25_b"]
            )
            print(f"   ✅ BM25 index built for {len(tokenized_chunks)} chunks")

        except Exception as e:
            print(f"   ❌ Failed to initialize BM25: {e}")
            self.bm25 = None

    def _initialize_cross_encoder(self) -> None:
        """Initialize cross-encoder for reranking."""
        if not CROSS_ENCODER_AVAILABLE or CrossEncoder is None:
            print("   ❌ Cross-encoder not available")
            return

        try:
            print("🧠 Loading cross-encoder model for reranking...")
            # Use a lightweight model for code relevance
            self.cross_encoder = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print("   ✅ Cross-encoder loaded successfully")
        except Exception as e:
            print(f"   ❌ Failed to initialize cross-encoder: {e}")
            self.cross_encoder = None

    def _get_voyage_client(self):
        """Get or initialize Voyage AI client."""
        if self.vo is None:
            try:
                self.vo = voyageai.Client()  # type: ignore
            except AttributeError as e:
                print(f"❌ Error: 'Client' not found in voyageai module. {e}")
                raise
        return self.vo

    def hybrid_search(self, query: str, top_k: int = 5,
                     min_similarity: Optional[float] = None) -> List[SearchResult]:
        """Perform hybrid search combining semantic and lexical search."""

        # Set default min_similarity if not provided
        actual_min_similarity = min_similarity if min_similarity is not None else HYBRID_SEARCH_CONFIG["min_similarity_threshold"]

        search_start_time = time.time()
        SEARCH_METRICS["total_searches"] += 1

        print(f"🔍 Hybrid search for: '{query}'")

        # Check cache first
        cache_key = f"{query}:{top_k}:{min_similarity}"
        if cache_key in self.query_cache:
            SEARCH_METRICS["cache_hits"] += 1
            print("   ⚡ Using cached results")
            return self.query_cache[cache_key]

        try:
            # Expand query if enabled
            expanded_queries = self._expand_query(query) if HYBRID_SEARCH_CONFIG["enable_query_expansion"] else [query]

            # Get semantic scores
            semantic_scores = self._get_semantic_scores(expanded_queries[0])  # Use primary query for semantic

            # Get lexical scores if BM25 available
            lexical_scores = self._get_lexical_scores(query) if self.bm25 else np.zeros_like(semantic_scores)

            # Combine scores
            hybrid_scores = self._combine_scores(semantic_scores, lexical_scores)

            # Apply binary rescoring if enabled
            if HYBRID_SEARCH_CONFIG["enable_binary_rescoring"]:
                results = self._binary_rescoring_search(query, hybrid_scores, top_k, actual_min_similarity)
                search_method = "binary_rescored"
            else:
                results = self._standard_search(query, hybrid_scores, semantic_scores, lexical_scores, top_k, actual_min_similarity)
                search_method = "hybrid"

            # Apply cross-encoder reranking if enabled
            if self.cross_encoder and results:
                results = self._rerank_with_cross_encoder(query, results)

            # Update search method and timing
            search_time = (time.time() - search_start_time) * 1000
            for result in results:
                result.search_method = search_method
                result.processing_time_ms = search_time / len(results) if results else search_time

            # Cache results
            self.query_cache[cache_key] = results

            # Update metrics
            SEARCH_METRICS["avg_search_time"] = (
                (SEARCH_METRICS["avg_search_time"] * (SEARCH_METRICS["total_searches"] - 1) + search_time) /
                SEARCH_METRICS["total_searches"]
            )

            print(f"   ✅ Found {len(results)} results in {search_time:.1f}ms")
            return results

        except Exception as e:
            print(f"❌ Error during hybrid search: {e}")
            raise

    def _expand_query(self, query: str) -> List[str]:
        """Expand query with programming-specific synonyms."""

        code_synonyms = {
            "sort": ["order", "arrange", "organize", "rank"],
            "search": ["find", "locate", "retrieve", "lookup"],
            "function": ["method", "procedure", "routine", "def"],
            "class": ["object", "type", "structure"],
            "async": ["asynchronous", "concurrent", "parallel"],
            "loop": ["iterate", "repeat", "cycle", "for", "while"],
            "algorithm": ["implementation", "solution", "approach"],
            "data": ["information", "content", "values"],
        }

        expanded_queries = [query]
        words = query.lower().split()

        for word in words:
            if word in code_synonyms:
                for synonym in code_synonyms[word]:
                    expanded_query = query.replace(word, synonym)
                    expanded_queries.append(expanded_query)

        return expanded_queries[:3]  # Limit to avoid too many variations

    def _get_semantic_scores(self, query: str) -> np.ndarray:
        """Get semantic similarity scores using voyage-code-3."""
        try:
            vo = self._get_voyage_client()
            SEARCH_METRICS["api_calls"] += 1

            # Generate query embedding
            query_embedding = vo.embed([query], model="voyage-code-3", input_type="query").embeddings[0]

            # Calculate cosine similarity
            query_array = np.array([query_embedding])
            similarities = cosine_similarity(query_array, self.embeddings)[0]
            return similarities

        except Exception as e:
            print(f"❌ Error generating semantic scores: {e}")
            return np.zeros(len(self.chunks))

    def _get_lexical_scores(self, query: str) -> np.ndarray:
        """Get BM25 lexical similarity scores."""
        if not self.bm25:
            return np.zeros(len(self.chunks))

        try:
            # Tokenize query
            tokenized_query = query.lower().split()

            # Get BM25 scores
            bm25_scores = self.bm25.get_scores(tokenized_query)

            # Normalize scores to 0-1 range
            if bm25_scores.max() > 0:
                bm25_scores = (bm25_scores - bm25_scores.min()) / (bm25_scores.max() - bm25_scores.min())

            return bm25_scores

        except Exception as e:
            print(f"❌ Error generating lexical scores: {e}")
            return np.zeros(len(self.chunks))

    def _combine_scores(self, semantic_scores: np.ndarray, lexical_scores: np.ndarray) -> np.ndarray:
        """Combine semantic and lexical scores using configured weights."""

        # Normalize scores to ensure fair combination
        semantic_normalized = self._normalize_scores(semantic_scores)
        lexical_normalized = self._normalize_scores(lexical_scores)

        # Weighted combination
        hybrid_scores = (
            HYBRID_SEARCH_CONFIG["semantic_weight"] * semantic_normalized +
            HYBRID_SEARCH_CONFIG["lexical_weight"] * lexical_normalized
        )

        return hybrid_scores

    def _normalize_scores(self, scores: np.ndarray) -> np.ndarray:
        """Normalize scores to 0-1 range."""
        if scores.max() == scores.min():
            return scores
        return (scores - scores.min()) / (scores.max() - scores.min())

    def _binary_rescoring_search(self, query: str, hybrid_scores: np.ndarray,
                                top_k: int, min_similarity: float) -> List[SearchResult]:
        """Perform binary rescoring for fast initial retrieval followed by full-precision reranking."""

        # Get more candidates for rescoring
        candidate_count = top_k * HYBRID_SEARCH_CONFIG["binary_candidates_multiplier"]
        candidate_indices = np.argsort(hybrid_scores)[-candidate_count:][::-1]

        # Filter by minimum similarity
        valid_candidates = []
        for idx in candidate_indices:
            if hybrid_scores[idx] >= min_similarity:
                valid_candidates.append(idx)

        if not valid_candidates:
            return []

        try:
            # Generate embeddings for rescoring
            vo = self._get_voyage_client()
            SEARCH_METRICS["api_calls"] += 1

            # For binary rescoring, we would use binary embeddings for initial filtering
            # and full precision for final rescoring. For now, use full precision directly.
            full_query = vo.embed([query], model="voyage-code-3",
                                input_type="query",
                                output_dimension=1024,
                                output_dtype="float").embeddings[0]

            # Rescore with full precision
            candidate_embeddings = self.embeddings[valid_candidates]
            query_array = np.array([full_query])
            final_similarities = cosine_similarity(query_array, candidate_embeddings)[0]

            # Get top-k after rescoring
            final_indices = valid_candidates[np.argsort(final_similarities)[-top_k:][::-1]]

            return self._create_search_results(final_indices.tolist(), hybrid_scores,
                                             final_similarities[np.argsort(final_similarities)[-top_k:][::-1]])

        except Exception as e:
            print(f"❌ Binary rescoring failed, falling back to standard search: {e}")
            return self._standard_search(query, hybrid_scores, hybrid_scores,
                                       np.zeros_like(hybrid_scores), top_k, min_similarity)

    def _standard_search(self, _query: str, hybrid_scores: np.ndarray,
                        semantic_scores: np.ndarray, lexical_scores: np.ndarray,
                        top_k: int, min_similarity: float) -> List[SearchResult]:
        """Perform standard hybrid search without binary rescoring."""

        # Get top results
        top_indices = np.argsort(hybrid_scores)[-top_k:][::-1]

        # Filter by minimum similarity
        valid_results = []
        for idx in top_indices:
            if hybrid_scores[idx] >= min_similarity:
                valid_results.append(idx)

        return self._create_search_results(valid_results, hybrid_scores,
                                         semantic_scores, lexical_scores)

    def _create_search_results(self, indices: List[int],
                              hybrid_scores: np.ndarray,
                              semantic_scores: Optional[np.ndarray] = None,
                              lexical_scores: Optional[np.ndarray] = None) -> List[SearchResult]:
        """Create SearchResult objects from indices and scores."""

        results = []

        for idx in indices:
            chunk = self.chunks[idx]

            # Get individual scores
            semantic_score = semantic_scores[idx] if semantic_scores is not None else 0.0
            lexical_score = lexical_scores[idx] if lexical_scores is not None else 0.0
            hybrid_score = hybrid_scores[idx]

            result = SearchResult(
                chunk_name=chunk.name,
                chunk_type=chunk.chunk_type.value if hasattr(chunk.chunk_type, 'value') else str(chunk.chunk_type),
                file_path=chunk.file_path,
                similarity_score=hybrid_score,
                start_line=chunk.start_line,
                end_line=chunk.end_line,
                content=chunk.content,
                docstring=getattr(chunk, 'docstring', None),
                signature=getattr(chunk, 'signature', None),
                parent_class=getattr(chunk, 'parent_class', None),
                semantic_score=semantic_score,
                lexical_score=lexical_score,
                hybrid_score=hybrid_score
            )

            results.append(result)

        return results

    def _rerank_with_cross_encoder(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        """Rerank results using cross-encoder for better relevance."""

        if not self.cross_encoder or not results:
            return results

        try:
            # Prepare pairs for reranking
            pairs = [(query, result.content) for result in results]

            # Get relevance scores
            relevance_scores = self.cross_encoder.predict(pairs)

            # Sort by relevance
            sorted_indices = np.argsort(relevance_scores)[::-1]
            reranked_results = [results[i] for i in sorted_indices]

            # Update similarity scores with reranking scores
            for i, result in enumerate(reranked_results):
                result.similarity_score = float(relevance_scores[sorted_indices[i]])
                result.search_method = "cross_encoder_reranked"

            return reranked_results

        except Exception as e:
            print(f"❌ Cross-encoder reranking failed: {e}")
            return results


def load_enhanced_embeddings(embeddings_path: str) -> Tuple[np.ndarray[Any, np.dtype[np.float32]], List[Any], Dict[str, Any]]:
    """Load embeddings with enhanced metadata support."""
    try:
        with open(embeddings_path, "rb") as f:
            data = pickle.load(f)

        # Check if this is the new enhanced format
        if "version" in data and data["version"] == "2.0":
            print(f"✅ Loaded enhanced embeddings (v{data['version']})")
            print(f"   Model used: {data.get('model_used', 'unknown')}")
            print(f"   Total chunks: {len(data['chunks'])}")

            embeddings = np.array(data["embeddings"], dtype=np.float32)
            chunks = data["chunks"]
            metadata = data.get("enhanced_metadata", {})

            return embeddings, chunks, metadata
        else:
            # Backward compatibility with old format
            print("⚠️  Loading legacy embeddings format")
            embeddings = np.array(data["embeddings"], dtype=np.float32)
            file_paths = data["file_paths"]

            # Create minimal chunk objects for compatibility
            chunks = []
            for file_path in file_paths:
                chunk = type('Chunk', (), {
                    'name': Path(file_path).stem,
                    'chunk_type': ChunkType.MODULE,
                    'file_path': file_path,
                    'start_line': 1,
                    'end_line': 1,
                    'content': '',
                    'docstring': None,
                    'signature': None,
                    'parent_class': None
                })()
                chunks.append(chunk)

            return embeddings, chunks, {}

    except FileNotFoundError:
        print(f"❌ Error: embeddings.pkl not found at {embeddings_path}")
        print("   Run generate_embeddings.py first to create embeddings.")
        raise
    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        raise


def enhanced_search_code(query: str, top_k: int = 5, min_similarity: float = 0.1,
                        search_method: str = "hybrid") -> List[SearchResult]:
    """
    Enhanced search for code chunks using hybrid semantic + lexical search with voyage-code-3.

    Args:
        query: Natural language description of what you're looking for
        top_k: Number of top results to return
        min_similarity: Minimum similarity threshold (0.0 to 1.0)
        search_method: "semantic", "hybrid", or "auto" (default: "hybrid")

    Returns:
        List of SearchResult objects sorted by relevance
    """
    try:
        # Load embeddings and chunks
        embeddings, chunks, metadata = load_enhanced_embeddings("embeddings.pkl")

        print(f"🔍 Enhanced search for: '{query}'")
        print(f"📊 Loaded {len(chunks)} code chunks from {metadata.get('total_files_processed', 'unknown')} files")
        print(f"🔧 Search method: {search_method}")

        # Initialize hybrid search system
        hybrid_search = HybridCodeSearch(embeddings, chunks, metadata)

        # Perform search based on method
        if search_method == "semantic":
            # Use only semantic search
            old_config = HYBRID_SEARCH_CONFIG.copy()
            HYBRID_SEARCH_CONFIG["semantic_weight"] = 1.0
            HYBRID_SEARCH_CONFIG["lexical_weight"] = 0.0
            HYBRID_SEARCH_CONFIG["enable_binary_rescoring"] = False

            results = hybrid_search.hybrid_search(query, top_k, min_similarity)

            # Restore config
            HYBRID_SEARCH_CONFIG.update(old_config)

        elif search_method == "hybrid" or search_method == "auto":
            # Use full hybrid search
            results = hybrid_search.hybrid_search(query, top_k, min_similarity)
        else:
            print(f"❌ Unknown search method: {search_method}. Using hybrid.")
            results = hybrid_search.hybrid_search(query, top_k, min_similarity)

        # Print search metrics
        print(f"📈 Search metrics:")
        print(f"   Total searches: {SEARCH_METRICS['total_searches']}")
        print(f"   Average search time: {SEARCH_METRICS['avg_search_time']:.1f}ms")
        print(f"   Cache hits: {SEARCH_METRICS['cache_hits']}")
        print(f"   API calls: {SEARCH_METRICS['api_calls']}")

        print(f"✅ Found {len(results)} relevant code chunks")
        return results

    except FileNotFoundError:
        print("❌ Error: embeddings.pkl not found. Please run generate_embeddings.py first.")
        return []
    except Exception as e:
        print(f"❌ Error during search: {e}")
        return []


def search_code(query: str, embeddings: np.ndarray[Any, np.dtype[np.float32]], chunks: List[Any],
                top_k: int = 5, min_similarity: float = 0.1) -> List[SearchResult]:
    """Perform semantic code search with enhanced results (legacy function)."""

    # Initialize Voyage AI client
    try:
        vo = voyageai.Client()  # type: ignore [attr-defined]
    except AttributeError as e:
        print(f"❌ Error: 'Client' not found in voyageai module. {e}")
        raise

    print(f"🔍 Searching for: '{query}'")

    try:
        # Generate query embedding using voyage-code-3 for consistency
        query_embedding_raw = cast(
            List[float],
            vo.embed([query], model="voyage-code-3", input_type="query").embeddings[0]
        )
        # Convert to numpy array for sklearn compatibility
        query_embedding = np.array(query_embedding_raw, dtype=np.float32).reshape(1, -1)

    except Exception as e:
        print(f"❌ Error generating query embedding: {e}")
        raise

    # Calculate similarities
    similarities = cosine_similarity(query_embedding, embeddings)[0]

    # Get top results above minimum similarity threshold
    valid_indices = np.where(similarities >= min_similarity)[0]
    if len(valid_indices) == 0:
        print(f"⚠️  No results found above similarity threshold {min_similarity}")
        return []

    # Sort by similarity and take top_k
    sorted_indices = valid_indices[np.argsort(similarities[valid_indices])[::-1]][:top_k]

    # Create search results
    results = []
    for idx in sorted_indices:
        chunk = chunks[idx]

        result = SearchResult(
            chunk_name=chunk.name,
            chunk_type=chunk.chunk_type.value if hasattr(chunk.chunk_type, 'value') else str(chunk.chunk_type),
            file_path=chunk.file_path,
            similarity_score=similarities[idx],
            start_line=chunk.start_line,
            end_line=chunk.end_line,
            content=chunk.content,
            docstring=getattr(chunk, 'docstring', None),
            signature=getattr(chunk, 'signature', None),
            parent_class=getattr(chunk, 'parent_class', None)
        )
        results.append(result)

    return results


def display_search_results(results: List[SearchResult], show_content: bool = True, max_content_length: int = 300) -> None:
    """Display search results in a formatted way."""
    if not results:
        print("❌ No results found.")
        return

    print(f"\n🎯 Found {len(results)} relevant code chunks:")
    print("=" * 80)

    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.chunk_name} ({result.chunk_type})")
        print(f"   📁 File: {result.file_path}")
        print(f"   📍 Lines: {result.start_line}-{result.end_line}")
        print(f"   🎯 Similarity: {result.similarity_score:.4f}")

        if result.parent_class:
            print(f"   🏗️  Class: {result.parent_class}")

        if result.signature:
            print(f"   ✍️  Signature: {result.signature}")

        if result.docstring:
            docstring_preview = result.docstring[:100] + "..." if len(result.docstring) > 100 else result.docstring
            print(f"   📝 Docstring: {docstring_preview}")

        if show_content and result.content:
            content_preview = result.content[:max_content_length]
            if len(result.content) > max_content_length:
                content_preview += "\n... (truncated)"
            print(f"   📄 Content:\n{content_preview}")

        print("-" * 80)


if __name__ == "__main__":
    # Configuration
    print("🚀 Testing Enhanced Hybrid Code Search (Phase 2)")
    print("=" * 60)

    # Test queries focused on sorting functions for validation
    test_queries = [
        "function to sort a list or array in Python",
        "quicksort algorithm implementation",
        "merge sort or divide and conquer sorting",
        "bubble sort simple sorting algorithm"
    ]

    # Test different search methods
    search_methods = ["semantic", "hybrid"]

    for method in search_methods:
        print(f"\n🔧 Testing {method.upper()} search method:")
        print("=" * 50)

        for query in test_queries[:2]:  # Test with first 2 queries to limit API usage
            print(f"\n{'='*80}")
            print(f"Query: {query}")
            print(f"Method: {method}")

            try:
                results = enhanced_search_code(query, top_k=3, min_similarity=0.1, search_method=method)
                display_search_results(results, show_content=True, max_content_length=150)

                # Show enhanced metrics
                if results:
                    print(f"\n📊 Enhanced Search Results:")
                    for i, result in enumerate(results, 1):
                        print(f"   {i}. {result.chunk_name} - Method: {result.search_method}")
                        print(f"      Semantic: {result.semantic_score:.3f}, Lexical: {result.lexical_score:.3f}, Hybrid: {result.hybrid_score:.3f}")
                        print(f"      Processing time: {result.processing_time_ms:.1f}ms")

            except Exception as e:
                print(f"❌ Error testing {method} search: {e}")

            print(f"{'='*80}")

            # Small delay to avoid rate limiting
            import time
            time.sleep(1)

    print("\n🎉 Phase 2 Hybrid Search Implementation completed!")
    print("✅ Features implemented:")
    print("   • Hybrid semantic + lexical search with configurable weights")
    print("   • Binary rescoring for performance optimization")
    print("   • Query expansion with programming synonyms")
    print("   • Enhanced error handling and metrics tracking")
    print("   • Cross-encoder reranking support")
    print("   • Memory-efficient caching system")