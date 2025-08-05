"""
Independent Tokenization Optimization for Hybrid Search Systems

This module implements separate, optimized tokenizers for BM25 lexical search 
and Voyage AI semantic search, replacing the unified tokenization approach
with independent optimization strategies based on 2025 best practices.

Key Features:
- BM25Tokenizer: Optimized for lexical search with stemming, stopword removal
- VoyageTokenizer: Dedicated semantic search tokenization via Voyage AI API
- TokenizationManager: Coordinates independent tokenization strategies
- Zero API calls for BM25 tokenization (local processing only)
- Maintains search quality while reducing costs and improving performance
"""

import os
import re
import string
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

# Lazy imports for optional dependencies
try:
    import nltk
    from nltk.stem import SnowballStemmer
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    SnowballStemmer = None
    stopwords = None

# Configure tokenizers to prevent warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


class BaseTokenizer(ABC):
    """Abstract base class for tokenizers."""
    
    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """Tokenize input text into a list of tokens."""
        pass
    
    @abstractmethod
    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """Tokenize a batch of texts."""
        pass


class BM25Tokenizer(BaseTokenizer):
    """
    Code-optimized tokenizer for BM25 lexical search.

    Implements 2025 best practices for code search BM25 optimization:
    - Light stemming optimized for code terms
    - Code-specific stopwords (reduced from natural language)
    - CamelCase and snake_case splitting
    - Programming pattern recognition
    - Local processing only (zero API calls)

    Code search optimizations:
    - CamelCase splitting: "calculateSimilarity" → ["calculate", "similarity"]
    - Acronym preservation: "BM25Tokenizer" → ["BM25", "tokenizer"]
    - Underscore splitting: "calculate_similarity" → ["calculate", "similarity"]
    - Reduced stemming to preserve code semantics
    """

    def __init__(self,
                 stemming: str = "light",  # "none", "light", "aggressive"
                 stopwords_lang: str = "english",
                 lowercase: bool = True,
                 min_token_length: int = 2,
                 remove_punctuation: bool = True,
                 split_camelcase: bool = True,
                 split_underscores: bool = True):
        """
        Initialize BM25 tokenizer with code search optimizations.

        Args:
            stemming: Stemming level - "none", "light", "aggressive"
            stopwords_lang: Language for stopword removal ("english")
            lowercase: Convert tokens to lowercase
            min_token_length: Minimum token length to keep
            remove_punctuation: Remove punctuation from tokens
            split_camelcase: Split camelCase terms
            split_underscores: Split underscore_separated terms
        """
        self.stemming = stemming
        self.stopwords_lang = stopwords_lang
        self.lowercase = lowercase
        self.min_token_length = min_token_length
        self.remove_punctuation = remove_punctuation
        self.split_camelcase = split_camelcase
        self.split_underscores = split_underscores
        
        # Initialize NLTK components
        self._initialize_nltk()
        
        # Compile regex patterns for performance
        self._compile_patterns()
        
        # Performance tracking
        self.tokenization_count = 0
        
    def _initialize_nltk(self) -> None:
        """Initialize NLTK components with code search optimizations."""
        if not NLTK_AVAILABLE:
            raise ImportError(
                "NLTK is required for BM25Tokenizer. Install with: pip install nltk"
            )

        try:
            # Download required NLTK data if not present
            try:
                nltk.data.find('corpora/stopwords')
            except LookupError:
                print("📥 Downloading NLTK stopwords data...")
                nltk.download('stopwords', quiet=True)

            # Initialize stemmer based on stemming level
            if self.stemming == "aggressive":
                self.stemmer = SnowballStemmer(self.stopwords_lang)
            elif self.stemming == "light":
                self.stemmer = SnowballStemmer(self.stopwords_lang)
                # Light stemming will be handled in tokenize method
            else:  # "none"
                self.stemmer = None

            # Initialize code-optimized stopwords (reduced set for code search)
            base_stopwords = set(stopwords.words(self.stopwords_lang))
            # Remove code-relevant words that are often in standard stopword lists
            code_relevant = {'a', 'an', 'as', 'at', 'be', 'by', 'for', 'from',
                           'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on',
                           'that', 'the', 'to', 'was', 'will', 'with'}
            # Keep only the most common stopwords for code search
            self.stop_words = base_stopwords & code_relevant

            print(f"✅ BM25Tokenizer initialized with {len(self.stop_words)} code-optimized stopwords")
            print(f"   Stemming level: {self.stemming}")
            print(f"   CamelCase splitting: {self.split_camelcase}")

        except Exception as e:
            print(f"❌ Failed to initialize NLTK components: {e}")
            # Fallback to basic tokenization
            self.stemmer = None
            self.stop_words = set()
    
    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficient code-aware text processing."""
        # Pattern for splitting text into tokens (code-aware)
        self.token_pattern = re.compile(r'\b\w+\b')

        # Pattern for camelCase splitting
        if self.split_camelcase:
            # Matches transitions from lowercase to uppercase
            self.camelcase_pattern = re.compile(r'([a-z])([A-Z])')
        else:
            self.camelcase_pattern = None

        # Pattern for underscore splitting
        if self.split_underscores:
            self.underscore_pattern = re.compile(r'_+')
        else:
            self.underscore_pattern = None

        # Pattern for removing punctuation (preserve underscores if splitting)
        if self.remove_punctuation:
            if self.split_underscores:
                # Remove punctuation except underscores (handled separately)
                punct_chars = string.punctuation.replace('_', '')
                self.punct_pattern = re.compile(f'[{re.escape(punct_chars)}]')
            else:
                self.punct_pattern = re.compile(f'[{re.escape(string.punctuation)}]')
        else:
            self.punct_pattern = None
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize single text with code-optimized BM25 processing.

        Args:
            text: Input text to tokenize

        Returns:
            List of code-optimized tokens for BM25 lexical search
        """
        if not text or not isinstance(text, str):
            return []

        self.tokenization_count += 1

        # Step 1: Code-aware preprocessing
        processed_text = self._preprocess_code_text(text)

        # Step 2: Extract tokens using regex
        tokens = self.token_pattern.findall(processed_text)

        # Step 3: Lowercase normalization
        if self.lowercase:
            tokens = [token.lower() for token in tokens]

        # Step 4: Remove punctuation
        if self.punct_pattern:
            tokens = [self.punct_pattern.sub('', token) for token in tokens]

        # Step 5: Filter by minimum length
        tokens = [token for token in tokens if len(token) >= self.min_token_length]

        # Step 6: Remove stopwords
        if self.stop_words:
            tokens = [token for token in tokens if token not in self.stop_words]

        # Step 7: Apply code-optimized stemming
        if self.stemmer:
            tokens = self._apply_code_stemming(tokens)

        return tokens

    def _preprocess_code_text(self, text: str) -> str:
        """Preprocess text with code-specific patterns."""
        processed = text

        # Split camelCase: "calculateSimilarity" → "calculate Similarity"
        if self.camelcase_pattern:
            processed = self.camelcase_pattern.sub(r'\1 \2', processed)

        # Split underscores: "calculate_similarity" → "calculate similarity"
        if self.underscore_pattern:
            processed = self.underscore_pattern.sub(' ', processed)

        return processed

    def _apply_code_stemming(self, tokens: List[str]) -> List[str]:
        """Apply code-optimized stemming."""
        if not self.stemmer:
            return tokens

        stemmed_tokens = []
        for token in tokens:
            if self.stemming == "light":
                # Light stemming: only stem common suffixes, preserve code terms
                stemmed = self._light_stem(token)
            else:  # aggressive
                stemmed = self.stemmer.stem(token)
            stemmed_tokens.append(stemmed)

        return stemmed_tokens

    def _light_stem(self, token: str) -> str:
        """Apply light stemming optimized for code search."""
        # Preserve short tokens and code-like patterns
        if len(token) <= 3:
            return token

        # Don't stem tokens that look like code (contain numbers, mixed case, etc.)
        if any(char.isdigit() for char in token):
            return token

        # Only stem common English suffixes
        light_suffixes = {
            'ing': '',
            'ed': '',
            'er': '',
            'est': '',
            's': ''  # Only remove trailing 's'
        }

        for suffix, replacement in light_suffixes.items():
            if token.endswith(suffix) and len(token) > len(suffix) + 2:
                return token[:-len(suffix)] + replacement

        return token
    
    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Tokenize batch of texts efficiently.
        
        Args:
            texts: List of input texts to tokenize
            
        Returns:
            List of token lists for each input text
        """
        return [self.tokenize(text) for text in texts]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tokenization statistics."""
        return {
            "tokenization_count": self.tokenization_count,
            "stemming_enabled": self.stemming,
            "stopwords_count": len(self.stop_words) if self.stop_words else 0,
            "min_token_length": self.min_token_length
        }


class VoyageTokenizer(BaseTokenizer):
    """
    Dedicated tokenizer for Voyage AI semantic search.
    
    Handles only semantic search tokenization using Voyage AI API.
    Maintains existing Voyage tokenization functionality but isolates
    it to semantic search operations only.
    """
    
    def __init__(self, 
                 model: str = "voyage-context-3",
                 batch_size: int = 100):
        """
        Initialize Voyage tokenizer for semantic search.
        
        Args:
            model: Voyage AI model for tokenization
            batch_size: Batch size for API calls
        """
        self.model = model
        self.batch_size = batch_size
        self.vo: Optional[Any] = None
        self.api_calls = 0
        
    def _get_voyage_client(self) -> Any:
        """Get or initialize Voyage AI client."""
        if self.vo is None:
            try:
                import voyageai
                api_key = os.getenv('VOYAGE_API_KEY')
                if not api_key:
                    raise ValueError(
                        "VOYAGE_API_KEY environment variable not set. "
                        "Please set it with: export VOYAGE_API_KEY='your-api-key'"
                    )
                self.vo = voyageai.Client(api_key=api_key)
            except ImportError:
                raise ImportError("voyageai package required for VoyageTokenizer")
        return self.vo
    
    def tokenize(self, text: str) -> List[str]:
        """
        Tokenize single text using Voyage AI.
        
        Args:
            text: Input text to tokenize
            
        Returns:
            List of Voyage AI tokens
        """
        if not text:
            return []
        
        try:
            vo = self._get_voyage_client()
            self.api_calls += 1
            
            result = vo.tokenize([text], model=self.model)
            return list(result[0].tokens)
            
        except Exception as e:
            print(f"❌ Voyage tokenization failed: {e}")
            # Fallback to simple tokenization
            return text.lower().split()
    
    def tokenize_batch(self, texts: List[str]) -> List[List[str]]:
        """
        Tokenize batch of texts using Voyage AI.
        
        Args:
            texts: List of input texts to tokenize
            
        Returns:
            List of token lists for each input text
        """
        if not texts:
            return []
        
        try:
            vo = self._get_voyage_client()
            all_tokenized = []
            
            # Process in batches for efficiency
            for i in range(0, len(texts), self.batch_size):
                batch_texts = texts[i:i + self.batch_size]
                self.api_calls += 1
                
                tokenized_results = vo.tokenize(batch_texts, model=self.model)
                batch_tokenized = [list(result.tokens) for result in tokenized_results]
                all_tokenized.extend(batch_tokenized)
            
            return all_tokenized
            
        except Exception as e:
            print(f"❌ Voyage batch tokenization failed: {e}")
            # Fallback to simple tokenization
            return [text.lower().split() for text in texts]
    
    def get_stats(self) -> Dict[str, Any]:
        """Get tokenization statistics."""
        return {
            "api_calls": self.api_calls,
            "model": self.model,
            "batch_size": self.batch_size
        }


class TokenizationManager:
    """
    Coordinates independent tokenization strategies for hybrid search.
    
    Provides clean interfaces for BM25 and Voyage tokenization while
    tracking performance metrics and managing configuration.
    """
    
    def __init__(self, 
                 bm25_config: Optional[Dict[str, Any]] = None,
                 voyage_config: Optional[Dict[str, Any]] = None):
        """
        Initialize tokenization manager with independent optimizers.
        
        Args:
            bm25_config: Configuration for BM25Tokenizer
            voyage_config: Configuration for VoyageTokenizer
        """
        # Default configurations optimized for code search (based on optimization results)
        default_bm25_config = {
            "stemming": "none",  # No stemming preserves code terms better
            "stopwords_lang": "english",
            "lowercase": True,
            "min_token_length": 2,
            "remove_punctuation": True,
            "split_camelcase": True,  # Essential for code search
            "split_underscores": True  # Essential for code search
        }
        
        default_voyage_config = {
            "model": "voyage-context-3",
            "batch_size": 100
        }
        
        # Merge with user configurations
        bm25_config = {**default_bm25_config, **(bm25_config or {})}
        voyage_config = {**default_voyage_config, **(voyage_config or {})}
        
        # Initialize tokenizers
        self.bm25_tokenizer = BM25Tokenizer(**bm25_config)
        self.voyage_tokenizer = VoyageTokenizer(**voyage_config)
        
        print("🔧 TokenizationManager initialized with independent optimizers:")
        print(f"   BM25: Local processing with {len(self.bm25_tokenizer.stop_words)} stopwords")
        print(f"   Voyage: API-based semantic tokenization ({voyage_config['model']})")
    
    def tokenize_for_bm25(self, text: str) -> List[str]:
        """Tokenize text for BM25 lexical search."""
        return self.bm25_tokenizer.tokenize(text)
    
    def tokenize_batch_for_bm25(self, texts: List[str]) -> List[List[str]]:
        """Tokenize batch of texts for BM25 lexical search."""
        return self.bm25_tokenizer.tokenize_batch(texts)
    
    def tokenize_for_voyage(self, text: str) -> List[str]:
        """Tokenize text for Voyage semantic search."""
        return self.voyage_tokenizer.tokenize(text)
    
    def tokenize_batch_for_voyage(self, texts: List[str]) -> List[List[str]]:
        """Tokenize batch of texts for Voyage semantic search."""
        return self.voyage_tokenizer.tokenize_batch(texts)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics."""
        return {
            "bm25_stats": self.bm25_tokenizer.get_stats(),
            "voyage_stats": self.voyage_tokenizer.get_stats(),
            "api_call_reduction": "BM25 tokenization uses zero API calls"
        }
