"""Modern Async-First Linguistic Feature Extractor (2025)

A completely modernized feature extraction component implementing 2025 best practices:
- Async-first architecture with full orchestrator integration
- Pydantic models for type safety and validation
- Structured logging with correlation IDs
- Health monitoring and metrics collection
- Dependency injection and configuration management
- Modern Python patterns (type hints, dataclasses, context managers)
"""

import asyncio
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Union
from decimal import Decimal

import numpy as np
from pydantic import BaseModel, Field, field_validator

from ....security.input_sanitization import InputSanitizer
from ....utils.redis_cache import RedisCache
from .english_nltk_manager import get_english_nltk_manager

# Modern async imports
try:
    import aiofiles
    import aioredis
    ASYNC_SUPPORT = True
except ImportError:
    ASYNC_SUPPORT = False

# Enhanced linguistic analysis
try:
    from ...analysis.linguistic_analyzer import (
        LinguisticAnalyzer,
        LinguisticConfig,
    )
    LINGUISTIC_ANALYSIS_AVAILABLE = True
except ImportError:
    LINGUISTIC_ANALYSIS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Modern Pydantic models for configuration and validation
class FeatureExtractionConfig(BaseModel):
    """Configuration model with validation and type safety."""
    
    weight: float = Field(default=1.0, ge=0.0, le=10.0)
    cache_enabled: bool = Field(default=True)
    cache_ttl_seconds: int = Field(default=3600, ge=60, le=86400)
    deterministic: bool = Field(default=True)
    max_text_length: int = Field(default=100000, ge=100, le=1000000)
    enable_metrics: bool = Field(default=True)
    async_batch_size: int = Field(default=10, ge=1, le=100)
    
    @field_validator('weight')
    @classmethod
    def validate_weight(cls, v):
        if not isinstance(v, (int, float, Decimal)):
            raise ValueError('Weight must be numeric')
        return float(v)

class FeatureExtractionRequest(BaseModel):
    """Request model for feature extraction with validation."""
    
    text: str = Field(min_length=1, max_length=100000)
    context: Optional[Dict[str, Any]] = Field(default=None)
    correlation_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    priority: int = Field(default=1, ge=1, le=3)
    
    @field_validator('text')
    @classmethod
    def validate_text_content(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty or whitespace only')
        return v.strip()

class FeatureExtractionResponse(BaseModel):
    """Response model with comprehensive metadata."""
    
    features: List[float] = Field(min_items=10, max_items=10)
    feature_names: List[str] = Field(min_items=10, max_items=10)
    extraction_time_ms: float = Field(ge=0)
    cache_hit: bool = Field(default=False)
    correlation_id: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    confidence_score: float = Field(ge=0.0, le=1.0, default=1.0)
    
    @field_validator('features')
    @classmethod
    def validate_feature_range(cls, v):
        if any(f < 0.0 or f > 1.0 for f in v):
            raise ValueError('All features must be in range [0.0, 1.0]')
        return v

@dataclass
class ExtractionMetrics:
    """Metrics collection for monitoring and performance analysis."""
    
    total_extractions: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_extraction_time_ms: float = 0.0
    errors: int = 0
    last_extraction: Optional[datetime] = None
    average_response_time_ms: float = 0.0
    
    def update_timing(self, extraction_time_ms: float, cache_hit: bool = False):
        """Update timing metrics."""
        self.total_extractions += 1
        self.total_extraction_time_ms += extraction_time_ms
        self.last_extraction = datetime.now(timezone.utc)
        
        if cache_hit:
            self.cache_hits += 1
        else:
            self.cache_misses += 1
            
        self.average_response_time_ms = self.total_extraction_time_ms / self.total_extractions
    
    def increment_errors(self):
        """Track error occurrences."""
        self.errors += 1
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate percentage."""
        if self.total_extractions == 0:
            return 0.0
        return (self.cache_hits / self.total_extractions) * 100
    
    @property
    def error_rate(self) -> float:
        """Calculate error rate percentage."""
        if self.total_extractions == 0:
            return 0.0
        return (self.errors / self.total_extractions) * 100

class LinguisticFeatureExtractor:
    """Modern async-first linguistic feature extractor with orchestrator integration.
    
    Implements 2025 best practices:
    - Async operations with proper context management
    - Pydantic models for type safety
    - Comprehensive health monitoring
    - Distributed caching with Redis
    - Structured logging with correlation IDs
    - Dependency injection
    - Performance metrics collection
    """
    
    def __init__(
        self,
        config: Optional[FeatureExtractionConfig] = None,
        input_sanitizer: Optional[InputSanitizer] = None,
        redis_cache: Optional[RedisCache] = None
    ):
        """Initialize with dependency injection and configuration."""
        self.config = config or FeatureExtractionConfig()
        self.input_sanitizer = input_sanitizer or InputSanitizer()
        self.redis_cache = redis_cache
        
        # Metrics and monitoring
        self.metrics = ExtractionMetrics()
        self._correlation_id = str(uuid.uuid4())
        
        # NLTK manager for fallback processing
        self.nltk_manager = get_english_nltk_manager()
        
        # Linguistic analyzer (if available)
        if LINGUISTIC_ANALYSIS_AVAILABLE:
            try:
                self.linguistic_analyzer = LinguisticAnalyzer(
                    config=LinguisticConfig(
                        enable_caching=self.config.cache_enabled,
                        cache_size=1000,
                        auto_download_nltk=False,
                        nltk_fallback_enabled=True
                    )
                )
                logger.info("Linguistic analyzer initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize linguistic analyzer: {e}")
                self.linguistic_analyzer = None
        else:
            self.linguistic_analyzer = None
        
        # Feature names (immutable)
        self._feature_names = [
            "readability_score",
            "lexical_diversity", 
            "entity_density",
            "syntactic_complexity",
            "sentence_structure_quality",
            "technical_term_ratio",
            "avg_sentence_length_norm",
            "instruction_clarity",
            "has_examples",
            "overall_linguistic_quality"
        ]
        
        logger.info(
            f"LinguisticFeatureExtractor initialized",
            extra={
                "component": "linguistic_feature_extractor",
                "config": self.config.model_dump(),
                "correlation_id": self._correlation_id
            }
        )
    
    async def extract_features(
        self, 
        request: Union[FeatureExtractionRequest, str],
        context: Optional[Dict[str, Any]] = None
    ) -> FeatureExtractionResponse:
        """Extract linguistic features asynchronously with full validation and monitoring.
        
        Args:
            request: Either a FeatureExtractionRequest object or text string
            context: Optional context for extraction (deprecated, use request object)
            
        Returns:
            FeatureExtractionResponse with features and metadata
            
        Raises:
            ValueError: If input validation fails
            RuntimeError: If extraction process fails
        """
        # Handle empty/invalid string inputs before Pydantic validation
        if isinstance(request, str):
            text = request.strip() if request else ""
            if not text:
                # Return default response for empty input without Pydantic validation
                correlation_id = str(uuid.uuid4())
                return self._create_default_response(correlation_id, 0.0, False)
            
            # Create request object for valid text
            try:
                request = FeatureExtractionRequest(
                    text=text,
                    context=context,
                    correlation_id=str(uuid.uuid4())
                )
            except Exception as e:
                # Fallback for any validation issues
                correlation_id = str(uuid.uuid4())
                return self._create_default_response(correlation_id, 0.0, False)
        
        start_time = time.perf_counter()
        cache_hit = False
        
        try:
            # Input validation and sanitization
            validated_text = await self._validate_and_sanitize_input(request.text)
            if not validated_text:
                return self._create_default_response(request.correlation_id, 0.0, cache_hit)
            
            # Check cache first
            cached_result = None
            if self.config.cache_enabled and self.redis_cache:
                cached_result = await self._get_cached_features(validated_text, request.correlation_id)
                if cached_result:
                    cache_hit = True
                    extraction_time = (time.perf_counter() - start_time) * 1000
                    self.metrics.update_timing(extraction_time, cache_hit=True)
                    return cached_result
            
            # Extract features using async processing
            features = await self._compute_linguistic_features_async(
                validated_text, 
                request.correlation_id
            )
            
            # Normalize and validate features
            normalized_features = self._normalize_features(features)
            
            # Calculate extraction time
            extraction_time = (time.perf_counter() - start_time) * 1000
            
            # Create response
            response = FeatureExtractionResponse(
                features=normalized_features,
                feature_names=self._feature_names,
                extraction_time_ms=extraction_time,
                cache_hit=cache_hit,
                correlation_id=request.correlation_id,
                confidence_score=self._calculate_confidence_score(features)
            )
            
            # Cache results asynchronously
            if self.config.cache_enabled and self.redis_cache:
                await self._cache_features_async(validated_text, response)
            
            # Update metrics
            self.metrics.update_timing(extraction_time, cache_hit=False)
            
            logger.info(
                f"features extracted successfully",
                extra={
                    "component": "linguistic_feature_extractor",
                    "extraction_time_ms": extraction_time,
                    "cache_hit": cache_hit,
                    "correlation_id": request.correlation_id,
                    "text_length": len(validated_text)
                }
            )
            
            return response
            
        except Exception as e:
            extraction_time = (time.perf_counter() - start_time) * 1000
            self.metrics.increment_errors()
            
            logger.error(
                f"Feature extraction failed: {e}",
                extra={
                    "component": "linguistic_feature_extractor",
                    "error": str(e),
                    "correlation_id": request.correlation_id,
                    "extraction_time_ms": extraction_time
                },
                exc_info=True
            )
            
            # Return default response on error
            return self._create_default_response(request.correlation_id, extraction_time, cache_hit)
    
    async def batch_extract_features(
        self, 
        requests: List[FeatureExtractionRequest]
    ) -> List[FeatureExtractionResponse]:
        """Extract features for multiple texts concurrently."""
        # Process in batches to avoid overwhelming the system
        batch_size = self.config.async_batch_size
        results = []
        
        for i in range(0, len(requests), batch_size):
            batch = requests[i:i + batch_size]
            batch_results = await asyncio.gather(
                *[self.extract_features(request) for request in batch],
                return_exceptions=True
            )
            
            # Handle exceptions in batch results
            for j, result in enumerate(batch_results):
                if isinstance(result, Exception):
                    logger.error(f"Batch extraction failed for request {i+j}: {result}")
                    results.append(self._create_default_response(
                        batch[j].correlation_id, 0.0, False
                    ))
                else:
                    results.append(result)
        
        return results
    
    async def _validate_and_sanitize_input(self, text: str) -> Optional[str]:
        """Async input validation and sanitization."""
        try:
            if not text or not isinstance(text, str):
                return None
            
            if len(text.strip()) == 0:
                return None
            
            if len(text) > self.config.max_text_length:
                logger.warning(f"Text truncated from {len(text)} to {self.config.max_text_length} characters")
                text = text[:self.config.max_text_length]
            
            # Async sanitization
            sanitized_text = await asyncio.to_thread(
                self.input_sanitizer.sanitize_html_input, 
                text
            )
            
            return sanitized_text.strip()
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return None
    
    async def _get_cached_features(
        self, 
        text: str, 
        correlation_id: str
    ) -> Optional[FeatureExtractionResponse]:
        """Retrieve cached features asynchronously."""
        if not self.redis_cache:
            return None
        
        try:
            cache_key = self._generate_cache_key(text)
            cached_data = await self.redis_cache.get_async(cache_key)
            
            if cached_data:
                # Reconstruct response from cached data
                response_data = cached_data.copy()
                response_data['correlation_id'] = correlation_id  # Update correlation ID
                response_data['cache_hit'] = True
                return FeatureExtractionResponse(**response_data)
            
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}", extra={"correlation_id": correlation_id})
        
        return None
    
    async def _cache_features_async(
        self, 
        text: str, 
        response: FeatureExtractionResponse
    ) -> None:
        """Cache features asynchronously."""
        if not self.redis_cache:
            return
        
        try:
            cache_key = self._generate_cache_key(text)
            cache_data = response.model_dump()
            cache_data['cache_hit'] = False  # Reset for caching
            
            await self.redis_cache.set_async(
                cache_key, 
                cache_data, 
                ttl=self.config.cache_ttl_seconds
            )
            
        except Exception as e:
            logger.warning(f"Failed to cache features: {e}")
    
    async def _compute_linguistic_features_async(
        self, 
        text: str, 
        correlation_id: str
    ) -> List[float]:
        """Compute linguistic features asynchronously."""
        if self.config.deterministic:
            # Set deterministic seeds for reproducible results
            np.random.seed(42)
        
        if self.linguistic_analyzer:
            return await self._extract_with_analyzer_async(text, correlation_id)
        else:
            return await self._extract_fallback_features_async(text, correlation_id)
    
    async def _extract_with_analyzer_async(
        self, 
        text: str, 
        correlation_id: str
    ) -> List[float]:
        """Extract features using linguistic analyzer asynchronously."""
        try:
            # Run analyzer in thread pool to avoid blocking
            linguistic_features = await asyncio.to_thread(
                self.linguistic_analyzer.analyze, 
                text
            )
            
            features = [
                linguistic_features.readability_score,
                linguistic_features.lexical_diversity,
                linguistic_features.entity_density,
                linguistic_features.syntactic_complexity,
                linguistic_features.sentence_structure_quality,
                linguistic_features.technical_term_ratio,
                min(1.0, linguistic_features.avg_sentence_length / 50.0),
                linguistic_features.instruction_clarity,
                1.0 if linguistic_features.has_examples else 0.0,
                linguistic_features.overall_quality
            ]
            
            return features
            
        except Exception as e:
            logger.error(
                f"Linguistic analysis failed: {e}",
                extra={"correlation_id": correlation_id}
            )
            return await self._extract_fallback_features_async(text, correlation_id)
    
    async def _extract_fallback_features_async(
        self, 
        text: str, 
        correlation_id: str
    ) -> List[float]:
        """Extract features using NLTK fallback asynchronously."""
        try:
            # Run NLTK processing in thread pool
            features = await asyncio.to_thread(
                self._compute_nltk_features, 
                text
            )
            return features
            
        except Exception as e:
            logger.error(
                f"Fallback feature extraction failed: {e}",
                extra={"correlation_id": correlation_id}
            )
            # Ultimate fallback
            return [0.5] * 10
    
    def _compute_nltk_features(self, text: str) -> List[float]:
        """Compute features using NLTK (synchronous)."""
        try:
            sentence_tokenizer = self.nltk_manager.get_sentence_tokenizer()
            word_tokenizer = self.nltk_manager.get_word_tokenizer()
            stopwords = self.nltk_manager.get_english_stopwords()

            sentences = sentence_tokenizer(text)
            words = word_tokenizer(text)
            content_words = [word for word in words if word.lower() not in stopwords]

            features = [
                # Readability (inverse of average sentence length)
                min(1.0, 1.0 - (sum(len(s.split()) for s in sentences) / len(sentences) / 30.0)) if sentences else 0.5,
                # Lexical diversity
                min(1.0, len(set(words)) / len(words)) if words else 0.0,
                # Entity density approximation
                self._estimate_entity_density(text, sentences),
                # Syntactic complexity
                min(1.0, (sum(len(s.split()) for s in sentences) / len(sentences)) / 25.0) if sentences else 0.0,
                # Sentence structure quality
                self._calculate_sentence_variety(sentences),
                # Technical term ratio
                self._calculate_technical_ratio(content_words),
                # Average sentence length (normalized)
                min(1.0, (sum(len(s.split()) for s in sentences) / len(sentences)) / 20.0) if sentences else 0.0,
                # Instruction clarity
                self._calculate_instruction_clarity(words),
                # Has examples
                1.0 if any(word in text.lower() for word in ['example', 'for instance', 'such as', 'e.g.', 'i.e.']) else 0.0,
                # Overall quality
                self._calculate_overall_quality(text, words, sentences)
            ]

            return features

        except Exception as e:
            logger.error(f"NLTK feature computation failed: {e}")
            return [0.5] * 10
    
    def _estimate_entity_density(self, text: str, sentences: List[str]) -> float:
        """Estimate entity density based on capitalization patterns."""
        try:
            import re
            capitalized_words = 0
            total_words = 0

            for sentence in sentences:
                words = sentence.split()
                if not words:
                    continue

                for word in words[1:]:  # Skip first word (sentence start)
                    total_words += 1
                    if re.match(r'^[A-Z][a-zA-Z]+', word):
                        capitalized_words += 1

            return min(1.0, capitalized_words / total_words) if total_words > 0 else 0.0

        except Exception:
            return 0.5
    
    def _calculate_sentence_variety(self, sentences: List[str]) -> float:
        """Calculate sentence length variety as quality indicator."""
        try:
            if len(sentences) < 2:
                return 0.5

            lengths = [len(s.split()) for s in sentences]
            if not lengths:
                return 0.5

            mean_length = sum(lengths) / len(lengths)
            if mean_length == 0:
                return 0.5

            variance = sum((l - mean_length) ** 2 for l in lengths) / len(lengths)
            std_dev = variance ** 0.5
            cv = std_dev / mean_length

            return min(1.0, cv / 0.5)

        except Exception:
            return 0.5
    
    def _calculate_technical_ratio(self, words: List[str]) -> float:
        """Calculate ratio of technical/specialized terms."""
        try:
            if not words:
                return 0.0

            technical_indicators = {
                'function', 'method', 'class', 'variable', 'algorithm', 'data', 'model',
                'system', 'process', 'analysis', 'implementation', 'framework', 'library',
                'api', 'database', 'server', 'client', 'interface', 'protocol', 'network'
            }

            technical_count = sum(1 for word in words if word.lower() in technical_indicators)
            return min(1.0, technical_count / len(words))

        except Exception:
            return 0.5
    
    def _calculate_instruction_clarity(self, words: List[str]) -> float:
        """Calculate instruction clarity based on action words."""
        try:
            if not words:
                return 0.0

            action_words = {
                'create', 'make', 'build', 'develop', 'implement', 'design', 'write',
                'generate', 'produce', 'construct', 'establish', 'form', 'execute',
                'perform', 'run', 'process', 'analyze', 'evaluate', 'assess', 'review'
            }

            action_count = sum(1 for word in words if word.lower() in action_words)
            clarity_score = min(1.0, action_count / max(1, len(words) / 10))

            return clarity_score

        except Exception:
            return 0.5
    
    def _calculate_overall_quality(self, text: str, words: List[str], sentences: List[str]) -> float:
        """Calculate overall linguistic quality composite score."""
        try:
            quality_factors = []

            # Length appropriateness
            text_length = len(text)
            if 50 <= text_length <= 500:
                quality_factors.append(1.0)
            elif text_length < 50:
                quality_factors.append(text_length / 50.0)
            else:
                quality_factors.append(max(0.3, 1.0 - (text_length - 500) / 1000.0))

            # Word variety
            if words:
                word_variety = len(set(words)) / len(words)
                quality_factors.append(word_variety)

            # Sentence structure
            if sentences:
                avg_sentence_length = sum(len(s.split()) for s in sentences) / len(sentences)
                if 10 <= avg_sentence_length <= 25:
                    quality_factors.append(1.0)
                else:
                    quality_factors.append(max(0.3, 1.0 - abs(avg_sentence_length - 17.5) / 17.5))

            return sum(quality_factors) / len(quality_factors) if quality_factors else 0.5

        except Exception:
            return 0.5
    
    def _normalize_features(self, features: List[float]) -> List[float]:
        """Normalize features to [0, 1] range with validation."""
        normalized = []
        for i, feature in enumerate(features):
            try:
                normalized_feature = min(1.0, max(0.0, float(feature)))
                normalized.append(normalized_feature)
            except (ValueError, TypeError):
                logger.warning(f"Invalid feature value at index {i}: {feature}, using 0.5")
                normalized.append(0.5)
        
        return normalized
    
    def _calculate_confidence_score(self, features: List[float]) -> float:
        """Calculate confidence score based on feature consistency."""
        if not features:
            return 0.0
        
        # Higher confidence for features that are not at extremes or defaults
        non_default_features = [f for f in features if f != 0.5]
        if not non_default_features:
            return 0.5  # All defaults, medium confidence
        
        # Calculate variance as indicator of meaningful processing
        mean_feature = sum(non_default_features) / len(non_default_features)
        variance = sum((f - mean_feature) ** 2 for f in non_default_features) / len(non_default_features)
        
        # Higher variance suggests more discriminative features
        confidence = min(1.0, 0.5 + variance * 2)
        return confidence
    
    def _generate_cache_key(self, text: str) -> str:
        """Generate cache key for text with configuration hash."""
        import hashlib
        
        # Include configuration in cache key to invalidate on config changes
        config_hash = hashlib.md5(
            self.config.model_dump_json().encode(), usedforsecurity=False
        ).hexdigest()[:8]
        
        text_hash = hashlib.md5(text.encode(), usedforsecurity=False).hexdigest()
        return f"linguistic_features:{config_hash}:{text_hash}"
    
    def _create_default_response(
        self, 
        correlation_id: str, 
        extraction_time: float, 
        cache_hit: bool
    ) -> FeatureExtractionResponse:
        """Create default response for error cases."""
        return FeatureExtractionResponse(
            features=[0.5] * 10,
            feature_names=self._feature_names,
            extraction_time_ms=extraction_time,
            cache_hit=cache_hit,
            correlation_id=correlation_id,
            confidence_score=0.0
        )
    
    # Orchestrator Integration Methods
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check endpoint for orchestrator monitoring."""
        try:
            # Test basic functionality
            test_request = FeatureExtractionRequest(
                text="Health check test text for feature extraction.",
                correlation_id="health-check"
            )
            
            start_time = time.perf_counter()
            response = await self.extract_features(test_request)
            health_check_time = (time.perf_counter() - start_time) * 1000
            
            return {
                "status": "healthy",
                "component": "linguistic_feature_extractor",
                "version": "2.0.0",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metrics": {
                    "total_extractions": self.metrics.total_extractions,
                    "cache_hit_rate": self.metrics.cache_hit_rate,
                    "error_rate": self.metrics.error_rate,
                    "average_response_time_ms": self.metrics.average_response_time_ms,
                    "health_check_time_ms": health_check_time
                },
                "configuration": {
                    "cache_enabled": self.config.cache_enabled,
                    "linguistic_analyzer_available": self.linguistic_analyzer is not None,
                    "async_support": ASYNC_SUPPORT
                }
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "component": "linguistic_feature_extractor",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive metrics for monitoring."""
        return {
            "component": "linguistic_feature_extractor",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": {
                "total_extractions": self.metrics.total_extractions,
                "cache_hits": self.metrics.cache_hits,
                "cache_misses": self.metrics.cache_misses,
                "cache_hit_rate": self.metrics.cache_hit_rate,
                "total_extraction_time_ms": self.metrics.total_extraction_time_ms,
                "average_response_time_ms": self.metrics.average_response_time_ms,
                "errors": self.metrics.errors,
                "error_rate": self.metrics.error_rate,
                "last_extraction": self.metrics.last_extraction.isoformat() if self.metrics.last_extraction else None
            },
            "configuration": self.config.model_dump()
        }
    
    def get_feature_names(self) -> List[str]:
        """Get feature names for model compatibility."""
        return self._feature_names.copy()
    
    def extract_features_sync(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[float]:
        """Synchronous wrapper for async feature extraction.
        
        Args:
            text: Input text to analyze
            context: Optional context for extraction
            
        Returns:
            List of extracted features
        """
        # Check if we're already in an async context
        try:
            loop = asyncio.get_running_loop()
            # If we're in an async context, we need to run in a thread pool
            import concurrent.futures
            
            # Create a new event loop in a separate thread
            def run_in_thread():
                new_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(new_loop)
                try:
                    request = FeatureExtractionRequest(text=text)
                    result = new_loop.run_until_complete(self.extract_features(request, context))
                    return result.features if hasattr(result, 'features') else result
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=30)  # 30 second timeout
                
        except RuntimeError:
            # No event loop running, we can create one
            request = FeatureExtractionRequest(text=text)
            result = asyncio.run(self.extract_features(request, context))
            return result.features if hasattr(result, 'features') else result
    
    async def clear_cache(self) -> Dict[str, Any]:
        """Clear cache and return operation result."""
        if not self.redis_cache:
            return {"status": "no_cache", "message": "Redis cache not configured"}
        
        try:
            # Clear all feature extraction cache keys
            pattern = "linguistic_features:*"
            cleared_count = await self.redis_cache.clear_pattern_async(pattern)
            
            return {
                "status": "success",
                "cleared_entries": cleared_count,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def shutdown(self) -> None:
        """Graceful shutdown with resource cleanup."""
        logger.info("Starting graceful shutdown of LinguisticFeatureExtractor")
        
        try:
            # Close Redis connections
            if self.redis_cache:
                await self.redis_cache.close_async()
            
            # Log final metrics
            final_metrics = await self.get_metrics()
            logger.info(
                "LinguisticFeatureExtractor shutdown complete",
                extra=final_metrics
            )
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    @asynccontextmanager
    async def extraction_context(self):
        """Context manager for batch processing with resource management."""
        logger.info("Starting extraction context")
        try:
            yield self
        finally:
            logger.info("Extraction context completed")
    
    def __repr__(self) -> str:
        """String representation for debugging."""
        return (
            f"LinguisticFeatureExtractor("
            f"weight={self.config.weight}, "
            f"cache_enabled={self.config.cache_enabled}, "
            f"deterministic={self.config.deterministic}, "
            f"extractions={self.metrics.total_extractions}"
            f")"
        )