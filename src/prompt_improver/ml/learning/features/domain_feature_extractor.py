"""Domain Feature Extractor

Modernized 2025 implementation for extracting domain-specific features from prompt text.
Features async/await patterns, Pydantic validation, circuit breaker, and orchestrator integration.

Features:
- Async-first architecture with non-blocking operations
- Pydantic models for configuration and data validation
- Circuit breaker pattern for fault tolerance
- Comprehensive observability and health monitoring
- ML Pipeline Orchestrator integration
- Type-safe configuration and validation
- Intelligent caching with TTL and cleanup
- Structured logging with correlation IDs
"""

import asyncio
import hashlib
import logging
import random
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import numpy as np
from pydantic import BaseModel, Field, field_validator

from ....security import InputValidator, ValidationError
from ....security.input_sanitization import InputSanitizer
from ....utils.datetime_utils import aware_utc_now
from ...analysis.domain_detector import PromptDomain

# Domain analysis integration with fallback
try:
    from ...analysis.domain_analyzer import DomainAnalyzer
    DOMAIN_ANALYSIS_AVAILABLE = True
except ImportError:
    DOMAIN_ANALYSIS_AVAILABLE = False

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    """Circuit breaker states for fault tolerance."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class DomainFeatureConfig(BaseModel):
    """Enhanced 2025 configuration model for domain feature extraction."""
    
    # Feature extraction configuration
    weight: float = Field(default=1.0, ge=0.0, le=10.0, description="Weight for domain features")
    deterministic: bool = Field(default=True, description="Use deterministic random seeds")
    
    # 2025: Advanced caching configuration
    cache_enabled: bool = Field(default=True, description="Enable result caching")
    cache_ttl_seconds: int = Field(default=3600, ge=0, description="Cache TTL in seconds")
    cache_max_size: int = Field(default=1000, ge=1, description="Maximum cache entries")
    cache_cleanup_interval: int = Field(default=300, ge=60, description="Cache cleanup interval in seconds")
    
    # 2025: Processing configuration with resource awareness
    timeout_seconds: float = Field(default=30.0, gt=0.0, description="Operation timeout")
    max_retries: int = Field(default=3, ge=0, le=10, description="Maximum retry attempts")
    memory_limit_mb: int = Field(default=512, gt=0, description="Memory limit in MB")
    
    # 2025: Circuit breaker configuration
    circuit_breaker_enabled: bool = Field(default=True, description="Enable circuit breaker")
    failure_threshold: int = Field(default=5, ge=1, description="Circuit breaker failure threshold")
    recovery_timeout: int = Field(default=60, ge=1, description="Circuit breaker recovery timeout")
    
    # Text processing quality thresholds
    min_text_length: int = Field(default=10, ge=0, description="Minimum text length required")
    max_text_length: int = Field(default=100000, gt=0, description="Maximum text length")
    default_feature_value: float = Field(default=0.5, ge=0.0, le=1.0, description="Default feature value")
    
    # Domain analysis configuration
    use_domain_analyzer: bool = Field(default=True, description="Use domain analyzer if available")
    fallback_on_analyzer_failure: bool = Field(default=True, description="Use fallback features if analyzer fails")
    
    # 2025: Observability configuration
    enable_metrics: bool = Field(default=True, description="Enable performance metrics")
    enable_health_checks: bool = Field(default=True, description="Enable health monitoring")
    log_level: str = Field(default="INFO", description="Logging level")
    correlation_id_enabled: bool = Field(default=True, description="Enable correlation ID tracking")
    
    @field_validator('log_level')
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid_levels = {'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'}
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of: {valid_levels}")
        return v.upper()


@dataclass
class DomainExtractionMetrics:
    """Metrics for domain feature extraction operations."""
    total_extractions: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    analyzer_extractions: int = 0
    fallback_extractions: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    circuit_breaker_trips: int = 0
    last_extraction_time: Optional[float] = None
    
    def update_success(self, processing_time: float, used_analyzer: bool = False):
        """Update metrics for successful extraction."""
        self.total_extractions += 1
        self.successful_extractions += 1
        if used_analyzer:
            self.analyzer_extractions += 1
        else:
            self.fallback_extractions += 1
        self.total_processing_time += processing_time
        self.average_processing_time = self.total_processing_time / self.total_extractions
        self.last_extraction_time = time.time()
    
    def update_failure(self):
        """Update metrics for failed extraction."""
        self.total_extractions += 1
        self.failed_extractions += 1
        self.last_extraction_time = time.time()
    
    def get_success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_extractions == 0:
            return 100.0
        return (self.successful_extractions / self.total_extractions) * 100.0
    
    def get_analyzer_usage_rate(self) -> float:
        """Get analyzer usage rate as percentage."""
        if self.successful_extractions == 0:
            return 0.0
        return (self.analyzer_extractions / self.successful_extractions) * 100.0
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status."""
        return {
            "total_extractions": self.total_extractions,
            "success_rate": self.get_success_rate(),
            "analyzer_usage_rate": self.get_analyzer_usage_rate(),
            "average_processing_time": self.average_processing_time,
            "cache_hit_rate": (self.cache_hits / (self.cache_hits + self.cache_misses) * 100
                              if (self.cache_hits + self.cache_misses) > 0 else 0),
            "circuit_breaker_trips": self.circuit_breaker_trips,
            "last_extraction_time": self.last_extraction_time
        }


class DomainFeatureExtractor:
    """Enhanced 2025 domain feature extractor with async operations and observability.
    
    Extracts 15 normalized domain-specific features:
    1. Domain confidence (0-1)
    2. Domain complexity score (0-1)
    3. Domain specificity score (0-1)
    4. Technical domain indicator (0/1)
    5. Creative domain indicator (0/1)
    6. Academic domain indicator (0/1)
    7. Business domain indicator (0/1)
    8. Technical feature density (0-1)
    9. Creative feature density (0-1)
    10. Academic feature density (0-1)
    11. Conversational politeness score (0-1)
    12. Urgency indicator (0/1)
    13. Question density (0-1)
    14. Instruction clarity (0-1)
    15. Domain hybrid indicator (0/1)
    
    This class implements 2025 best practices:
    - Async/await for non-blocking operations
    - Circuit breaker pattern for fault tolerance
    - Comprehensive metrics and health monitoring
    - Pydantic configuration with validation
    - Intelligent caching with TTL and cleanup
    - ML Pipeline Orchestrator integration
    """
    
    def __init__(self, config: Optional[DomainFeatureConfig] = None):
        """Initialize enhanced domain feature extractor.
        
        Args:
            config: Configuration for domain feature extraction
        """
        self.config = config or DomainFeatureConfig()
        
        # Core dependencies
        self.input_validator = InputValidator()
        self.input_sanitizer = InputSanitizer()
        
        # Initialize domain analyzer if available and enabled
        if DOMAIN_ANALYSIS_AVAILABLE and self.config.use_domain_analyzer:
            try:
                self.domain_analyzer = DomainAnalyzer()
                logger.info("Domain analyzer initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize domain analyzer: {e}")
                self.domain_analyzer = None
        else:
            self.domain_analyzer = None
            if not DOMAIN_ANALYSIS_AVAILABLE:
                logger.warning("Domain analyzer not available, using fallback features")
        
        # 2025: Enhanced state management
        self.metrics = DomainExtractionMetrics()
        self.circuit_breaker_state = CircuitBreakerState.CLOSED
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure: Optional[float] = None
        
        # 2025: Intelligent caching with TTL
        self._feature_cache: Dict[str, Dict[str, Any]] = {} if self.config.cache_enabled else {}
        self._last_cache_cleanup = time.time()
        
        # 2025: Async processing semaphore
        self._extraction_semaphore = asyncio.Semaphore(1)  # Single extractor, limit concurrent access
        
        # Domain categorization (unchanged for compatibility)
        self._technical_domains = {
            PromptDomain.SOFTWARE_DEVELOPMENT,
            PromptDomain.DATA_SCIENCE,
            PromptDomain.AI_ML,
            PromptDomain.WEB_DEVELOPMENT,
            PromptDomain.SYSTEM_ADMIN,
            PromptDomain.API_DOCUMENTATION,
        }
        
        self._creative_domains = {
            PromptDomain.CREATIVE_WRITING,
            PromptDomain.MARKETING,
            PromptDomain.CONTENT_CREATION,
            PromptDomain.STORYTELLING,
        }
        
        self._academic_domains = {
            PromptDomain.RESEARCH,
            PromptDomain.EDUCATION,
            PromptDomain.ACADEMIC_WRITING,
            PromptDomain.SCIENTIFIC,
        }
        
        self._business_domains = {
            PromptDomain.BUSINESS_ANALYSIS,
            PromptDomain.PROJECT_MANAGEMENT,
            PromptDomain.MARKETING,
            PromptDomain.CUSTOMER_SERVICE,
            PromptDomain.SALES,
        }
        
        # Configure logging level
        logger.setLevel(getattr(logging, self.config.log_level))
        
        logger.info(
            f"Enhanced DomainFeatureExtractor initialized with config: "
            f"weight={self.config.weight}, cache={self.config.cache_enabled}, "
            f"analyzer={self.domain_analyzer is not None}, "
            f"circuit_breaker={self.config.circuit_breaker_enabled}, "
            f"timeout={self.config.timeout_seconds}s"
        )
    
    def extract_features(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[float]:
        """Synchronous wrapper for async feature extraction.
        
        Args:
            text: Input text to analyze
            context: Optional context for feature extraction
            
        Returns:
            List of 15 normalized domain features
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
                    return new_loop.run_until_complete(self.extract_features_async(text, context))
                finally:
                    new_loop.close()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(run_in_thread)
                return future.result(timeout=self.config.timeout_seconds + 5)
                
        except RuntimeError:
            # No event loop running, we can create one
            return asyncio.run(self.extract_features_async(text, context))
    
    async def extract_features_async(self, text: str, context: Optional[Dict[str, Any]] = None) -> List[float]:
        """Enhanced 2025 async feature extraction with circuit breaker and observability.
        
        Args:
            text: Input text to analyze
            context: Optional context for feature extraction
            
        Returns:
            List of 15 normalized domain features
        """
        start_time = time.time()
        correlation_id = self._generate_correlation_id() if self.config.correlation_id_enabled else None
        
        # Circuit breaker check
        if not self._is_circuit_breaker_closed():
            logger.warning(f"Circuit breaker is open [correlation_id={correlation_id}]")
            self.metrics.circuit_breaker_trips += 1
            return self._get_default_features()
        
        # Input validation
        validated_text = await self._validate_input_async(text)
        if not validated_text:
            logger.warning(f"Input validation failed, using defaults [correlation_id={correlation_id}]")
            return self._get_default_features()
        
        # Check cache first
        if self.config.cache_enabled:
            cache_key = self._generate_cache_key(validated_text)
            cached_result = await self._get_cached_features_async(cache_key)
            if cached_result is not None:
                self.metrics.cache_hits += 1
                logger.debug(f"Cache hit for domain feature extraction [correlation_id={correlation_id}]")
                return cached_result
            self.metrics.cache_misses += 1
        
        try:
            async with self._extraction_semaphore:
                # Use timeout for the entire operation
                result, used_analyzer = await asyncio.wait_for(
                    self._perform_extraction_async(validated_text, correlation_id),
                    timeout=self.config.timeout_seconds
                )
                
                # Cache the result
                if self.config.cache_enabled and result:
                    await self._cache_features_async(cache_key, result)
                
                processing_time = time.time() - start_time
                self.metrics.update_success(processing_time, used_analyzer)
                
                logger.info(
                    f"Domain feature extraction completed in {processing_time:.3f}s "
                    f"(analyzer={used_analyzer}) [correlation_id={correlation_id}]"
                )
                return result
        
        except asyncio.TimeoutError:
            logger.error(
                f"Domain feature extraction timed out after {self.config.timeout_seconds}s "
                f"[correlation_id={correlation_id}]"
            )
            self._handle_circuit_breaker_failure()
            self.metrics.update_failure()
            return self._get_default_features()
        
        except Exception as e:
            logger.error(
                f"Async domain feature extraction failed: {e} [correlation_id={correlation_id}]"
            )
            self._handle_circuit_breaker_failure()
            self.metrics.update_failure()
            return self._get_default_features()
    
    async def _validate_input_async(self, text: str) -> Optional[str]:
        """Async validation and sanitization of input text."""
        try:
            if not text or not isinstance(text, str):
                logger.debug("Empty or invalid text provided")
                return None
            
            # Check text length limits
            if len(text.strip()) < self.config.min_text_length:
                logger.debug(f"Text too short: {len(text)} < {self.config.min_text_length}")
                return None
            
            if len(text) > self.config.max_text_length:
                logger.warning(
                    f"Text too long: {len(text)} > {self.config.max_text_length}, truncating"
                )
                text = text[:self.config.max_text_length]
            
            # Run sanitization in executor to avoid blocking
            sanitized_text = await asyncio.get_event_loop().run_in_executor(
                None, self.input_sanitizer.sanitize_html_input, text
            )
            
            return sanitized_text.strip()
            
        except Exception as e:
            logger.error(f"Async text validation failed: {e}")
            return None
    
    def _generate_cache_key(self, text: str) -> str:
        """Generate cache key for text using secure hashing."""
        # Create comprehensive key from text and configuration
        key_elements = [
            text[:500],  # First 500 chars to avoid huge keys
            str(self.config.weight),
            str(self.config.deterministic),
            str(self.domain_analyzer is not None),
            str(len(text))  # Include full text length for uniqueness
        ]
        key_string = '|'.join(key_elements)
        
        # Use secure hash for consistent, collision-resistant keys
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]
    
    def _generate_correlation_id(self) -> str:
        """Generate correlation ID for request tracing."""
        return hashlib.md5(f"{time.time()}_{id(self)}".encode(), usedforsecurity=False).hexdigest()[:8]
    
    async def _get_cached_features_async(self, cache_key: str) -> Optional[List[float]]:
        """Get cached features if available with TTL check."""
        if not self.config.cache_enabled or cache_key not in self._feature_cache:
            return None
        
        cache_entry = self._feature_cache[cache_key]
        current_time = time.time()
        
        # Check TTL
        if current_time - cache_entry['timestamp'] > self.config.cache_ttl_seconds:
            del self._feature_cache[cache_key]
            logger.debug(f"Cache entry expired and removed: {cache_key}")
            return None
        
        # Update access time for LRU
        cache_entry['last_access'] = current_time
        return cache_entry['features']
    
    async def _cache_features_async(self, cache_key: str, features: List[float]) -> None:
        """Cache extracted features with TTL and size management."""
        if not self.config.cache_enabled:
            return
        
        current_time = time.time()
        
        # Cleanup cache if needed
        await self._cleanup_cache_if_needed()
        
        # Add new cache entry
        self._feature_cache[cache_key] = {
            'features': features,
            'timestamp': current_time,
            'last_access': current_time
        }
        
        logger.debug(f"Cached domain features for key: {cache_key}")
    
    async def _cleanup_cache_if_needed(self) -> None:
        """Cleanup cache based on size limits and TTL."""
        current_time = time.time()
        
        # Check if cleanup is needed
        if (current_time - self._last_cache_cleanup < self.config.cache_cleanup_interval and 
            len(self._feature_cache) <= self.config.cache_max_size):
            return
        
        # Remove expired entries
        expired_keys = [
            key for key, entry in self._feature_cache.items()
            if current_time - entry['timestamp'] > self.config.cache_ttl_seconds
        ]
        
        for key in expired_keys:
            del self._feature_cache[key]
        
        # Remove oldest entries if still over limit (LRU)
        if len(self._feature_cache) > self.config.cache_max_size:
            # Sort by last access time and remove oldest
            sorted_entries = sorted(
                self._feature_cache.items(),
                key=lambda x: x[1]['last_access']
            )
            
            entries_to_remove = len(self._feature_cache) - self.config.cache_max_size
            for i in range(entries_to_remove):
                key = sorted_entries[i][0]
                del self._feature_cache[key]
        
        self._last_cache_cleanup = current_time
        logger.debug(f"Domain cache cleanup completed, {len(expired_keys)} expired entries removed")
    
    async def _perform_extraction_async(self, text: str, correlation_id: Optional[str]) -> tuple[List[float], bool]:
        """Perform the actual domain feature extraction.
        
        Returns:
            Tuple of (features, used_analyzer)
        """
        logger.debug(f"Starting domain feature extraction [correlation_id={correlation_id}]")
        
        # Set deterministic seeds if enabled
        if self.config.deterministic:
            random.seed(42)
            np.random.seed(42)
        
        used_analyzer = False
        
        # Try domain analyzer first if available
        if self.domain_analyzer and self.config.use_domain_analyzer:
            try:
                # Run analyzer in executor to avoid blocking
                features = await asyncio.get_event_loop().run_in_executor(
                    None, self._extract_with_analyzer, text
                )
                normalized_features = self._normalize_features(features)
                used_analyzer = True
                
                logger.debug(
                    f"Domain analyzer extracted {len(normalized_features)} features "
                    f"[correlation_id={correlation_id}]"
                )
                
                return normalized_features, used_analyzer
                
            except Exception as e:
                logger.warning(
                    f"Domain analyzer failed: {e}, falling back to basic features "
                    f"[correlation_id={correlation_id}]"
                )
                
                # Fall through to fallback if configured
                if not self.config.fallback_on_analyzer_failure:
                    raise
        
        # Use fallback feature extraction
        features = await asyncio.get_event_loop().run_in_executor(
            None, self._extract_fallback_features, text
        )
        normalized_features = self._normalize_features(features)
        
        logger.debug(
            f"Fallback extraction completed with {len(normalized_features)} features "
            f"[correlation_id={correlation_id}]"
        )
        
        return normalized_features, used_analyzer
    
    def _extract_with_analyzer(self, text: str) -> List[float]:
        """Extract features using domain analyzer."""
        try:
            # Use the domain analyzer interface
            domain_features = self.domain_analyzer.extract_domain_features(text)

            features = [
                # Basic domain metrics
                domain_features.confidence,
                domain_features.complexity_score,
                domain_features.specificity_score,

                # Domain type indicators
                1.0 if domain_features.domain in self._technical_domains else 0.0,
                1.0 if domain_features.domain in self._creative_domains else 0.0,
                1.0 if domain_features.domain in self._academic_domains else 0.0,
                1.0 if domain_features.domain in self._business_domains else 0.0,

                # Feature densities from technical_features dict
                domain_features.technical_features.get('density', 0.5),
                domain_features.creative_features.get('density', 0.5) if hasattr(domain_features, 'creative_features') else 0.5,
                domain_features.academic_features.get('density', 0.5) if hasattr(domain_features, 'academic_features') else 0.5,

                # Communication features
                domain_features.conversational_features.get('politeness_score', 0.5) if hasattr(domain_features, 'conversational_features') else 0.5,
                1.0 if (hasattr(domain_features, 'conversational_features') and domain_features.conversational_features.get('has_urgency', False)) else 0.0,
                domain_features.conversational_features.get('question_density', 0.5) if hasattr(domain_features, 'conversational_features') else 0.5,
                domain_features.conversational_features.get('instruction_clarity', 0.5) if hasattr(domain_features, 'conversational_features') else 0.5,

                # Hybrid indicator
                1.0 if domain_features.hybrid_domain else 0.0,
            ]

            return features

        except Exception as e:
            logger.error(f"Domain analysis failed: {e}")
            return self._extract_fallback_features(text)
    
    def _extract_fallback_features(self, text: str) -> List[float]:
        """Extract basic features when analyzer is not available."""
        try:
            text_lower = text.lower()
            
            # Technical indicators
            technical_keywords = ['code', 'function', 'api', 'database', 'algorithm', 'programming']
            technical_score = sum(1 for keyword in technical_keywords if keyword in text_lower) / len(technical_keywords)
            
            # Creative indicators
            creative_keywords = ['creative', 'story', 'design', 'artistic', 'imagination', 'narrative']
            creative_score = sum(1 for keyword in creative_keywords if keyword in text_lower) / len(creative_keywords)
            
            # Academic indicators
            academic_keywords = ['research', 'study', 'analysis', 'theory', 'academic', 'scholarly']
            academic_score = sum(1 for keyword in academic_keywords if keyword in text_lower) / len(academic_keywords)
            
            # Business indicators
            business_keywords = ['business', 'strategy', 'market', 'customer', 'revenue', 'management']
            business_score = sum(1 for keyword in business_keywords if keyword in text_lower) / len(business_keywords)
            
            # Question density
            question_count = text.count('?')
            question_density = min(1.0, question_count / 10.0)
            
            # Urgency indicators
            urgency_keywords = ['urgent', 'asap', 'immediately', 'quickly', 'deadline']
            has_urgency = any(keyword in text_lower for keyword in urgency_keywords)
            
            features = [
                0.5,  # Domain confidence (default)
                min(1.0, len(text) / 1000.0),  # Complexity proxy
                0.5,  # Specificity (default)
                1.0 if technical_score > 0.3 else 0.0,  # Technical domain
                1.0 if creative_score > 0.3 else 0.0,   # Creative domain
                1.0 if academic_score > 0.3 else 0.0,   # Academic domain
                1.0 if business_score > 0.3 else 0.0,   # Business domain
                technical_score,  # Technical density
                creative_score,   # Creative density
                academic_score,   # Academic density
                0.5,  # Politeness (default)
                1.0 if has_urgency else 0.0,  # Urgency
                question_density,  # Question density
                0.5,  # Instruction clarity (default)
                1.0 if sum([technical_score > 0.2, creative_score > 0.2, academic_score > 0.2, business_score > 0.2]) > 1 else 0.0  # Hybrid
            ]
            
            return features
            
        except Exception as e:
            logger.error(f"Fallback domain feature extraction failed: {e}")
            return [self.config.default_feature_value] * 15
    
    def _normalize_features(self, features: List[float]) -> List[float]:
        """Normalize features to 0-1 range."""
        normalized = []
        for feature in features:
            # Ensure feature is in 0-1 range
            normalized_feature = min(1.0, max(0.0, float(feature)))
            normalized.append(normalized_feature)
        
        return normalized
    
    def _get_default_features(self) -> List[float]:
        """Get default feature vector when extraction fails."""
        return [self.config.default_feature_value] * 15
    
    # 2025: Circuit breaker implementation
    
    def _is_circuit_breaker_closed(self) -> bool:
        """Check if circuit breaker allows operations."""
        if not self.config.circuit_breaker_enabled:
            return True
        
        if self.circuit_breaker_state == CircuitBreakerState.CLOSED:
            return True
        elif self.circuit_breaker_state == CircuitBreakerState.OPEN:
            # Check if we should transition to half-open
            if (self.circuit_breaker_last_failure and
                time.time() - self.circuit_breaker_last_failure > self.config.recovery_timeout):
                self.circuit_breaker_state = CircuitBreakerState.HALF_OPEN
                logger.info("Domain extractor circuit breaker transitioning to half-open")
                return True
            return False
        else:  # HALF_OPEN
            return True
    
    def _handle_circuit_breaker_failure(self):
        """Handle circuit breaker failure."""
        if not self.config.circuit_breaker_enabled:
            return
        
        self.circuit_breaker_failures += 1
        self.circuit_breaker_last_failure = time.time()
        
        if self.circuit_breaker_failures >= self.config.failure_threshold:
            self.circuit_breaker_state = CircuitBreakerState.OPEN
            logger.warning(
                f"Domain extractor circuit breaker opened after {self.circuit_breaker_failures} failures"
            )
        elif self.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
            # Failed during half-open, go back to open
            self.circuit_breaker_state = CircuitBreakerState.OPEN
            logger.warning("Domain extractor circuit breaker failed during half-open, returning to open")
    
    def _handle_circuit_breaker_success(self):
        """Handle successful operation for circuit breaker."""
        if not self.config.circuit_breaker_enabled:
            return
        
        if self.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
            # Successful operation during half-open, close the circuit
            self.circuit_breaker_state = CircuitBreakerState.CLOSED
            self.circuit_breaker_failures = 0
            logger.info("Domain extractor circuit breaker closed after successful operation")
    
    def get_feature_names(self) -> List[str]:
        """Get names of extracted features with domain prefix."""
        return [
            "domain_confidence",
            "domain_complexity",
            "domain_specificity",
            "domain_technical_indicator",
            "domain_creative_indicator",
            "domain_academic_indicator",
            "domain_business_indicator",
            "domain_technical_density",
            "domain_creative_density",
            "domain_academic_density",
            "domain_conversational_politeness",
            "domain_urgency_indicator",
            "domain_question_density",
            "domain_instruction_clarity",
            "domain_hybrid_indicator"
        ]
    
    # 2025: Health monitoring and observability
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get comprehensive health status for monitoring."""
        return {
            "status": "healthy" if self.circuit_breaker_state == CircuitBreakerState.CLOSED else "degraded",
            "circuit_breaker_state": self.circuit_breaker_state.value,
            "metrics": self.metrics.get_health_status(),
            "cache_stats": self.get_cache_stats(),
            "analyzer_status": {
                "available": self.domain_analyzer is not None,
                "enabled": self.config.use_domain_analyzer,
                "fallback_enabled": self.config.fallback_on_analyzer_failure
            },
            "config": {
                "weight": self.config.weight,
                "timeout_seconds": self.config.timeout_seconds,
                "failure_threshold": self.config.failure_threshold,
                "cache_enabled": self.config.cache_enabled,
                "deterministic": self.config.deterministic
            },
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    
    def reset_metrics(self) -> None:
        """Reset all metrics for fresh monitoring."""
        self.metrics = DomainExtractionMetrics()
        logger.info("Domain extractor metrics reset")
    
    def reset_circuit_breaker(self) -> None:
        """Manually reset circuit breaker to closed state."""
        self.circuit_breaker_state = CircuitBreakerState.CLOSED
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = None
        logger.info("Domain extractor circuit breaker manually reset to closed state")
    
    # 2025: ML Pipeline Orchestrator integration
    
    async def run_orchestrated_analysis(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Orchestrator-compatible interface for domain feature extraction (2025 pattern).
        
        Args:
            config: Orchestrator configuration containing:
                - text: Text to analyze for domain features
                - context: Optional context information
                - operation: Operation type ("extract_features")
                - output_path: Local path for output files (optional)
                - correlation_id: Request correlation ID (optional)
        
        Returns:
            Dictionary with extraction results and metadata
        """
        try:
            operation = config.get("operation", "extract_features")
            text = config.get("text", "")
            context = config.get("context")
            correlation_id = config.get("correlation_id")
            
            if correlation_id:
                logger.info(f"Starting orchestrated domain analysis [correlation_id={correlation_id}]")
            
            if operation == "extract_features":
                features = await self.extract_features_async(text, context)
                
                # Create result with comprehensive metadata
                result = {
                    "features": features,
                    "feature_names": self.get_feature_names(),
                    "feature_count": len(features),
                    "extraction_metadata": {
                        "extractor_type": "domain",
                        "config_weight": self.config.weight,
                        "text_length": len(text) if text else 0,
                        "analyzer_available": self.domain_analyzer is not None,
                        "analyzer_usage_rate": self.metrics.get_analyzer_usage_rate(),
                        "circuit_breaker_state": self.circuit_breaker_state.value,
                        "cache_enabled": self.config.cache_enabled,
                        "deterministic": self.config.deterministic
                    }
                }
                
                # Save to output path if specified
                output_path = config.get("output_path")
                if output_path:
                    import json
                    import os
                    
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    with open(output_path, 'w') as f:
                        json.dump(result, f, indent=2)
                    
                    result['output_path'] = output_path
                    logger.info(f"Domain features saved to {output_path}")
                
                return {
                    "status": "success",
                    "operation": operation,
                    "result": result,
                    "component": "domain_feature_extractor",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "correlation_id": correlation_id
                }
            
            elif operation == "health_check":
                health_status = self.get_health_status()
                return {
                    "status": "success",
                    "operation": operation,
                    "result": health_status,
                    "component": "domain_feature_extractor",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "correlation_id": correlation_id
                }
            
            else:
                return {
                    "status": "error",
                    "error": f"Unsupported operation: {operation}",
                    "component": "domain_feature_extractor",
                    "correlation_id": correlation_id
                }
        
        except Exception as e:
            logger.error(f"Orchestrated domain analysis failed: {e} [correlation_id={correlation_id}]")
            return {
                "status": "error",
                "error": str(e),
                "component": "domain_feature_extractor",
                "correlation_id": correlation_id
            }
    
    # 2025: Configuration management
    
    def update_config(self, new_config: DomainFeatureConfig) -> None:
        """Update configuration and apply changes."""
        old_config = self.config
        self.config = new_config
        
        # Update logging level if changed
        if old_config.log_level != new_config.log_level:
            logger.setLevel(getattr(logging, new_config.log_level))
        
        # Reinitialize domain analyzer if settings changed
        if (old_config.use_domain_analyzer != new_config.use_domain_analyzer and
            DOMAIN_ANALYSIS_AVAILABLE and new_config.use_domain_analyzer):
            try:
                self.domain_analyzer = DomainAnalyzer()
                logger.info("Domain analyzer reinitialized due to configuration change")
            except Exception as e:
                logger.warning(f"Failed to reinitialize domain analyzer: {e}")
                self.domain_analyzer = None
        
        # Clear cache if cache settings changed
        if (old_config.cache_enabled != new_config.cache_enabled or
            old_config.cache_ttl_seconds != new_config.cache_ttl_seconds or
            old_config.cache_max_size != new_config.cache_max_size):
            self.clear_cache()
            logger.info("Cache cleared due to configuration change")
        
        # Reset circuit breaker if thresholds changed
        if (old_config.failure_threshold != new_config.failure_threshold or
            old_config.recovery_timeout != new_config.recovery_timeout):
            self.reset_circuit_breaker()
            logger.info("Circuit breaker reset due to configuration change")
        
        logger.info("DomainFeatureExtractor configuration updated")
    
    def get_config_dict(self) -> Dict[str, Any]:
        """Get current configuration as dictionary."""
        return self.config.model_dump()
    
    def clear_cache(self) -> int:
        """Clear feature cache and return number of entries cleared."""
        count = len(self._feature_cache)
        self._feature_cache.clear()
        self._last_cache_cleanup = time.time()
        logger.info(f"Cleared {count} cached domain features")
        return count
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        if not self.config.cache_enabled:
            return {
                "cache_enabled": False,
                "analyzer_available": self.domain_analyzer is not None
            }
        
        current_time = time.time()
        valid_entries = 0
        expired_entries = 0
        
        for entry in self._feature_cache.values():
            if current_time - entry['timestamp'] <= self.config.cache_ttl_seconds:
                valid_entries += 1
            else:
                expired_entries += 1
        
        return {
            "cache_enabled": True,
            "cache_size": len(self._feature_cache),
            "cache_max_size": self.config.cache_max_size,
            "valid_entries": valid_entries,
            "expired_entries": expired_entries,
            "cache_hits": self.metrics.cache_hits,
            "cache_misses": self.metrics.cache_misses,
            "hit_rate_percent": (
                (self.metrics.cache_hits / (self.metrics.cache_hits + self.metrics.cache_misses) * 100)
                if (self.metrics.cache_hits + self.metrics.cache_misses) > 0 else 0
            ),
            "ttl_seconds": self.config.cache_ttl_seconds,
            "last_cleanup": self._last_cache_cleanup,
            "analyzer_available": self.domain_analyzer is not None
        }
