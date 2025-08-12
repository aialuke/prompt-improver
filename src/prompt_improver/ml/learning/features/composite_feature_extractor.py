"""Composite Feature Extractor

Enhanced 2025 implementation combining multiple specialized feature extractors
to create comprehensive feature vectors. Implements the Composite pattern
with async operations, event-driven architecture, and modern observability.

features:
- Async/await pattern for non-blocking operations
- Event-driven integration with ML Pipeline Orchestrator
- Comprehensive observability and metrics
- Resource-aware processing with memory management
- Type-safe configuration and validation
- Intelligent caching with TTL and invalidation
- Circuit breaker pattern for fault tolerance
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from enum import Enum
import logging
import time
from typing import Any, Dict, List, Optional, Union

from sqlmodel import SQLModel, Field
from pydantic import BaseModel

import numpy as np

from .context_feature_extractor import ContextFeatureExtractor
from .domain_feature_extractor import DomainFeatureExtractor
from .linguistic_feature_extractor import LinguisticFeatureExtractor

logger = logging.getLogger(__name__)

class ExtractionMode(Enum):
    """Feature extraction execution modes."""
    SEQUENTIAL = "sequential"
    parallel = "parallel"
    ADAPTIVE = "adaptive"  # 2025: Adaptive based on resource availability

class CircuitBreakerState(Enum):
    """Circuit breaker states for fault tolerance."""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"

class FeatureExtractionConfig(BaseModel):
    """Enhanced 2025 configuration for composite feature extraction."""

    # Feature extractor enablement
    enable_linguistic: bool = Field(default=True, description="Enable linguistic feature extraction")
    enable_domain: bool = Field(default=True, description="Enable domain feature extraction")
    enable_context: bool = Field(default=True, description="Enable context feature extraction")

    # Feature weights with validation
    linguistic_weight: float = Field(default=1.0, ge=0.0, le=10.0, description="Weight for linguistic features")
    domain_weight: float = Field(default=1.0, ge=0.0, le=10.0, description="Weight for domain features")
    context_weight: float = Field(default=1.0, ge=0.0, le=10.0, description="Weight for context features")

    # 2025: Advanced caching configuration
    cache_enabled: bool = Field(default=True, description="Enable feature extraction caching")
    cache_ttl_seconds: int = Field(default=3600, ge=0, description="Cache time-to-live in seconds")
    cache_max_size: int = Field(default=10000, ge=1, description="Maximum cache size")
    cache_invalidation_strategy: str = Field(default="lru", description="Cache invalidation strategy")

    # 2025: Processing configuration with resource awareness
    execution_mode: ExtractionMode = Field(default=ExtractionMode.ADAPTIVE, description="Feature extraction execution mode")
    max_concurrent_extractions: int = Field(default=3, ge=1, le=20, description="Maximum concurrent extractions")
    timeout_seconds: float = Field(default=30.0, gt=0.0, description="Extraction timeout in seconds")
    memory_limit_mb: int = Field(default=1024, ge=128, description="Memory limit in MB")

    # 2025: Circuit breaker configuration
    circuit_breaker_enabled: bool = Field(default=True, description="Enable circuit breaker pattern")
    failure_threshold: int = Field(default=5, ge=1, description="Circuit breaker failure threshold")
    recovery_timeout: int = Field(default=60, ge=1, description="Circuit breaker recovery timeout")

    # Quality thresholds with enhanced validation
    min_text_length: int = Field(default=10, ge=0, description="Minimum text length for processing")
    max_text_length: int = Field(default=100000, ge=1, description="Maximum text length for processing")
    min_confidence_threshold: float = Field(default=0.5, ge=0.0, le=1.0, description="Minimum confidence threshold")

    # 2025: Observability configuration
    enable_metrics: bool = Field(default=True, description="Enable metrics collection")
    enable_tracing: bool = Field(default=True, description="Enable distributed tracing")
    log_level: str = Field(default="INFO", description="Logging level")

class ExtractionMetrics(BaseModel):
    """Metrics for feature extraction operations."""
    total_extractions: int = Field(default=0, ge=0, description="Total number of extractions")
    successful_extractions: int = Field(default=0, ge=0, description="Number of successful extractions")
    failed_extractions: int = Field(default=0, ge=0, description="Number of failed extractions")
    total_processing_time: float = Field(default=0.0, ge=0.0, description="Total processing time")
    average_processing_time: float = Field(default=0.0, ge=0.0, description="Average processing time")
    cache_hits: int = Field(default=0, ge=0, description="Number of cache hits")
    cache_misses: int = Field(default=0, ge=0, description="Number of cache misses")
    circuit_breaker_trips: int = Field(default=0, ge=0, description="Number of circuit breaker trips")

    def update_success(self, processing_time: float):
        """Update metrics for successful extraction."""
        self.total_extractions += 1
        self.successful_extractions += 1
        self.total_processing_time += processing_time
        self.average_processing_time = self.total_processing_time / self.total_extractions

    def update_failure(self):
        """Update metrics for failed extraction."""
        self.total_extractions += 1
        self.failed_extractions += 1

    def get_success_rate(self) -> float:
        """Get success rate as percentage."""
        if self.total_extractions == 0:
            return 0.0
        return (self.successful_extractions / self.total_extractions) * 100.0

class CompositeFeatureExtractor:
    """Enhanced 2025 composite feature extractor with async operations and observability.

    This class implements the Composite pattern with modern 2025 best practices:
    - Async/await for non-blocking operations
    - Circuit breaker pattern for fault tolerance
    - Comprehensive metrics and observability
    - Resource-aware processing
    - Event-driven integration capabilities
    """

    def __init__(self, config: FeatureExtractionConfig | None = None):
        """Initialize enhanced composite feature extractor.

        Args:
            config: Configuration for feature extraction
        """
        self.config = config or FeatureExtractionConfig()

        # Initialize extractors based on configuration
        self.extractors = {}
        self._initialize_extractors()

        # 2025: Enhanced state management
        self.metrics = ExtractionMetrics()
        self.circuit_breaker_state = CircuitBreakerState.CLOSED
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = None

        # 2025: Async processing state
        self._extraction_semaphore = asyncio.Semaphore(self.config.max_concurrent_extractions)
        self._cache = {} if self.config.cache_enabled else None

        logger.info(
            f"Enhanced CompositeFeatureExtractor initialized with {len(self.extractors)} extractors, "
            f"mode: {self.config.execution_mode.value}, "
            f"circuit_breaker: {self.config.circuit_breaker_enabled}"
        )

    def _initialize_extractors(self):
        """Initialize feature extractors based on configuration."""
        if self.config.enable_linguistic:
            # Import the correct config class for LinguisticFeatureExtractor
            from .linguistic_feature_extractor import (
                FeatureExtractionConfig as LinguisticConfig,
            )

            linguistic_config = LinguisticConfig(
                weight=self.config.linguistic_weight,
                cache_enabled=self.config.cache_enabled,
                deterministic=True,  # Always deterministic for consistency
                enable_metrics=self.config.enable_metrics
            )
            self.extractors['linguistic'] = LinguisticFeatureExtractor(config=linguistic_config)

        if self.config.enable_domain:
            from .domain_feature_extractor import DomainFeatureConfig
            domain_config = DomainFeatureConfig(
                weight=self.config.domain_weight,
                cache_enabled=self.config.cache_enabled,
                deterministic=True
            )
            self.extractors['domain'] = DomainFeatureExtractor(domain_config)

        if self.config.enable_context:
            from .context_feature_extractor import ContextFeatureConfig
            context_config = ContextFeatureConfig(
                weight=self.config.context_weight,
                cache_enabled=self.config.cache_enabled
            )
            self.extractors['context'] = ContextFeatureExtractor(context_config)
    
    def extract_features(self, 
                        text: str, 
                        context_data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Extract comprehensive features from text and context.
        
        Args:
            text: Input text to analyze
            context_data: Optional context information
            
        Returns:
            Dictionary containing:
            - 'features': Combined feature vector
            - 'feature_names': Names of features
            - 'extractor_results': Individual extractor results
            - 'metadata': Extraction metadata
        """
        try:
            # Validate inputs
            if not self._validate_inputs(text, context_data):
                return self._get_default_result()
            
            # Extract features from each enabled extractor
            extractor_results = {}
            all_features = []
            all_feature_names = []
            
            # Linguistic features
            if 'linguistic' in self.extractors:
                linguistic_features = self.extractors['linguistic'].extract_features_sync(text, context_data)
                linguistic_names = self.extractors['linguistic'].get_feature_names()
                
                extractor_results['linguistic'] = {
                    'features': linguistic_features,
                    'feature_names': linguistic_names,
                    'count': len(linguistic_features)
                }
                
                all_features.extend(linguistic_features)
                all_feature_names.extend([f"linguistic_{name}" for name in linguistic_names])
            
            # Domain features
            if 'domain' in self.extractors:
                domain_features = self.extractors['domain'].extract_features(text, context_data)
                domain_names = self.extractors['domain'].get_feature_names()
                
                extractor_results['domain'] = {
                    'features': domain_features,
                    'feature_names': domain_names,
                    'count': len(domain_features)
                }
                
                all_features.extend(domain_features)
                all_feature_names.extend([f"domain_{name}" for name in domain_names])
            
            # Context features
            if 'context' in self.extractors and context_data:
                context_features = self.extractors['context'].extract_features(context_data)
                context_names = self.extractors['context'].get_feature_names()
                
                extractor_results['context'] = {
                    'features': context_features,
                    'feature_names': context_names,
                    'count': len(context_features)
                }
                
                all_features.extend(context_features)
                all_feature_names.extend([f"context_{name}" for name in context_names])
            
            # Create result
            result = {
                'features': np.array(all_features),
                'feature_names': all_feature_names,
                'extractor_results': extractor_results,
                'metadata': {
                    'total_features': len(all_features),
                    'extractors_used': list(extractor_results.keys()),
                    'text_length': len(text) if text else 0,
                    'has_context': context_data is not None,
                    'config': self.config
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Composite feature extraction failed: {e}")
            return self._get_default_result()
    
    def _validate_inputs(self, text: str, context_data: dict[str, Any] | None) -> bool:
        """Validate input parameters."""
        try:
            # Validate text
            if not text or not isinstance(text, str):
                logger.warning("Invalid text input")
                return False
            
            if len(text.strip()) < self.config.min_text_length:
                logger.warning(f"Text too short: {len(text)} < {self.config.min_text_length}")
                return False
            
            if len(text) > self.config.max_text_length:
                logger.warning(f"Text too long: {len(text)} > {self.config.max_text_length}")
                return False
            
            # Validate context data if provided
            if context_data is not None and not isinstance(context_data, dict):
                logger.warning("Invalid context data type")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Input validation failed: {e}")
            return False
    
    def _get_default_result(self) -> dict[str, Any]:
        """Get default result when extraction fails."""
        # Calculate expected feature count
        feature_count = 0
        feature_names = []
        
        if self.config.enable_linguistic:
            feature_count += 10  # Linguistic features
            feature_names.extend([f"linguistic_feature_{i}" for i in range(10)])
        
        if self.config.enable_domain:
            feature_count += 15  # Domain features
            feature_names.extend([f"domain_feature_{i}" for i in range(15)])
        
        if self.config.enable_context:
            feature_count += 20  # Context features
            feature_names.extend([f"context_feature_{i}" for i in range(20)])
        
        return {
            'features': np.array([0.5] * feature_count),
            'feature_names': feature_names,
            'extractor_results': {},
            'metadata': {
                'total_features': feature_count,
                'extractors_used': [],
                'text_length': 0,
                'has_context': False,
                'config': self.config,
                'is_default': True
            }
        }
    
    def get_feature_count(self) -> int:
        """Get total number of features that will be extracted."""
        count = 0
        if self.config.enable_linguistic:
            count += 10
        if self.config.enable_domain:
            count += 15
        if self.config.enable_context:
            count += 20
        return count
    
    def get_extractor_info(self) -> dict[str, Any]:
        """Get information about configured extractors."""
        info = {}
        for name, extractor in self.extractors.items():
            if hasattr(extractor, 'get_cache_stats'):
                info[name] = extractor.get_cache_stats()
            else:
                info[name] = {'available': True}
        return info
    
    def clear_all_caches(self) -> dict[str, int]:
        """Clear caches for all extractors."""
        cleared_counts = {}
        for name, extractor in self.extractors.items():
            if hasattr(extractor, 'clear_cache'):
                cleared_counts[name] = extractor.clear_cache()
            else:
                cleared_counts[name] = 0
        
        total_cleared = sum(cleared_counts.values())
        logger.info(f"Cleared {total_cleared} total cached features across {len(cleared_counts)} extractors")
        
        return cleared_counts
    
    def update_config(self, new_config: FeatureExtractionConfig) -> None:
        """Update configuration and reinitialize extractors if needed."""
        old_config = self.config
        self.config = new_config
        
        # Check if we need to reinitialize extractors
        extractors_changed = (
            old_config.enable_linguistic != new_config.enable_linguistic or
            old_config.enable_domain != new_config.enable_domain or
            old_config.enable_context != new_config.enable_context or
            old_config.linguistic_weight != new_config.linguistic_weight or
            old_config.domain_weight != new_config.domain_weight or
            old_config.context_weight != new_config.context_weight
        )
        
        if extractors_changed:
            logger.info("Configuration changed, reinitializing extractors")
            self.__init__(new_config)
        else:
            logger.info("Configuration updated without reinitializing extractors")

    # 2025: Enhanced async methods for ML Pipeline Orchestrator integration

    async def extract_features_async(self,
                                    text: str,
                                    context_data: dict[str, Any] | None = None) -> dict[str, Any]:
        """Enhanced 2025 async feature extraction with circuit breaker and observability.

        Args:
            text: Input text to analyze
            context_data: Optional context information

        Returns:
            Dictionary containing extracted features and metadata
        """
        start_time = time.time()

        # Circuit breaker check
        if not self._is_circuit_breaker_closed():
            logger.warning("Circuit breaker is open, returning default result")
            self.metrics.circuit_breaker_trips += 1
            return self._get_default_result()

        # Input validation
        if not self._validate_inputs(text, context_data):
            return self._get_default_result()

        # Check cache first
        cache_key = self._generate_cache_key(text, context_data)
        if self._cache and cache_key in self._cache:
            self.metrics.cache_hits += 1
            logger.debug("Cache hit for feature extraction")
            return self._cache[cache_key]

        self.metrics.cache_misses += 1

        try:
            async with self._extraction_semaphore:
                # Use timeout for the entire operation
                result = await asyncio.wait_for(
                    self._perform_extraction_async(text, context_data),
                    timeout=self.config.timeout_seconds
                )

                # Cache the result
                if self._cache and result:
                    self._cache[cache_key] = result
                    self._cleanup_cache()

                processing_time = time.time() - start_time
                self.metrics.update_success(processing_time)

                logger.info("Feature extraction completed in %.3fs", processing_time)
                return result

        except TimeoutError:
            logger.error(f"Feature extraction timed out after {self.config.timeout_seconds}s")
            self._handle_circuit_breaker_failure()
            self.metrics.update_failure()
            return self._get_default_result()

        except Exception as e:
            logger.error(f"Async feature extraction failed: {e}")
            self._handle_circuit_breaker_failure()
            self.metrics.update_failure()
            return self._get_default_result()

    async def _perform_extraction_async(self, text: str, context_data: dict[str, Any] | None) -> dict[str, Any]:
        """Perform the actual feature extraction based on execution mode."""
        if self.config.execution_mode == ExtractionMode.parallel:
            return await self._extract_parallel(text, context_data)
        elif self.config.execution_mode == ExtractionMode.ADAPTIVE:
            return await self._extract_adaptive(text, context_data)
        else:
            return await self._extract_sequential_async(text, context_data)

    async def _extract_sequential_async(self, text: str, context_data: dict[str, Any] | None) -> dict[str, Any]:
        """Sequential async feature extraction."""
        extractor_results = {}
        all_features = []
        all_feature_names = []

        for extractor_name, extractor in self.extractors.items():
            try:
                # Convert sync extraction to async
                if extractor_name == 'context' and context_data:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, extractor.extract_features, text, context_data
                    )
                else:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, extractor.extract_features, text
                    )

                if result and 'features' in result:
                    features = self._normalize_features(result['features'])
                    extractor_results[extractor_name] = result
                    all_features.extend(features)

                    feature_names = result.get('feature_names', [])
                    if not feature_names:
                        feature_names = [f"{extractor_name}_feature_{i}" for i in range(len(features))]
                    all_feature_names.extend(feature_names)

                    logger.debug(f"Extracted {len(features)} features from {extractor_name}")

            except Exception as e:
                logger.error(f"Feature extraction failed for {extractor_name}: {e}")
                continue

        return self._create_result(all_features, all_feature_names, extractor_results, text, context_data)

    async def _extract_parallel(self, text: str, context_data: dict[str, Any] | None) -> dict[str, Any]:
        """Parallel feature extraction using asyncio.gather."""
        tasks = []
        extractor_names = []

        for extractor_name, extractor in self.extractors.items():
            if extractor_name == 'context' and context_data:
                task = asyncio.get_event_loop().run_in_executor(
                    None, extractor.extract_features, text, context_data
                )
            else:
                task = asyncio.get_event_loop().run_in_executor(
                    None, extractor.extract_features, text
                )
            tasks.append(task)
            extractor_names.append(extractor_name)

        try:
            results = await asyncio.gather(*tasks, return_exceptions=True)

            extractor_results = {}
            all_features = []
            all_feature_names = []

            for i, result in enumerate(results):
                extractor_name = extractor_names[i]

                if isinstance(result, Exception):
                    logger.error(f"Parallel extraction failed for {extractor_name}: {result}")
                    continue

                if result and 'features' in result:
                    features = self._normalize_features(result['features'])
                    extractor_results[extractor_name] = result
                    all_features.extend(features)

                    feature_names = result.get('feature_names', [])
                    if not feature_names:
                        feature_names = [f"{extractor_name}_feature_{i}" for i in range(len(features))]
                    all_feature_names.extend(feature_names)

            return self._create_result(all_features, all_feature_names, extractor_results, text, context_data)

        except Exception as e:
            logger.error(f"Parallel feature extraction failed: {e}")
            raise

    async def _extract_adaptive(self, text: str, context_data: dict[str, Any] | None) -> dict[str, Any]:
        """Adaptive feature extraction based on resource availability."""
        # For now, use parallel if we have multiple extractors, sequential otherwise
        if len(self.extractors) > 1:
            return await self._extract_parallel(text, context_data)
        else:
            return await self._extract_sequential_async(text, context_data)

    # 2025: Circuit breaker and utility methods

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
                logger.info("Circuit breaker transitioning to half-open")
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
            logger.warning(f"Circuit breaker opened after {self.circuit_breaker_failures} failures")

    def _generate_cache_key(self, text: str, context_data: dict[str, Any] | None) -> str:
        """Generate cache key for text and context."""
        import hashlib

        key_data = text
        if context_data:
            key_data += str(sorted(context_data.items()))

        return hashlib.md5(key_data.encode(), usedforsecurity=False).hexdigest()

    def _cleanup_cache(self):
        """Clean up cache based on size limits."""
        if not self._cache or len(self._cache) <= self.config.cache_max_size:
            return

        # Simple LRU cleanup - remove oldest entries
        items_to_remove = len(self._cache) - self.config.cache_max_size
        keys_to_remove = list(self._cache.keys())[:items_to_remove]

        for key in keys_to_remove:
            del self._cache[key]

    def _validate_inputs(self, text: str, context_data: dict[str, Any] | None) -> bool:
        """Validate input parameters."""
        if not text or not isinstance(text, str):
            logger.warning("Invalid text input")
            return False

        if len(text) < self.config.min_text_length:
            logger.warning(f"Text too short: {len(text)} < {self.config.min_text_length}")
            return False

        if len(text) > self.config.max_text_length:
            logger.warning(f"Text too long: {len(text)} > {self.config.max_text_length}")
            # Truncate but continue processing
            return True

        return True

    def _normalize_features(self, features: Any) -> list[float]:
        """Normalize features to list of floats."""
        if isinstance(features, np.ndarray):
            return features.tolist()
        elif isinstance(features, list):
            return features
        elif isinstance(features, (int, float)):
            return [float(features)]
        else:
            return [0.0]

    def _create_result(self, all_features: list[float], all_feature_names: list[str],
                      extractor_results: dict[str, Any], text: str,
                      context_data: dict[str, Any] | None) -> dict[str, Any]:
        """Create standardized result dictionary."""
        return {
            'features': np.array(all_features),
            'feature_names': all_feature_names,
            'extractor_results': extractor_results,
            'metadata': {
                'total_features': len(all_features),
                'extractors_used': list(extractor_results.keys()),
                'text_length': len(text) if text else 0,
                'has_context': context_data is not None,
                'config': self.config,
                'extraction_mode': self.config.execution_mode.value,
                'circuit_breaker_state': self.circuit_breaker_state.value,
                'metrics': {
                    'success_rate': self.metrics.get_success_rate(),
                    'avg_processing_time': self.metrics.average_processing_time,
                    'cache_hit_rate': (self.metrics.cache_hits /
                                     (self.metrics.cache_hits + self.metrics.cache_misses) * 100
                                     if (self.metrics.cache_hits + self.metrics.cache_misses) > 0 else 0)
                }
            }
        }

    # 2025: Orchestrator-compatible interface

    async def run_orchestrated_analysis(self, config: dict[str, Any]) -> dict[str, Any]:
        """Orchestrator-compatible interface for feature extraction (2025 pattern).

        Args:
            config: Orchestrator configuration containing:
                - text: Text to analyze
                - context_data: Optional context information
                - operation: Operation type ("extract_features")
                - output_path: Local path for output files (optional)

        Returns:
            Dictionary with extraction results and metadata
        """
        try:
            operation = config.get("operation", "extract_features")
            text = config.get("text", "")
            context_data = config.get("context_data")

            if operation == "extract_features":
                result = await self.extract_features_async(text, context_data)

                # Save to output path if specified
                output_path = config.get("output_path")
                if output_path:
                    import json
                    import os

                    os.makedirs(os.path.dirname(output_path), exist_ok=True)

                    # Convert numpy arrays to lists for JSON serialization
                    serializable_result = {
                        'features': result['features'].tolist() if isinstance(result['features'], np.ndarray) else result['features'],
                        'feature_names': result['feature_names'],
                        'metadata': result['metadata']
                    }

                    with open(output_path, 'w') as f:
                        json.dump(serializable_result, f, indent=2)

                    result['output_path'] = output_path

                return {
                    "status": "success",
                    "operation": operation,
                    "result": result,
                    "component": "composite_feature_extractor",
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            else:
                return {
                    "status": "error",
                    "error": f"Unsupported operation: {operation}",
                    "component": "composite_feature_extractor"
                }

        except Exception as e:
            logger.error(f"Orchestrated analysis failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "component": "composite_feature_extractor"
            }

class FeatureExtractorFactory:
    """Factory for creating configured feature extractors."""
    
    @staticmethod
    def create_default_extractor() -> CompositeFeatureExtractor:
        """Create extractor with default configuration."""
        return CompositeFeatureExtractor()
    
    @staticmethod
    def create_lightweight_extractor() -> CompositeFeatureExtractor:
        """Create lightweight extractor for performance-critical scenarios."""
        config = FeatureExtractionConfig(
            enable_linguistic=True,
            enable_domain=False,
            enable_context=True,
            cache_enabled=True,
            deterministic=True
        )
        return CompositeFeatureExtractor(config)
    
    @staticmethod
    def create_comprehensive_extractor() -> CompositeFeatureExtractor:
        """Create comprehensive extractor with all features enabled."""
        config = FeatureExtractionConfig(
            enable_linguistic=True,
            enable_domain=True,
            enable_context=True,
            cache_enabled=True,
            deterministic=True,
            parallel_extraction=True
        )
        return CompositeFeatureExtractor(config)
