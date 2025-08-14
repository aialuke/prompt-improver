"""Context Feature Extractor

Modernized 2025 implementation for extracting context-specific features from prompt data.
features async/await patterns, Pydantic validation, circuit breaker, and orchestrator integration.

features:
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
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import logging
import time
from typing import Any, Dict, List, Optional, Union
from sqlmodel import SQLModel, Field
from pydantic import BaseModel
import numpy as np
from ....security import OWASP2025InputValidator, ValidationError
from ....security.input_sanitization import InputSanitizer
from ....utils.datetime_utils import aware_utc_now
logger = logging.getLogger(__name__)

class CircuitBreakerState(Enum):
    """Circuit breaker states for fault tolerance."""
    CLOSED = 'closed'
    OPEN = 'open'
    HALF_OPEN = 'half_open'

class ContextFeatureConfig(BaseModel):
    """Enhanced 2025 configuration model for context feature extraction."""
    weight: float = Field(default=1.0, ge=0.0, le=10.0, description='Weight for context features')
    cache_enabled: bool = Field(default=True, description='Enable result caching')
    cache_ttl_seconds: int = Field(default=3600, ge=0, description='Cache TTL in seconds')
    cache_max_size: int = Field(default=1000, ge=1, description='Maximum cache entries')
    cache_cleanup_interval: int = Field(default=300, ge=60, description='Cache cleanup interval in seconds')
    timeout_seconds: float = Field(default=30.0, gt=0.0, description='Operation timeout')
    max_retries: int = Field(default=3, ge=0, le=10, description='Maximum retry attempts')
    memory_limit_mb: int = Field(default=512, gt=0, description='Memory limit in MB')
    circuit_breaker_enabled: bool = Field(default=True, description='Enable circuit breaker')
    failure_threshold: int = Field(default=5, ge=1, description='Circuit breaker failure threshold')
    recovery_timeout: int = Field(default=60, ge=1, description='Circuit breaker recovery timeout')
    min_context_fields: int = Field(default=1, ge=0, description='Minimum context fields required')
    max_context_size: int = Field(default=10000, gt=0, description='Maximum context size in characters')
    default_feature_value: float = Field(default=0.5, ge=0.0, le=1.0, description='Default feature value')
    enable_metrics: bool = Field(default=True, description='Enable performance metrics')
    enable_health_checks: bool = Field(default=True, description='Enable health monitoring')
    log_level: str = Field(default='INFO', description='Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)')
    correlation_id_enabled: bool = Field(default=True, description='Enable correlation ID tracking')

@dataclass
class ContextExtractionMetrics:
    """Metrics for context feature extraction operations."""
    total_extractions: int = 0
    successful_extractions: int = 0
    failed_extractions: int = 0
    total_processing_time: float = 0.0
    average_processing_time: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    circuit_breaker_trips: int = 0
    last_extraction_time: float | None = None

    def update_success(self, processing_time: float):
        """Update metrics for successful extraction."""
        self.total_extractions += 1
        self.successful_extractions += 1
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
        return self.successful_extractions / self.total_extractions * 100.0

    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status."""
        return {'total_extractions': self.total_extractions, 'success_rate': self.get_success_rate(), 'average_processing_time': self.average_processing_time, 'cache_hit_rate': self.cache_hits / (self.cache_hits + self.cache_misses) * 100 if self.cache_hits + self.cache_misses > 0 else 0, 'circuit_breaker_trips': self.circuit_breaker_trips, 'last_extraction_time': self.last_extraction_time}

class ContextFeatureExtractor:
    """Enhanced 2025 context feature extractor with async operations and observability.
    
    Extracts 20 normalized context features:
    - Performance metrics (5 features): improvement_score, user_satisfaction, rule_effectiveness, response_time_score, quality_score
    - Context metadata (5 features): project_type_encoding, complexity_score, technical_level, context_richness, data_quality  
    - User interaction patterns (5 features): session_length_norm, iteration_count_norm, feedback_frequency, user_engagement_score, success_rate
    - Temporal features (5 features): time_of_day_norm, day_of_week_norm, session_recency_norm, usage_frequency_norm, trend_indicator
    
    This class implements 2025 best practices:
    - Async/await for non-blocking operations
    - Circuit breaker pattern for fault tolerance
    - Comprehensive metrics and health monitoring
    - Pydantic configuration with validation
    - Intelligent caching with TTL and cleanup
    - ML Pipeline Orchestrator integration
    """

    def __init__(self, config: ContextFeatureConfig | None=None):
        """Initialize enhanced context feature extractor.
        
        Args:
            config: Configuration for context feature extraction
        """
        self.config = config or ContextFeatureConfig()
        self.input_validator = OWASP2025InputValidator()
        self.input_sanitizer = InputSanitizer()
        self.metrics = ContextExtractionMetrics()
        self.circuit_breaker_state = CircuitBreakerState.CLOSED
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure: float | None = None
        self._feature_cache: dict[str, dict[str, Any]] = {} if self.config.cache_enabled else {}
        self._last_cache_cleanup = time.time()
        self._extraction_semaphore = asyncio.Semaphore(1)
        logger.setLevel(getattr(logging, self.config.log_level))
        logger.info('Enhanced ContextFeatureExtractor initialized with config: weight=%s, cache=%s, circuit_breaker=%s, timeout=%ss', self.config.weight, self.config.cache_enabled, self.config.circuit_breaker_enabled, self.config.timeout_seconds)

    def extract_features(self, context_data: dict[str, Any]) -> list[float]:
        """Synchronous wrapper for async feature extraction.
        
        Args:
            context_data: Context information including performance metrics,
                         user data, session info, etc.
            
        Returns:
            List of 20 normalized context features
        """
        try:
            loop = asyncio.get_running_loop()
            return asyncio.run_coroutine_threadsafe(self.extract_features_async(context_data), loop).result()
        except RuntimeError:
            return asyncio.run(self.extract_features_async(context_data))

    async def extract_features_async(self, context_data: dict[str, Any]) -> list[float]:
        """Enhanced 2025 async feature extraction with circuit breaker and observability.
        
        Args:
            context_data: Context information including performance metrics,
                         user data, session info, etc.
            
        Returns:
            List of 20 normalized context features
        """
        start_time = time.time()
        correlation_id = self._generate_correlation_id() if self.config.correlation_id_enabled else None
        if not self._is_circuit_breaker_closed():
            logger.warning('Circuit breaker is open [correlation_id=%s]', correlation_id)
            self.metrics.circuit_breaker_trips += 1
            return self._get_default_features()
        validated_data = await self._validate_input_async(context_data)
        if not validated_data:
            logger.warning('Input validation failed, using defaults [correlation_id=%s]', correlation_id)
            return self._get_default_features()
        if self.config.cache_enabled:
            cache_key = self._generate_cache_key(validated_data)
            cached_result = await self._get_cached_features_async(cache_key)
            if cached_result is not None:
                self.metrics.cache_hits += 1
                logger.debug('Cache hit for feature extraction [correlation_id=%s]', correlation_id)
                return cached_result
            self.metrics.cache_misses += 1
        try:
            async with self._extraction_semaphore:
                result = await asyncio.wait_for(self._perform_extraction_async(validated_data, correlation_id), timeout=self.config.timeout_seconds)
                if self.config.cache_enabled and result:
                    await self._cache_features_async(cache_key, result)
                processing_time = time.time() - start_time
                self.metrics.update_success(processing_time)
                logger.info('Context feature extraction completed in %ss [correlation_id=%s]', format(processing_time, '.3f'), correlation_id)
                return result
        except TimeoutError:
            logger.error('Context feature extraction timed out after %ss [correlation_id=%s]', self.config.timeout_seconds, correlation_id)
            self._handle_circuit_breaker_failure()
            self.metrics.update_failure()
            return self._get_default_features()
        except Exception as e:
            logger.error('Async context feature extraction failed: %s [correlation_id=%s]', e, correlation_id)
            self._handle_circuit_breaker_failure()
            self.metrics.update_failure()
            return self._get_default_features()

    async def _perform_extraction_async(self, context_data: dict[str, Any], correlation_id: str | None) -> list[float]:
        """Perform the actual feature extraction."""
        logger.debug('Starting feature extraction [correlation_id=%s]', correlation_id)
        features = self._compute_context_features(context_data)
        normalized_features = self._normalize_features(features)
        logger.debug('Extracted %s features [correlation_id=%s]', len(normalized_features), correlation_id)
        return normalized_features

    async def _validate_input_async(self, context_data: dict[str, Any]) -> dict[str, Any] | None:
        """Async validation and sanitization of input context data."""
        try:
            if not context_data or not isinstance(context_data, dict):
                logger.debug('Empty or invalid context data provided')
                return {}
            context_str = str(context_data)
            if len(context_str) > self.config.max_context_size:
                logger.warning('Context data too large: %s > %s', len(context_str), self.config.max_context_size)
                return {}
            if len(context_data) < self.config.min_context_fields:
                logger.debug('Insufficient context fields: {len(context_data)} < %s', self.config.min_context_fields)
                return {}
            sanitized_data = await asyncio.get_event_loop().run_in_executor(None, self.input_sanitizer.sanitize_json_input, context_data)
            return sanitized_data
        except Exception as e:
            logger.error('Async context validation failed: %s', e)
            return {}

    def _generate_cache_key(self, context_data: dict[str, Any]) -> str:
        """Generate cache key for context data using secure hashing."""
        key_elements = [str(context_data.get('user_id', 'unknown')), str(context_data.get('session_id', 'unknown')), str(context_data.get('project_type', 'unknown')), str(self.config.weight), str(len(context_data)), str(sorted(context_data.keys()))]
        key_string = '|'.join(key_elements)
        return hashlib.sha256(key_string.encode()).hexdigest()[:16]

    def _generate_correlation_id(self) -> str:
        """Generate correlation ID for request tracing."""
        return hashlib.md5(f'{time.time()}_{id(self)}'.encode(), usedforsecurity=False).hexdigest()[:8]

    async def _get_cached_features_async(self, cache_key: str) -> list[float] | None:
        """Get cached features if available with TTL check."""
        if not self.config.cache_enabled or cache_key not in self._feature_cache:
            return None
        cache_entry = self._feature_cache[cache_key]
        current_time = time.time()
        if current_time - cache_entry['timestamp'] > self.config.cache_ttl_seconds:
            del self._feature_cache[cache_key]
            logger.debug('Cache entry expired and removed: %s', cache_key)
            return None
        cache_entry['last_access'] = current_time
        return cache_entry['features']

    async def _cache_features_async(self, cache_key: str, features: list[float]) -> None:
        """Cache extracted features with TTL and size management."""
        if not self.config.cache_enabled:
            return
        current_time = time.time()
        await self._cleanup_cache_if_needed()
        self._feature_cache[cache_key] = {'features': features, 'timestamp': current_time, 'last_access': current_time}
        logger.debug('Cached features for key: %s', cache_key)

    async def _cleanup_cache_if_needed(self) -> None:
        """Cleanup cache based on size limits and TTL."""
        current_time = time.time()
        if current_time - self._last_cache_cleanup < self.config.cache_cleanup_interval and len(self._feature_cache) <= self.config.cache_max_size:
            return
        expired_keys = [key for key, entry in self._feature_cache.items() if current_time - entry['timestamp'] > self.config.cache_ttl_seconds]
        for key in expired_keys:
            del self._feature_cache[key]
        if len(self._feature_cache) > self.config.cache_max_size:
            sorted_entries = sorted(self._feature_cache.items(), key=lambda x: x[1]['last_access'])
            entries_to_remove = len(self._feature_cache) - self.config.cache_max_size
            for i in range(entries_to_remove):
                key = sorted_entries[i][0]
                del self._feature_cache[key]
        self._last_cache_cleanup = current_time
        logger.debug('Cache cleanup completed, %s expired entries removed', len(expired_keys))

    def _compute_context_features(self, context_data: dict[str, Any]) -> list[float]:
        """Compute raw context features."""
        features = []
        features.extend(self._extract_performance_features(context_data))
        features.extend(self._extract_metadata_features(context_data))
        features.extend(self._extract_interaction_features(context_data))
        features.extend(self._extract_temporal_features(context_data))
        return features

    def _extract_performance_features(self, context_data: dict[str, Any]) -> list[float]:
        """Extract performance-related features."""
        try:
            performance = context_data.get('performance', {})
            features = [float(performance.get('improvement_score', 0.5)), float(performance.get('user_satisfaction', 0.5)), float(performance.get('rule_effectiveness', 0.5)), float(performance.get('response_time_score', 0.5)), float(performance.get('quality_score', 0.5))]
            return features
        except Exception as e:
            logger.error('Performance feature extraction failed: %s', e)
            return [0.5] * 5

    def _extract_metadata_features(self, context_data: dict[str, Any]) -> list[float]:
        """Extract metadata features."""
        try:
            features = []
            project_type = context_data.get('project_type', 'unknown')
            project_types = ['web', 'mobile', 'data', 'ai', 'other']
            project_encoding = 0.0
            if project_type.lower() in project_types:
                project_encoding = project_types.index(project_type.lower()) / len(project_types)
            features.append(project_encoding)
            features.append(float(context_data.get('complexity_score', 0.5)))
            features.append(float(context_data.get('technical_level', 0.5)))
            context_fields = len([k for k, v in context_data.items() if v is not None])
            features.append(min(1.0, context_fields / 20.0))
            non_empty_fields = len([k for k, v in context_data.items() if v is not None and str(v).strip()])
            data_quality = non_empty_fields / max(1, len(context_data))
            features.append(data_quality)
            return features
        except Exception as e:
            logger.error('Metadata feature extraction failed: %s', e)
            return [0.5] * 5

    def _extract_interaction_features(self, context_data: dict[str, Any]) -> list[float]:
        """Extract user interaction pattern features."""
        try:
            interaction = context_data.get('interaction', {})
            features = [float(interaction.get('session_length_norm', 0.5)), float(interaction.get('iteration_count_norm', 0.5)), float(interaction.get('feedback_frequency', 0.5)), float(interaction.get('user_engagement_score', 0.5)), float(interaction.get('success_rate', 0.5))]
            return features
        except Exception as e:
            logger.error('Interaction feature extraction failed: %s', e)
            return [0.5] * 5

    def _extract_temporal_features(self, context_data: dict[str, Any]) -> list[float]:
        """Extract temporal pattern features."""
        try:
            temporal = context_data.get('temporal', {})
            features = [float(temporal.get('time_of_day_norm', 0.5)), float(temporal.get('day_of_week_norm', 0.5)), float(temporal.get('session_recency_norm', 0.5)), float(temporal.get('usage_frequency_norm', 0.5)), float(temporal.get('trend_indicator', 0.5))]
            return features
        except Exception as e:
            logger.error('Temporal feature extraction failed: %s', e)
            return [0.5] * 5

    def _normalize_features(self, features: list[float]) -> list[float]:
        """Normalize features to 0-1 range."""
        normalized = []
        for feature in features:
            normalized_feature = min(1.0, max(0.0, float(feature)))
            normalized.append(normalized_feature)
        return normalized

    def _get_default_features(self) -> list[float]:
        """Get default feature vector when extraction fails."""
        return [self.config.default_feature_value] * 20

    def _is_circuit_breaker_closed(self) -> bool:
        """Check if circuit breaker allows operations."""
        if not self.config.circuit_breaker_enabled:
            return True
        if self.circuit_breaker_state == CircuitBreakerState.CLOSED:
            return True
        elif self.circuit_breaker_state == CircuitBreakerState.OPEN:
            if self.circuit_breaker_last_failure and time.time() - self.circuit_breaker_last_failure > self.config.recovery_timeout:
                self.circuit_breaker_state = CircuitBreakerState.HALF_OPEN
                logger.info('Context extractor circuit breaker transitioning to half-open')
                return True
            return False
        else:
            return True

    def _handle_circuit_breaker_failure(self):
        """Handle circuit breaker failure."""
        if not self.config.circuit_breaker_enabled:
            return
        self.circuit_breaker_failures += 1
        self.circuit_breaker_last_failure = time.time()
        if self.circuit_breaker_failures >= self.config.failure_threshold:
            self.circuit_breaker_state = CircuitBreakerState.OPEN
            logger.warning('Context extractor circuit breaker opened after %s failures', self.circuit_breaker_failures)
        elif self.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
            self.circuit_breaker_state = CircuitBreakerState.OPEN
            logger.warning('Context extractor circuit breaker failed during half-open, returning to open')

    def _handle_circuit_breaker_success(self):
        """Handle successful operation for circuit breaker."""
        if not self.config.circuit_breaker_enabled:
            return
        if self.circuit_breaker_state == CircuitBreakerState.HALF_OPEN:
            self.circuit_breaker_state = CircuitBreakerState.CLOSED
            self.circuit_breaker_failures = 0
            logger.info('Context extractor circuit breaker closed after successful operation')

    def get_health_status(self) -> dict[str, Any]:
        """Get comprehensive health status for monitoring."""
        return {'status': 'healthy' if self.circuit_breaker_state == CircuitBreakerState.CLOSED else 'degraded', 'circuit_breaker_state': self.circuit_breaker_state.value, 'metrics': self.metrics.get_health_status(), 'cache_stats': self.get_cache_stats(), 'config': {'weight': self.config.weight, 'timeout_seconds': self.config.timeout_seconds, 'failure_threshold': self.config.failure_threshold, 'cache_enabled': self.config.cache_enabled}, 'timestamp': datetime.now(timezone.utc).isoformat()}

    def reset_metrics(self) -> None:
        """Reset all metrics for fresh monitoring."""
        self.metrics = ContextExtractionMetrics()
        logger.info('Context extractor metrics reset')

    def reset_circuit_breaker(self) -> None:
        """Manually reset circuit breaker to closed state."""
        self.circuit_breaker_state = CircuitBreakerState.CLOSED
        self.circuit_breaker_failures = 0
        self.circuit_breaker_last_failure = None
        logger.info('Context extractor circuit breaker manually reset to closed state')

    async def run_orchestrated_analysis(self, config: dict[str, Any]) -> dict[str, Any]:
        """Orchestrator-compatible interface for context feature extraction (2025 pattern).
        
        Args:
            config: Orchestrator configuration containing:
                - context_data: Context information to extract features from
                - operation: Operation type ("extract_features")
                - output_path: Local path for output files (optional)
                - correlation_id: Request correlation ID (optional)
        
        Returns:
            Dictionary with extraction results and metadata
        """
        try:
            operation = config.get('operation', 'extract_features')
            context_data = config.get('context_data', {})
            correlation_id = config.get('correlation_id')
            if correlation_id:
                logger.info('Starting orchestrated context analysis [correlation_id=%s]', correlation_id)
            if operation == 'extract_features':
                features = await self.extract_features_async(context_data)
                result = {'features': features, 'feature_names': self.get_feature_names(), 'feature_count': len(features), 'extraction_metadata': {'extractor_type': 'context', 'config_weight': self.config.weight, 'context_fields_count': len(context_data), 'circuit_breaker_state': self.circuit_breaker_state.value, 'cache_enabled': self.config.cache_enabled}}
                output_path = config.get('output_path')
                if output_path:
                    import json
                    import os
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    with open(output_path, 'w') as f:
                        json.dump(result, f, indent=2)
                    result['output_path'] = output_path
                    logger.info('Context features saved to %s', output_path)
                return {'status': 'success', 'operation': operation, 'result': result, 'component': 'context_feature_extractor', 'timestamp': datetime.now(timezone.utc).isoformat(), 'correlation_id': correlation_id}
            elif operation == 'health_check':
                health_status = self.get_health_status()
                return {'status': 'success', 'operation': operation, 'result': health_status, 'component': 'context_feature_extractor', 'timestamp': datetime.now(timezone.utc).isoformat(), 'correlation_id': correlation_id}
            else:
                return {'status': 'error', 'error': f'Unsupported operation: {operation}', 'component': 'context_feature_extractor', 'correlation_id': correlation_id}
        except Exception as e:
            logger.error('Orchestrated context analysis failed: {e} [correlation_id=%s]', correlation_id)
            return {'status': 'error', 'error': str(e), 'component': 'context_feature_extractor', 'correlation_id': correlation_id}

    def update_config(self, new_config: ContextFeatureConfig) -> None:
        """Update configuration and apply changes."""
        old_config = self.config
        self.config = new_config
        if old_config.log_level != new_config.log_level:
            logger.setLevel(getattr(logging, new_config.log_level))
        if old_config.cache_enabled != new_config.cache_enabled or old_config.cache_ttl_seconds != new_config.cache_ttl_seconds or old_config.cache_max_size != new_config.cache_max_size:
            self.clear_cache()
            logger.info('Cache cleared due to configuration change')
        if old_config.failure_threshold != new_config.failure_threshold or old_config.recovery_timeout != new_config.recovery_timeout:
            self.reset_circuit_breaker()
            logger.info('Circuit breaker reset due to configuration change')
        logger.info('ContextFeatureExtractor configuration updated')

    def get_config_dict(self) -> dict[str, Any]:
        """Get current configuration as dictionary."""
        return self.config.model_dump()

    def get_feature_names(self) -> list[str]:
        """Get names of extracted features with context prefix."""
        return ['context_improvement_score', 'context_user_satisfaction', 'context_rule_effectiveness', 'context_response_time_score', 'context_quality_score', 'context_project_type_encoding', 'context_complexity_score', 'context_technical_level', 'context_richness', 'context_data_quality', 'context_session_length_norm', 'context_iteration_count_norm', 'context_feedback_frequency', 'context_user_engagement_score', 'context_success_rate', 'context_time_of_day_norm', 'context_day_of_week_norm', 'context_session_recency_norm', 'context_usage_frequency_norm', 'context_trend_indicator']

    def clear_cache(self) -> int:
        """Clear feature cache and return number of entries cleared."""
        count = len(self._feature_cache)
        self._feature_cache.clear()
        self._last_cache_cleanup = time.time()
        logger.info('Cleared %s cached context features', count)
        return count

    def get_cache_stats(self) -> dict[str, Any]:
        """Get comprehensive cache statistics."""
        if not self.config.cache_enabled:
            return {'cache_enabled': False}
        current_time = time.time()
        valid_entries = 0
        expired_entries = 0
        for entry in self._feature_cache.values():
            if current_time - entry['timestamp'] <= self.config.cache_ttl_seconds:
                valid_entries += 1
            else:
                expired_entries += 1
        return {'cache_enabled': True, 'cache_size': len(self._feature_cache), 'cache_max_size': self.config.cache_max_size, 'valid_entries': valid_entries, 'expired_entries': expired_entries, 'cache_hits': self.metrics.cache_hits, 'cache_misses': self.metrics.cache_misses, 'hit_rate_percent': self.metrics.cache_hits / (self.metrics.cache_hits + self.metrics.cache_misses) * 100 if self.metrics.cache_hits + self.metrics.cache_misses > 0 else 0, 'ttl_seconds': self.config.cache_ttl_seconds, 'last_cleanup': self._last_cache_cleanup}
