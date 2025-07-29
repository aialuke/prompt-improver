"""Modern Async Tests for ML Learning Components (2025)

Demonstrates modern testing patterns with async/await, Pydantic validation,
and comprehensive integration testing following 2025 best practices.
"""

import asyncio
import pytest
import pytest_asyncio
import numpy as np
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timezone

from prompt_improver.ml.learning.features.linguistic_feature_extractor import (
    LinguisticFeatureExtractor,
    FeatureExtractionConfig,
    FeatureExtractionRequest,
    FeatureExtractionResponse,
    ExtractionMetrics
)
from pydantic import ValidationError


class TestFeatureExtractionConfig:
    """Test configuration model validation."""
    
    def test_valid_config(self):
        """Test valid configuration creation."""
        config = FeatureExtractionConfig(
            weight=1.5,
            cache_enabled=True,
            cache_ttl_seconds=7200,
            deterministic=False,
            async_batch_size=20
        )
        assert config.weight == 1.5
        assert config.cache_enabled is True
        assert config.cache_ttl_seconds == 7200
        assert config.deterministic is False
        assert config.async_batch_size == 20
    
    def test_config_validation_bounds(self):
        """Test configuration validation bounds."""
        # Test weight bounds
        with pytest.raises(ValidationError):
            FeatureExtractionConfig(weight=-1.0)  # Below minimum
        
        with pytest.raises(ValidationError):
            FeatureExtractionConfig(weight=11.0)  # Above maximum
            
        # Test cache TTL bounds
        with pytest.raises(ValidationError):
            FeatureExtractionConfig(cache_ttl_seconds=30)  # Below minimum
    
    def test_config_defaults(self):
        """Test default configuration values."""
        config = FeatureExtractionConfig()
        assert config.weight == 1.0
        assert config.cache_enabled is True
        assert config.cache_ttl_seconds == 3600
        assert config.deterministic is True
        assert config.max_text_length == 100000
        assert config.enable_metrics is True
        assert config.async_batch_size == 10


class TestFeatureExtractionRequest:
    """Test request model validation."""
    
    def test_valid_request(self):
        """Test valid request creation."""
        request = FeatureExtractionRequest(
            text="This is a test prompt for feature extraction.",
            context={"source": "test"},
            priority=2
        )
        assert request.text == "This is a test prompt for feature extraction."
        assert request.context == {"source": "test"}
        assert request.priority == 2
        assert isinstance(request.correlation_id, str)
        assert len(request.correlation_id) > 0
    
    def test_text_validation(self):
        """Test text field validation."""
        # Test empty text
        with pytest.raises(ValidationError):
            FeatureExtractionRequest(text="")
        
        # Test whitespace-only text
        with pytest.raises(ValidationError):
            FeatureExtractionRequest(text="   ")
    
    def test_text_stripping(self):
        """Test automatic text stripping."""
        request = FeatureExtractionRequest(text="  test text  ")
        assert request.text == "test text"
    
    def test_priority_validation(self):
        """Test priority validation bounds."""
        with pytest.raises(ValidationError):
            FeatureExtractionRequest(text="test", priority=0)  # Below minimum
        
        with pytest.raises(ValidationError):
            FeatureExtractionRequest(text="test", priority=4)  # Above maximum


class TestFeatureExtractionResponse:
    """Test response model validation."""
    
    def test_valid_response(self):
        """Test valid response creation."""
        response = FeatureExtractionResponse(
            features=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            feature_names=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"],
            extraction_time_ms=123.45,
            correlation_id="test-123",
            confidence_score=0.85
        )
        assert len(response.features) == 10
        assert len(response.feature_names) == 10
        assert response.extraction_time_ms == 123.45
        assert response.confidence_score == 0.85
    
    def test_feature_range_validation(self):
        """Test feature range validation."""
        with pytest.raises(ValidationError):
            FeatureExtractionResponse(
                features=[-0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # Invalid negative
                feature_names=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"],
                extraction_time_ms=100.0,
                correlation_id="test"
            )
        
        with pytest.raises(ValidationError):
            FeatureExtractionResponse(
                features=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.1],  # Invalid > 1.0
                feature_names=["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"],
                extraction_time_ms=100.0,
                correlation_id="test"
            )


class TestExtractionMetrics:
    """Test metrics collection functionality."""
    
    def test_metrics_initialization(self):
        """Test metrics initialization."""
        metrics = ExtractionMetrics()
        assert metrics.total_extractions == 0
        assert metrics.cache_hits == 0
        assert metrics.cache_misses == 0
        assert metrics.errors == 0
        assert metrics.cache_hit_rate == 0.0
        assert metrics.error_rate == 0.0
    
    def test_timing_updates(self):
        """Test timing metric updates."""
        metrics = ExtractionMetrics()
        
        # Test cache miss
        metrics.update_timing(100.0, cache_hit=False)
        assert metrics.total_extractions == 1
        assert metrics.cache_misses == 1
        assert metrics.cache_hits == 0
        assert metrics.average_response_time_ms == 100.0
        
        # Test cache hit
        metrics.update_timing(50.0, cache_hit=True)
        assert metrics.total_extractions == 2
        assert metrics.cache_hits == 1
        assert metrics.cache_misses == 1
        assert metrics.average_response_time_ms == 75.0
    
    def test_error_tracking(self):
        """Test error tracking."""
        metrics = ExtractionMetrics()
        metrics.update_timing(100.0)
        metrics.increment_errors()
        
        assert metrics.errors == 1
        assert metrics.error_rate == 100.0


@pytest_asyncio.fixture
async def mock_input_sanitizer():
    """Mock input sanitizer for testing."""
    sanitizer = Mock()
    sanitizer.sanitize_html_input = Mock(return_value="sanitized text")
    return sanitizer


@pytest_asyncio.fixture
async def mock_redis_cache():
    """Mock Redis cache for testing."""
    cache = Mock()
    cache.get_async = AsyncMock(return_value=None)
    cache.set_async = AsyncMock()
    cache.clear_pattern_async = AsyncMock(return_value=5)
    cache.close_async = AsyncMock()
    return cache


@pytest_asyncio.fixture
async def extractor(mock_input_sanitizer, mock_redis_cache):
    """Create linguistic feature extractor for testing."""
    config = FeatureExtractionConfig(
        cache_enabled=True,
        deterministic=True,
        async_batch_size=5
    )
    return LinguisticFeatureExtractor(
        config=config,
        input_sanitizer=mock_input_sanitizer,
        redis_cache=mock_redis_cache
    )


class TestLinguisticFeatureExtractor:
    """Test modern async linguistic feature extractor."""
    
    @pytest.mark.asyncio
    async def test_initialization(self, extractor):
        """Test extractor initialization."""
        assert extractor.config.weight == 1.0
        assert extractor.config.cache_enabled is True
        assert len(extractor.get_feature_names()) == 10
        
        # Test feature names are correct
        expected_names = [
            "readability_score", "lexical_diversity", "entity_density",
            "syntactic_complexity", "sentence_structure_quality", 
            "technical_term_ratio", "avg_sentence_length_norm",
            "instruction_clarity", "has_examples", "overall_linguistic_quality"
        ]
        assert extractor.get_feature_names() == expected_names
    
    @pytest.mark.asyncio
    async def test_feature_extraction_with_request_object(self, extractor):
        """Test feature extraction using request object."""
        request = FeatureExtractionRequest(
            text="This is a comprehensive test prompt for advanced feature extraction analysis.",
            context={"source": "test"}
        )
        
        response = await extractor.extract_features(request)
        
        # Validate response structure
        assert isinstance(response, FeatureExtractionResponse)
        assert len(response.features) == 10
        assert len(response.feature_names) == 10
        assert all(0.0 <= f <= 1.0 for f in response.features)
        assert response.correlation_id == request.correlation_id
        assert response.extraction_time_ms >= 0
        assert 0.0 <= response.confidence_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_feature_extraction_with_string_input(self, extractor):
        """Test backward compatibility with string input."""
        text = "This is a test prompt for feature extraction."
        
        response = await extractor.extract_features(text)
        
        assert isinstance(response, FeatureExtractionResponse)
        assert len(response.features) == 10
        assert all(0.0 <= f <= 1.0 for f in response.features)
        assert response.correlation_id is not None
    
    @pytest.mark.asyncio
    async def test_invalid_input_handling(self, extractor):
        """Test handling of invalid inputs."""
        # Test empty string
        response = await extractor.extract_features("")
        assert response.features == [0.5] * 10
        assert response.confidence_score == 0.0
        
        # Test whitespace only
        response = await extractor.extract_features("   ")
        assert response.features == [0.5] * 10
        assert response.confidence_score == 0.0
    
    @pytest.mark.asyncio
    async def test_cache_functionality(self, extractor, mock_redis_cache):
        """Test caching functionality."""
        request = FeatureExtractionRequest(text="Test caching functionality")
        
        # First call - cache miss
        response1 = await extractor.extract_features(request)
        assert not response1.cache_hit
        
        # Verify cache write was called
        mock_redis_cache.set_async.assert_called_once()
        
    @pytest.mark.asyncio
    async def test_cache_hit_scenario(self, extractor, mock_redis_cache):
        """Test cache hit scenario."""
        # Mock cached data
        cached_data = {
            "features": [0.1] * 10,
            "feature_names": ["f1", "f2", "f3", "f4", "f5", "f6", "f7", "f8", "f9", "f10"],
            "extraction_time_ms": 50.0,
            "cache_hit": False,
            "correlation_id": "cached-id",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "confidence_score": 0.8
        }
        mock_redis_cache.get_async.return_value = cached_data
        
        request = FeatureExtractionRequest(text="Cached text")
        response = await extractor.extract_features(request)
        
        assert response.cache_hit is True
        assert response.features == [0.1] * 10
        assert response.correlation_id == request.correlation_id  # Should be updated
    
    @pytest.mark.asyncio
    async def test_batch_extraction(self, extractor):
        """Test batch feature extraction."""
        requests = [
            FeatureExtractionRequest(text=f"Test text {i}", correlation_id=f"test-{i}")
            for i in range(3)
        ]
        
        responses = await extractor.batch_extract_features(requests)
        
        assert len(responses) == 3
        for i, response in enumerate(responses):
            assert response.correlation_id == f"test-{i}"
            assert len(response.features) == 10
            assert all(0.0 <= f <= 1.0 for f in response.features)
    
    @pytest.mark.asyncio
    async def test_health_check(self, extractor):
        """Test health check functionality."""
        health = await extractor.health_check()
        
        assert health["status"] == "healthy"
        assert health["component"] == "linguistic_feature_extractor"
        assert health["version"] == "2.0.0"
        assert "metrics" in health
        assert "configuration" in health
        assert "timestamp" in health
    
    @pytest.mark.asyncio
    async def test_metrics_collection(self, extractor):
        """Test metrics collection."""
        # Perform some extractions
        await extractor.extract_features("Test text 1")
        await extractor.extract_features("Test text 2")
        
        metrics = await extractor.get_metrics()
        
        assert metrics["component"] == "linguistic_feature_extractor"
        assert metrics["metrics"]["total_extractions"] == 2
        assert "timestamp" in metrics
        assert "configuration" in metrics
    
    @pytest.mark.asyncio
    async def test_cache_clearing(self, extractor, mock_redis_cache):
        """Test cache clearing functionality."""
        result = await extractor.clear_cache()
        
        assert result["status"] == "success"
        assert result["cleared_entries"] == 5  # Mock return value
        mock_redis_cache.clear_pattern_async.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown(self, extractor, mock_redis_cache):
        """Test graceful shutdown."""
        await extractor.shutdown()
        
        # Verify Redis connection was closed
        mock_redis_cache.close_async.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_extraction_context_manager(self, extractor):
        """Test extraction context manager."""
        async with extractor.extraction_context() as ctx:
            assert ctx is extractor
            response = await ctx.extract_features("Test context manager")
            assert len(response.features) == 10
    
    @pytest.mark.asyncio
    async def test_feature_consistency(self, extractor):
        """Test feature extraction consistency."""
        text = "This is a consistent test for feature extraction validation."
        
        # Extract features multiple times
        responses = []
        for _ in range(3):
            response = await extractor.extract_features(text)
            responses.append(response.features)
        
        # All responses should be identical due to deterministic=True
        for i in range(1, len(responses)):
            assert responses[i] == responses[0], "Features should be consistent across extractions"
    
    def test_repr_method(self, extractor):
        """Test string representation."""
        repr_str = repr(extractor)
        assert "LinguisticFeatureExtractor" in repr_str
        assert "weight=1.0" in repr_str
        assert "cache_enabled=True" in repr_str
        assert "deterministic=True" in repr_str


@pytest.mark.asyncio
async def test_integration_with_orchestrator():
    """Integration test demonstrating orchestrator compatibility."""
    # This test shows how the component integrates with the orchestrator
    config = FeatureExtractionConfig(
        weight=1.5,
        cache_enabled=True,
        deterministic=True
    )
    
    extractor = LinguisticFeatureExtractor(config=config)
    
    # Test health check endpoint
    health = await extractor.health_check()
    assert health["status"] == "healthy"
    
    # Test metrics endpoint
    metrics = await extractor.get_metrics()
    assert "metrics" in metrics
    
    # Test feature extraction
    request = FeatureExtractionRequest(
        text="Integration test for orchestrator compatibility and modern async patterns."
    )
    response = await extractor.extract_features(request)
    
    assert isinstance(response, FeatureExtractionResponse)
    assert len(response.features) == 10
    assert response.correlation_id == request.correlation_id
    
    # Cleanup
    await extractor.shutdown()
