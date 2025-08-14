"""Integration tests for RuleAnalysisService.

Tests the real behavior of the extracted rule analysis logic with mock dependencies.
Validates performance, error handling, and circuit breaker integration.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock
from typing import Dict, Any, List

from src.prompt_improver.ml.services.intelligence.rule_analysis_service import RuleAnalysisService
from src.prompt_improver.ml.services.intelligence.protocols.intelligence_service_protocols import (
    IntelligenceResult,
)


class MockMLRepository:
    """Mock ML repository for testing."""
    
    def __init__(self):
        self.prompt_characteristics_data = [
            {
                "session_id": "test_session_1",
                "original_prompt": "Test prompt",
                "improved_prompt": "Improved test prompt",
                "improvement_score": 0.8,
                "quality_score": 0.75,
                "confidence_level": 0.85,
            }
        ]
        
        self.rule_performance_data = [
            {
                "rule_id": "rule_001",
                "usage_count": 25,
                "success_count": 20,
                "effectiveness_ratio": 0.8,
                "confidence_score": 0.85,
                "avg_improvement": 0.15,
            },
            {
                "rule_id": "rule_002", 
                "usage_count": 15,
                "success_count": 10,
                "effectiveness_ratio": 0.67,
                "confidence_score": 0.70,
                "avg_improvement": 0.10,
            }
        ]
        
        self.combinations_data = [
            {
                "rule_combination": ["rule_001", "rule_002"],
                "avg_improvement": 0.75,
                "avg_quality": 0.80,
                "usage_count": 12,
                "last_used": "2025-08-14T10:00:00Z",
            }
        ]
        
        self.cached_rules = set()
        self.cached_intelligence = []
        self.cached_combinations = []
    
    async def get_prompt_characteristics_batch(self, batch_size: int = 100) -> List[Dict[str, Any]]:
        return self.prompt_characteristics_data[:batch_size]
    
    async def get_rule_performance_data(self, batch_size: int = 100) -> List[Dict[str, Any]]:
        return self.rule_performance_data[:batch_size]
    
    async def cache_rule_intelligence(self, intelligence_data: List[Dict[str, Any]]) -> None:
        self.cached_intelligence.extend(intelligence_data)
    
    async def get_rule_combinations_data(self, batch_size: int = 100) -> List[Dict[str, Any]]:
        return self.combinations_data[:batch_size]
    
    async def cache_combination_intelligence(self, combination_data: List[Dict[str, Any]]) -> None:
        self.cached_combinations.extend(combination_data)
    
    async def check_rule_intelligence_freshness(self, rule_id: str) -> bool:
        return rule_id in self.cached_rules
    
    async def get_rule_historical_performance(self, rule_id: str) -> List[Dict[str, Any]]:
        # Simulate historical performance data
        return [
            {"effectiveness": 0.75, "timestamp": "2025-08-10T10:00:00Z"},
            {"effectiveness": 0.80, "timestamp": "2025-08-11T10:00:00Z"},
            {"effectiveness": 0.85, "timestamp": "2025-08-12T10:00:00Z"},
            {"effectiveness": 0.82, "timestamp": "2025-08-13T10:00:00Z"},
        ] if rule_id == "rule_001" else []


class MockPatternDiscovery:
    """Mock pattern discovery component."""
    
    async def discover_advanced_patterns(self, **kwargs) -> Dict[str, Any]:
        return {
            "parameter_patterns": {"pattern_1": 0.85},
            "performance_patterns": {"trend_up": 0.70},
            "ensemble_analysis": {"confidence": 0.80}
        }


class MockRuleOptimizer:
    """Mock rule optimizer component."""
    
    async def optimize_rule(self, rule_id: str, performance_data: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "recommendations": [f"Optimize parameters for {rule_id}"],
            "trend": "stable",
            "optimization_score": 0.75
        }


class MockCircuitBreaker:
    """Mock circuit breaker service."""
    
    def __init__(self):
        self.call_count = 0
        self.failure_mode = False
    
    async def call_with_breaker(self, component: str, operation, *args, **kwargs):
        self.call_count += 1
        if self.failure_mode:
            raise Exception("Circuit breaker test failure")
        return await operation(*args, **kwargs)
    
    def set_failure_mode(self, enabled: bool):
        self.failure_mode = enabled


@pytest.fixture
def mock_dependencies():
    """Create mock dependencies for testing."""
    return {
        "ml_repository": MockMLRepository(),
        "pattern_discovery": MockPatternDiscovery(),
        "rule_optimizer": MockRuleOptimizer(),
        "circuit_breaker": MockCircuitBreaker(),
    }


@pytest.fixture
def rule_analysis_service(mock_dependencies):
    """Create RuleAnalysisService with mock dependencies."""
    return RuleAnalysisService(
        ml_repository=mock_dependencies["ml_repository"],
        pattern_discovery=mock_dependencies["pattern_discovery"],
        rule_optimizer=mock_dependencies["rule_optimizer"],
        circuit_breaker=mock_dependencies["circuit_breaker"],
        batch_size=10,
        cache_ttl_hours=24,
        performance_threshold_ms=50.0,
    )


@pytest.mark.asyncio
async def test_process_rule_intelligence_success(rule_analysis_service, mock_dependencies):
    """Test successful rule intelligence processing."""
    # Execute
    result = await rule_analysis_service.process_rule_intelligence()
    
    # Verify
    assert isinstance(result, IntelligenceResult)
    assert result.success is True
    assert result.confidence == 0.85
    assert result.processing_time_ms > 0
    assert result.processing_time_ms < 100  # Should be fast with mocks
    assert result.data["rules_processed"] == 2
    assert result.data["intelligence_cached"] == 2
    
    # Verify intelligence was cached
    cached_intelligence = mock_dependencies["ml_repository"].cached_intelligence
    assert len(cached_intelligence) == 2
    assert cached_intelligence[0]["rule_id"] == "rule_001"
    assert cached_intelligence[1]["rule_id"] == "rule_002"


@pytest.mark.asyncio
async def test_process_rule_intelligence_with_specific_rule_ids(rule_analysis_service, mock_dependencies):
    """Test rule intelligence processing with specific rule IDs."""
    # Execute
    result = await rule_analysis_service.process_rule_intelligence(rule_ids=["rule_001"])
    
    # Verify
    assert result.success is True
    assert result.data["rules_processed"] == 1
    
    # Verify only specified rule was processed
    cached_intelligence = mock_dependencies["ml_repository"].cached_intelligence
    assert len(cached_intelligence) == 1
    assert cached_intelligence[0]["rule_id"] == "rule_001"


@pytest.mark.asyncio
async def test_process_rule_intelligence_with_cached_rules(rule_analysis_service, mock_dependencies):
    """Test rule intelligence processing with cached rules."""
    # Setup - mark rule as cached
    mock_dependencies["ml_repository"].cached_rules.add("rule_001")
    
    # Execute
    result = await rule_analysis_service.process_rule_intelligence()
    
    # Verify
    assert result.success is True
    assert result.data["cache_hits"] == 1
    assert result.data["rules_processed"] == 1  # Only rule_002 processed
    assert result.cache_hit is True


@pytest.mark.asyncio
async def test_process_combination_intelligence_success(rule_analysis_service, mock_dependencies):
    """Test successful combination intelligence processing."""
    # Execute
    result = await rule_analysis_service.process_combination_intelligence()
    
    # Verify
    assert isinstance(result, IntelligenceResult)
    assert result.success is True
    assert result.confidence == 0.80
    assert result.processing_time_ms > 0
    assert result.data["combinations_generated"] == 1
    assert result.data["intelligence_cached"] == 1
    
    # Verify combination intelligence was cached
    cached_combinations = mock_dependencies["ml_repository"].cached_combinations
    assert len(cached_combinations) == 1
    assert cached_combinations[0]["rule_combination"] == ["rule_001", "rule_002"]
    assert cached_combinations[0]["synergy_score"] > 0


@pytest.mark.asyncio
async def test_generate_intelligence_data(rule_analysis_service):
    """Test intelligence data generation."""
    # Setup
    rule_characteristics = {
        "rule_id": "rule_001",
        "effectiveness_ratio": 0.80,
        "usage_count": 25,
        "confidence_score": 0.85,
        "avg_improvement": 0.15,
    }
    
    # Execute
    result = await rule_analysis_service.generate_intelligence_data(rule_characteristics)
    
    # Verify
    assert "effectiveness_prediction" in result
    assert "confidence_score" in result
    assert "context_compatibility" in result
    assert "usage_recommendations" in result
    assert "characteristics_hash" in result
    assert result["effectiveness_prediction"] > 0.8  # Should be enhanced
    assert result["confidence_score"] == 0.85


@pytest.mark.asyncio
async def test_analyze_rule_effectiveness(rule_analysis_service, mock_dependencies):
    """Test rule effectiveness analysis."""
    # Execute
    result = await rule_analysis_service.analyze_rule_effectiveness(
        "rule_001", 
        {"usage_pattern": "frequent"}
    )
    
    # Verify
    assert result["rule_id"] == "rule_001"
    assert "effectiveness_score" in result
    assert "trend_direction" in result
    assert "confidence_level" in result
    assert "recommendations" in result
    assert result["effectiveness_score"] > 0.7  # Should be good
    assert result["trend_direction"] == "improving"  # Based on mock data


@pytest.mark.asyncio
async def test_analyze_rule_effectiveness_insufficient_data(rule_analysis_service):
    """Test rule effectiveness analysis with insufficient data."""
    # Execute
    result = await rule_analysis_service.analyze_rule_effectiveness(
        "unknown_rule", 
        {"usage_pattern": "rare"}
    )
    
    # Verify
    assert result["rule_id"] == "unknown_rule"
    assert result["analysis_status"] == "insufficient_data"
    assert "recommendation" in result


@pytest.mark.asyncio
async def test_circuit_breaker_failure_handling(rule_analysis_service, mock_dependencies):
    """Test circuit breaker failure handling."""
    # Setup - enable failure mode
    mock_dependencies["circuit_breaker"].set_failure_mode(True)
    
    # Execute
    result = await rule_analysis_service.process_rule_intelligence()
    
    # Verify
    assert result.success is False
    assert result.error_message == "Circuit breaker test failure"
    assert result.confidence == 0.0
    assert result.processing_time_ms > 0


@pytest.mark.asyncio
async def test_performance_threshold_monitoring(rule_analysis_service, mock_dependencies):
    """Test performance threshold monitoring."""
    # Setup - reduce threshold to trigger warning
    rule_analysis_service.performance_threshold_ms = 0.1  # Very low threshold
    
    # Execute
    result = await rule_analysis_service.process_rule_intelligence()
    
    # Verify - should still succeed but log warning
    assert result.success is True
    assert result.processing_time_ms > rule_analysis_service.performance_threshold_ms


@pytest.mark.asyncio
async def test_performance_metrics_tracking(rule_analysis_service):
    """Test performance metrics tracking."""
    # Execute multiple operations
    await rule_analysis_service.process_rule_intelligence()
    await rule_analysis_service.process_combination_intelligence()
    
    # Get metrics
    metrics = rule_analysis_service.get_performance_metrics()
    
    # Verify
    assert metrics["total_rules_processed"] == 2
    assert metrics["total_combinations_processed"] == 1
    assert metrics["avg_processing_time_ms"] > 0
    assert "performance_threshold_ms" in metrics
    assert "batch_size" in metrics


@pytest.mark.asyncio
async def test_concurrent_processing(rule_analysis_service):
    """Test concurrent rule processing."""
    # Execute multiple concurrent operations
    tasks = [
        rule_analysis_service.process_rule_intelligence(),
        rule_analysis_service.process_combination_intelligence(),
        rule_analysis_service.generate_intelligence_data({
            "rule_id": "test_rule",
            "effectiveness_ratio": 0.75,
            "usage_count": 20,
        }),
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Verify all operations completed successfully
    assert len(results) == 3
    assert all(not isinstance(r, Exception) for r in results)
    assert results[0].success is True
    assert results[1].success is True
    assert isinstance(results[2], dict)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])