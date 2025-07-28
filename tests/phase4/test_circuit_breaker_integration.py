"""
Test Circuit Breaker Integration in Phase 4 Implementation

Tests the 2025 circuit breaker patterns integrated into the MLIntelligenceProcessor
for graceful degradation and resilience.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock

from prompt_improver.ml.background.intelligence_processor import MLIntelligenceProcessor
from prompt_improver.performance.monitoring.health.circuit_breaker import CircuitBreakerOpen


class TestCircuitBreakerIntegration:
    """Test circuit breaker integration in ML Intelligence Processor."""

    @pytest.fixture
    def processor(self):
        """Create MLIntelligenceProcessor instance for testing."""
        return MLIntelligenceProcessor()

    @pytest.mark.asyncio
    async def test_circuit_breaker_initialization(self, processor):
        """Test that circuit breakers are properly initialized."""
        # Verify circuit breakers exist
        assert hasattr(processor, 'pattern_discovery_breaker')
        assert hasattr(processor, 'rule_optimizer_breaker')
        assert hasattr(processor, 'database_breaker')
        
        # Verify circuit breaker names
        assert processor.pattern_discovery_breaker.name == "pattern_discovery"
        assert processor.rule_optimizer_breaker.name == "rule_optimizer"
        assert processor.database_breaker.name == "database_operations"
        
        # Verify initial state is closed
        from prompt_improver.performance.monitoring.health.circuit_breaker import CircuitState
        assert processor.pattern_discovery_breaker.state == CircuitState.closed
        assert processor.rule_optimizer_breaker.state == CircuitState.closed
        assert processor.database_breaker.state == CircuitState.closed

    @pytest.mark.asyncio
    async def test_graceful_degradation_pattern_discovery_failure(self, processor):
        """Test graceful degradation when pattern discovery circuit breaker opens."""
        # Mock the pattern discovery to always fail
        with patch.object(processor.pattern_discovery_breaker, 'call', side_effect=CircuitBreakerOpen("Pattern discovery circuit open")):
            with patch.object(processor, '_process_rule_intelligence_protected', return_value={"rules_processed": 5}):
                with patch.object(processor, '_process_combination_intelligence_protected', return_value={"combinations_generated": 3}):
                    with patch.object(processor, '_process_ml_predictions_protected', return_value={"predictions_generated": 2}):
                        with patch.object(processor, '_cleanup_expired_cache', return_value={"cache_cleaned": 1}):
                            with patch.object(processor.db_manager, 'get_session') as mock_session:
                                mock_session.return_value.__aenter__ = AsyncMock()
                                mock_session.return_value.__aexit__ = AsyncMock()
                                mock_session.return_value.commit = AsyncMock()
                                
                                # Run processing
                                result = await processor.run_intelligence_processing()
                                
                                # Verify graceful degradation
                                assert result["status"] == "partial_success_degraded"
                                assert result["degraded_mode"] is True
                                assert "pattern_discovery_skipped" in result["circuit_breaker_events"]
                                assert result["patterns_discovered"] == 0
                                
                                # Verify other components still worked
                                assert result["rules_processed"] == 5
                                assert result["combinations_generated"] == 3

    @pytest.mark.asyncio
    async def test_complete_failure_database_circuit_breaker(self, processor):
        """Test complete failure when database circuit breaker opens."""
        # Mock the database circuit breaker to be open
        with patch.object(processor.database_breaker, 'call', side_effect=CircuitBreakerOpen("Database circuit open")):
            # Run processing
            result = await processor.run_intelligence_processing()
            
            # Verify complete failure with graceful handling
            assert result["status"] == "failed_circuit_breaker"
            assert result["degraded_mode"] is True
            assert "database_unavailable" in result["circuit_breaker_events"]
            assert "error" in result
            assert result["rules_processed"] == 0
            assert result["combinations_generated"] == 0
            assert result["patterns_discovered"] == 0

    @pytest.mark.asyncio
    async def test_circuit_breaker_state_change_logging(self, processor):
        """Test that circuit breaker state changes are properly logged."""
        from prompt_improver.performance.monitoring.health.circuit_breaker import CircuitState
        
        # Test state change callback
        with patch('prompt_improver.ml.background.intelligence_processor.logger') as mock_logger:
            processor._on_breaker_state_change("test_component", CircuitState.open)
            
            # Verify logging
            mock_logger.warning.assert_called_once()
            mock_logger.error.assert_called_once()
            
            # Verify stats update
            assert processor.processing_stats.get("test_component_circuit_open") is True

    @pytest.mark.asyncio
    async def test_multiple_circuit_breaker_failures(self, processor):
        """Test behavior when multiple circuit breakers fail."""
        # Mock multiple circuit breakers to fail
        with patch.object(processor.pattern_discovery_breaker, 'call', side_effect=CircuitBreakerOpen("Pattern discovery circuit open")):
            with patch.object(processor.rule_optimizer_breaker, 'call', side_effect=CircuitBreakerOpen("Rule optimizer circuit open")):
                with patch.object(processor, '_cleanup_expired_cache', return_value={"cache_cleaned": 1}):
                    with patch.object(processor.db_manager, 'get_session') as mock_session:
                        mock_session.return_value.__aenter__ = AsyncMock()
                        mock_session.return_value.__aexit__ = AsyncMock()
                        mock_session.return_value.commit = AsyncMock()
                        
                        # Run processing
                        result = await processor.run_intelligence_processing()
                        
                        # Verify multiple failures are handled gracefully
                        assert result["status"] == "partial_success_degraded"
                        assert result["degraded_mode"] is True
                        
                        # Verify all expected circuit breaker events
                        expected_events = [
                            "rule_intelligence_skipped",
                            "combination_intelligence_skipped", 
                            "pattern_discovery_skipped",
                            "ml_predictions_skipped"
                        ]
                        for event in expected_events:
                            assert event in result["circuit_breaker_events"]

    def test_circuit_breaker_configuration_2025_patterns(self, processor):
        """Test that circuit breakers are configured with 2025 best practices."""
        # Pattern discovery should be more lenient (exploratory ML)
        pattern_config = processor.pattern_discovery_breaker.config
        assert pattern_config.failure_threshold == 3
        assert pattern_config.success_rate_threshold == 0.7
        assert pattern_config.response_time_threshold_ms == 30000
        
        # Rule optimizer should be stricter (production optimization)
        optimizer_config = processor.rule_optimizer_breaker.config
        assert optimizer_config.failure_threshold == 2
        assert optimizer_config.success_rate_threshold == 0.85
        assert optimizer_config.response_time_threshold_ms == 15000
        
        # Database should be highly reliable
        database_config = processor.database_breaker.config
        assert database_config.failure_threshold == 5
        assert database_config.success_rate_threshold == 0.95
        assert database_config.response_time_threshold_ms == 5000
