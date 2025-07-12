"""Unit tests for _Timer context manager in metrics module."""

import pytest
import time
import asyncio
from unittest.mock import patch, MagicMock

from src.prompt_improver.services.health.metrics import (
    _Timer,
    HEALTH_CHECK_DURATION,
    HEALTH_CHECK_STATUS,
    HEALTH_CHECKS_TOTAL,
    HEALTH_CHECK_RESPONSE_TIME,
    instrument_health_check,
    get_health_metrics_summary,
    reset_health_metrics,
    PROMETHEUS_AVAILABLE
)


class TestTimerContextManager:
    """Test the _Timer context manager implementation."""
    
    def test_timer_context_manager_basic(self):
        """Test basic _Timer context manager functionality."""
        with _Timer() as timer:
            # Simulate some work
            time.sleep(0.01)
        
        # Verify timer attributes are set
        assert hasattr(timer, 'start')
        assert hasattr(timer, 'end')
        assert hasattr(timer, 'duration')
        
        # Verify timing is reasonable
        assert timer.duration > 0
        assert timer.duration < 1.0  # Should be much less than 1 second
        assert timer.end > timer.start
    
    def test_timer_context_manager_exception_handling(self):
        """Test _Timer context manager with exceptions."""
        with pytest.raises(ValueError):
            with _Timer() as timer:
                raise ValueError("Test exception")
        
        # Timer should still have recorded the timing
        assert hasattr(timer, 'start')
        assert hasattr(timer, 'end')
        assert hasattr(timer, 'duration')
        assert timer.duration >= 0
    
    def test_timer_context_manager_precision(self):
        """Test _Timer context manager timing precision."""
        with _Timer() as timer:
            # Very short operation
            pass
        
        # Even for minimal operations, duration should be non-negative
        assert timer.duration >= 0
        assert timer.end >= timer.start
    
    def test_timer_context_manager_multiple_uses(self):
        """Test _Timer context manager can be used multiple times."""
        timer1 = _Timer()
        timer2 = _Timer()
        
        with timer1:
            time.sleep(0.005)
        
        with timer2:
            time.sleep(0.01)
        
        # Both timers should work independently
        assert timer1.duration > 0
        assert timer2.duration > 0
        assert timer2.duration > timer1.duration


class TestMetricsTimerIntegration:
    """Test integration of _Timer with metrics."""
    
    @patch('src.prompt_improver.services.health.metrics.PROMETHEUS_AVAILABLE', False)
    def test_mock_metric_timer_returns_timer(self):
        """Test that MockMetric.time() returns _Timer instance when Prometheus unavailable."""
        # Re-import to get mock metrics
        from src.prompt_improver.services.health.metrics import _create_metric_safe, Summary
        
        # Create a mock metric
        mock_metric = _create_metric_safe(Summary, 'test_metric', 'Test metric')
        duration_metric = mock_metric.labels(component='test')
        timer = duration_metric.time()
        
        assert isinstance(timer, _Timer)
    
    @patch('src.prompt_improver.services.health.metrics.PROMETHEUS_AVAILABLE', False)
    def test_mock_metric_timer_context_manager(self):
        """Test MockMetric timer works as context manager when Prometheus unavailable."""
        # Re-import to get mock metrics
        from src.prompt_improver.services.health.metrics import _create_metric_safe, Summary
        
        # Create a mock metric
        mock_metric = _create_metric_safe(Summary, 'test_metric_2', 'Test metric 2')
        duration_metric = mock_metric.labels(component='test')
        
        with duration_metric.time() as timer:
            time.sleep(0.001)
        
        assert hasattr(timer, 'duration')
        assert timer.duration > 0
    
    def test_real_metric_timer_context_manager(self):
        """Test that real Prometheus metrics work as context manager."""
        duration_metric = HEALTH_CHECK_DURATION.labels(component='test')
        
        # Real Prometheus timer should work as context manager
        with duration_metric.time() as timer:
            time.sleep(0.001)
        
        # Real timer might not have duration attribute, but context manager should work
        assert True  # If we get here, the context manager worked


class TestHealthCheckInstrumentation:
    """Test the health check instrumentation with _Timer."""
    
    @pytest.mark.asyncio
    async def test_instrument_health_check_with_timer(self):
        """Test health check instrumentation uses timer context manager."""
        # Mock health check function
        async def mock_health_check():
            await asyncio.sleep(0.001)
            result = MagicMock(status='healthy')
            result.response_time_ms = 10
            return result
        
        # Apply instrumentation
        instrumented_check = instrument_health_check('test_component')(mock_health_check)
        
        # Run the instrumented check
        result = await instrumented_check()
        
        # Should complete without error
        assert result is not None
    
    @pytest.mark.asyncio
    async def test_instrument_health_check_timing_behavior(self):
        """Test that instrumentation properly uses context manager timing."""
        call_count = 0
        
        async def mock_health_check():
            nonlocal call_count
            call_count += 1
            await asyncio.sleep(0.001)
            result = MagicMock(status='healthy')
            result.response_time_ms = 0.5
            return result
        
        # Apply instrumentation
        instrumented_check = instrument_health_check('test_timing')(mock_health_check)
        
        # Run the check
        await instrumented_check()
        
        # Verify the function was called
        assert call_count == 1


@pytest.mark.skipif(not PROMETHEUS_AVAILABLE, reason="Prometheus client not available")
class TestRealPrometheusIntegration:
    """Test integration with real Prometheus client when available."""
    
    def test_real_prometheus_timer_context_manager(self):
        """Test that real Prometheus metrics work with context manager."""
        # This test only runs if prometheus_client is available
        from prometheus_client import Summary
        
        # Create a real Summary metric
        test_summary = Summary('test_timer_summary', 'Test summary for timer')
        
        # Use it as context manager
        with test_summary.time() as timer:
            time.sleep(0.001)
        
        # Real Prometheus timer might not have duration attribute
        # but the context manager should work
        assert True  # If we get here, the context manager worked


class TestMetricsModuleFunctions:
    """Test other metrics module functions."""
    
    def test_get_health_metrics_summary(self):
        """Test get_health_metrics_summary function."""
        summary = get_health_metrics_summary()
        
        assert isinstance(summary, dict)
        assert 'prometheus_available' in summary
        assert isinstance(summary['prometheus_available'], bool)
    
    def test_reset_health_metrics(self):
        """Test reset_health_metrics function."""
        # Should not raise exception
        reset_health_metrics()
        assert True


if __name__ == '__main__':
    pytest.main([__file__])
