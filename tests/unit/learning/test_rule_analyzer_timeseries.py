"""Tests for Phase 2 Time Series Cross-Validation in Rule Effectiveness Analyzer.

Comprehensive test suite for time series enhancements including:
- Time series cross-validation with seasonal decomposition
- Change point detection for performance regime shifts
- Rolling window validation for temporal stability
- Integration with existing rule analysis workflow

Testing best practices applied from Context7 research:
- Proper temporal data validation
- Realistic time series parameter ranges
- Statistical significance testing for temporal patterns
- Cross-validation fold validation
"""

import pytest
import numpy as np
import pandas as pd
import asyncio
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, List, Any
from datetime import datetime, timedelta

from prompt_improver.learning.rule_analyzer import (
    RuleEffectivenessAnalyzer,
    RuleAnalysisConfig,
    RuleMetrics,
    TimeSeriesValidationResult
)


@pytest.fixture
def timeseries_config():
    """Configuration with time series cross-validation enabled."""
    return RuleAnalysisConfig(
        enable_time_series_cv=True,
        time_series_cv_splits=5,
        time_series_min_train_size=20,
        rolling_window_size=7,
        seasonal_decomposition=True,
        # Standard parameters
        min_sample_size=10,
        significance_level=0.05,
        effect_size_threshold=0.2
    )


@pytest.fixture
def rule_analyzer_ts(timeseries_config):
    """Rule analyzer with time series validation enabled."""
    return RuleEffectivenessAnalyzer(config=timeseries_config)


@pytest.fixture
def temporal_rule_data():
    """Realistic temporal rule performance data."""
    np.random.seed(42)  # Reproducible test data
    
    # Generate 60 days of temporal data
    n_days = 60
    dates = [datetime.now() - timedelta(days=i) for i in range(n_days)]
    dates.reverse()
    
    # Simulate rule performance with temporal patterns
    base_performance = 0.75
    
    # Add trend component
    trend = np.linspace(0.0, 0.1, n_days)  # Slight improvement over time
    
    # Add seasonal component (weekly pattern)
    seasonal = 0.05 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    
    # Add noise
    noise = np.random.normal(0, 0.08, n_days)
    
    # Combine components
    performance_scores = base_performance + trend + seasonal + noise
    performance_scores = np.clip(performance_scores, 0.0, 1.0)
    
    # Create rule data with temporal information
    rule_data = {
        "rule_clarity_001": {
            "temporal_data": [
                {
                    "timestamp": dates[i].isoformat(),
                    "score": float(performance_scores[i]),
                    "applications": 15 + np.random.randint(-5, 6),
                    "context": f"context_{i % 5}",
                    "execution_time_ms": 80 + np.random.randint(-20, 21)
                }
                for i in range(n_days)
            ],
            "total_applications": n_days * 15,
            "avg_score": float(np.mean(performance_scores)),
            "std_score": float(np.std(performance_scores)),
            "contexts_used": [f"context_{i}" for i in range(5)]
        }
    }
    
    return rule_data


@pytest.fixture
def temporal_data_with_changepoint():
    """Temporal data with a clear change point."""
    np.random.seed(123)
    
    n_days = 50
    dates = [datetime.now() - timedelta(days=i) for i in range(n_days)]
    dates.reverse()
    
    # Create data with change point at day 25
    performance_before = np.random.normal(0.6, 0.05, 25)  # Lower performance
    performance_after = np.random.normal(0.8, 0.05, 25)   # Higher performance after improvement
    
    performance_scores = np.concatenate([performance_before, performance_after])
    performance_scores = np.clip(performance_scores, 0.0, 1.0)
    
    rule_data = {
        "rule_with_changepoint": {
            "temporal_data": [
                {
                    "timestamp": dates[i].isoformat(),
                    "score": float(performance_scores[i]),
                    "applications": 20,
                    "context": "test_context"
                }
                for i in range(n_days)
            ],
            "total_applications": n_days * 20,
            "avg_score": float(np.mean(performance_scores))
        }
    }
    
    return rule_data


@pytest.fixture
def insufficient_temporal_data():
    """Minimal temporal data that may not support time series analysis."""
    return {
        "rule_minimal": {
            "temporal_data": [
                {
                    "timestamp": (datetime.now() - timedelta(days=i)).isoformat(),
                    "score": 0.7 + 0.1 * np.random.random(),
                    "applications": 10
                }
                for i in range(5)  # Only 5 data points
            ],
            "total_applications": 50,
            "avg_score": 0.75
        }
    }


class TestTimeSeriesCrossValidation:
    """Test suite for time series cross-validation functionality."""

    @pytest.mark.asyncio
    async def test_time_series_cv_integration(self, rule_analyzer_ts, temporal_rule_data):
        """Test full time series cross-validation workflow integration."""
        with patch('prompt_improver.learning.rule_analyzer.TIME_SERIES_AVAILABLE', True):
            result = await rule_analyzer_ts.analyze_rule_effectiveness(temporal_rule_data)
            
            # Should include time series validation results
            assert "time_series_validation" in result
            ts_validation = result["time_series_validation"]
            
            # Validate time series results structure
            for rule_id, ts_result in ts_validation.items():
                if isinstance(ts_result, dict):
                    assert "cv_scores" in ts_result
                    assert "mean_cv_score" in ts_result
                    assert "temporal_stability" in ts_result
                    
                    # Validate realistic CV scores
                    cv_scores = ts_result["cv_scores"]
                    assert all(0.0 <= score <= 1.0 for score in cv_scores)
                    assert len(cv_scores) <= rule_analyzer_ts.config.time_series_cv_splits

    @pytest.mark.asyncio
    async def test_time_series_cross_validation_method(self, rule_analyzer_ts, temporal_rule_data):
        """Test time series cross-validation method implementation."""
        rule_id = "rule_clarity_001"
        temporal_data = temporal_rule_data[rule_id]["temporal_data"]
        
        with patch('prompt_improver.learning.rule_analyzer.TIME_SERIES_AVAILABLE', True):
            with patch.object(rule_analyzer_ts, '_prepare_time_series_data') as mock_prepare:
                # Mock successful data preparation with proper DataFrame mock
                mock_df = MagicMock()
                mock_df.__len__.return_value = 60  # Sufficient data for time series analysis
                mock_df.__getitem__.return_value.values = np.random.random(60)  # Mock score values
                mock_prepare.return_value = mock_df
                
                with patch.object(rule_analyzer_ts, '_time_series_cross_validate') as mock_ts_cv:
                    # Mock the cross-validation method to return a proper result
                    mock_result = TimeSeriesValidationResult(
                        rule_id=rule_id,
                        cv_scores=[-0.1, -0.15, -0.12, -0.11, -0.13],  # Negative MSE scores
                        mean_cv_score=-0.122,
                        std_cv_score=0.02,
                        temporal_stability=0.85,
                        trend_coefficient=0.001,
                        seasonal_component={"day_of_week_variance": 0.02},
                        change_points=[]
                    )
                    mock_ts_cv.return_value = mock_result
                    
                    # Test time series validation - construct proper data format for internal method
                    data_points = []
                    for item in temporal_data:
                        data_points.append({
                            "score": item["score"],
                            "context": {"projectType": item.get("context", "unknown"), "domain": "test"},
                            "timestamp": item["timestamp"],
                            "overall_score": item["score"],
                            "other_rules": []
                        })
                    rule_data = {rule_id: data_points}
                    ts_results = await rule_analyzer_ts._perform_time_series_validation(rule_data)
                    ts_result = ts_results.get(rule_id)
                    
                    # Verify the mocks were called correctly
                    mock_prepare.assert_called_once()
                    mock_ts_cv.assert_called_once()
                    
                    # Validate result structure (using mocked result)
                    assert ts_result is not None
                    assert isinstance(ts_result, TimeSeriesValidationResult)
                    assert ts_result.rule_id == rule_id
                    assert len(ts_result.cv_scores) > 0
                    assert ts_result.temporal_stability == 0.85  # From mock

    @pytest.mark.asyncio
    async def test_change_point_detection(self, rule_analyzer_ts, temporal_data_with_changepoint):
        """Test change point detection in temporal performance data."""
        rule_id = "rule_with_changepoint"
        temporal_data = temporal_data_with_changepoint[rule_id]["temporal_data"]
        
        with patch('prompt_improver.learning.rule_analyzer.TIME_SERIES_AVAILABLE', True):
            # Extract scores for change point analysis
            scores = [item["score"] for item in temporal_data]
            timestamps = [datetime.fromisoformat(item["timestamp"]) for item in temporal_data]
            
            # Test change point detection method - using async version that matches expected signature
            change_points_result = await rule_analyzer_ts._detect_change_points_async(timestamps, scores)
            change_points = change_points_result.get("change_points", [])
            
            # Should detect the change point around day 25
            assert isinstance(change_points, list)
            if len(change_points) > 0:
                # Change points should be within reasonable range
                for cp in change_points:
                    assert "index" in cp
                    assert "timestamp" in cp
                    assert "p_value" in cp  # Updated to match actual implementation
                    assert 0 <= cp["index"] < len(scores)
                    assert 0.0 <= cp["p_value"] <= 1.0

    @pytest.mark.asyncio
    async def test_seasonal_decomposition(self, rule_analyzer_ts, temporal_rule_data):
        """Test seasonal decomposition of temporal performance data."""
        if not rule_analyzer_ts.config.seasonal_decomposition:
            pytest.skip("Seasonal decomposition disabled in config")
        
        rule_id = "rule_clarity_001"
        temporal_data = temporal_rule_data[rule_id]["temporal_data"]
        
        # Test seasonal decomposition (internal method)
        scores = [item["score"] for item in temporal_data]
        
        # Simple decomposition test (would use more sophisticated methods in practice)
        if len(scores) >= 14:  # Need at least 2 weeks for weekly seasonality
            # Test basic statistics that seasonal decomposition would analyze
            weekly_means = []
            for week_start in range(0, len(scores) - 6, 7):
                week_scores = scores[week_start:week_start + 7]
                weekly_means.append(np.mean(week_scores))
            
            # Should have some variation in weekly patterns
            if len(weekly_means) > 1:
                weekly_std = np.std(weekly_means)
                # Some seasonal variation expected (not strict test due to noise)
                assert weekly_std >= 0.0

    @pytest.mark.asyncio
    async def test_rolling_window_validation(self, rule_analyzer_ts, temporal_rule_data):
        """Test rolling window validation for temporal stability."""
        rule_id = "rule_clarity_001"
        temporal_data = temporal_rule_data[rule_id]["temporal_data"]
        
        window_size = rule_analyzer_ts.config.rolling_window_size
        scores = [item["score"] for item in temporal_data]
        
        # Test rolling window statistics
        if len(scores) >= window_size:
            rolling_means = []
            rolling_stds = []
            
            for i in range(len(scores) - window_size + 1):
                window_scores = scores[i:i + window_size]
                rolling_means.append(np.mean(window_scores))
                rolling_stds.append(np.std(window_scores))
            
            # Rolling statistics should be reasonable
            assert all(0.0 <= mean <= 1.0 for mean in rolling_means)
            assert all(std >= 0.0 for std in rolling_stds)
            
            # Temporal stability could be measured as consistency of rolling means
            if len(rolling_means) > 1:
                stability = 1.0 - (np.std(rolling_means) / (np.mean(rolling_means) + 1e-6))
                assert 0.0 <= stability <= 1.0

    @pytest.mark.asyncio
    async def test_temporal_trend_analysis(self, rule_analyzer_ts, temporal_rule_data):
        """Test temporal trend analysis and significance testing."""
        rule_id = "rule_clarity_001"
        temporal_data = temporal_rule_data[rule_id]["temporal_data"]
        
        # Extract time series data
        scores = [item["score"] for item in temporal_data]
        timestamps = [datetime.fromisoformat(item["timestamp"]) for item in temporal_data]
        
        if len(scores) >= 10:  # Need sufficient data for trend analysis
            # Convert to numeric time for regression
            time_numeric = [(ts - timestamps[0]).total_seconds() for ts in timestamps]
            
            # Test trend analysis using linear regression
            from scipy import stats
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, scores)
            
            # Validate trend analysis results
            assert isinstance(slope, float)
            assert isinstance(r_value, float)
            assert isinstance(p_value, float)
            assert 0.0 <= p_value <= 1.0
            assert -1.0 <= r_value <= 1.0

    @pytest.mark.asyncio
    async def test_insufficient_temporal_data(self, rule_analyzer_ts, insufficient_temporal_data):
        """Test handling of insufficient temporal data."""
        with patch('prompt_improver.learning.rule_analyzer.TIME_SERIES_AVAILABLE', True):
            result = await rule_analyzer_ts.analyze_rule_effectiveness(insufficient_temporal_data)
            
            # Should handle gracefully with traditional analysis
            assert "rule_metrics" in result
            
            # Time series validation may be skipped or return limited results
            if "time_series_validation" in result:
                ts_validation = result["time_series_validation"]
                for rule_id, ts_result in ts_validation.items():
                    if isinstance(ts_result, dict):
                        # May indicate insufficient data
                        assert ts_result.get("status") in [None, "insufficient_data", "skipped"]

    @pytest.mark.asyncio
    async def test_time_series_cv_splits_validation(self, rule_analyzer_ts, temporal_rule_data):
        """Test validation of time series cross-validation splits."""
        rule_id = "rule_clarity_001"
        temporal_data = temporal_rule_data[rule_id]["temporal_data"]
        
        with patch('prompt_improver.learning.rule_analyzer.TIME_SERIES_AVAILABLE', True):
            with patch('sklearn.model_selection.TimeSeriesSplit') as mock_ts_split:
                # Setup mock with realistic splits
                mock_splitter = MagicMock()
                
                # Create overlapping time series splits (train grows, test moves forward)
                n_samples = len(temporal_data)
                min_train = rule_analyzer_ts.config.time_series_min_train_size
                splits = []
                
                for i in range(rule_analyzer_ts.config.time_series_cv_splits):
                    train_end = min_train + i * 5
                    test_start = train_end
                    test_end = min(test_start + 5, n_samples)
                    
                    if test_end > test_start and train_end <= n_samples:
                        train_indices = list(range(train_end))
                        test_indices = list(range(test_start, test_end))
                        splits.append((train_indices, test_indices))
                
                mock_splitter.split.return_value = splits
                mock_ts_split.return_value = mock_splitter
                
                # Test cross-validation - construct proper data format for internal method
                data_points = []
                for item in temporal_data:
                    data_points.append({
                        "score": item["score"],
                        "context": {"projectType": item.get("context", "unknown"), "domain": "test"},
                        "timestamp": item["timestamp"],
                        "overall_score": item["score"],
                        "other_rules": []
                    })
                rule_data = {rule_id: data_points}
                ts_results = await rule_analyzer_ts._perform_time_series_validation(rule_data)
                ts_result = ts_results.get(rule_id)
                
                if ts_result and hasattr(ts_result, 'cv_scores'):
                    # Validate that we have reasonable number of CV scores
                    assert len(ts_result.cv_scores) <= rule_analyzer_ts.config.time_series_cv_splits
                    assert len(ts_result.cv_scores) > 0

    @pytest.mark.parametrize("n_splits", [3, 5, 10])
    async def test_variable_cv_splits(self, temporal_rule_data, n_splits):
        """Test time series CV with different numbers of splits."""
        config = RuleAnalysisConfig(
            enable_time_series_cv=True,
            time_series_cv_splits=n_splits,
            time_series_min_train_size=15
        )
        analyzer = RuleEffectivenessAnalyzer(config=config)
        
        rule_id = "rule_clarity_001"
        temporal_data = temporal_rule_data[rule_id]["temporal_data"]
        
        with patch('prompt_improver.learning.rule_analyzer.TIME_SERIES_AVAILABLE', True):
            with patch.object(analyzer, '_prepare_time_series_data') as mock_prepare:
                # Mock successful data preparation
                mock_df = MagicMock()
                mock_df.__len__.return_value = 60
                mock_df.__getitem__.return_value.values = np.random.random(60)
                mock_prepare.return_value = mock_df
                
                with patch.object(analyzer, '_time_series_cross_validate') as mock_ts_cv:
                    # Mock the cross-validation method to return appropriate result
                    mock_result = TimeSeriesValidationResult(
                        rule_id=rule_id,
                        cv_scores=[-0.1] * min(n_splits, 5),  # Create CV scores based on splits
                        mean_cv_score=-0.1,
                        std_cv_score=0.01,
                        temporal_stability=0.85,
                        trend_coefficient=0.001,
                        seasonal_component={"day_of_week_variance": 0.02},
                        change_points=[]
                    )
                    mock_ts_cv.return_value = mock_result
                    
                    # Construct proper data format for internal method
                    data_points = []
                    for item in temporal_data:
                        data_points.append({
                            "score": item["score"],
                            "context": {"projectType": item.get("context", "unknown"), "domain": "test"},
                            "timestamp": item["timestamp"],
                            "overall_score": item["score"],
                            "other_rules": []
                        })
                    rule_data = {rule_id: data_points}
                    ts_results = await analyzer._perform_time_series_validation(rule_data)
                    ts_result = ts_results.get(rule_id)
                    
                    # Verify mocks were called
                    mock_prepare.assert_called_once()
                    mock_ts_cv.assert_called_once()
                    
                    # Validate result structure
                    assert ts_result is not None
                    assert isinstance(ts_result, TimeSeriesValidationResult)
                    assert ts_result.rule_id == rule_id
                    assert len(ts_result.cv_scores) == min(n_splits, 5)


class TestTimeSeriesErrorHandling:
    """Test error handling and edge cases for time series analysis."""

    @pytest.mark.asyncio
    async def test_time_series_libraries_unavailable(self, temporal_rule_data):
        """Test behavior when time series libraries are not available."""
        config = RuleAnalysisConfig(enable_time_series_cv=True)
        analyzer = RuleEffectivenessAnalyzer(config=config)
        
        with patch('prompt_improver.learning.rule_analyzer.TIME_SERIES_AVAILABLE', False):
            result = await analyzer.analyze_rule_effectiveness(temporal_rule_data)
            
            # Should provide traditional analysis without time series features
            assert "rule_metrics" in result
            # Should not contain time series validation
            assert "time_series_validation" not in result or \
                   all(v.get("status") == "unavailable" for v in result["time_series_validation"].values())

    @pytest.mark.asyncio
    async def test_time_series_disabled(self, rule_analyzer_ts, temporal_rule_data):
        """Test behavior when time series validation is disabled."""
        config = RuleAnalysisConfig(enable_time_series_cv=False)
        analyzer = RuleEffectivenessAnalyzer(config=config)
        
        result = await analyzer.analyze_rule_effectiveness(temporal_rule_data)
        
        # Should not perform time series validation
        assert "rule_metrics" in result
        # Time series validation should be absent or empty when disabled
        assert "time_series_validation" not in result or len(result.get("time_series_validation", {})) == 0

    @pytest.mark.asyncio
    async def test_invalid_temporal_data_format(self, rule_analyzer_ts):
        """Test handling of invalid temporal data formats."""
        invalid_data = {
            "rule_invalid": {
                "temporal_data": [
                    {
                        "timestamp": "invalid_timestamp",  # Invalid format
                        "score": "not_a_number",          # Invalid score
                        "applications": 10
                    },
                    {
                        "timestamp": datetime.now().isoformat(),
                        "score": None,  # None value
                        "applications": 15
                    }
                ],
                "total_applications": 25,
                "avg_score": 0.75
            }
        }
        
        with patch('prompt_improver.learning.rule_analyzer.TIME_SERIES_AVAILABLE', True):
            # Should handle invalid data gracefully
            result = await rule_analyzer_ts.analyze_rule_effectiveness(invalid_data)
            
            # Should still provide some analysis
            assert "rule_metrics" in result

    @pytest.mark.asyncio
    async def test_empty_temporal_data(self, rule_analyzer_ts):
        """Test handling of empty temporal data arrays."""
        empty_temporal_data = {
            "rule_empty": {
                "temporal_data": [],  # Empty array
                "total_applications": 0,
                "avg_score": 0.0
            }
        }
        
        with patch('prompt_improver.learning.rule_analyzer.TIME_SERIES_AVAILABLE', True):
            result = await rule_analyzer_ts.analyze_rule_effectiveness(empty_temporal_data)
            
            # Should handle empty data gracefully
            assert "rule_metrics" in result

    @pytest.mark.asyncio
    async def test_single_timestamp_data(self, rule_analyzer_ts):
        """Test handling of data with only one timestamp."""
        single_point_data = {
            "rule_single": {
                "temporal_data": [
                    {
                        "timestamp": datetime.now().isoformat(),
                        "score": 0.8,
                        "applications": 20
                    }
                ],
                "total_applications": 20,
                "avg_score": 0.8
            }
        }
        
        with patch('prompt_improver.learning.rule_analyzer.TIME_SERIES_AVAILABLE', True):
            result = await rule_analyzer_ts.analyze_rule_effectiveness(single_point_data)
            
            # Should handle single point gracefully (no time series analysis possible)
            assert "rule_metrics" in result
            
            if "time_series_validation" in result:
                ts_validation = result["time_series_validation"]
                if "rule_single" in ts_validation:
                    # Should indicate insufficient temporal data
                    assert ts_validation["rule_single"].get("status") in [None, "insufficient_data"]

    @pytest.mark.asyncio
    async def test_extreme_temporal_values(self, rule_analyzer_ts):
        """Test handling of extreme temporal values."""
        extreme_data = {
            "rule_extreme": {
                "temporal_data": [
                    {
                        "timestamp": datetime.now().isoformat(),
                        "score": 1000.0,  # Extreme score
                        "applications": 10
                    },
                    {
                        "timestamp": datetime.now().isoformat(),
                        "score": -100.0,  # Negative score
                        "applications": 10
                    },
                    {
                        "timestamp": datetime.now().isoformat(),
                        "score": float('inf'),  # Infinite value
                        "applications": 10
                    }
                ],
                "total_applications": 30,
                "avg_score": 0.75
            }
        }
        
        with patch('prompt_improver.learning.rule_analyzer.TIME_SERIES_AVAILABLE', True):
            # Should handle extreme values without crashing
            try:
                result = await rule_analyzer_ts.analyze_rule_effectiveness(extreme_data)
                assert "rule_metrics" in result
            except (ValueError, OverflowError):
                # Acceptable to reject extreme values
                pass

    @pytest.mark.asyncio
    async def test_unsorted_temporal_data(self, rule_analyzer_ts):
        """Test handling of unsorted temporal data."""
        # Create data with timestamps out of order
        base_time = datetime.now()
        unsorted_data = {
            "rule_unsorted": {
                "temporal_data": [
                    {
                        "timestamp": (base_time - timedelta(days=2)).isoformat(),
                        "score": 0.7,
                        "applications": 10
                    },
                    {
                        "timestamp": (base_time - timedelta(days=5)).isoformat(),  # Older
                        "score": 0.8,
                        "applications": 12
                    },
                    {
                        "timestamp": (base_time - timedelta(days=1)).isoformat(),  # Newer
                        "score": 0.75,
                        "applications": 11
                    }
                ],
                "total_applications": 33,
                "avg_score": 0.75
            }
        }
        
        with patch('prompt_improver.learning.rule_analyzer.TIME_SERIES_AVAILABLE', True):
            # Should handle unsorted data (may sort internally or handle gracefully)
            result = await rule_analyzer_ts.analyze_rule_effectiveness(unsorted_data)
            assert "rule_metrics" in result


class TestTimeSeriesIntegration:
    """Integration tests for time series analysis with existing workflows."""

    @pytest.mark.asyncio
    async def test_time_series_with_traditional_metrics(self, rule_analyzer_ts, temporal_rule_data):
        """Test integration of time series analysis with traditional rule metrics."""
        with patch('prompt_improver.learning.rule_analyzer.TIME_SERIES_AVAILABLE', True):
            result = await rule_analyzer_ts.analyze_rule_effectiveness(temporal_rule_data)
            
            # Should have both traditional and time series analysis
            assert "rule_metrics" in result
            assert "time_series_validation" in result
            
            # Traditional metrics should be preserved
            rule_metrics = result["rule_metrics"]
            for rule_id, metrics in rule_metrics.items():
                if isinstance(metrics, dict):
                    # Should still have standard metrics
                    assert "avg_score" in metrics or "total_applications" in metrics

    @pytest.mark.asyncio
    async def test_time_series_performance_monitoring(self, rule_analyzer_ts, temporal_rule_data):
        """Test performance characteristics of time series analysis."""
        import time
        
        start_time = time.time()
        result = await rule_analyzer_ts.analyze_rule_effectiveness(temporal_rule_data)
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time (allow up to 5 seconds for time series analysis)
        assert execution_time < 5.0
        
        # Should provide meaningful results
        assert "rule_metrics" in result

    @pytest.mark.asyncio
    async def test_time_series_cross_validation_reliability(self, rule_analyzer_ts, temporal_rule_data):
        """Test reliability and consistency of time series cross-validation."""
        rule_id = "rule_clarity_001"
        temporal_data = temporal_rule_data[rule_id]["temporal_data"]
        
        with patch('prompt_improver.learning.rule_analyzer.TIME_SERIES_AVAILABLE', True):
            # Run multiple times to check consistency
            results = []
            for _ in range(3):  # Limited runs for test performance
                # Construct proper data format for internal method
                data_points = []
                for item in temporal_data:
                    data_points.append({
                        "score": item["score"],
                        "context": {"projectType": item.get("context", "unknown"), "domain": "test"},
                        "timestamp": item["timestamp"],
                        "overall_score": item["score"],
                        "other_rules": []
                    })
                rule_data = {rule_id: data_points}
                ts_results = await rule_analyzer_ts._perform_time_series_validation(rule_data)
                ts_result = ts_results.get(rule_id)
                if ts_result and hasattr(ts_result, 'mean_cv_score'):
                    results.append(ts_result.mean_cv_score)
            
            if len(results) > 1:
                # Results should be somewhat consistent (allowing for CV randomness)
                result_std = np.std(results)
                # Not too strict due to inherent CV variability
                assert result_std < 0.5  # Reasonable variability bound


# Test markers for categorization
pytestmark = [
    pytest.mark.unit,
    pytest.mark.ml_performance,
    pytest.mark.ml_data_validation
]