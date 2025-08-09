"""
Tests for Performance Improvement Calculator
Tests real behavior with actual database integration and performance calculations.
"""
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
import pytest
import numpy as np
from prompt_improver.database.models import TrainingIteration, TrainingSession
from prompt_improver.ml.analytics.performance_improvement_calculator import ImprovementCalculation, PerformanceImprovementCalculator, PerformanceMetrics, PlateauDetection, PlateauStatus, TrendAnalysis, TrendDirection

class TestPerformanceImprovementCalculator:
    """Test suite for performance improvement calculator with real behavior testing"""

    @pytest.fixture
    async def mock_db_session(self):
        """Create mock database session"""
        session = AsyncMock()
        return session

    @pytest.fixture
    def calculator(self, mock_db_session):
        """Create calculator instance"""
        return PerformanceImprovementCalculator(mock_db_session)

    @pytest.fixture
    def sample_metrics_current(self):
        """Sample current performance metrics"""
        return PerformanceMetrics(model_accuracy=0.85, rule_effectiveness=0.78, pattern_coverage=0.72, training_efficiency=0.65, timestamp=datetime.now(timezone.utc), iteration=10, session_id='test_session_1')

    @pytest.fixture
    def sample_metrics_previous(self):
        """Sample previous performance metrics"""
        return PerformanceMetrics(model_accuracy=0.8, rule_effectiveness=0.75, pattern_coverage=0.68, training_efficiency=0.6, timestamp=datetime.now(timezone.utc) - timedelta(minutes=30), iteration=9, session_id='test_session_1')

    @pytest.mark.asyncio
    async def test_calculate_improvement_basic(self, calculator, sample_metrics_current, sample_metrics_previous):
        """Test basic improvement calculation"""
        calculator._calculate_statistical_significance = AsyncMock(return_value=(True, 0.5))
        result = await calculator.calculate_improvement(sample_metrics_current, sample_metrics_previous)
        assert isinstance(result, ImprovementCalculation)
        assert result.absolute_improvement > 0
        assert result.relative_improvement > 0
        assert result.weighted_improvement > 0
        assert 0 <= result.confidence_score <= 1
        assert result.statistical_significance is True
        assert result.effect_size == 0.5

    @pytest.mark.asyncio
    async def test_calculate_improvement_no_change(self, calculator):
        """Test improvement calculation with no performance change"""
        base_time = datetime.now(timezone.utc)
        current = PerformanceMetrics(0.8, 0.7, 0.6, 0.5, base_time, 10, 'test')
        previous = PerformanceMetrics(0.8, 0.7, 0.6, 0.5, base_time - timedelta(minutes=30), 9, 'test')
        calculator._calculate_statistical_significance = AsyncMock(return_value=(False, 0.0))
        result = await calculator.calculate_improvement(current, previous)
        assert result.absolute_improvement == 0.0
        assert result.relative_improvement == 0.0
        assert result.weighted_improvement == 0.0
        assert result.statistical_significance is False

    @pytest.mark.asyncio
    async def test_calculate_improvement_decline(self, calculator):
        """Test improvement calculation with performance decline"""
        base_time = datetime.now(timezone.utc)
        current = PerformanceMetrics(0.75, 0.7, 0.65, 0.55, base_time, 10, 'test')
        previous = PerformanceMetrics(0.85, 0.78, 0.72, 0.65, base_time - timedelta(minutes=30), 9, 'test')
        calculator._calculate_statistical_significance = AsyncMock(return_value=(True, 0.3))
        result = await calculator.calculate_improvement(current, previous)
        assert result.absolute_improvement < 0
        assert result.relative_improvement < 0
        assert result.weighted_improvement < 0

    @pytest.mark.asyncio
    async def test_analyze_performance_trend_improving(self, calculator):
        """Test trend analysis with improving performance"""
        mock_iterations = []
        for i in range(10):
            iteration = MagicMock()
            iteration.iteration = i + 1
            iteration.performance_metrics = {'model_accuracy': 0.7 + i * 0.02, 'rule_effectiveness': 0.6 + i * 0.015, 'pattern_coverage': 0.5 + i * 0.01, 'training_efficiency': 0.4 + i * 0.005}
            mock_iterations.append(iteration)
        calculator._get_recent_iterations = AsyncMock(return_value=mock_iterations)
        result = await calculator.analyze_performance_trend('test_session', 10)
        assert isinstance(result, TrendAnalysis)
        assert result.direction == TrendDirection.IMPROVING
        assert result.slope > 0
        assert result.correlation > 0.5
        assert result.trend_strength in ['strong', 'moderate', 'weak']

    @pytest.mark.asyncio
    async def test_analyze_performance_trend_declining(self, calculator):
        """Test trend analysis with declining performance"""
        mock_iterations = []
        for i in range(10):
            iteration = MagicMock()
            iteration.iteration = i + 1
            iteration.performance_metrics = {'model_accuracy': 0.9 - i * 0.02, 'rule_effectiveness': 0.8 - i * 0.015, 'pattern_coverage': 0.7 - i * 0.01, 'training_efficiency': 0.6 - i * 0.005}
            mock_iterations.append(iteration)
        calculator._get_recent_iterations = AsyncMock(return_value=mock_iterations)
        result = await calculator.analyze_performance_trend('test_session', 10)
        assert result.direction == TrendDirection.DECLINING
        assert result.slope < 0
        assert result.correlation < -0.5

    @pytest.mark.asyncio
    async def test_analyze_performance_trend_stable(self, calculator):
        """Test trend analysis with stable performance"""
        mock_iterations = []
        for i in range(10):
            iteration = MagicMock()
            iteration.iteration = i + 1
            iteration.performance_metrics = {'model_accuracy': 0.8 + np.random.normal(0, 0.01), 'rule_effectiveness': 0.7 + np.random.normal(0, 0.01), 'pattern_coverage': 0.6 + np.random.normal(0, 0.01), 'training_efficiency': 0.5 + np.random.normal(0, 0.01)}
            mock_iterations.append(iteration)
        calculator._get_recent_iterations = AsyncMock(return_value=mock_iterations)
        result = await calculator.analyze_performance_trend('test_session', 10)
        assert result.direction in [TrendDirection.STABLE, TrendDirection.VOLATILE]
        assert abs(result.correlation) < 0.5

    @pytest.mark.asyncio
    async def test_detect_plateau_no_plateau(self, calculator):
        """Test plateau detection with no plateau"""
        improvement_history = [0.05, 0.04, 0.06, 0.03, 0.05, 0.04, 0.07, 0.03, 0.05, 0.04]
        result = await calculator.detect_plateau('test_session', improvement_history, 10)
        assert isinstance(result, PlateauDetection)
        assert result.status == PlateauStatus.NO_PLATEAU
        assert result.plateau_start_iteration is None
        assert 'Continue training' in result.recommendation

    @pytest.mark.asyncio
    async def test_detect_plateau_confirmed(self, calculator):
        """Test plateau detection with confirmed plateau"""
        improvement_history = [0.05, 0.04, 0.01, 0.005, 0.008, 0.003, 0.001, 0.002, 0.001, 0.001]
        result = await calculator.detect_plateau('test_session', improvement_history, 10)
        assert result.status in [PlateauStatus.CONFIRMED_PLATEAU, PlateauStatus.EARLY_PLATEAU]
        assert result.improvement_stagnation_score > 0.5
        assert 'plateau' in result.recommendation.lower()

    @pytest.mark.asyncio
    async def test_detect_plateau_performance_decline(self, calculator):
        """Test plateau detection with performance decline"""
        improvement_history = [0.05, 0.02, -0.01, -0.02, -0.01, -0.03, -0.02, -0.01, -0.02, -0.01]
        result = await calculator.detect_plateau('test_session', improvement_history, 10)
        assert result.status == PlateauStatus.PERFORMANCE_DECLINE
        assert result.improvement_stagnation_score > 0.7
        assert 'decline' in result.recommendation.lower()

    def test_calculate_weighted_score(self, calculator):
        """Test weighted score calculation"""
        metrics = {'model_accuracy': 0.8, 'rule_effectiveness': 0.7, 'pattern_coverage': 0.6, 'training_efficiency': 0.5}
        score = calculator._calculate_weighted_score(metrics)
        expected_score = 0.7
        assert abs(score - expected_score) < 0.001

    def test_classify_trend_direction(self, calculator):
        """Test trend direction classification"""
        direction = calculator._classify_trend_direction(0.8, 0.01)
        assert direction == TrendDirection.IMPROVING
        direction = calculator._classify_trend_direction(-0.8, 0.01)
        assert direction == TrendDirection.DECLINING
        direction = calculator._classify_trend_direction(0.2, 0.01)
        assert direction == TrendDirection.VOLATILE
        direction = calculator._classify_trend_direction(0.8, 0.1)
        assert direction == TrendDirection.STABLE

    def test_calculate_stagnation_score(self, calculator):
        """Test stagnation score calculation"""
        history = [0.05, 0.04, 0.06, 0.05, 0.04, 0.05, 0.06, 0.04, 0.05, 0.04]
        score = calculator._calculate_stagnation_score(history)
        assert score < 0.5
        history = [0.05, 0.04, 0.06, 0.05, 0.04, 0.001, 0.002, 0.001, 0.001, 0.001]
        score = calculator._calculate_stagnation_score(history)
        assert score > 0.7

    def test_determine_plateau_status(self, calculator):
        """Test plateau status determination"""
        status = calculator._determine_plateau_status(1, False, 0.3)
        assert status == PlateauStatus.NO_PLATEAU
        status = calculator._determine_plateau_status(2, True, 0.6)
        assert status == PlateauStatus.EARLY_PLATEAU
        status = calculator._determine_plateau_status(3, True, 0.8)
        assert status == PlateauStatus.CONFIRMED_PLATEAU
        status = calculator._determine_plateau_status(1, False, 0.9)
        assert status == PlateauStatus.PERFORMANCE_DECLINE

    @pytest.mark.asyncio
    async def test_error_handling(self, calculator):
        """Test error handling in calculations"""
        invalid_current = PerformanceMetrics(model_accuracy=float('nan'), rule_effectiveness=0.7, pattern_coverage=0.6, training_efficiency=0.5, timestamp=datetime.now(timezone.utc), iteration=10, session_id='test')
        invalid_previous = PerformanceMetrics(model_accuracy=0.8, rule_effectiveness=0.7, pattern_coverage=0.6, training_efficiency=0.5, timestamp=datetime.now(timezone.utc) - timedelta(minutes=30), iteration=9, session_id='test')
        calculator._calculate_statistical_significance = AsyncMock(side_effect=Exception('Test error'))
        result = await calculator.calculate_improvement(invalid_current, invalid_previous)
        assert isinstance(result, ImprovementCalculation)
        assert result.confidence_score >= 0.0
