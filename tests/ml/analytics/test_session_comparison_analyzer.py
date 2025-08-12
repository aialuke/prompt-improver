"""
Tests for Session Comparison Analyzer
Tests real behavior with actual database integration and comprehensive comparison functionality.
"""
import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock
import pytest
import numpy as np
from prompt_improver.database.models import TrainingIteration, TrainingSession
from prompt_improver.ml.analytics.session_comparison_analyzer import BenchmarkResult, ComparisonDimension, ComparisonMethod, MultiSessionAnalysis, SessionComparisonAnalyzer, SessionComparisonResult

class TestSessionComparisonAnalyzer:
    """Test suite for session comparison analyzer with real behavior testing"""

    @pytest.fixture
    async def db_session(self, postgres_container):
        """Create real database session with PostgreSQL testcontainer"""
        async with postgres_container.get_session() as session:
            yield session

    @pytest.fixture
    def analyzer(self, db_session):
        """Create analyzer instance with real database session"""
        return SessionComparisonAnalyzer(db_session)

    @pytest.fixture
    def sample_session_a(self):
        """Sample training session A (high performer)"""
        return MagicMock(session_id='session_a', status='completed', started_at=datetime.now(timezone.utc) - timedelta(hours=3), completed_at=datetime.now(timezone.utc) - timedelta(hours=1), current_iteration=15, initial_performance=0.6, current_performance=0.85, best_performance=0.87, total_training_time_seconds=7200, continuous_mode=True, improvement_threshold=0.01, max_iterations=50, timeout_seconds=86400)

    @pytest.fixture
    def sample_session_b(self):
        """Sample training session B (lower performer)"""
        return MagicMock(session_id='session_b', status='completed', started_at=datetime.now(timezone.utc) - timedelta(hours=4), completed_at=datetime.now(timezone.utc) - timedelta(hours=1), current_iteration=20, initial_performance=0.65, current_performance=0.75, best_performance=0.78, total_training_time_seconds=10800, continuous_mode=True, improvement_threshold=0.01, max_iterations=50, timeout_seconds=86400)

    @pytest.fixture
    def sample_iterations_a(self):
        """Sample iterations for session A (high performer)"""
        iterations = []
        base_time = datetime.now(timezone.utc) - timedelta(hours=3)
        for i in range(15):
            iteration = MagicMock()
            iteration.iteration = i + 1
            iteration.started_at = base_time + timedelta(minutes=i * 8)
            iteration.duration_seconds = 400 + i * 5
            iteration.status = 'completed'
            iteration.improvement_score = 0.04 + i * 0.005
            iteration.error_message = None
            iteration.performance_metrics = {'model_accuracy': 0.6 + i * 0.017, 'rule_effectiveness': 0.55 + i * 0.02, 'memory_usage_mb': 800 + i * 20, 'quality_score': 0.85 + i * 0.005}
            iterations.append(iteration)
        return iterations

    @pytest.fixture
    def sample_iterations_b(self):
        """Sample iterations for session B (lower performer)"""
        iterations = []
        base_time = datetime.now(timezone.utc) - timedelta(hours=4)
        for i in range(20):
            iteration = MagicMock()
            iteration.iteration = i + 1
            iteration.started_at = base_time + timedelta(minutes=i * 9)
            iteration.duration_seconds = 600 + i * 15
            iteration.status = 'completed' if i < 16 else 'failed'
            iteration.improvement_score = 0.02 + i * 0.003 if i < 16 else 0.0
            iteration.error_message = 'Test error' if i >= 16 else None
            iteration.performance_metrics = {'model_accuracy': 0.65 + i * 0.008, 'rule_effectiveness': 0.6 + i * 0.01, 'memory_usage_mb': 1200 + i * 40, 'quality_score': 0.75 + i * 0.003} if i < 16 else {}
            iterations.append(iteration)
        return iterations

    @pytest.mark.asyncio
    async def test_compare_sessions_performance(self, analyzer, sample_session_a, sample_session_b, sample_iterations_a, sample_iterations_b):
        """Test basic session comparison for performance dimension"""
        analyzer._get_session_with_iterations = AsyncMock()
        analyzer._get_session_with_iterations.side_effect = [(sample_session_a, sample_iterations_a), (sample_session_b, sample_iterations_b)]
        result = await analyzer.compare_sessions('session_a', 'session_b', ComparisonDimension.PERFORMANCE, ComparisonMethod.T_TEST)
        assert isinstance(result, SessionComparisonResult)
        assert result.session_a_id == 'session_a'
        assert result.session_b_id == 'session_b'
        assert result.comparison_dimension == ComparisonDimension.PERFORMANCE
        assert result.session_a_score > result.session_b_score
        assert len(result.insights) > 0
        assert len(result.recommendations) > 0

    @pytest.mark.asyncio
    async def test_compare_sessions_efficiency(self, analyzer, sample_session_a, sample_session_b, sample_iterations_a, sample_iterations_b):
        """Test session comparison for efficiency dimension"""
        analyzer._get_session_with_iterations = AsyncMock()
        analyzer._get_session_with_iterations.side_effect = [(sample_session_a, sample_iterations_a), (sample_session_b, sample_iterations_b)]
        result = await analyzer.compare_sessions('session_a', 'session_b', ComparisonDimension.EFFICIENCY, ComparisonMethod.T_TEST)
        assert isinstance(result, SessionComparisonResult)
        assert result.comparison_dimension == ComparisonDimension.EFFICIENCY
        assert result.session_a_score > result.session_b_score

    @pytest.mark.asyncio
    async def test_compare_sessions_stability(self, analyzer, sample_session_a, sample_session_b, sample_iterations_a, sample_iterations_b):
        """Test session comparison for stability dimension"""
        analyzer._get_session_with_iterations = AsyncMock()
        analyzer._get_session_with_iterations.side_effect = [(sample_session_a, sample_iterations_a), (sample_session_b, sample_iterations_b)]
        result = await analyzer.compare_sessions('session_a', 'session_b', ComparisonDimension.STABILITY, ComparisonMethod.T_TEST)
        assert isinstance(result, SessionComparisonResult)
        assert result.comparison_dimension == ComparisonDimension.STABILITY
        assert result.session_a_score >= 0
        assert result.session_b_score >= 0
        assert result.winner in ['session_a', 'session_b', 'no_significant_difference']

    @pytest.mark.asyncio
    async def test_statistical_comparison_methods(self, analyzer):
        """Test different statistical comparison methods"""
        metrics_a = [0.8, 0.82, 0.85, 0.83, 0.87]
        metrics_b = [0.7, 0.72, 0.75, 0.73, 0.77]
        result_ttest = await analyzer._perform_statistical_comparison(metrics_a, metrics_b, ComparisonMethod.T_TEST)
        assert 'significant' in result_ttest
        assert 'p_value' in result_ttest
        assert 'effect_size' in result_ttest
        assert 'confidence_interval' in result_ttest
        result_mw = await analyzer._perform_statistical_comparison(metrics_a, metrics_b, ComparisonMethod.MANN_WHITNEY)
        assert 'significant' in result_mw
        assert 'p_value' in result_mw
        assert 'effect_size' in result_mw

    @pytest.mark.asyncio
    async def test_analyze_multiple_sessions(self, analyzer):
        """Test multi-session analysis functionality"""
        sessions_data = []
        for i in range(5):
            session = MagicMock()
            session.session_id = f'session_{i}'
            session.started_at = datetime.now(timezone.utc) - timedelta(hours=i + 1)
            session.completed_at = datetime.now(timezone.utc) - timedelta(minutes=30)
            session.current_performance = 0.7 + i * 0.05
            session.initial_performance = 0.6
            session.best_performance = session.current_performance + 0.02
            session.total_training_time_seconds = 3600 + i * 600
            iterations = []
            for j in range(10):
                iteration = MagicMock()
                iteration.iteration = j + 1
                iteration.duration_seconds = 300 + j * 10
                iteration.status = 'completed'
                iteration.improvement_score = 0.03 + j * 0.002
                iteration.performance_metrics = {'model_accuracy': 0.6 + j * 0.01, 'memory_usage_mb': 1000 + j * 50, 'quality_score': 0.8}
                iterations.append(iteration)
            sessions_data.append((session, iterations))
        analyzer._get_multiple_sessions_by_ids = AsyncMock(return_value=sessions_data)
        result = await analyzer.analyze_multiple_sessions(session_ids=[f'session_{i}' for i in range(5)])
        assert isinstance(result, MultiSessionAnalysis)
        assert len(result.session_ids) == 5
        assert len(result.performance_ranking) == 5
        assert len(result.efficiency_ranking) == 5
        assert len(result.stability_ranking) == 5
        assert isinstance(result.high_performers, list)
        assert isinstance(result.low_performers, list)
        assert isinstance(result.optimization_recommendations, list)
        assert isinstance(result.performance_distribution, dict)
        assert isinstance(result.correlation_matrix, dict)

    @pytest.mark.asyncio
    async def test_benchmark_session(self, analyzer, sample_session_a, sample_iterations_a):
        """Test session benchmarking functionality"""
        historical_sessions = []
        for i in range(10):
            session = MagicMock()
            session.session_id = f'historical_{i}'
            session.started_at = datetime.now(timezone.utc) - timedelta(days=i + 1)
            session.completed_at = session.started_at + timedelta(hours=2)
            session.current_performance = 0.6 + i * 0.02
            session.initial_performance = 0.5
            session.best_performance = session.current_performance + 0.01
            session.total_training_time_seconds = 7200
            iterations = []
            for j in range(8):
                iteration = MagicMock()
                iteration.iteration = j + 1
                iteration.duration_seconds = 400
                iteration.status = 'completed'
                iteration.improvement_score = 0.03
                iteration.performance_metrics = {'model_accuracy': 0.5 + j * 0.015, 'memory_usage_mb': 1000, 'quality_score': 0.8}
                iterations.append(iteration)
            historical_sessions.append((session, iterations))
        analyzer._get_session_with_iterations = AsyncMock(return_value=(sample_session_a, sample_iterations_a))
        analyzer._get_sessions_by_date_range = AsyncMock(return_value=historical_sessions)
        result = await analyzer.benchmark_session('session_a')
        assert isinstance(result, BenchmarkResult)
        assert result.session_id == 'session_a'
        assert 0 <= result.performance_percentile <= 100
        assert 0 <= result.efficiency_percentile <= 100
        assert 0 <= result.speed_percentile <= 100
        assert result.performance_tier in ['excellent', 'good', 'average', 'below_average', 'poor']
        assert 0 <= result.improvement_potential <= 1
        assert isinstance(result.strengths, list)
        assert isinstance(result.weaknesses, list)
        assert isinstance(result.recommendations, list)

    @pytest.mark.asyncio
    async def test_extract_session_features(self, analyzer, sample_session_a, sample_iterations_a):
        """Test session feature extraction"""
        features = await analyzer._extract_session_features(sample_session_a, sample_iterations_a)
        assert isinstance(features, dict)
        assert 'total_iterations' in features
        assert 'duration_hours' in features
        assert 'initial_performance' in features
        assert 'final_performance' in features
        assert 'total_improvement' in features
        assert 'improvement_rate' in features
        assert 'avg_iteration_duration' in features
        assert 'success_rate' in features
        assert 'avg_efficiency' in features
        assert 'performance_stability' in features
        assert 'avg_memory_usage' in features
        assert features['total_iterations'] == 15
        assert features['success_rate'] == 1.0
        assert features['total_improvement'] == 0.25

    @pytest.mark.asyncio
    async def test_extract_comparison_metrics(self, analyzer, sample_session_a, sample_iterations_a):
        """Test comparison metrics extraction for different dimensions"""
        perf_metrics = await analyzer._extract_comparison_metrics(sample_session_a, sample_iterations_a, ComparisonDimension.PERFORMANCE)
        assert len(perf_metrics) > 0
        assert all(isinstance(m, (int, float)) for m in perf_metrics)
        eff_metrics = await analyzer._extract_comparison_metrics(sample_session_a, sample_iterations_a, ComparisonDimension.EFFICIENCY)
        assert len(eff_metrics) > 0
        stab_metrics = await analyzer._extract_comparison_metrics(sample_session_a, sample_iterations_a, ComparisonDimension.STABILITY)
        assert len(stab_metrics) >= 0
        speed_metrics = await analyzer._extract_comparison_metrics(sample_session_a, sample_iterations_a, ComparisonDimension.SPEED)
        assert len(speed_metrics) > 0

    @pytest.mark.asyncio
    async def test_pattern_identification(self, analyzer):
        """Test pattern identification in session features"""
        session_features = {'high_perf_1': {'final_performance': 0.9, 'success_rate': 0.95, 'avg_efficiency': 0.8}, 'high_perf_2': {'final_performance': 0.88, 'success_rate': 0.92, 'avg_efficiency': 0.75}, 'low_perf_1': {'final_performance': 0.4, 'success_rate': 0.6, 'avg_efficiency': 0.3}, 'low_perf_2': {'final_performance': 0.35, 'success_rate': 0.55, 'avg_efficiency': 0.25}, 'average_1': {'final_performance': 0.7, 'success_rate': 0.8, 'avg_efficiency': 0.5}}
        patterns = await analyzer._identify_patterns(session_features)
        assert isinstance(patterns, dict)
        assert 'high_performers' in patterns
        assert 'low_performers' in patterns
        assert 'outliers' in patterns
        assert 'success_patterns' in patterns
        assert 'failure_patterns' in patterns
        assert len(patterns['high_performers']) >= 1
        assert len(patterns['low_performers']) >= 1

    @pytest.mark.asyncio
    async def test_clustering_analysis(self, analyzer):
        """Test clustering analysis functionality"""
        session_features = {}
        for i in range(6):
            session_features[f'session_{i}'] = {'final_performance': 0.5 + i * 0.1, 'total_improvement': 0.1 + i * 0.02, 'improvement_rate': 0.05 + i * 0.01, 'success_rate': 0.7 + i * 0.05, 'avg_efficiency': 0.3 + i * 0.1}
        clusters = await analyzer._perform_clustering_analysis(session_features)
        assert isinstance(clusters, dict)
        assert len(clusters) <= 3
        total_sessions = sum(len(sessions) for sessions in clusters.values())
        assert total_sessions == 6

    @pytest.mark.asyncio
    async def test_correlation_matrix_calculation(self, analyzer):
        """Test correlation matrix calculation"""
        session_features = {}
        for i in range(10):
            session_features[f'session_{i}'] = {'final_performance': 0.5 + i * 0.05, 'total_improvement': 0.1 + i * 0.01, 'improvement_rate': 0.05 + i * 0.005, 'success_rate': 0.7 + i * 0.03, 'avg_efficiency': 0.3 + i * 0.07, 'performance_stability': 0.8 + i * 0.02}
        correlation_matrix = await analyzer._calculate_correlation_matrix(session_features)
        assert isinstance(correlation_matrix, dict)
        expected_features = ['final_performance', 'total_improvement', 'improvement_rate', 'success_rate', 'avg_efficiency', 'performance_stability']
        for feature in expected_features:
            assert feature in correlation_matrix
            assert isinstance(correlation_matrix[feature], dict)
            assert abs(correlation_matrix[feature][feature] - 1.0) < 0.001

    @pytest.mark.asyncio
    async def test_performance_tier_classification(self, analyzer):
        """Test performance tier classification"""
        target_features = {'final_performance': 0.85, 'total_improvement': 0.2, 'avg_efficiency': 0.7, 'success_rate': 0.9}
        historical_features = {}
        for i in range(20):
            historical_features[f'hist_{i}'] = {'final_performance': 0.5 + i * 0.02, 'total_improvement': 0.1 + i * 0.01, 'avg_efficiency': 0.3 + i * 0.03, 'success_rate': 0.6 + i * 0.02}
        tier = await analyzer._classify_performance_tier(target_features, historical_features)
        assert tier in ['excellent', 'good', 'average', 'below_average', 'poor']
        assert tier in ['excellent', 'good']

    @pytest.mark.asyncio
    async def test_error_handling(self, analyzer):
        """Test error handling in comparison operations"""
        analyzer._get_session_with_iterations = AsyncMock(return_value=None)
        with pytest.raises(ValueError, match='One or both sessions not found'):
            await analyzer.compare_sessions('nonexistent_a', 'nonexistent_b')
        result = await analyzer._perform_statistical_comparison([], [], ComparisonMethod.T_TEST)
        assert not result['significant']
        assert result['p_value'] == 1.0

    def test_significance_thresholds(self, analyzer):
        """Test statistical significance thresholds"""
        assert analyzer.significance_level == 0.05
        assert 'small' in analyzer.effect_size_thresholds
        assert 'medium' in analyzer.effect_size_thresholds
        assert 'large' in analyzer.effect_size_thresholds
        assert analyzer.effect_size_thresholds['small'] == 0.2
        assert analyzer.effect_size_thresholds['medium'] == 0.5
        assert analyzer.effect_size_thresholds['large'] == 0.8

    def test_performance_tier_thresholds(self, analyzer):
        """Test performance tier classification thresholds"""
        tiers = analyzer.performance_tiers
        assert 'excellent' in tiers
        assert 'good' in tiers
        assert 'average' in tiers
        assert 'below_average' in tiers
        assert 'poor' in tiers
        assert tiers['excellent'] > tiers['good']
        assert tiers['good'] > tiers['average']
        assert tiers['average'] > tiers['below_average']
        assert tiers['below_average'] > tiers['poor']
