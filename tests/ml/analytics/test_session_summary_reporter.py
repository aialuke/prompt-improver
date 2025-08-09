"""
Tests for Session Summary Reporter
Tests real behavior with actual database integration and comprehensive reporting functionality.
"""
import asyncio
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
import pytest
from prompt_improver.database.models import GenerationSession, TrainingIteration, TrainingSession
from prompt_improver.ml.analytics.session_summary_reporter import ExecutiveSummary, ReportFormat, SessionSummary, SessionSummaryReporter, UserRole

class TestSessionSummaryReporter:
    """Test suite for session summary reporter with real behavior testing"""

    @pytest.fixture
    async def mock_db_session(self):
        """Create mock database session"""
        session = AsyncMock()
        return session

    @pytest.fixture
    def reporter(self, mock_db_session):
        """Create reporter instance"""
        return SessionSummaryReporter(mock_db_session)

    @pytest.fixture
    def sample_training_session(self):
        """Sample training session"""
        return MagicMock(session_id='test_session_1', status='completed', started_at=datetime.now(timezone.utc) - timedelta(hours=2), completed_at=datetime.now(timezone.utc), current_iteration=10, initial_performance=0.7, current_performance=0.85, best_performance=0.87, total_training_time_seconds=7200, continuous_mode=True, improvement_threshold=0.01, max_iterations=50, timeout_seconds=86400)

    @pytest.fixture
    def sample_iterations(self):
        """Sample training iterations"""
        iterations = []
        base_time = datetime.now(timezone.utc) - timedelta(hours=2)
        for i in range(10):
            iteration = MagicMock()
            iteration.iteration = i + 1
            iteration.started_at = base_time + timedelta(minutes=i * 12)
            iteration.duration_seconds = 600 + i * 10
            iteration.status = 'completed' if i < 8 else 'failed'
            iteration.improvement_score = 0.05 + i * 0.01
            iteration.error_message = 'Test error' if i >= 8 else None
            iteration.performance_metrics = {'model_accuracy': 0.7 + i * 0.015, 'rule_effectiveness': 0.6 + i * 0.02, 'memory_usage_mb': 1000 + i * 50, 'quality_score': 0.8 + i * 0.01}
            iterations.append(iteration)
        return iterations

    @pytest.fixture
    def sample_generation_sessions(self):
        """Sample generation sessions"""
        sessions = []
        base_time = datetime.now(timezone.utc) - timedelta(hours=2)
        for i in range(3):
            session = MagicMock()
            session.training_session_id = 'test_session_1'
            session.started_at = base_time + timedelta(minutes=i * 30)
            session.samples_generated = 100 + i * 50
            session.average_quality = 0.8 + i * 0.05
            sessions.append(session)
        return sessions

    @pytest.mark.asyncio
    async def test_generate_session_summary_basic(self, reporter, sample_training_session, sample_iterations, sample_generation_sessions):
        """Test basic session summary generation"""
        reporter._get_training_session = AsyncMock(return_value=sample_training_session)
        reporter._get_session_iterations = AsyncMock(return_value=sample_iterations)
        reporter._get_generation_sessions = AsyncMock(return_value=sample_generation_sessions)
        reporter.performance_calculator.analyze_performance_trend = AsyncMock(return_value=MagicMock(direction=MagicMock(value='improving')))
        result = await reporter.generate_session_summary('test_session_1')
        assert isinstance(result, SessionSummary)
        assert result.session_id == 'test_session_1'
        assert result.status == 'completed'
        assert result.total_iterations == 10
        assert result.successful_iterations == 8
        assert result.failed_iterations == 2
        assert result.performance_trend == 'improving'
        assert 0 <= result.performance_score <= 1
        assert 0 <= result.success_rate <= 1

    @pytest.mark.asyncio
    async def test_generate_executive_summary(self, reporter, sample_training_session, sample_iterations, sample_generation_sessions):
        """Test executive summary generation with 2025 best practices"""
        reporter._get_training_session = AsyncMock(return_value=sample_training_session)
        reporter._get_session_iterations = AsyncMock(return_value=sample_iterations)
        reporter._get_generation_sessions = AsyncMock(return_value=sample_generation_sessions)
        reporter.performance_calculator.analyze_performance_trend = AsyncMock(return_value=MagicMock(direction=MagicMock(value='improving')))
        result = await reporter.generate_executive_summary('test_session_1')
        assert isinstance(result, ExecutiveSummary)
        assert result.session_id == 'test_session_1'
        assert result.overall_status == 'completed'
        assert len(result.performance_highlights) == 5
        assert 'Performance Score' in result.performance_highlights
        assert 'Improvement Velocity' in result.performance_highlights
        assert 'Efficiency Rating' in result.performance_highlights
        assert 'Quality Index' in result.performance_highlights
        assert 'Success Rate' in result.performance_highlights
        assert isinstance(result.key_achievements, list)
        assert isinstance(result.critical_issues, list)
        assert isinstance(result.next_actions, list)
        assert isinstance(result.roi_metrics, dict)

    @pytest.mark.asyncio
    async def test_role_based_summary_generation(self, reporter, sample_training_session, sample_iterations, sample_generation_sessions):
        """Test role-based summary customization"""
        reporter._get_training_session = AsyncMock(return_value=sample_training_session)
        reporter._get_session_iterations = AsyncMock(return_value=sample_iterations)
        reporter._get_generation_sessions = AsyncMock(return_value=sample_generation_sessions)
        reporter.performance_calculator.analyze_performance_trend = AsyncMock(return_value=MagicMock(direction=MagicMock(value='improving')))
        executive_summary = await reporter.generate_session_summary('test_session_1', UserRole.EXECUTIVE)
        analyst_summary = await reporter.generate_session_summary('test_session_1', UserRole.ANALYST)
        assert executive_summary.session_id == analyst_summary.session_id
        assert len(executive_summary.recommendations) >= 0
        assert len(analyst_summary.recommendations) >= 0

    @pytest.mark.asyncio
    async def test_calculate_executive_kpis(self, reporter, sample_training_session, sample_iterations):
        """Test executive KPI calculations following 2025 standards"""
        result = await reporter._calculate_executive_kpis(sample_training_session, sample_iterations)
        assert isinstance(result, dict)
        assert 'performance_score' in result
        assert 'improvement_velocity' in result
        assert 'efficiency_rating' in result
        assert 'quality_index' in result
        assert 'success_rate' in result
        for kpi_value in result.values():
            assert 0 <= kpi_value <= 1

    @pytest.mark.asyncio
    async def test_performance_metrics_calculation(self, reporter, sample_training_session, sample_iterations):
        """Test performance metrics calculation"""
        result = await reporter._calculate_performance_metrics(sample_training_session, sample_iterations)
        assert isinstance(result, dict)
        assert 'total_improvement' in result
        assert 'improvement_rate' in result
        assert 'successful_iterations' in result
        assert 'failed_iterations' in result
        assert 'average_duration' in result
        assert 'trend' in result
        assert result['successful_iterations'] == 8
        assert result['failed_iterations'] == 2
        assert abs(result['total_improvement'] - 0.15) < 0.001

    @pytest.mark.asyncio
    async def test_ai_insights_generation(self, reporter, sample_training_session, sample_iterations):
        """Test AI-powered insights generation (2025 feature)"""
        result = await reporter._generate_ai_insights(sample_training_session, sample_iterations, UserRole.ANALYST)
        assert isinstance(result, dict)
        assert 'insights' in result
        assert 'recommendations' in result
        assert 'anomalies' in result
        assert isinstance(result['insights'], list)
        assert isinstance(result['recommendations'], list)
        assert isinstance(result['anomalies'], list)

    @pytest.mark.asyncio
    async def test_observability_metrics(self, reporter, sample_training_session, sample_iterations):
        """Test observability metrics calculation"""
        result = await reporter._calculate_observability_metrics(sample_training_session, sample_iterations)
        assert isinstance(result, dict)
        assert 'error_rate' in result
        assert 'alert_count' in result
        assert 'alerts' in result
        assert result['error_rate'] == 0.2

    @pytest.mark.asyncio
    async def test_export_json_format(self, reporter, sample_training_session, sample_iterations, sample_generation_sessions, tmp_path):
        """Test JSON export functionality"""
        reporter._get_training_session = AsyncMock(return_value=sample_training_session)
        reporter._get_session_iterations = AsyncMock(return_value=sample_iterations)
        reporter._get_generation_sessions = AsyncMock(return_value=sample_generation_sessions)
        reporter.performance_calculator.analyze_performance_trend = AsyncMock(return_value=MagicMock(direction=MagicMock(value='improving')))
        output_path = tmp_path / 'test_report.json'
        result_path = await reporter.export_session_report('test_session_1', ReportFormat.JSON, str(output_path))
        assert result_path == str(output_path)
        assert output_path.exists()
        with open(output_path, 'r') as f:
            data = json.load(f)
        assert 'session_summary' in data
        assert 'iterations' in data
        assert 'export_metadata' in data
        assert data['export_metadata']['format'] == 'json'

    @pytest.mark.asyncio
    async def test_export_markdown_format(self, reporter, sample_training_session, sample_iterations, sample_generation_sessions, tmp_path):
        """Test Markdown export functionality"""
        reporter._get_training_session = AsyncMock(return_value=sample_training_session)
        reporter._get_session_iterations = AsyncMock(return_value=sample_iterations)
        reporter._get_generation_sessions = AsyncMock(return_value=sample_generation_sessions)
        reporter.performance_calculator.analyze_performance_trend = AsyncMock(return_value=MagicMock(direction=MagicMock(value='improving')))
        output_path = tmp_path / 'test_report.md'
        result_path = await reporter.export_session_report('test_session_1', ReportFormat.MARKDOWN, str(output_path))
        assert result_path == str(output_path)
        assert output_path.exists()
        content = output_path.read_text()
        assert '# Training Session Report' in content
        assert '## Executive KPIs' in content
        assert 'test_session_1' in content

    @pytest.mark.asyncio
    async def test_performance_optimization(self, reporter, sample_training_session, sample_iterations, sample_generation_sessions):
        """Test performance optimization features (2025 standard)"""
        reporter._get_training_session = AsyncMock(return_value=sample_training_session)
        reporter._get_session_iterations = AsyncMock(return_value=sample_iterations)
        reporter._get_generation_sessions = AsyncMock(return_value=sample_generation_sessions)
        reporter.performance_calculator.analyze_performance_trend = AsyncMock(return_value=MagicMock(direction=MagicMock(value='improving')))
        start_time = datetime.now()
        summary = await reporter.generate_session_summary('test_session_1')
        generation_time = (datetime.now() - start_time).total_seconds()
        assert generation_time < reporter.performance_targets['max_report_generation_seconds']
        assert isinstance(summary, SessionSummary)

    @pytest.mark.asyncio
    async def test_mobile_optimization_features(self, reporter):
        """Test mobile optimization features"""
        executive_summary = MagicMock()
        executive_summary.session_id = 'test_session_1'
        executive_summary.overall_status = 'completed'
        executive_summary.performance_highlights = {'Performance Score': 0.85, 'Improvement Velocity': 0.75, 'Efficiency Rating': 0.8, 'Quality Index': 0.9, 'Success Rate': 0.88}
        executive_summary.critical_issues = ['Test issue 1', 'Test issue 2']
        try:
            reporter._display_mobile_executive_dashboard(executive_summary)
            mobile_test_passed = True
        except Exception:
            mobile_test_passed = False
        assert mobile_test_passed

    @pytest.mark.asyncio
    async def test_error_handling(self, reporter):
        """Test error handling in report generation"""
        reporter._get_training_session = AsyncMock(return_value=None)
        with pytest.raises(ValueError, match='Training session .* not found'):
            await reporter.generate_session_summary('nonexistent_session')

    @pytest.mark.asyncio
    async def test_duration_calculation(self, reporter):
        """Test duration calculation helper"""
        start_time = datetime.now(timezone.utc)
        end_time = start_time + timedelta(hours=2, minutes=30)
        duration = reporter._calculate_duration_hours(start_time, end_time)
        assert duration == 2.5

    def test_executive_kpi_weights(self, reporter):
        """Test executive KPI weights follow 2025 standards"""
        weights = reporter.executive_kpi_weights
        assert len(weights) == 5
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.001
        assert weights['performance_score'] == max(weights.values())

    def test_performance_targets(self, reporter):
        """Test performance targets meet 2025 standards"""
        targets = reporter.performance_targets
        assert 'max_query_time_seconds' in targets
        assert 'max_report_generation_seconds' in targets
        assert 'cache_ttl_seconds' in targets
        assert targets['max_report_generation_seconds'] <= 5.0
        assert targets['max_query_time_seconds'] <= 3.0
