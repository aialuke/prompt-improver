"""
Phase 3 tests for PromptImprovementService focusing on ML optimization methods.
Tests the 3 TODO methods that were completed for Phase 3 continuous learning.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timedelta

from prompt_improver.services.prompt_improvement import PromptImprovementService
from prompt_improver.database.models import (
    UserFeedback, RulePerformance, RuleMetadata, 
    MLModelPerformance, ABExperiment, DiscoveredPattern
)


class TestPhase3MLOptimization:
    """Test Phase 3 ML optimization methods in PromptImprovementService."""
    
    @pytest.fixture
    def prompt_service(self):
        """Create PromptImprovementService instance."""
        return PromptImprovementService()
    


class TestTriggerOptimization:
    """Test trigger_optimization method (lines 538-634)."""
    
    @pytest.mark.asyncio
    async def test_trigger_optimization_success(self, prompt_service, mock_db_session, sample_user_feedback, sample_rule_performance):
        """Test successful ML optimization trigger from feedback."""
        # Mock feedback query result
        feedback_result = MagicMock()
        feedback_result.scalar_one_or_none.return_value = sample_user_feedback[0]
        
        # Mock performance query result
        perf_result = MagicMock()
        perf_result.fetchall.return_value = sample_rule_performance
        
        mock_db_session.execute.side_effect = [feedback_result, perf_result]
        
        # Mock ML service
        mock_ml_service = AsyncMock()
        mock_ml_service.optimize_rules.return_value = {
            "status": "success",
            "best_score": 0.85,
            "model_id": "optimized_model_123"
        }
        
        with patch('prompt_improver.services.prompt_improvement.get_ml_service', return_value=mock_ml_service):
            result = await prompt_service.trigger_optimization(123, mock_db_session)
        
        assert result["status"] == "success"
        assert result["performance_score"] == 0.85
        assert result["training_samples"] == 20
        assert result["model_id"] == "optimized_model_123"
        
        # Verify ML service was called with correct parameters
        mock_ml_service.optimize_rules.assert_called_once()
        call_args = mock_ml_service.optimize_rules.call_args[0]
        training_data = call_args[0]
        assert "features" in training_data
        assert "effectiveness_scores" in training_data
        assert len(training_data["features"]) == 20

    @pytest.mark.asyncio
    async def test_trigger_optimization_feedback_not_found(self, prompt_service, mock_db_session):
        """Test trigger optimization with non-existent feedback."""
        # Mock empty feedback result
        feedback_result = MagicMock()
        feedback_result.scalar_one_or_none.return_value = None
        mock_db_session.execute.return_value = feedback_result
        
        result = await prompt_service.trigger_optimization(999, mock_db_session)
        
        assert result["status"] == "error"
        assert "Feedback 999 not found" in result["message"]

    @pytest.mark.asyncio
    async def test_trigger_optimization_insufficient_data(self, prompt_service, mock_db_session, sample_user_feedback):
        """Test trigger optimization with insufficient performance data."""
        # Mock feedback query result
        feedback_result = MagicMock()
        feedback_result.scalar_one_or_none.return_value = sample_user_feedback[0]
        
        # Mock insufficient performance data (less than 10 samples)
        perf_result = MagicMock()
        perf_result.fetchall.return_value = []
        
        mock_db_session.execute.side_effect = [feedback_result, perf_result]
        
        result = await prompt_service.trigger_optimization(123, mock_db_session)
        
        assert result["status"] == "error"
        assert "No performance data available" in result["message"]

    @pytest.mark.asyncio
    async def test_trigger_optimization_ml_failure(self, prompt_service, mock_db_session, sample_user_feedback, sample_rule_performance):
        """Test trigger optimization with ML service failure."""
        # Mock successful queries
        feedback_result = MagicMock()
        feedback_result.scalar_one_or_none.return_value = sample_user_feedback[0]
        
        perf_result = MagicMock()
        perf_result.fetchall.return_value = sample_rule_performance
        
        mock_db_session.execute.side_effect = [feedback_result, perf_result]
        
        # Mock ML service failure
        mock_ml_service = AsyncMock()
        mock_ml_service.optimize_rules.return_value = {
            "status": "error",
            "error": "Model training failed"
        }
        
        with patch('prompt_improver.services.prompt_improvement.get_ml_service', return_value=mock_ml_service):
            result = await prompt_service.trigger_optimization(123, mock_db_session)
        
        assert result["status"] == "error"
        assert "Optimization failed: Model training failed" in result["message"]


class TestRunMLOptimization:
    """Test run_ml_optimization method (lines 636-738)."""
    
    @pytest.mark.asyncio
    async def test_run_ml_optimization_success(self, prompt_service, mock_db_session, sample_rule_performance):
        """Test successful ML optimization run."""
        # Mock performance query result
        perf_result = MagicMock()
        perf_result.fetchall.return_value = sample_rule_performance
        mock_db_session.execute.return_value = perf_result
        
        # Mock ML service success
        mock_ml_service = AsyncMock()
        mock_ml_service.optimize_rules.return_value = {
            "status": "success",
            "best_score": 0.88,
            "model_id": "ml_model_456"
        }
        
        with patch('prompt_improver.services.prompt_improvement.get_ml_service', return_value=mock_ml_service), \
             patch('prompt_improver.services.prompt_improvement.datetime') as mock_datetime:
            
            mock_datetime.utcnow.return_value = datetime(2025, 1, 1)
            
            result = await prompt_service.run_ml_optimization(["clarity_rule"], mock_db_session)
        
        assert result["status"] == "success"
        assert result["best_score"] == 0.88
        assert result["model_id"] == "ml_model_456"
        
        # Verify enhanced feature engineering (10 features)
        call_args = mock_ml_service.optimize_rules.call_args[0]
        training_data = call_args[0]
        features = training_data["features"]
        assert len(features[0]) == 10  # 10 features as specified

    @pytest.mark.asyncio
    async def test_run_ml_optimization_with_ensemble(self, prompt_service, mock_db_session):
        """Test ML optimization with ensemble when enough data available."""
        # Create 50+ samples for ensemble
        large_performance_data = [
            MagicMock(
                rule_id="test_rule",
                improvement_score=0.8,
                execution_time_ms=150,
                user_satisfaction_score=0.9,
                parameters={"weight": 1.0},
                weight=1.0,
                priority=5
            )
        ] * 60
        
        perf_result = MagicMock()
        perf_result.fetchall.return_value = large_performance_data
        mock_db_session.execute.return_value = perf_result
        
        # Mock ML service
        mock_ml_service = AsyncMock()
        mock_ml_service.optimize_rules.return_value = {
            "status": "success",
            "best_score": 0.88
        }
        mock_ml_service.optimize_ensemble_rules.return_value = {
            "status": "success", 
            "ensemble_score": 0.92
        }
        
        with patch('prompt_improver.services.prompt_improvement.get_ml_service', return_value=mock_ml_service), \
             patch('prompt_improver.services.prompt_improvement.datetime') as mock_datetime:
            
            mock_datetime.utcnow.return_value = datetime(2025, 1, 1)
            
            result = await prompt_service.run_ml_optimization(None, mock_db_session)
        
        assert result["status"] == "success"
        assert "ensemble" in result
        assert result["ensemble"]["ensemble_score"] == 0.92
        
        # Verify both optimization methods were called
        mock_ml_service.optimize_rules.assert_called_once()
        mock_ml_service.optimize_ensemble_rules.assert_called_once()

    @pytest.mark.asyncio
    async def test_run_ml_optimization_insufficient_data(self, prompt_service, mock_db_session):
        """Test ML optimization with insufficient data (< 20 samples)."""
        # Mock insufficient data
        insufficient_data = [MagicMock()] * 15  # Only 15 samples
        
        perf_result = MagicMock()
        perf_result.fetchall.return_value = insufficient_data
        mock_db_session.execute.return_value = perf_result
        
        with patch('prompt_improver.services.prompt_improvement.datetime') as mock_datetime:
            mock_datetime.utcnow.return_value = datetime(2025, 1, 1)
            
            result = await prompt_service.run_ml_optimization(["clarity_rule"], mock_db_session)
        
        assert result["status"] == "insufficient_data"
        assert "Need at least 20 samples" in result["message"]
        assert result["samples_found"] == 15

    @pytest.mark.asyncio
    async def test_run_ml_optimization_feature_engineering(self, prompt_service, mock_db_session):
        """Test enhanced feature engineering with 10 features."""
        # Create sample data with all required fields
        sample_data = [
            MagicMock(
                improvement_score=0.8,
                execution_time_ms=150,
                weight=1.0,
                priority=5,
                user_satisfaction_score=0.9,
                parameters={
                    "confidence_threshold": 0.7,
                    "enabled": True,
                    "min_length": 10,
                    "max_length": 500
                }
            )
        ] * 25
        
        perf_result = MagicMock()
        perf_result.fetchall.return_value = sample_data
        mock_db_session.execute.return_value = perf_result
        
        mock_ml_service = AsyncMock()
        mock_ml_service.optimize_rules.return_value = {"status": "success"}
        
        with patch('prompt_improver.services.prompt_improvement.get_ml_service', return_value=mock_ml_service), \
             patch('prompt_improver.services.prompt_improvement.datetime') as mock_datetime:
            
            mock_datetime.utcnow.return_value = datetime(2025, 1, 1)
            
            await prompt_service.run_ml_optimization(None, mock_db_session)
        
        # Verify feature engineering
        call_args = mock_ml_service.optimize_rules.call_args[0]
        training_data = call_args[0]
        features = training_data["features"][0]
        
        # Verify 10 features as specified in implementation
        assert len(features) == 10
        assert features[0] == 0.8  # improvement_score
        assert features[1] == 150  # execution_time_ms
        assert features[2] == 1.0  # weight
        assert features[3] == 5    # priority
        assert features[4] == 0.9  # user_satisfaction_score
        assert features[5] == 4    # len(parameters)
        assert features[6] == 0.7  # confidence_threshold
        assert features[7] == 1.0  # enabled status
        assert features[8] == 0.1  # normalized min_length
        assert features[9] == 0.5  # normalized max_length


class TestDiscoverPatterns:
    """Test discover_patterns method (lines 740-781)."""
    
    @pytest.mark.asyncio
    async def test_discover_patterns_success(self, prompt_service, mock_db_session):
        """Test successful pattern discovery."""
        # Mock ML service
        mock_ml_service = AsyncMock()
        mock_ml_service.discover_patterns.return_value = {
            "status": "success",
            "patterns_discovered": 3,
            "patterns": [
                {"parameters": {"weight": 1.0}, "avg_effectiveness": 0.85},
                {"parameters": {"weight": 0.8}, "avg_effectiveness": 0.80},
                {"parameters": {"weight": 0.9}, "avg_effectiveness": 0.82}
            ],
            "total_analyzed": 100,
            "processing_time_ms": 1500
        }
        
        with patch('prompt_improver.services.prompt_improvement.get_ml_service', return_value=mock_ml_service):
            result = await prompt_service.discover_patterns(0.7, 5, mock_db_session)
        
        assert result["status"] == "success"
        assert result["patterns_discovered"] == 3
        assert len(result["patterns"]) == 3
        assert result["total_analyzed"] == 100
        assert result["processing_time_ms"] == 1500
        
        # Verify ML service was called with correct parameters
        mock_ml_service.discover_patterns.assert_called_once_with(
            db_session=mock_db_session,
            min_effectiveness=0.7,
            min_support=5
        )

    @pytest.mark.asyncio
    async def test_discover_patterns_insufficient_data(self, prompt_service, mock_db_session):
        """Test pattern discovery with insufficient data."""
        mock_ml_service = AsyncMock()
        mock_ml_service.discover_patterns.return_value = {
            "status": "insufficient_data",
            "message": "Only 3 high-performing samples found (minimum: 5)"
        }
        
        with patch('prompt_improver.services.prompt_improvement.get_ml_service', return_value=mock_ml_service):
            result = await prompt_service.discover_patterns(0.8, 5, mock_db_session)
        
        assert result["status"] == "insufficient_data"
        assert "minimum: 5" in result["message"]

    @pytest.mark.asyncio
    async def test_discover_patterns_creates_ab_experiments(self, prompt_service, mock_db_session):
        """Test that pattern discovery creates A/B experiments for top patterns."""
        # Mock successful pattern discovery
        mock_ml_service = AsyncMock()
        mock_ml_service.discover_patterns.return_value = {
            "status": "success",
            "patterns_discovered": 5,
            "patterns": [
                {"parameters": {"weight": 1.0}, "avg_effectiveness": 0.90},
                {"parameters": {"weight": 0.9}, "avg_effectiveness": 0.85},
                {"parameters": {"weight": 0.8}, "avg_effectiveness": 0.80}
            ] * 2,  # 6 total patterns
            "total_analyzed": 150
        }
        
        with patch('prompt_improver.services.prompt_improvement.get_ml_service', return_value=mock_ml_service):
            await prompt_service.discover_patterns(0.7, 5, mock_db_session)
        
        # Verify A/B experiments were created for top 3 patterns
        # Check database interactions for A/B experiment creation
        assert mock_db_session.execute.call_count >= 3  # At least 3 A/B experiments
        assert mock_db_session.add.call_count >= 3
        assert mock_db_session.commit.call_count >= 1


class TestPhase3HelperMethods:
    """Test Phase 3 helper methods for ML integration."""
    
    @pytest.mark.asyncio
    async def test_store_optimization_trigger(self, prompt_service, mock_db_session, sample_user_feedback):
        """Test storing optimization trigger event."""
        # Mock feedback query
        feedback_result = MagicMock()
        feedback_result.scalar_one_or_none.return_value = sample_user_feedback[0]
        mock_db_session.execute.return_value = feedback_result
        
        await prompt_service._store_optimization_trigger(
            mock_db_session, 123, "model_456", 25, 0.88
        )
        
        # Verify feedback was updated with ML optimization info
        assert sample_user_feedback[0].ml_optimized == True
        assert sample_user_feedback[0].model_id == "model_456"
        mock_db_session.add.assert_called_with(sample_user_feedback[0])
        mock_db_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_store_ml_optimization_results(self, prompt_service, mock_db_session):
        """Test storing ML optimization results."""
        optimization_result = {
            "model_id": "test_model_789",
            "best_score": 0.85,
            "accuracy": 0.90,
            "precision": 0.88,
            "recall": 0.92
        }
        
        with patch('prompt_improver.services.prompt_improvement.datetime') as mock_datetime, \
             patch('prompt_improver.services.prompt_improvement.MLModelPerformance') as mock_perf_class:
            
            mock_datetime.utcnow.return_value = datetime(2025, 1, 1)
            mock_performance_record = MagicMock()
            mock_perf_class.return_value = mock_performance_record
            
            await prompt_service._store_ml_optimization_results(
                mock_db_session, ["clarity_rule"], optimization_result, 30
            )
        
        # Verify performance record was created with correct data
        mock_perf_class.assert_called_once()
        mock_db_session.add.assert_called_with(mock_performance_record)
        mock_db_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_create_ab_experiments_from_patterns(self, prompt_service, mock_db_session):
        """Test creating A/B experiments from discovered patterns."""
        patterns = [
            {"avg_effectiveness": 0.90, "parameters": {"weight": 1.0}},
            {"avg_effectiveness": 0.85, "parameters": {"weight": 0.9}},
            {"avg_effectiveness": 0.80, "parameters": {"weight": 0.8}}
        ]
        
        # Mock existing experiment check
        experiment_result = MagicMock()
        experiment_result.scalar_one_or_none.return_value = None  # No existing experiments
        mock_db_session.execute.return_value = experiment_result
        
        with patch('prompt_improver.services.prompt_improvement.ABExperiment') as mock_ab_class, \
             patch('prompt_improver.services.prompt_improvement.datetime') as mock_datetime:
            
            mock_datetime.utcnow.return_value = datetime(2025, 1, 1)
            mock_experiment = MagicMock()
            mock_ab_class.return_value = mock_experiment
            
            await prompt_service._create_ab_experiments_from_patterns(mock_db_session, patterns)
        
        # Verify 3 A/B experiments were created
        assert mock_ab_class.call_count == 3
        assert mock_db_session.add.call_count == 3
        mock_db_session.commit.assert_called()

    @pytest.mark.asyncio
    async def test_store_pattern_discovery_results(self, prompt_service, mock_db_session):
        """Test storing pattern discovery results."""
        discovery_result = {
            "patterns": [
                {"avg_effectiveness": 0.88, "parameters": {"weight": 1.0}, "support_count": 10},
                {"avg_effectiveness": 0.82, "parameters": {"weight": 0.9}, "support_count": 8}
            ]
        }
        
        with patch('prompt_improver.services.prompt_improvement.DiscoveredPattern') as mock_pattern_class, \
             patch('prompt_improver.services.prompt_improvement.datetime') as mock_datetime:
            
            mock_datetime.utcnow.return_value = datetime(2025, 1, 1)
            mock_pattern_record = MagicMock()
            mock_pattern_class.return_value = mock_pattern_record
            
            await prompt_service._store_pattern_discovery_results(mock_db_session, discovery_result)
        
        # Verify pattern records were created
        assert mock_pattern_class.call_count == 2
        assert mock_db_session.add.call_count == 2
        mock_db_session.commit.assert_called()


class TestPhase3Integration:
    """Test Phase 3 integration with existing functionality."""
    
    @pytest.mark.asyncio
    async def test_phase3_methods_integration(self, prompt_service, mock_db_session):
        """Test that Phase 3 methods integrate properly with existing service."""
        # Verify Phase 3 methods exist and are callable
        assert hasattr(prompt_service, 'trigger_optimization')
        assert hasattr(prompt_service, 'run_ml_optimization') 
        assert hasattr(prompt_service, 'discover_patterns')
        
        # Verify methods are async
        assert asyncio.iscoroutinefunction(prompt_service.trigger_optimization)
        assert asyncio.iscoroutinefunction(prompt_service.run_ml_optimization)
        assert asyncio.iscoroutinefunction(prompt_service.discover_patterns)

    @pytest.mark.asyncio
    async def test_ml_service_import_integration(self, prompt_service):
        """Test that ML service import works correctly."""
        # This tests the 'from .ml_integration import get_ml_service' import
        with patch('prompt_improver.services.prompt_improvement.get_ml_service') as mock_get_ml:
            mock_ml_service = AsyncMock()
            mock_get_ml.return_value = mock_ml_service
            
            # Import should work without errors
            from prompt_improver.services.prompt_improvement import get_ml_service
            
            service = await get_ml_service()
            assert service is mock_ml_service


if __name__ == "__main__":
    pytest.main([__file__, "-v"])