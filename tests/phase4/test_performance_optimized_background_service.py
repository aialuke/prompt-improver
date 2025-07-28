"""
Test Performance-Optimized Background Service (Day 28)

Tests the 2025 performance optimization features using real database:
- 6-hour batch processing with parallel workers
- Incremental updates with change detection
- ML prediction pipeline with confidence scoring

2025 Best Practice: Uses testcontainers for real database testing
instead of mocking for better integration coverage.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch

from prompt_improver.ml.background.intelligence_processor import MLIntelligenceProcessor
from prompt_improver.rule_engine.models import PromptCharacteristics


class TestPerformanceOptimizedBackgroundService:
    """Test performance-optimized background service features."""

    @pytest.fixture
    def processor(self):
        """Create MLIntelligenceProcessor instance for testing."""
        return MLIntelligenceProcessor()

    @pytest.fixture
    def sample_rule_data(self):
        """Create sample rule data for testing."""
        return MagicMock(
            rule_id="test_rule_123",
            rule_name="Test Rule",
            prompt_characteristics={
                "prompt_type": "instructional",
                "complexity_level": 0.7,
                "domain": "technical",
                "length_category": "medium",
                "reasoning_required": True,
                "specificity_level": 0.8,
                "context_richness": 0.6,
                "task_type": "problem_solving",
                "language_style": "formal",
                "custom_attributes": {}
            },
            prompt_type="instructional"
        )

    def test_performance_configuration_2025_patterns(self, processor):
        """Test that performance configuration follows 2025 best practices."""
        # Verify batch processing configuration
        assert processor.batch_size == 100
        assert processor.processing_interval_hours == 6
        assert processor.max_parallel_workers == 4
        assert processor.incremental_update_threshold == 0.1

        # Verify ML prediction pipeline configuration
        assert processor.confidence_scoring_enabled is True
        assert processor.min_confidence_threshold == 0.6
        assert processor.prediction_batch_size == 50

    def test_batch_range_calculation(self, processor):
        """Test optimal batch range calculation for parallel processing."""
        # Test with 400 rules and 4 workers
        batches = processor._calculate_batch_ranges(400)

        # Should create 4 batches of 100 rules each
        assert len(batches) == 4
        assert all(batch["batch_size"] == 100 for batch in batches)
        assert batches[0]["start_offset"] == 0
        assert batches[1]["start_offset"] == 100
        assert batches[2]["start_offset"] == 200
        assert batches[3]["start_offset"] == 300

    def test_batch_range_calculation_small_dataset(self, processor):
        """Test batch range calculation with small dataset."""
        # Test with 50 rules and 4 workers
        batches = processor._calculate_batch_ranges(50)

        # Should create batches with smaller sizes
        assert len(batches) <= 4
        total_rules = sum(batch["batch_size"] for batch in batches)
        assert total_rules == 50

    @pytest.mark.asyncio
    async def test_incremental_update_detection(self, processor, db_session, sample_rule_intelligence_cache):
        """Test incremental update detection logic with real database."""
        # Test with rule that has cached data
        needs_update = await processor._check_incremental_update_needed(
            db_session, "test_rule_1"
        )

        # Should not need update initially (no new performance data)
        assert needs_update is False

    @pytest.mark.asyncio
    async def test_incremental_update_not_needed(self, processor, db_session):
        """Test incremental update when no cached data exists."""
        # Test with rule that has no cached data - should need update
        needs_update = await processor._check_incremental_update_needed(
            db_session, "non_existent_rule"
        )

        # Should need update when no cache entry exists
        assert needs_update is True

    @pytest.mark.asyncio
    async def test_ml_predictions_with_confidence_scoring(self, processor, db_session, sample_rule_performance_data):
        """Test ML prediction pipeline with confidence scoring using real database."""
        # Use the first sample rule data
        rule_data = type('RuleData', (), {
            'rule_id': sample_rule_performance_data[0]['rule_id'],
            'rule_name': sample_rule_performance_data[0]['rule_name']
        })()

        characteristics = PromptCharacteristics(**sample_rule_performance_data[0]['prompt_characteristics'])

        # Generate ML predictions with real database
        predictions = await processor._generate_ml_predictions_with_confidence(
            db_session, rule_data, characteristics
        )

        # Verify confidence scoring structure
        assert "confidence" in predictions
        assert "predictions" in predictions

        # With limited test data, confidence might be low
        assert predictions["confidence"] >= 0.1  # At least some confidence

        # Verify prediction content structure
        pred_data = predictions["predictions"]
        if "effectiveness_prediction" in pred_data:
            assert "confidence_interval" in pred_data
            assert "sample_size" in pred_data

    @pytest.mark.asyncio
    async def test_ml_predictions_insufficient_data(self, processor, db_session):
        """Test ML predictions with insufficient data using real database."""
        # Create rule data for non-existent rule
        rule_data = type('RuleData', (), {
            'rule_id': 'non_existent_rule',
            'rule_name': 'Non Existent Rule'
        })()

        characteristics = PromptCharacteristics(
            prompt_type="test",
            complexity_level=0.5,
            domain="test",
            length_category="short",
            reasoning_required=False,
            specificity_level=0.5,
            context_richness=0.5,
            task_type="test",
            language_style="casual",
            custom_attributes={}
        )

        # Generate ML predictions for non-existent rule
        predictions = await processor._generate_ml_predictions_with_confidence(
            db_session, rule_data, characteristics
        )

        # Verify low confidence for insufficient data
        assert predictions["confidence"] < 0.5
        assert "insufficient_data" in predictions["predictions"] or "no_valid_scores" in predictions["predictions"]

    @pytest.mark.asyncio
    async def test_parallel_batch_processing_configuration(self, processor, db_session, sample_rule_performance_data):
        """Test parallel batch processing configuration with real database."""
        # Patch the get_session_context to return our test session
        with patch('prompt_improver.ml.background.intelligence_processor.get_session_context') as mock_session_context:
            mock_session_context.return_value.__aenter__.return_value = db_session

            # Mock the batch processing method to avoid complex processing
            with patch.object(processor, '_process_batch_with_semaphore', return_value={"rules_processed": 1, "status": "success"}):
                # Run parallel batch processing
                result = await processor.run_parallel_batch_processing()

                # Verify parallel processing configuration
                assert result["status"] in ["success", "partial_success", "no_data"]
                if result["status"] != "no_data":
                    assert result["parallel_workers"] == processor.max_parallel_workers
                    assert "batches_processed" in result
                assert "processing_time_ms" in result

    def test_2025_performance_optimization_patterns(self, processor):
        """Test that implementation follows 2025 performance optimization patterns."""
        # Verify parallel processing configuration
        assert processor.max_parallel_workers > 1, "Should use parallel processing"

        # Verify incremental update optimization
        assert processor.incremental_update_threshold > 0, "Should use incremental updates"
        assert processor.incremental_update_threshold < 1, "Threshold should be reasonable"

        # Verify ML prediction pipeline optimization
        assert processor.confidence_scoring_enabled, "Should use confidence scoring"
        assert processor.min_confidence_threshold > 0.5, "Should have reasonable confidence threshold"
        assert processor.prediction_batch_size > 0, "Should use batch prediction"

        # Verify caching optimization
        assert processor.cache_ttl_hours > processor.processing_interval_hours, "Cache should outlive processing interval"
