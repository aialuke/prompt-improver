"""
Tests for the refactored MLMetricsCollector using the new base class.

Tests real behavior without mocks and verifies JSONB compatibility.
"""

import asyncio
import json
import pytest
from datetime import datetime, timezone, timedelta
from typing import Dict, Any

from prompt_improver.metrics.ml_metrics import (
    MLMetricsCollector,
    PromptImprovementMetric,
    ModelInferenceMetric,
    FeatureFlagMetric,
    MLPipelineMetric,
    PromptCategory,
    ModelInferenceStage,
    get_ml_metrics_collector
)
from prompt_improver.metrics.base_metrics_collector import MetricsConfig


class TestRefactoredMLMetricsCollector:
    """Test suite for refactored MLMetricsCollector."""
    
    @pytest.fixture
    async def ml_collector(self):
        """Create an ML metrics collector instance."""
        config = MetricsConfig(
            aggregation_window_minutes=1,
            retention_hours=2,
            max_metrics_per_type=100
        )
        collector = MLMetricsCollector(config)
        yield collector
        await collector.stop_collection()
    
    def test_ml_collector_inheritance(self, ml_collector):
        """Test that MLMetricsCollector properly inherits from base class."""
        # Should have base class attributes
        assert hasattr(ml_collector, 'config')
        assert hasattr(ml_collector, 'collection_stats')
        assert hasattr(ml_collector, '_metrics_storage')
        assert hasattr(ml_collector, 'metrics_registry')
        
        # Should have ML-specific attributes
        assert hasattr(ml_collector, 'ml_stats')
        
        # Should have mixin methods
        assert hasattr(ml_collector, 'create_counter')
        assert hasattr(ml_collector, 'create_histogram')
        assert hasattr(ml_collector, 'store_metric')
        assert hasattr(ml_collector, 'get_recent_metrics')
    
    async def test_prompt_improvement_metric_collection(self, ml_collector):
        """Test collecting prompt improvement metrics."""
        metric = PromptImprovementMetric(
            category=PromptCategory.TECHNICAL_WRITING,
            original_length=100,
            improved_length=80,
            improvement_ratio=1.25,
            success=True,
            processing_time_ms=150.5,
            confidence_score=0.85,
            rule_combination_count=3,
            user_id="test_user",
            session_id="test_session",
            timestamp=datetime.now(timezone.utc),
            quality_metrics={"clarity": 0.9, "conciseness": 0.8},
            feature_flags_used=["new_algorithm", "enhanced_rules"]
        )
        
        # Collect the metric
        ml_collector.collect_metric(metric)
        
        # Verify storage
        assert len(ml_collector._metrics_storage["prompt"]) == 1
        stored_metric = ml_collector._metrics_storage["prompt"][0]
        assert stored_metric.category == PromptCategory.TECHNICAL_WRITING
        assert stored_metric.improvement_ratio == 1.25
        assert stored_metric.success is True
        
        # Verify ML-specific stats
        assert ml_collector.ml_stats["prompt_improvements_tracked"] == 1
    
    async def test_model_inference_metric_collection(self, ml_collector):
        """Test collecting model inference metrics."""
        metric = ModelInferenceMetric(
            model_name="gpt-4",
            inference_stage=ModelInferenceStage.PREPROCESSING,
            input_tokens=500,
            output_tokens=200,
            latency_ms=250.0,
            memory_usage_mb=128.5,
            gpu_utilization_percent=75.0,
            batch_size=4,
            success=True,
            error_type=None,
            confidence_distribution=[0.8, 0.9, 0.85, 0.92],
            timestamp=datetime.now(timezone.utc),
            request_id="req_123"
        )
        
        # Collect the metric
        ml_collector.collect_metric(metric)
        
        # Verify storage
        assert len(ml_collector._metrics_storage["inference"]) == 1
        stored_metric = ml_collector._metrics_storage["inference"][0]
        assert stored_metric.model_name == "gpt-4"
        assert stored_metric.inference_stage == ModelInferenceStage.PREPROCESSING
        assert stored_metric.latency_ms == 250.0
        
        # Verify ML-specific stats
        assert ml_collector.ml_stats["model_inferences_tracked"] == 1
    
    async def test_feature_flag_metric_collection(self, ml_collector):
        """Test collecting feature flag metrics."""
        metric = FeatureFlagMetric(
            flag_name="new_ml_algorithm",
            enabled=True,
            user_id="test_user",
            session_id="test_session",
            rollout_percentage=25.0,
            performance_impact_ms=5.2,
            success_rate=0.95,
            conversion_rate=0.12,
            error_count=2,
            timestamp=datetime.now(timezone.utc),
            experiment_variant="variant_a"
        )
        
        # Collect the metric
        ml_collector.collect_metric(metric)
        
        # Verify storage
        assert len(ml_collector._metrics_storage["feature_flag"]) == 1
        stored_metric = ml_collector._metrics_storage["feature_flag"][0]
        assert stored_metric.flag_name == "new_ml_algorithm"
        assert stored_metric.enabled is True
        assert stored_metric.rollout_percentage == 25.0
        
        # Verify ML-specific stats
        assert ml_collector.ml_stats["feature_flags_tracked"] == 1
    
    async def test_ml_pipeline_metric_collection(self, ml_collector):
        """Test collecting ML pipeline metrics."""
        metric = MLPipelineMetric(
            pipeline_name="prompt_optimization",
            stage_name="feature_extraction",
            processing_time_ms=500.0,
            throughput_items_per_second=100.0,
            memory_peak_mb=256.0,
            cpu_utilization_percent=80.0,
            queue_depth=5,
            success=True,
            retry_count=0,
            data_quality_score=0.92,
            timestamp=datetime.now(timezone.utc)
        )
        
        # Collect the metric
        ml_collector.collect_metric(metric)
        
        # Verify storage
        assert len(ml_collector._metrics_storage["ml_pipeline"]) == 1
        stored_metric = ml_collector._metrics_storage["ml_pipeline"][0]
        assert stored_metric.pipeline_name == "prompt_optimization"
        assert stored_metric.stage_name == "feature_extraction"
        assert stored_metric.throughput_items_per_second == 100.0
        
        # Verify ML-specific stats
        assert ml_collector.ml_stats["pipeline_operations_tracked"] == 1
    
    async def test_collection_stats_integration(self, ml_collector):
        """Test that collection stats properly integrate base and ML-specific stats."""
        # Collect various metrics
        prompt_metric = PromptImprovementMetric(
            category=PromptCategory.CREATIVE_WRITING,
            original_length=50, improved_length=45, improvement_ratio=1.1,
            success=True, processing_time_ms=100.0, confidence_score=0.8,
            rule_combination_count=2, user_id=None, session_id=None,
            timestamp=datetime.now(timezone.utc), quality_metrics={},
            feature_flags_used=[]
        )
        
        inference_metric = ModelInferenceMetric(
            model_name="claude", inference_stage=ModelInferenceStage.INFERENCE,
            input_tokens=100, output_tokens=50, latency_ms=200.0,
            memory_usage_mb=64.0, gpu_utilization_percent=None, batch_size=1,
            success=True, error_type=None, confidence_distribution=[0.9],
            timestamp=datetime.now(timezone.utc), request_id="req_456"
        )
        
        ml_collector.collect_metric(prompt_metric)
        ml_collector.collect_metric(inference_metric)
        
        # Get collection stats
        stats = ml_collector.get_collection_stats()
        
        # Should have base class stats
        assert "total_metrics_collected" in stats
        assert "is_running" in stats
        assert "config" in stats
        
        # Should have ML-specific stats
        assert stats["prompt_improvements_tracked"] == 1
        assert stats["model_inferences_tracked"] == 1
        assert stats["feature_flags_tracked"] == 0
        assert stats["pipeline_operations_tracked"] == 0
        
        # Should have current metrics count
        assert stats["current_metrics_count"]["prompt_improvements"] == 1
        assert stats["current_metrics_count"]["model_inferences"] == 1
    
    async def test_jsonb_compatibility(self, ml_collector):
        """Test JSONB compatibility of collected metrics."""
        # Create a complex metric with nested data
        metric = PromptImprovementMetric(
            category=PromptCategory.BUSINESS_COMMUNICATION,
            original_length=200,
            improved_length=150,
            improvement_ratio=1.33,
            success=True,
            processing_time_ms=300.0,
            confidence_score=0.88,
            rule_combination_count=5,
            user_id="user_789",
            session_id="session_abc",
            timestamp=datetime.now(timezone.utc),
            quality_metrics={
                "clarity": 0.9,
                "conciseness": 0.85,
                "engagement": 0.92,
                "technical_accuracy": 0.88
            },
            feature_flags_used=["advanced_nlp", "context_aware", "style_optimization"]
        )
        
        ml_collector.collect_metric(metric)
        
        # Get collection stats (should be JSONB compatible)
        stats = ml_collector.get_collection_stats()
        
        # Test JSON serialization
        json_str = json.dumps(stats, default=str)  # default=str for datetime objects
        assert isinstance(json_str, str)
        
        # Test round-trip
        parsed_stats = json.loads(json_str)
        assert parsed_stats["prompt_improvements_tracked"] == 1
        assert "current_metrics_count" in parsed_stats
    
    async def test_factory_function(self):
        """Test the factory function for ML metrics collector."""
        config = MetricsConfig(max_metrics_per_type=500)
        
        # Get collector using factory function
        collector1 = get_ml_metrics_collector(config)
        collector2 = get_ml_metrics_collector(config)
        
        # Should return the same instance (singleton pattern)
        assert collector1 is collector2
        assert isinstance(collector1, MLMetricsCollector)
        
        # Clean up
        await collector1.stop_collection()
    
    async def test_prometheus_metrics_initialization(self, ml_collector):
        """Test that Prometheus metrics are properly initialized."""
        # Should have ML-specific Prometheus metrics
        assert hasattr(ml_collector, 'prompt_success_rate')
        assert hasattr(ml_collector, 'prompt_processing_time')
        assert hasattr(ml_collector, 'prompt_improvement_ratio')
        assert hasattr(ml_collector, 'model_inference_latency')
        
        # Test that metrics were created using mixin methods
        assert ml_collector.prompt_success_rate is not None
        assert ml_collector.prompt_processing_time is not None
    
    async def test_real_behavior_workflow(self, ml_collector):
        """Test a realistic ML metrics collection workflow."""
        await ml_collector.start_collection()
        
        # Simulate a prompt improvement workflow
        start_time = datetime.now(timezone.utc)
        
        # 1. Feature flag check
        flag_metric = FeatureFlagMetric(
            flag_name="enhanced_prompt_optimization",
            enabled=True,
            user_id="workflow_user",
            session_id="workflow_session",
            rollout_percentage=50.0,
            performance_impact_ms=2.0,
            success_rate=0.98,
            conversion_rate=None,
            error_count=0,
            timestamp=start_time,
            experiment_variant="control"
        )
        ml_collector.collect_metric(flag_metric)
        
        # 2. Model inference for preprocessing
        inference_metric = ModelInferenceMetric(
            model_name="preprocessing_model",
            inference_stage=ModelInferenceStage.PREPROCESSING,
            input_tokens=300,
            output_tokens=280,
            latency_ms=150.0,
            memory_usage_mb=96.0,
            gpu_utilization_percent=60.0,
            batch_size=1,
            success=True,
            error_type=None,
            confidence_distribution=[0.87],
            timestamp=start_time + timedelta(milliseconds=50),
            request_id="workflow_req_1"
        )
        ml_collector.collect_metric(inference_metric)
        
        # 3. Prompt improvement
        improvement_metric = PromptImprovementMetric(
            category=PromptCategory.TECHNICAL_WRITING,
            original_length=300,
            improved_length=250,
            improvement_ratio=1.2,
            success=True,
            processing_time_ms=200.0,
            confidence_score=0.87,
            rule_combination_count=4,
            user_id="workflow_user",
            session_id="workflow_session",
            timestamp=start_time + timedelta(milliseconds=200),
            quality_metrics={"clarity": 0.9, "conciseness": 0.85},
            feature_flags_used=["enhanced_prompt_optimization"]
        )
        ml_collector.collect_metric(improvement_metric)
        
        # 4. Pipeline completion
        pipeline_metric = MLPipelineMetric(
            pipeline_name="prompt_improvement_pipeline",
            stage_name="completion",
            processing_time_ms=350.0,
            throughput_items_per_second=2.86,  # 1 item / 0.35 seconds
            memory_peak_mb=128.0,
            cpu_utilization_percent=75.0,
            queue_depth=0,
            success=True,
            retry_count=0,
            data_quality_score=0.91,
            timestamp=start_time + timedelta(milliseconds=350)
        )
        ml_collector.collect_metric(pipeline_metric)
        
        # Verify all metrics were collected
        stats = ml_collector.get_collection_stats()
        assert stats["feature_flags_tracked"] == 1
        assert stats["model_inferences_tracked"] == 1
        assert stats["prompt_improvements_tracked"] == 1
        assert stats["pipeline_operations_tracked"] == 1
        assert stats["total_metrics_collected"] == 4
        
        # Test recent metrics retrieval
        recent_prompts = ml_collector.get_recent_metrics("prompt", hours=1)
        assert len(recent_prompts) == 1
        assert recent_prompts[0].category == PromptCategory.TECHNICAL_WRITING
        
        await ml_collector.stop_collection()
