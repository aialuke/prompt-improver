"""
ML-Specific Business Metrics Collectors for Prompt Improvement.

Tracks prompt improvement success rates, model inference performance,
feature flag effectiveness, and ML pipeline throughput with real-time aggregation.
"""

import asyncio
import logging
import time
from typing import Dict, Any, Optional, List, DefaultDict, Union
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from enum import Enum
import statistics
from collections import defaultdict, deque

# Lazy import to avoid circular dependency
# from ..performance.monitoring.metrics_registry import get_metrics_registry
from .base_metrics_collector import (
    BaseMetricsCollector, MetricsConfig, PrometheusMetricsMixin,
    MetricsStorageMixin
)

class PromptCategory(Enum):
    """Categories for prompt improvements."""
    CLARITY = "clarity"
    SPECIFICITY = "specificity"
    CONTEXT = "context"
    STRUCTURE = "structure"
    ROLE_BASED = "role_based"
    FEW_SHOT = "few_shot"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    XML_ENHANCEMENT = "xml_enhancement"

class ModelInferenceStage(Enum):
    """Stages of model inference pipeline."""
    INPUT_PREPROCESSING = "input_preprocessing"
    TOKENIZATION = "tokenization"
    MODEL_FORWARD = "model_forward"
    POSTPROCESSING = "postprocessing"
    OUTPUT_VALIDATION = "output_validation"

@dataclass
class PromptImprovementMetric:
    """Metrics for a single prompt improvement operation."""
    category: PromptCategory
    original_length: int
    improved_length: int
    improvement_ratio: float
    success: bool
    processing_time_ms: float
    confidence_score: float
    rule_combination_count: int
    user_id: Optional[str]
    session_id: Optional[str]
    timestamp: datetime
    quality_metrics: Dict[str, float]
    feature_flags_used: List[str]

@dataclass
class ModelInferenceMetric:
    """Metrics for model inference operations."""
    model_name: str
    inference_stage: ModelInferenceStage
    input_tokens: int
    output_tokens: int
    latency_ms: float
    memory_usage_mb: float
    gpu_utilization_percent: Optional[float]
    batch_size: int
    success: bool
    error_type: Optional[str]
    confidence_distribution: List[float]
    timestamp: datetime
    request_id: str

@dataclass
class FeatureFlagMetric:
    """Metrics for feature flag usage and effectiveness."""
    flag_name: str
    enabled: bool
    user_id: Optional[str]
    session_id: Optional[str]
    rollout_percentage: float
    performance_impact_ms: float
    success_rate: float
    conversion_rate: Optional[float]
    error_count: int
    timestamp: datetime
    experiment_variant: Optional[str]

@dataclass
class MLPipelineMetric:
    """Metrics for ML pipeline processing."""
    pipeline_name: str
    stage_name: str
    processing_time_ms: float
    throughput_items_per_second: float
    memory_peak_mb: float
    cpu_utilization_percent: float
    queue_depth: int
    success: bool
    retry_count: int
    data_quality_score: float
    timestamp: datetime

class MLMetricsCollector(
    BaseMetricsCollector[Union[PromptImprovementMetric, ModelInferenceMetric,
                              FeatureFlagMetric, MLPipelineMetric]],
    PrometheusMetricsMixin,
    MetricsStorageMixin
):
    """
    Collects and aggregates ML-specific business metrics.

    Provides real-time tracking of prompt improvement effectiveness,
    model performance, and feature rollout success rates.

    Uses modern 2025 Python patterns with composition over inheritance.
    """

    def __init__(self, config: Optional[Union[Dict[str, Any], MetricsConfig]] = None):
        """Initialize ML metrics collector with modern base class."""
        # Initialize base class with dependency injection
        super().__init__(config)

        # ML-specific storage using base class storage
        self._metrics_storage.update({
            "prompt": deque(maxlen=self.config.get("max_prompt_metrics", 10000)),
            "inference": deque(maxlen=self.config.get("max_inference_metrics", 5000)),
            "feature_flag": deque(maxlen=self.config.get("max_flag_metrics", 10000)),
            "ml_pipeline": deque(maxlen=self.config.get("max_pipeline_metrics", 10000))
        })

        # ML-specific collection statistics
        self.ml_stats = {
            "prompt_improvements_tracked": 0,
            "model_inferences_tracked": 0,
            "feature_flags_tracked": 0,
            "pipeline_operations_tracked": 0,
        }

    def collect_metric(self, metric: Union[PromptImprovementMetric, ModelInferenceMetric,
                                         FeatureFlagMetric, MLPipelineMetric]) -> None:
        """Collect an ML metric using the base class storage."""
        if isinstance(metric, PromptImprovementMetric):
            self.store_metric("prompt", metric)
            self.ml_stats["prompt_improvements_tracked"] += 1
        elif isinstance(metric, ModelInferenceMetric):
            self.store_metric("inference", metric)
            self.ml_stats["model_inferences_tracked"] += 1
        elif isinstance(metric, FeatureFlagMetric):
            self.store_metric("feature_flag", metric)
            self.ml_stats["feature_flags_tracked"] += 1
        elif isinstance(metric, MLPipelineMetric):
            self.store_metric("ml_pipeline", metric)
            self.ml_stats["pipeline_operations_tracked"] += 1

    def _initialize_prometheus_metrics(self) -> None:
        """Initialize Prometheus metrics for ML operations using mixins."""
        # Prompt improvement metrics using mixin methods
        self.prompt_success_rate = self.create_gauge(
            "ml_prompt_improvement_success_rate",
            "Success rate of prompt improvements by category",
            ["category", "time_window"]
        )

        self.prompt_processing_time = self.create_histogram(
            "ml_prompt_processing_time_seconds",
            "Time spent processing prompt improvements",
            ["category"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )

        self.prompt_improvement_ratio = self.create_histogram(
            "ml_prompt_improvement_ratio",
            "Ratio of improvement in prompt quality",
            ["category"],
            buckets=[0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 5.0]
        )

        # Model inference metrics
        self.model_inference_latency = self.create_histogram(
            "ml_model_inference_latency_seconds",
            "Model inference latency by stage",
            ["model_name", "stage"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )

        self.model_token_throughput = self.metrics_registry.get_or_create_gauge(
            "ml_model_token_throughput",
            "Tokens processed per second by model",
            ["model_name", "token_type"]
        )

        self.model_confidence_distribution = self.metrics_registry.get_or_create_histogram(
            "ml_model_confidence_scores",
            "Distribution of model confidence scores",
            ["model_name"],
            buckets=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
        )

        # Feature flag metrics
        self.feature_flag_adoption_rate = self.metrics_registry.get_or_create_gauge(
            "ml_feature_flag_adoption_rate",
            "Adoption rate of feature flags",
            ["flag_name", "variant"]
        )

        self.feature_flag_performance_impact = self.metrics_registry.get_or_create_histogram(
            "ml_feature_flag_performance_impact_seconds",
            "Performance impact of feature flags",
            ["flag_name"],
            buckets=[0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5]
        )

        # Pipeline metrics
        self.pipeline_throughput = self.metrics_registry.get_or_create_gauge(
            "ml_pipeline_throughput_items_per_second",
            "ML pipeline processing throughput",
            ["pipeline_name", "stage"]
        )

        self.pipeline_queue_depth = self.metrics_registry.get_or_create_gauge(
            "ml_pipeline_queue_depth",
            "Current queue depth for ML pipelines",
            ["pipeline_name"]
        )

    async def start_aggregation(self) -> None:
        """Start background metrics aggregation."""
        if self.is_running:
            return

        self.is_running = True
        self.aggregation_task = asyncio.create_task(self._aggregation_loop())
        self.logger.info("Started ML metrics aggregation")

    async def stop_aggregation(self) -> None:
        """Stop background metrics aggregation."""
        if not self.is_running:
            return

        self.is_running = False
        if self.aggregation_task:
            self.aggregation_task.cancel()
            try:
                await self.aggregation_task
            except asyncio.CancelledError:
                pass

        self.logger.info("Stopped ML metrics aggregation")

    async def _aggregation_loop(self) -> None:
        """Background aggregation of metrics."""
        try:
            while self.is_running:
                await self._aggregate_metrics()
                await asyncio.sleep(self.aggregation_window_minutes * 60)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.logger.error(f"Error in ML metrics aggregation: {e}")

    async def record_prompt_improvement(self, metric: PromptImprovementMetric) -> None:
        """Record a prompt improvement metric."""
        try:
            self.prompt_metrics.append(metric)
            self.collection_stats["prompt_improvements_tracked"] += 1

            # Update Prometheus metrics immediately
            category_str = metric.category.value

            # Success rate (using 1 for success, 0 for failure)
            success_value = 1.0 if metric.success else 0.0
            self.prompt_success_rate.labels(
                category=category_str,
                time_window="realtime"
            ).set(success_value)

            # Processing time
            self.prompt_processing_time.labels(
                category=category_str
            ).observe(metric.processing_time_ms / 1000.0)

            # Improvement ratio
            if metric.improvement_ratio > 0:
                self.prompt_improvement_ratio.labels(
                    category=category_str
                ).observe(metric.improvement_ratio)

            self.logger.debug(f"Recorded prompt improvement metric: {category_str}")

        except Exception as e:
            self.logger.error(f"Error recording prompt improvement metric: {e}")

    async def record_model_inference(self, metric: ModelInferenceMetric) -> None:
        """Record a model inference metric."""
        try:
            self.inference_metrics.append(metric)
            self.collection_stats["model_inferences_tracked"] += 1

            # Update Prometheus metrics
            model_name = metric.model_name
            stage_name = metric.inference_stage.value

            # Latency
            self.model_inference_latency.labels(
                model_name=model_name,
                stage=stage_name
            ).observe(metric.latency_ms / 1000.0)

            # Token throughput
            if metric.latency_ms > 0:
                input_throughput = (metric.input_tokens / metric.latency_ms) * 1000
                output_throughput = (metric.output_tokens / metric.latency_ms) * 1000

                self.model_token_throughput.labels(
                    model_name=model_name,
                    token_type="input"
                ).set(input_throughput)

                self.model_token_throughput.labels(
                    model_name=model_name,
                    token_type="output"
                ).set(output_throughput)

            # Confidence distribution
            for confidence in metric.confidence_distribution:
                self.model_confidence_distribution.labels(
                    model_name=model_name
                ).observe(confidence)

            self.logger.debug(f"Recorded model inference metric: {model_name}/{stage_name}")

        except Exception as e:
            self.logger.error(f"Error recording model inference metric: {e}")

    async def record_feature_flag(self, metric: FeatureFlagMetric) -> None:
        """Record a feature flag metric."""
        try:
            self.feature_flag_metrics.append(metric)
            self.collection_stats["feature_flags_tracked"] += 1

            # Update Prometheus metrics
            flag_name = metric.flag_name
            variant = metric.experiment_variant or "default"

            # Adoption rate
            adoption_value = metric.rollout_percentage / 100.0
            self.feature_flag_adoption_rate.labels(
                flag_name=flag_name,
                variant=variant
            ).set(adoption_value)

            # Performance impact
            if metric.performance_impact_ms > 0:
                self.feature_flag_performance_impact.labels(
                    flag_name=flag_name
                ).observe(metric.performance_impact_ms / 1000.0)

            self.logger.debug(f"Recorded feature flag metric: {flag_name}")

        except Exception as e:
            self.logger.error(f"Error recording feature flag metric: {e}")

    async def record_pipeline_operation(self, metric: MLPipelineMetric) -> None:
        """Record an ML pipeline operation metric."""
        try:
            self.pipeline_metrics.append(metric)
            self.collection_stats["pipeline_operations_tracked"] += 1

            # Update Prometheus metrics
            pipeline_name = metric.pipeline_name
            stage_name = metric.stage_name

            # Throughput
            self.pipeline_throughput.labels(
                pipeline_name=pipeline_name,
                stage=stage_name
            ).set(metric.throughput_items_per_second)

            # Queue depth
            self.pipeline_queue_depth.labels(
                pipeline_name=pipeline_name
            ).set(metric.queue_depth)

            self.logger.debug(f"Recorded pipeline metric: {pipeline_name}/{stage_name}")

        except Exception as e:
            self.logger.error(f"Error recording pipeline metric: {e}")

    async def _aggregate_metrics(self) -> None:
        """Aggregate metrics over time windows."""
        try:
            current_time = datetime.now(timezone.utc)
            window_start = current_time - timedelta(minutes=self.aggregation_window_minutes)

            # Aggregate prompt improvement metrics
            await self._aggregate_prompt_metrics(window_start, current_time)

            # Aggregate model inference metrics
            await self._aggregate_inference_metrics(window_start, current_time)

            # Aggregate feature flag metrics
            await self._aggregate_feature_flag_metrics(window_start, current_time)

            # Aggregate pipeline metrics
            await self._aggregate_pipeline_metrics(window_start, current_time)

            # Clean up old metrics
            await self._cleanup_old_metrics(current_time)

            self.collection_stats["last_aggregation"] = current_time

        except Exception as e:
            self.logger.error(f"Error in metrics aggregation: {e}")

    async def _aggregate_prompt_metrics(self, window_start: datetime, window_end: datetime) -> None:
        """Aggregate prompt improvement metrics."""
        window_metrics = [
            m for m in self.prompt_metrics
            if window_start <= m.timestamp <= window_end
        ]

        if not window_metrics:
            return

        # Group by category
        category_groups: DefaultDict[PromptCategory, List[PromptImprovementMetric]] = defaultdict(list)
        for metric in window_metrics:
            category_groups[metric.category].append(metric)

        # Calculate aggregated success rates
        for category, metrics in category_groups.items():
            if not metrics:
                continue

            success_rate = sum(1 for m in metrics if m.success) / len(metrics)
            avg_improvement_ratio = statistics.mean(m.improvement_ratio for m in metrics)
            avg_processing_time = statistics.mean(m.processing_time_ms for m in metrics)
            avg_confidence = statistics.mean(m.confidence_score for m in metrics)

            # Update aggregated metrics
            self.prompt_success_rate.labels(
                category=category.value,
                time_window=f"{self.aggregation_window_minutes}m"
            ).set(success_rate)

            self.logger.debug(
                f"Aggregated prompt metrics for {category.value}: "
                f"success_rate={success_rate:.3f}, "
                f"avg_improvement={avg_improvement_ratio:.3f}, "
                f"avg_time={avg_processing_time:.1f}ms, "
                f"avg_confidence={avg_confidence:.3f}"
            )

    async def _aggregate_inference_metrics(self, window_start: datetime, window_end: datetime) -> None:
        """Aggregate model inference metrics."""
        window_metrics = [
            m for m in self.inference_metrics
            if window_start <= m.timestamp <= window_end
        ]

        if not window_metrics:
            return

        # Group by model and stage
        model_stage_groups: DefaultDict[tuple[str, ModelInferenceStage], List[ModelInferenceMetric]] = defaultdict(list)
        for metric in window_metrics:
            key = (metric.model_name, metric.inference_stage)
            model_stage_groups[key].append(metric)

        # Calculate throughput and performance metrics
        for (model_name, stage), metrics in model_stage_groups.items():
            if not metrics:
                continue

            total_tokens = sum(m.input_tokens + m.output_tokens for m in metrics)
            total_time_seconds = sum(m.latency_ms for m in metrics) / 1000.0

            if total_time_seconds > 0:
                throughput = total_tokens / total_time_seconds

                self.model_token_throughput.labels(
                    model_name=model_name,
                    token_type="aggregated"
                ).set(throughput)
            else:
                throughput = 0.0

            avg_latency = statistics.mean(m.latency_ms for m in metrics)
            success_rate = sum(1 for m in metrics if m.success) / len(metrics)

            self.logger.debug(
                f"Aggregated inference metrics for {model_name}/{stage.value}: "
                f"throughput={throughput:.1f} tokens/s, "
                f"avg_latency={avg_latency:.1f}ms, "
                f"success_rate={success_rate:.3f}"
            )

    async def _aggregate_feature_flag_metrics(self, window_start: datetime, window_end: datetime) -> None:
        """Aggregate feature flag metrics."""
        window_metrics = [
            m for m in self.feature_flag_metrics
            if window_start <= m.timestamp <= window_end
        ]

        if not window_metrics:
            return

        # Group by flag name
        flag_groups: DefaultDict[str, List[FeatureFlagMetric]] = defaultdict(list)
        for metric in window_metrics:
            flag_groups[metric.flag_name].append(metric)

        # Calculate effectiveness metrics
        for flag_name, metrics in flag_groups.items():
            if not metrics:
                continue

            enabled_count = sum(1 for m in metrics if m.enabled)
            total_count = len(metrics)
            adoption_rate = enabled_count / total_count if total_count > 0 else 0

            avg_performance_impact = statistics.mean(m.performance_impact_ms for m in metrics)
            avg_success_rate = statistics.mean(m.success_rate for m in metrics)

            self.feature_flag_adoption_rate.labels(
                flag_name=flag_name,
                variant="aggregated"
            ).set(adoption_rate)

            self.logger.debug(
                f"Aggregated feature flag metrics for {flag_name}: "
                f"adoption_rate={adoption_rate:.3f}, "
                f"avg_performance_impact={avg_performance_impact:.1f}ms, "
                f"avg_success_rate={avg_success_rate:.3f}"
            )

    async def _aggregate_pipeline_metrics(self, window_start: datetime, window_end: datetime) -> None:
        """Aggregate ML pipeline metrics."""
        window_metrics = [
            m for m in self.pipeline_metrics
            if window_start <= m.timestamp <= window_end
        ]

        if not window_metrics:
            return

        # Group by pipeline and stage
        pipeline_groups: DefaultDict[tuple[str, str], List[MLPipelineMetric]] = defaultdict(list)
        for metric in window_metrics:
            key = (metric.pipeline_name, metric.stage_name)
            pipeline_groups[key].append(metric)

        # Calculate throughput and performance metrics
        for (pipeline_name, stage_name), metrics in pipeline_groups.items():
            if not metrics:
                continue

            avg_throughput = statistics.mean(m.throughput_items_per_second for m in metrics)
            avg_queue_depth = statistics.mean(m.queue_depth for m in metrics)
            avg_data_quality = statistics.mean(m.data_quality_score for m in metrics)
            success_rate = sum(1 for m in metrics if m.success) / len(metrics)

            self.pipeline_throughput.labels(
                pipeline_name=pipeline_name,
                stage=stage_name
            ).set(avg_throughput)

            self.pipeline_queue_depth.labels(
                pipeline_name=pipeline_name
            ).set(avg_queue_depth)

            self.logger.debug(
                f"Aggregated pipeline metrics for {pipeline_name}/{stage_name}: "
                f"throughput={avg_throughput:.1f} items/s, "
                f"queue_depth={avg_queue_depth:.1f}, "
                f"data_quality={avg_data_quality:.3f}, "
                f"success_rate={success_rate:.3f}"
            )

    async def _cleanup_old_metrics(self, current_time: datetime) -> None:
        """Clean up metrics older than retention period."""
        cutoff_time = current_time - timedelta(hours=self.retention_hours)

        # Clean prompt metrics
        original_prompt_count = len(self.prompt_metrics)
        self.prompt_metrics = deque(
            (m for m in self.prompt_metrics if m.timestamp > cutoff_time),
            maxlen=self.prompt_metrics.maxlen
        )

        # Clean inference metrics
        original_inference_count = len(self.inference_metrics)
        self.inference_metrics = deque(
            (m for m in self.inference_metrics if m.timestamp > cutoff_time),
            maxlen=self.inference_metrics.maxlen
        )

        # Clean feature flag metrics
        original_flag_count = len(self.feature_flag_metrics)
        self.feature_flag_metrics = deque(
            (m for m in self.feature_flag_metrics if m.timestamp > cutoff_time),
            maxlen=self.feature_flag_metrics.maxlen
        )

        # Clean pipeline metrics
        original_pipeline_count = len(self.pipeline_metrics)
        self.pipeline_metrics = deque(
            (m for m in self.pipeline_metrics if m.timestamp > cutoff_time),
            maxlen=self.pipeline_metrics.maxlen
        )

        cleaned_total = (
            (original_prompt_count - len(self.prompt_metrics)) +
            (original_inference_count - len(self.inference_metrics)) +
            (original_flag_count - len(self.feature_flag_metrics)) +
            (original_pipeline_count - len(self.pipeline_metrics))
        )

        if cleaned_total > 0:
            self.logger.debug(f"Cleaned up {cleaned_total} old ML metrics")

    async def get_prompt_improvement_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get summary of prompt improvement metrics."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_metrics = [m for m in self.prompt_metrics if m.timestamp > cutoff_time]

        if not recent_metrics:
            return {"status": "no_data", "hours": hours}

        # Group by category
        category_stats = {}
        for category in PromptCategory:
            category_metrics = [m for m in recent_metrics if m.category == category]
            if category_metrics:
                category_stats[category.value] = {
                    "count": len(category_metrics),
                    "success_rate": sum(1 for m in category_metrics if m.success) / len(category_metrics),
                    "avg_improvement_ratio": statistics.mean(m.improvement_ratio for m in category_metrics),
                    "avg_processing_time_ms": statistics.mean(m.processing_time_ms for m in category_metrics),
                    "avg_confidence": statistics.mean(m.confidence_score for m in category_metrics)
                }

        return {
            "total_improvements": len(recent_metrics),
            "overall_success_rate": sum(1 for m in recent_metrics if m.success) / len(recent_metrics),
            "category_breakdown": category_stats,
            "time_window_hours": hours,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

    async def get_model_performance_summary(self, hours: int = 1) -> Dict[str, Any]:
        """Get summary of model inference performance."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_metrics = [m for m in self.inference_metrics if m.timestamp > cutoff_time]

        if not recent_metrics:
            return {"status": "no_data", "hours": hours}

        # Group by model
        model_stats: Dict[str, List[ModelInferenceMetric]] = {}
        for metric in recent_metrics:
            model_name = metric.model_name
            if model_name not in model_stats:
                model_stats[model_name] = []
            model_stats[model_name].append(metric)

        model_summary = {}
        for model_name, metrics in model_stats.items():
            total_tokens = sum(m.input_tokens + m.output_tokens for m in metrics)
            total_time_seconds = sum(m.latency_ms for m in metrics) / 1000.0

            model_summary[model_name] = {
                "inference_count": len(metrics),
                "total_tokens": total_tokens,
                "tokens_per_second": total_tokens / total_time_seconds if total_time_seconds > 0 else 0,
                "avg_latency_ms": statistics.mean(m.latency_ms for m in metrics),
                "success_rate": sum(1 for m in metrics if m.success) / len(metrics),
                "avg_memory_usage_mb": statistics.mean(m.memory_usage_mb for m in metrics),
                "confidence_stats": {
                    "mean": statistics.mean([conf for m in metrics for conf in m.confidence_distribution]),
                    "min": min([conf for m in metrics for conf in m.confidence_distribution]),
                    "max": max([conf for m in metrics for conf in m.confidence_distribution])
                } if any(m.confidence_distribution for m in metrics) else None
            }

        return {
            "total_inferences": len(recent_metrics),
            "model_breakdown": model_summary,
            "time_window_hours": hours,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

    async def get_feature_flag_effectiveness(self, hours: int = 24) -> Dict[str, Any]:
        """Get summary of feature flag effectiveness."""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours)
        recent_metrics = [m for m in self.feature_flag_metrics if m.timestamp > cutoff_time]

        if not recent_metrics:
            return {"status": "no_data", "hours": hours}

        # Group by flag
        flag_stats: Dict[str, List[FeatureFlagMetric]] = {}
        for metric in recent_metrics:
            flag_name = metric.flag_name
            if flag_name not in flag_stats:
                flag_stats[flag_name] = []
            flag_stats[flag_name].append(metric)

        flag_summary = {}
        for flag_name, metrics in flag_stats.items():
            enabled_metrics = [m for m in metrics if m.enabled]

            flag_summary[flag_name] = {
                "total_exposures": len(metrics),
                "enabled_exposures": len(enabled_metrics),
                "adoption_rate": len(enabled_metrics) / len(metrics) if metrics else 0,
                "avg_rollout_percentage": statistics.mean(m.rollout_percentage for m in metrics),
                "avg_performance_impact_ms": statistics.mean(m.performance_impact_ms for m in metrics),
                "success_rate": statistics.mean(m.success_rate for m in metrics),
                "error_rate": sum(m.error_count for m in metrics) / len(metrics) if metrics else 0,
                "conversion_rate": statistics.mean(
                    m.conversion_rate for m in metrics if m.conversion_rate is not None
                ) if any(m.conversion_rate is not None for m in metrics) else None
            }

        return {
            "total_flag_exposures": len(recent_metrics),
            "flag_breakdown": flag_summary,
            "time_window_hours": hours,
            "generated_at": datetime.now(timezone.utc).isoformat()
        }

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get current collection statistics."""
        # Get base class stats and merge with ML-specific stats
        base_stats = super().get_collection_stats()
        base_stats.update({
            **self.ml_stats,
            "current_metrics_count": {
                "prompt_improvements": len(self._metrics_storage["prompt"]),
                "model_inferences": len(self._metrics_storage["inference"]),
                "feature_flags": len(self._metrics_storage["feature_flag"]),
                "pipeline_operations": len(self._metrics_storage["ml_pipeline"])
            }
        })
        return base_stats

def get_ml_metrics_collector(
    config: Optional[Union[Dict[str, Any], MetricsConfig]] = None
) -> MLMetricsCollector:
    """Get global ML metrics collector instance using modern factory pattern."""
    from .base_metrics_collector import get_or_create_collector
    collector = get_or_create_collector(MLMetricsCollector, config)
    return collector  # type: ignore[return-value]

async def record_prompt_improvement(
    category: PromptCategory,
    original_length: int,
    improved_length: int,
    improvement_ratio: float,
    success: bool,
    processing_time_ms: float,
    confidence_score: float,
    rule_combination_count: int = 1,
    user_id: Optional[str] = None,
    session_id: Optional[str] = None,
    quality_metrics: Optional[Dict[str, float]] = None,
    feature_flags_used: Optional[List[str]] = None
) -> None:
    """Record a prompt improvement metric (convenience function)."""
    collector = get_ml_metrics_collector()
    metric = PromptImprovementMetric(
        category=category,
        original_length=original_length,
        improved_length=improved_length,
        improvement_ratio=improvement_ratio,
        success=success,
        processing_time_ms=processing_time_ms,
        confidence_score=confidence_score,
        rule_combination_count=rule_combination_count,
        user_id=user_id,
        session_id=session_id,
        timestamp=datetime.now(timezone.utc),
        quality_metrics=quality_metrics or {},
        feature_flags_used=feature_flags_used or []
    )
    await collector.record_prompt_improvement(metric)

async def record_model_inference(
    model_name: str,
    inference_stage: ModelInferenceStage,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float,
    memory_usage_mb: float,
    batch_size: int = 1,
    success: bool = True,
    error_type: Optional[str] = None,
    gpu_utilization_percent: Optional[float] = None,
    confidence_distribution: Optional[List[float]] = None,
    request_id: Optional[str] = None
) -> None:
    """Record a model inference metric (convenience function)."""
    collector = get_ml_metrics_collector()
    metric = ModelInferenceMetric(
        model_name=model_name,
        inference_stage=inference_stage,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        memory_usage_mb=memory_usage_mb,
        gpu_utilization_percent=gpu_utilization_percent,
        batch_size=batch_size,
        success=success,
        error_type=error_type,
        confidence_distribution=confidence_distribution or [],
        timestamp=datetime.now(timezone.utc),
        request_id=request_id or f"req_{int(time.time() * 1000)}"
    )
    await collector.record_model_inference(metric)
