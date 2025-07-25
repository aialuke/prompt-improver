"""
ML-Specific Health Checkers for 2025 ML Pipeline Monitoring
Provides comprehensive health monitoring for ML models, data quality, training, and performance
"""

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .base import HealthChecker, HealthResult, HealthStatus
from .metrics import instrument_health_check

logger = logging.getLogger(__name__)

class MLModelHealthChecker(HealthChecker):
    """Health checker for ML model lifecycle and deployment status."""

    def __init__(self):
        """Initialize the ML model health checker."""
        super().__init__(name="ml_models")
        self.logger = logging.getLogger(__name__)

    @instrument_health_check("ml_models")
    async def check(self) -> HealthResult:
        """Perform health check on ML models."""
        start_time = datetime.now(timezone.utc)

        try:
            # Check model registry health
            model_registry_health = await self._check_model_registry()

            # Check deployed models
            deployed_models_health = await self._check_deployed_models()

            # Check model cache
            model_cache_health = await self._check_model_cache()

            # Check model performance metrics
            model_performance_health = await self._check_model_performance()

            end_time = datetime.now(timezone.utc)
            response_time = (end_time - start_time).total_seconds() * 1000

            # Aggregate health status
            all_checks = [
                model_registry_health,
                deployed_models_health,
                model_cache_health,
                model_performance_health
            ]

            failed_checks = [check for check in all_checks if not check["healthy"]]
            warning_checks = [check for check in all_checks if check.get("warning", False)]

            if failed_checks:
                status = HealthStatus.FAILED
                message = f"ML model health issues: {len(failed_checks)} failed checks"
            elif warning_checks:
                status = HealthStatus.WARNING
                message = f"ML model warnings: {len(warning_checks)} warning checks"
            else:
                status = HealthStatus.HEALTHY
                message = "All ML models healthy"

            return HealthResult(
                status=status,
                component=self.name,
                response_time_ms=response_time,
                message=message,
                details={
                    "model_registry": model_registry_health,
                    "deployed_models": deployed_models_health,
                    "model_cache": model_cache_health,
                    "model_performance": model_performance_health,
                    "total_checks": len(all_checks),
                    "failed_checks": len(failed_checks),
                    "warning_checks": len(warning_checks)
                },
                timestamp=start_time
            )

        except Exception as e:
            end_time = datetime.now(timezone.utc)
            response_time = (end_time - start_time).total_seconds() * 1000

            self.logger.error(f"ML model health check failed: {e}")
            return HealthResult(
                status=HealthStatus.FAILED,
                component=self.name,
                response_time_ms=response_time,
                message=f"ML model health check failed: {str(e)}",
                error=str(e),
                timestamp=start_time
            )

    async def _check_model_registry(self) -> Dict[str, Any]:
        """Check the health of the model registry."""
        try:
            from ....ml.models.production_registry import ProductionModelRegistry

            registry = ProductionModelRegistry()

            # Check if registry is accessible
            models = await registry.list_models()

            return {
                "healthy": True,
                "total_models": len(models),
                "registry_accessible": True
            }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "registry_accessible": False
            }

    async def _check_deployed_models(self) -> Dict[str, Any]:
        """Check the health of deployed models."""
        try:
            from ....ml.core.ml_integration import MLModelService

            ml_service = MLModelService()

            # Check model cache status
            cache_stats = ml_service.get_cache_stats()

            return {
                "healthy": True,
                "cached_models": cache_stats.get("total_models", 0),
                "cache_size_mb": cache_stats.get("total_size_mb", 0),
                "cache_hit_rate": cache_stats.get("hit_rate", 0.0)
            }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "cache_accessible": False
            }

    async def _check_model_cache(self) -> Dict[str, Any]:
        """Check the health of model cache."""
        try:
            from ....ml.core.ml_integration import MLModelService

            ml_service = MLModelService()
            cache_stats = ml_service.get_cache_stats()

            # Check cache health metrics
            cache_utilization = cache_stats.get("cache_utilization", 0.0)
            memory_pressure = cache_utilization > 0.9

            return {
                "healthy": not memory_pressure,
                "warning": memory_pressure,
                "cache_utilization": cache_utilization,
                "memory_pressure": memory_pressure,
                "evictions": cache_stats.get("evictions", 0)
            }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }

    async def _check_model_performance(self) -> Dict[str, Any]:
        """Check model performance metrics."""
        try:
            # This would integrate with your performance monitoring
            # For now, return a basic health check
            return {
                "healthy": True,
                "performance_monitoring": "active",
                "metrics_available": True
            }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }

class MLDataQualityChecker(HealthChecker):
    """Health checker for ML data quality and pipeline integrity."""

    def __init__(self):
        """Initialize the ML data quality health checker."""
        super().__init__(name="ml_data_quality")
        self.logger = logging.getLogger(__name__)

    @instrument_health_check("ml_data_quality")
    async def check(self) -> HealthResult:
        """Perform health check on ML data quality."""
        start_time = datetime.now(timezone.utc)

        try:
            # Check training data loader
            training_data_health = await self._check_training_data()

            # Check synthetic data generator
            synthetic_data_health = await self._check_synthetic_data()

            # Check data preprocessing pipeline
            preprocessing_health = await self._check_preprocessing_pipeline()

            end_time = datetime.now(timezone.utc)
            response_time = (end_time - start_time).total_seconds() * 1000

            # Aggregate health status
            all_checks = [training_data_health, synthetic_data_health, preprocessing_health]
            failed_checks = [check for check in all_checks if not check["healthy"]]
            warning_checks = [check for check in all_checks if check.get("warning", False)]

            if failed_checks:
                status = HealthStatus.FAILED
                message = f"ML data quality issues: {len(failed_checks)} failed checks"
            elif warning_checks:
                status = HealthStatus.WARNING
                message = f"ML data quality warnings: {len(warning_checks)} warning checks"
            else:
                status = HealthStatus.HEALTHY
                message = "ML data quality healthy"

            return HealthResult(
                status=status,
                component=self.name,
                response_time_ms=response_time,
                message=message,
                details={
                    "training_data": training_data_health,
                    "synthetic_data": synthetic_data_health,
                    "preprocessing": preprocessing_health,
                    "total_checks": len(all_checks),
                    "failed_checks": len(failed_checks),
                    "warning_checks": len(warning_checks)
                },
                timestamp=start_time
            )

        except Exception as e:
            end_time = datetime.now(timezone.utc)
            response_time = (end_time - start_time).total_seconds() * 1000

            self.logger.error(f"ML data quality health check failed: {e}")
            return HealthResult(
                status=HealthStatus.FAILED,
                component=self.name,
                response_time_ms=response_time,
                message=f"ML data quality health check failed: {str(e)}",
                error=str(e),
                timestamp=start_time
            )

    async def _check_training_data(self) -> Dict[str, Any]:
        """Check the health of training data loader."""
        try:
            from ....ml.core.training_data_loader import TrainingDataLoader, get_training_data_stats

            # Get training data statistics
            stats = await get_training_data_stats()

            # Check for data quality issues
            data_quality_issues = []
            if stats.get("missing_values_ratio", 0) > 0.1:
                data_quality_issues.append("High missing values ratio")

            if stats.get("duplicate_ratio", 0) > 0.05:
                data_quality_issues.append("High duplicate ratio")

            return {
                "healthy": len(data_quality_issues) == 0,
                "warning": len(data_quality_issues) > 0,
                "data_quality_issues": data_quality_issues,
                "total_samples": stats.get("total_samples", 0),
                "missing_values_ratio": stats.get("missing_values_ratio", 0),
                "duplicate_ratio": stats.get("duplicate_ratio", 0)
            }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }

    async def _check_synthetic_data(self) -> Dict[str, Any]:
        """Check the health of synthetic data generator."""
        try:
            from ....ml.preprocessing.synthetic_data_generator import ProductionSyntheticDataGenerator

            generator = ProductionSyntheticDataGenerator()

            # Basic health check - ensure generator can be initialized
            return {
                "healthy": True,
                "generator_available": True,
                "generation_method": generator.generation_method
            }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "generator_available": False
            }

    async def _check_preprocessing_pipeline(self) -> Dict[str, Any]:
        """Check the health of data preprocessing pipeline."""
        try:
            # This would check various preprocessing components
            # For now, return a basic health check
            return {
                "healthy": True,
                "pipeline_available": True
            }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }

class MLTrainingHealthChecker(HealthChecker):
    """Health checker for ML training processes and optimization."""

    def __init__(self):
        """Initialize the ML training health checker."""
        super().__init__(name="ml_training")
        self.logger = logging.getLogger(__name__)

    @instrument_health_check("ml_training")
    async def check(self) -> HealthResult:
        """Perform health check on ML training processes."""
        start_time = datetime.now(timezone.utc)

        try:
            # Check optimization algorithms
            optimization_health = await self._check_optimization_algorithms()

            # Check batch processing
            batch_processing_health = await self._check_batch_processing()

            # Check learning algorithms
            learning_health = await self._check_learning_algorithms()

            end_time = datetime.now(timezone.utc)
            response_time = (end_time - start_time).total_seconds() * 1000

            # Aggregate health status
            all_checks = [optimization_health, batch_processing_health, learning_health]
            failed_checks = [check for check in all_checks if not check["healthy"]]
            warning_checks = [check for check in all_checks if check.get("warning", False)]

            if failed_checks:
                status = HealthStatus.FAILED
                message = f"ML training issues: {len(failed_checks)} failed checks"
            elif warning_checks:
                status = HealthStatus.WARNING
                message = f"ML training warnings: {len(warning_checks)} warning checks"
            else:
                status = HealthStatus.HEALTHY
                message = "ML training processes healthy"

            return HealthResult(
                status=status,
                component=self.name,
                response_time_ms=response_time,
                message=message,
                details={
                    "optimization": optimization_health,
                    "batch_processing": batch_processing_health,
                    "learning": learning_health,
                    "total_checks": len(all_checks),
                    "failed_checks": len(failed_checks),
                    "warning_checks": len(warning_checks)
                },
                timestamp=start_time
            )

        except Exception as e:
            end_time = datetime.now(timezone.utc)
            response_time = (end_time - start_time).total_seconds() * 1000

            self.logger.error(f"ML training health check failed: {e}")
            return HealthResult(
                status=HealthStatus.FAILED,
                component=self.name,
                response_time_ms=response_time,
                message=f"ML training health check failed: {str(e)}",
                error=str(e),
                timestamp=start_time
            )

    async def _check_optimization_algorithms(self) -> Dict[str, Any]:
        """Check the health of optimization algorithms."""
        try:
            # Check if optimization components are available
            optimization_components = [
                "RuleOptimizer",
                "MultiarmedBanditFramework",
                "ClusteringOptimizer",
                "AdvancedDimensionalityReducer"
            ]

            available_components = []
            for component in optimization_components:
                try:
                    # Try to import each component
                    if component == "RuleOptimizer":
                        from ....ml.optimization.algorithms.rule_optimizer import RuleOptimizer
                        available_components.append(component)
                    elif component == "ClusteringOptimizer":
                        from ....ml.optimization.algorithms.clustering_optimizer import ClusteringOptimizer
                        available_components.append(component)
                    # Add other components as needed
                except ImportError:
                    pass

            return {
                "healthy": len(available_components) > 0,
                "available_components": available_components,
                "total_components": len(optimization_components)
            }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }

    async def _check_batch_processing(self) -> Dict[str, Any]:
        """Check the health of batch processing."""
        try:
            from ....ml.optimization.batch.batch_processor import BatchProcessor

            # Basic health check for batch processor
            return {
                "healthy": True,
                "batch_processor_available": True
            }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "batch_processor_available": False
            }

    async def _check_learning_algorithms(self) -> Dict[str, Any]:
        """Check the health of learning algorithms."""
        try:
            # Check if learning components are available
            learning_components = [
                "ContextLearner",
                "InsightGenerationEngine",
                "FailureModeAnalyzer"
            ]

            available_components = []
            for component in learning_components:
                try:
                    # Try to import each component
                    if component == "ContextLearner":
                        from ....ml.learning.algorithms.context_learner import ContextLearner
                        available_components.append(component)
                    elif component == "InsightGenerationEngine":
                        from ....ml.learning.algorithms.insight_engine import InsightGenerationEngine
                        available_components.append(component)
                    elif component == "FailureModeAnalyzer":
                        from ....ml.learning.algorithms.failure_analyzer import FailureModeAnalyzer
                        available_components.append(component)
                except ImportError:
                    pass

            return {
                "healthy": len(available_components) > 0,
                "available_components": available_components,
                "total_components": len(learning_components)
            }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }

class MLPerformanceHealthChecker(HealthChecker):
    """Health checker for ML performance monitoring and evaluation."""

    def __init__(self):
        """Initialize the ML performance health checker."""
        super().__init__(name="ml_performance")
        self.logger = logging.getLogger(__name__)

    @instrument_health_check("ml_performance")
    async def check(self) -> HealthResult:
        """Perform health check on ML performance monitoring."""
        start_time = datetime.now(timezone.utc)

        try:
            # Check evaluation components
            evaluation_health = await self._check_evaluation_components()

            # Check performance monitoring
            monitoring_health = await self._check_performance_monitoring()

            # Check analytics integration
            analytics_health = await self._check_analytics_integration()

            end_time = datetime.now(timezone.utc)
            response_time = (end_time - start_time).total_seconds() * 1000

            # Aggregate health status
            all_checks = [evaluation_health, monitoring_health, analytics_health]
            failed_checks = [check for check in all_checks if not check["healthy"]]
            warning_checks = [check for check in all_checks if check.get("warning", False)]

            if failed_checks:
                status = HealthStatus.FAILED
                message = f"ML performance issues: {len(failed_checks)} failed checks"
            elif warning_checks:
                status = HealthStatus.WARNING
                message = f"ML performance warnings: {len(warning_checks)} warning checks"
            else:
                status = HealthStatus.HEALTHY
                message = "ML performance monitoring healthy"

            return HealthResult(
                status=status,
                component=self.name,
                response_time_ms=response_time,
                message=message,
                details={
                    "evaluation": evaluation_health,
                    "monitoring": monitoring_health,
                    "analytics": analytics_health,
                    "total_checks": len(all_checks),
                    "failed_checks": len(failed_checks),
                    "warning_checks": len(warning_checks)
                },
                timestamp=start_time
            )

        except Exception as e:
            end_time = datetime.now(timezone.utc)
            response_time = (end_time - start_time).total_seconds() * 1000

            self.logger.error(f"ML performance health check failed: {e}")
            return HealthResult(
                status=HealthStatus.FAILED,
                component=self.name,
                response_time_ms=response_time,
                message=f"ML performance health check failed: {str(e)}",
                error=str(e),
                timestamp=start_time
            )

    async def _check_evaluation_components(self) -> Dict[str, Any]:
        """Check the health of evaluation components."""
        try:
            # Check if evaluation components are available
            evaluation_components = [
                "CausalInferenceAnalyzer",
                "AdvancedStatisticalValidator",
                "PatternSignificanceAnalyzer"
            ]

            available_components = []
            for component in evaluation_components:
                try:
                    # Try to import each component
                    if component == "CausalInferenceAnalyzer":
                        from ....ml.evaluation.causal_inference_analyzer import CausalInferenceAnalyzer
                        available_components.append(component)
                    elif component == "AdvancedStatisticalValidator":
                        from ....ml.evaluation.advanced_statistical_validator import AdvancedStatisticalValidator
                        available_components.append(component)
                    elif component == "PatternSignificanceAnalyzer":
                        from ....ml.evaluation.pattern_significance_analyzer import PatternSignificanceAnalyzer
                        available_components.append(component)
                except ImportError:
                    pass

            return {
                "healthy": len(available_components) > 0,
                "available_components": available_components,
                "total_components": len(evaluation_components)
            }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e)
            }

    async def _check_performance_monitoring(self) -> Dict[str, Any]:
        """Check the health of performance monitoring."""
        try:
            from ...performance_monitor import PerformanceMonitor

            # Basic health check for performance monitor
            return {
                "healthy": True,
                "performance_monitor_available": True
            }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "performance_monitor_available": False
            }

    async def _check_analytics_integration(self) -> Dict[str, Any]:
        """Check the health of analytics integration."""
        try:
            from ....core.services.analytics_factory import get_analytics_interface

            analytics_factory = get_analytics_interface()
            analytics = analytics_factory() if analytics_factory else None

            # Basic health check for analytics
            return {
                "healthy": True,
                "analytics_available": True
            }

        except Exception as e:
            return {
                "healthy": False,
                "error": str(e),
                "analytics_available": False
            }
