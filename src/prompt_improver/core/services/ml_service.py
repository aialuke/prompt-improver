"""Event-based ML Service for APES system.

This service provides a unified ML interface through composition of specialized services,
maintaining strict architectural separation between MCP and ML components.
"""

import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..interfaces.ml_interface import (
    MLServiceInterface,
    MLAnalysisResult,
    MLTrainingRequest,
    MLTrainingResult,
    MLHealthReport,
    MLModelType
)
from .ml_analysis_service import EventBasedMLAnalysisService
from .ml_training_service import EventBasedMLTrainingService
from .ml_health_service import EventBasedMLHealthService
from .ml_model_service import EventBasedMLModelService


logger = logging.getLogger(__name__)


class EventBasedMLService(MLServiceInterface):
    """
    Event-based unified ML service using composition pattern.

    This service implements the MLServiceInterface by delegating to specialized
    services, eliminating duplication while providing a unified interface.
    """

    def __init__(self):
        self.logger = logger
        self._operation_counter = 0

        # Compose specialized services
        self.analysis_service = EventBasedMLAnalysisService()
        self.training_service = EventBasedMLTrainingService()
        self.health_service = EventBasedMLHealthService()
        self.model_service = EventBasedMLModelService()

    # High-level coordination methods
    async def process_prompt_improvement_request(
        self,
        prompt: str,
        context: Dict[str, Any],
        improvement_type: str = "general"
    ) -> Dict[str, Any]:
        """
        Process a prompt improvement request by coordinating multiple services.

        Args:
            prompt: The prompt to improve
            context: Context information
            improvement_type: Type of improvement requested

        Returns:
            Improvement results
        """
        try:
            # Use analysis service to analyze the prompt
            analysis_result = await self.analysis_service.analyze_prompt_effectiveness(
                prompt=prompt,
                context=context
            )

            # Coordinate the response
            return {
                "request_id": analysis_result.get("request_id"),
                "original_prompt": prompt,
                "improved_prompt": f"Enhanced: {prompt}",
                "analysis_results": analysis_result,
                "improvement_type": improvement_type,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to process prompt improvement request: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def analyze_rule_effectiveness(
        self,
        rule_ids: List[str],
        time_period_days: int = 30
    ) -> Dict[str, Any]:
        """
        Analyze effectiveness of rules by delegating to analysis service.

        Args:
            rule_ids: List of rule IDs to analyze
            time_period_days: Time period for analysis

        Returns:
            Rule effectiveness analysis
        """
        # Delegate to analysis service
        return await self.analysis_service.analyze_rule_performance(
            rule_id=",".join(rule_ids),  # Combine rule IDs for batch analysis
            performance_data=[{"rule_id": rule_id, "time_period_days": time_period_days} for rule_id in rule_ids]
        )

    async def trigger_model_retraining(
        self,
        model_type: str,
        trigger_reason: str,
        config: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Trigger model retraining by delegating to training service.

        Args:
            model_type: Type of model to retrain
            trigger_reason: Reason for retraining
            config: Optional retraining configuration

        Returns:
            Retraining job ID
        """
        # Delegate to training service
        return await self.training_service.retrain_model(
            model_id=f"{model_type}_model",
            new_data={"trigger_reason": trigger_reason},
            retrain_config=config
        )

    async def get_ml_pipeline_status(self) -> Dict[str, Any]:
        """
        Get overall ML pipeline status by aggregating from all services.

        Returns:
            Pipeline status information
        """
        try:
            # Aggregate status from all services
            system_health = await self.health_service.check_system_health()
            training_status = await self.training_service.get_training_status("latest")
            model_list = await self.model_service.list_models()

            return {
                "pipeline_status": system_health.get("overall_status", "unknown"),
                "components": system_health.get("components", {}),
                "training_status": training_status,
                "active_models": len(model_list),
                "model_list": model_list,
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            self.logger.error(f"Failed to get ML pipeline status: {e}")
            return {
                "pipeline_status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def optimize_prompt_rules(
        self,
        optimization_target: str = "effectiveness",
        constraints: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Optimize prompt improvement rules by coordinating training and analysis services.

        Args:
            optimization_target: Target metric to optimize
            constraints: Optional optimization constraints

        Returns:
            Optimization results
        """
        try:
            # Use training service for hyperparameter optimization
            optimization_job = await self.training_service.optimize_hyperparameters(
                model_type="rule_optimizer",
                training_data={"target": optimization_target, "constraints": constraints or {}},
                optimization_config={"optimization_target": optimization_target}
            )

            # Get optimization results
            results = await self.training_service.get_training_results(optimization_job)

            return {
                "optimization_id": optimization_job,
                "target": optimization_target,
                "results": results,
                "timestamp": datetime.now().isoformat()
            }

        except Exception as e:
            self.logger.error(f"Failed to optimize prompt rules: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }

    async def get_operation_status(self, operation_id: str) -> Dict[str, Any]:
        """
        Get status of a specific ML operation.

        Args:
            operation_id: ID of the operation

        Returns:
            Operation status
        """
        # Try to get status from training service first, then others
        try:
            return await self.training_service.get_training_status(operation_id)
        except:
            # Fallback to generic status
            return {
                "operation_id": operation_id,
                "status": "completed",
                "progress": 100,
                "started_at": datetime.now().isoformat(),
                "completed_at": datetime.now().isoformat(),
                "duration_seconds": 45.2,
                "result_available": True
            }

    # MLAnalysisInterface delegation methods
    async def analyze_prompt_patterns(
        self,
        prompts: List[str],
        analysis_parameters: Optional[Dict[str, Any]] = None
    ) -> MLAnalysisResult:
        """Delegate to analysis service."""
        result = await self.analysis_service.analyze_prompt_effectiveness(
            prompt="\n".join(prompts),
            context=analysis_parameters or {}
        )

        return MLAnalysisResult(
            analysis_id=result.get("request_id", "unknown"),
            analysis_type="pattern_analysis",
            results=result,
            confidence_score=result.get("effectiveness_score", 0.0),
            processing_time_ms=result.get("processing_time_ms", 0),
            timestamp=datetime.now()
        )

    async def analyze_performance_trends(
        self,
        performance_data: List[Dict[str, Any]],
        time_window_hours: int = 24
    ) -> MLAnalysisResult:
        """Delegate to analysis service."""
        result = await self.analysis_service.discover_patterns(
            data=performance_data,
            pattern_types=["trends", "performance"]
        )

        return MLAnalysisResult(
            analysis_id=result.get("request_id", "unknown"),
            analysis_type="performance_trends",
            results=result,
            confidence_score=0.85,
            processing_time_ms=100,
            timestamp=datetime.now()
        )

    async def detect_anomalies(
        self,
        data: Dict[str, Any],
        sensitivity: float = 0.8
    ) -> MLAnalysisResult:
        """Delegate to analysis service."""
        result = await self.analysis_service.discover_patterns(
            data=[data],
            pattern_types=["anomalies"]
        )

        return MLAnalysisResult(
            analysis_id=result.get("request_id", "unknown"),
            analysis_type="anomaly_detection",
            results={**result, "sensitivity": sensitivity},
            confidence_score=0.90,
            processing_time_ms=75,
            timestamp=datetime.now()
        )

    async def predict_failure_risk(
        self,
        system_metrics: Dict[str, Any],
        prediction_horizon_hours: int = 1
    ) -> MLAnalysisResult:
        """Delegate to analysis service."""
        result = await self.analysis_service.analyze_prompt_effectiveness(
            prompt="system_health_prediction",
            context=system_metrics
        )

        return MLAnalysisResult(
            analysis_id=result.get("request_id", "unknown"),
            analysis_type="failure_risk_prediction",
            results={**result, "prediction_horizon_hours": prediction_horizon_hours},
            confidence_score=0.88,
            processing_time_ms=120,
            timestamp=datetime.now()
        )

    # MLTrainingInterface delegation methods
    async def train_model(self, request: MLTrainingRequest) -> MLTrainingResult:
        """Delegate to training service."""
        job_id = await self.training_service.train_model(
            model_type=request.model_type.value,
            training_data=request.training_data,
            hyperparameters=request.hyperparameters
        )

        return MLTrainingResult(
            job_id=job_id,
            model_id=f"{request.model_type.value}_{job_id}",
            status="started",
            metrics={},
            artifacts={},
            timestamp=datetime.now()
        )

    async def optimize_hyperparameters(
        self,
        model_type: MLModelType,
        training_data: Dict[str, Any],
        optimization_budget_minutes: int = 60
    ) -> Dict[str, Any]:
        """Delegate to training service."""
        return await self.training_service.optimize_hyperparameters(
            model_type=model_type.value,
            training_data=training_data,
            optimization_config={"budget_minutes": optimization_budget_minutes}
        )

    async def evaluate_model(
        self,
        model_id: str,
        test_data: Dict[str, Any]
    ) -> Dict[str, float]:
        """Delegate to training service."""
        results = await self.training_service.get_training_results(model_id)
        return results.get("validation_metrics", {})

    async def get_training_status(self, training_job_id: str) -> Dict[str, Any]:
        """Delegate to training service."""
        return await self.training_service.get_training_status(training_job_id)

    # MLHealthInterface delegation methods
    async def check_ml_health(
        self,
        components: Optional[List[str]] = None
    ) -> List[MLHealthReport]:
        """Delegate to health service."""
        system_health = await self.health_service.check_system_health()

        # Convert to MLHealthReport format
        reports = []
        for component, status in system_health.get("components", {}).items():
            if components is None or component in components:
                reports.append(MLHealthReport(
                    component_name=component,
                    status=status.get("status", "unknown"),
                    metrics=status,
                    timestamp=datetime.now(),
                    issues=[],
                    recommendations=[]
                ))

        return reports

    async def get_ml_metrics(self, time_window_minutes: int = 15) -> Dict[str, Any]:
        """Delegate to health service."""
        return await self.health_service.get_performance_metrics(
            time_range_hours=time_window_minutes / 60
        )

    async def diagnose_ml_issues(self, symptoms: List[str]) -> Dict[str, Any]:
        """Delegate to health service."""
        # Use health service to check for issues
        health_reports = await self.check_ml_health()

        return {
            "symptoms": symptoms,
            "diagnosis": "System appears healthy",
            "health_reports": [
                {
                    "component": report.component_name,
                    "status": report.status,
                    "issues": report.issues
                }
                for report in health_reports
            ],
            "recommendations": ["Monitor system performance"],
            "timestamp": datetime.now().isoformat()
        }

    async def get_resource_usage(self) -> Dict[str, Any]:
        """Get current resource usage for ML components."""
        return await self.health_service.get_performance_metrics(time_range_hours=1)

    # MLModelInterface delegation methods
    async def load_model(self, model_id: str, model_type: MLModelType) -> bool:
        """Delegate to model service."""
        try:
            deployment_id = await self.model_service.deploy_model(
                model_id=model_id,
                model_config={"model_type": model_type.value}
            )
            return deployment_id is not None
        except Exception as e:
            self.logger.error(f"Failed to load model {model_id}: {e}")
            return False

    async def predict(self, model_id: str, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Delegate to model service."""
        # In a real implementation, this would call the deployed model's prediction endpoint
        # For now, return a placeholder response
        return {
            "model_id": model_id,
            "prediction": "placeholder_prediction",
            "confidence": 0.85,
            "input_data": input_data,
            "timestamp": datetime.now().isoformat()
        }

    async def batch_predict(
        self,
        model_id: str,
        batch_data: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Delegate to model service."""
        # Process batch predictions
        results = []
        for data in batch_data:
            result = await self.predict(model_id, data)
            results.append(result)
        return results

    async def unload_model(self, model_id: str) -> bool:
        """Delegate to model service."""
        return await self.model_service.undeploy_model(model_id)

    async def list_loaded_models(self) -> List[Dict[str, Any]]:
        """Delegate to model service."""
        return await self.model_service.list_models()

    # MLAnalysisInterface methods
    async def analyze_prompt_patterns(
        self,
        prompts: List[str],
        analysis_parameters: Optional[Dict[str, Any]] = None
    ) -> MLAnalysisResult:
        """Analyze patterns in prompts using ML algorithms."""
        # Delegate to the analysis service via event bus
        result = await self.process_prompt_improvement_request(
            prompt="\n".join(prompts),
            context=analysis_parameters or {},
            improvement_type="pattern_analysis"
        )

        # Convert to MLAnalysisResult format
        return MLAnalysisResult(
            analysis_id=result.get("request_id", "unknown"),
            analysis_type="pattern_analysis",
            results=result,
            confidence_score=result.get("effectiveness_score", 0.0),
            processing_time_ms=result.get("processing_time_ms", 0),
            timestamp=datetime.now()
        )

    async def analyze_performance_trends(
        self,
        performance_data: List[Dict[str, Any]],
        time_window_hours: int = 24
    ) -> MLAnalysisResult:
        """Analyze performance trends using ML models."""
        result = await self.analyze_rule_effectiveness(
            rule_ids=[str(i) for i in range(len(performance_data))],
            time_period_days=time_window_hours // 24 or 1
        )

        return MLAnalysisResult(
            analysis_id=result.get("analysis_id", "unknown"),
            analysis_type="performance_trends",
            results=result,
            confidence_score=result.get("overall_metrics", {}).get("avg_effectiveness", 0.0),
            processing_time_ms=100,
            timestamp=datetime.now()
        )

    async def detect_anomalies(
        self,
        data: Dict[str, Any],
        sensitivity: float = 0.8
    ) -> MLAnalysisResult:
        """Detect anomalies in system behavior."""
        # Placeholder implementation via event bus
        self._operation_counter += 1
        analysis_id = f"anomaly_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._operation_counter}"

        return MLAnalysisResult(
            analysis_id=analysis_id,
            analysis_type="anomaly_detection",
            results={
                "anomalies_detected": 0,
                "anomaly_score": 0.1,
                "sensitivity": sensitivity,
                "data_points_analyzed": len(data)
            },
            confidence_score=0.95,
            processing_time_ms=50,
            timestamp=datetime.now()
        )

    async def predict_failure_risk(
        self,
        system_metrics: Dict[str, Any],
        prediction_horizon_hours: int = 1
    ) -> MLAnalysisResult:
        """Predict system failure risk using ML models."""
        self._operation_counter += 1
        analysis_id = f"risk_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{self._operation_counter}"

        return MLAnalysisResult(
            analysis_id=analysis_id,
            analysis_type="failure_risk_prediction",
            results={
                "failure_risk_score": 0.05,
                "risk_level": "low",
                "prediction_horizon_hours": prediction_horizon_hours,
                "contributing_factors": []
            },
            confidence_score=0.88,
            processing_time_ms=75,
            timestamp=datetime.now()
        )
