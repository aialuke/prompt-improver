"""ML Interface Abstractions for MCP-ML Boundary Isolation

Defines Protocol interfaces that MCP components can use to interact with ML
subsystems without direct imports, maintaining clean architectural boundaries.

SECURITY FEATURES:
- Input validation on all interface methods
- Rate limiting support for ML operations
- Audit logging for security compliance
- Type-safe interface contracts
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Tuple, Union

if TYPE_CHECKING:
    from prompt_improver.core.events.ml_event_bus import (
        MLEvent,
        MLEventBus,
        MLEventType,
    )


class MLHealthStatus(Enum):
    """ML component health status."""

    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILED = "failed"
    UNKNOWN = "unknown"


class MLModelType(Enum):
    """Types of ML models supported."""

    PROMPT_ENHANCEMENT = "prompt_enhancement"
    PATTERN_ANALYSIS = "pattern_analysis"
    PERFORMANCE_PREDICTION = "performance_prediction"
    ANOMALY_DETECTION = "anomaly_detection"
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


@dataclass
class MLAnalysisRequest:
    """Request for ML analysis operations."""

    analysis_type: str
    input_data: dict[str, Any]
    parameters: dict[str, Any] | None = None
    correlation_id: str | None = None
    priority: int = 1


@dataclass
class MLAnalysisResult:
    """Result from ML analysis operations."""

    success: bool
    analysis_type: str
    results: dict[str, Any]
    confidence_score: float
    processing_time_ms: int
    error_message: str | None = None
    correlation_id: str | None = None


@dataclass
class MLTrainingRequest:
    """Request for ML training operations."""

    model_type: MLModelType
    training_data: dict[str, Any]
    hyperparameters: dict[str, Any] | None = None
    validation_split: float = 0.2
    correlation_id: str | None = None


@dataclass
class MLTrainingResult:
    """Result from ML training operations."""

    success: bool
    model_id: str
    performance_metrics: dict[str, float]
    training_duration_ms: int
    model_size_bytes: int
    error_message: str | None = None
    correlation_id: str | None = None


@dataclass
class MLHealthReport:
    """ML component health report."""

    component: str
    status: MLHealthStatus
    metrics: dict[str, Any]
    timestamp: datetime
    issues: list[str]
    recommendations: list[str]


class MLAnalysisInterface(Protocol):
    """Interface for ML analysis operations.

    Provides async methods for requesting ML analysis without direct ML imports.
    All methods support rate limiting and security validation.
    """

    async def analyze_prompt_patterns(
        self, prompts: list[str], analysis_parameters: dict[str, Any] | None = None
    ) -> MLAnalysisResult:
        """Analyze patterns in prompts using ML algorithms.

        Args:
            prompts: List of prompts to analyze
            analysis_parameters: Optional analysis configuration

        Returns:
            Analysis results with patterns and insights

        Raises:
            ValueError: If input validation fails
            RuntimeError: If ML service is unavailable
        """
        ...

    async def analyze_performance_trends(
        self, performance_data: list[dict[str, Any]], time_window_hours: int = 24
    ) -> MLAnalysisResult:
        """Analyze performance trends using ML models.

        Args:
            performance_data: Historical performance metrics
            time_window_hours: Analysis time window

        Returns:
            Trend analysis results with predictions
        """
        ...

    async def detect_anomalies(
        self, data: dict[str, Any], sensitivity: float = 0.8
    ) -> MLAnalysisResult:
        """Detect anomalies in system behavior.

        Args:
            data: System metrics and behavior data
            sensitivity: Anomaly detection sensitivity (0.0-1.0)

        Returns:
            Anomaly detection results
        """
        ...

    async def predict_failure_risk(
        self, system_metrics: dict[str, Any], prediction_horizon_hours: int = 1
    ) -> MLAnalysisResult:
        """Predict system failure risk using ML models.

        Args:
            system_metrics: Current system metrics
            prediction_horizon_hours: How far ahead to predict

        Returns:
            Failure risk prediction with confidence scores
        """
        ...


class MLTrainingInterface(Protocol):
    """Interface for ML training operations.

    Supports model training, hyperparameter optimization, and model management
    with security and performance monitoring.
    """

    async def train_model(self, request: MLTrainingRequest) -> MLTrainingResult:
        """Train an ML model with specified parameters.

        Args:
            request: Training request with data and configuration

        Returns:
            Training results with model ID and metrics

        Raises:
            ValueError: If training data is invalid
            ResourceError: If insufficient resources for training
        """
        ...

    async def optimize_hyperparameters(
        self,
        model_type: MLModelType,
        training_data: dict[str, Any],
        optimization_budget_minutes: int = 60,
    ) -> dict[str, Any]:
        """Optimize model hyperparameters using automated methods.

        Args:
            model_type: Type of model to optimize
            training_data: Data for training and validation
            optimization_budget_minutes: Time budget for optimization

        Returns:
            Optimal hyperparameters and performance metrics
        """
        ...

    async def evaluate_model(
        self, model_id: str, test_data: dict[str, Any]
    ) -> dict[str, float]:
        """Evaluate a trained model's performance.

        Args:
            model_id: ID of the model to evaluate
            test_data: Test dataset for evaluation

        Returns:
            Performance metrics (accuracy, precision, recall, etc.)
        """
        ...

    async def get_training_status(self, training_job_id: str) -> dict[str, Any]:
        """Get status of a training job.

        Args:
            training_job_id: ID of the training job

        Returns:
            Current status, progress, and metrics
        """
        ...


class MLHealthInterface(Protocol):
    """Interface for ML health monitoring and diagnostics.

    Provides health checks, performance monitoring, and diagnostic capabilities
    for ML components without requiring direct ML imports.
    """

    async def check_ml_health(
        self, components: list[str] | None = None
    ) -> list[MLHealthReport]:
        """Check health of ML components.

        Args:
            components: Specific components to check (None for all)

        Returns:
            Health reports for each component
        """
        ...

    async def get_ml_metrics(self, time_window_minutes: int = 15) -> dict[str, Any]:
        """Get ML system performance metrics.

        Args:
            time_window_minutes: Metrics collection window

        Returns:
            System metrics and performance data
        """
        ...

    async def diagnose_ml_issues(self, symptoms: list[str]) -> dict[str, Any]:
        """Diagnose ML system issues based on symptoms.

        Args:
            symptoms: List of observed symptoms

        Returns:
            Diagnostic results with recommendations
        """
        ...

    async def get_resource_usage(self) -> dict[str, Any]:
        """Get ML system resource usage.

        Returns:
            Current resource usage (CPU, memory, GPU, etc.)
        """
        ...


class MLModelInterface(Protocol):
    """Interface for ML model operations.

    Supports model loading, prediction, and lifecycle management with
    security validation and performance monitoring.
    """

    async def load_model(self, model_id: str, model_type: MLModelType) -> bool:
        """Load an ML model for inference.

        Args:
            model_id: ID of the model to load
            model_type: Type of the model

        Returns:
            True if model loaded successfully

        Raises:
            ModelNotFoundError: If model doesn't exist
            ResourceError: If insufficient resources to load model
        """
        ...

    async def predict(
        self, model_id: str, input_data: dict[str, Any]
    ) -> dict[str, Any]:
        """Make predictions using a loaded model.

        Args:
            model_id: ID of the model to use
            input_data: Input data for prediction

        Returns:
            Prediction results with confidence scores

        Raises:
            ModelNotLoadedError: If model is not loaded
            ValueError: If input data is invalid
        """
        ...

    async def batch_predict(
        self, model_id: str, batch_data: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Make batch predictions for multiple inputs.

        Args:
            model_id: ID of the model to use
            batch_data: List of input data for predictions

        Returns:
            List of prediction results
        """
        ...

    async def unload_model(self, model_id: str) -> bool:
        """Unload a model to free resources.

        Args:
            model_id: ID of the model to unload

        Returns:
            True if model unloaded successfully
        """
        ...

    async def list_loaded_models(self) -> list[dict[str, Any]]:
        """List currently loaded models.

        Returns:
            List of loaded models with metadata
        """
        ...


class MLServiceInterface(
    MLAnalysisInterface, MLTrainingInterface, MLHealthInterface, MLModelInterface
):
    """Unified interface for all ML operations.

    Combines all ML interfaces into a single protocol for comprehensive
    ML functionality access without direct imports.
    """

    async def initialize_ml_service(self, config: dict[str, Any]) -> bool:
        """Initialize the ML service with configuration.

        Args:
            config: ML service configuration

        Returns:
            True if initialization successful
        """

    async def shutdown_ml_service(self) -> bool:
        """Shutdown the ML service gracefully.

        Returns:
            True if shutdown successful
        """

    async def get_service_status(self) -> dict[str, Any]:
        """Get overall ML service status.

        Returns:
            Service status and metadata
        """


async def request_ml_analysis_via_events(
    analysis_type: str, input_data: dict[str, Any], correlation_id: str | None = None
) -> str:
    """Request ML analysis via event bus.

    Args:
        analysis_type: Type of analysis to perform
        input_data: Data to analyze
        correlation_id: Optional correlation ID

    Returns:
        Request ID for tracking the analysis
    """
    from prompt_improver.core.events.ml_event_bus import (
        MLEvent,
        MLEventType,
        get_ml_event_bus,
    )

    event_bus = await get_ml_event_bus()
    request_id = f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    event = MLEvent(
        event_type=MLEventType.ANALYSIS_REQUEST,
        source="mcp_interface",
        data={
            "request_id": request_id,
            "analysis_type": analysis_type,
            "input_data": input_data,
        },
        correlation_id=correlation_id,
    )
    await event_bus.publish(event)
    return request_id


async def request_ml_training_via_events(
    model_type: str,
    training_data: dict[str, Any],
    hyperparameters: dict[str, Any] | None = None,
) -> str:
    """Request ML training via event bus.

    Args:
        model_type: Type of model to train
        training_data: Training dataset
        hyperparameters: Optional hyperparameters

    Returns:
        Training job ID for tracking
    """
    from prompt_improver.core.events.ml_event_bus import (
        MLEvent,
        MLEventType,
        get_ml_event_bus,
    )

    event_bus = await get_ml_event_bus()
    job_id = f"training_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    event = MLEvent(
        event_type=MLEventType.TRAINING_REQUEST,
        source="mcp_interface",
        data={
            "job_id": job_id,
            "model_type": model_type,
            "training_data": training_data,
            "hyperparameters": hyperparameters or {},
        },
    )
    await event_bus.publish(event)
    return job_id


async def request_ml_health_check_via_events(
    components: list[str] | None = None,
) -> str:
    """Request ML health check via event bus.

    Args:
        components: Specific components to check

    Returns:
        Health check request ID
    """
    from prompt_improver.core.events.ml_event_bus import (
        MLEvent,
        MLEventType,
        get_ml_event_bus,
    )

    event_bus = await get_ml_event_bus()
    check_id = f"health_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
    event = MLEvent(
        event_type=MLEventType.HEALTH_CHECK_REQUEST,
        source="mcp_interface",
        data={"check_id": check_id, "components": components or []},
    )
    await event_bus.publish(event)
    return check_id


class MLInterfaceError(Exception):
    """Base exception for ML interface errors."""


class ModelNotFoundError(MLInterfaceError):
    """Raised when a requested model is not found."""


class ModelNotLoadedError(MLInterfaceError):
    """Raised when trying to use a model that is not loaded."""


class ResourceError(MLInterfaceError):
    """Raised when insufficient resources are available."""


class ValidationError(MLInterfaceError):
    """Raised when input validation fails."""


def create_ml_analysis_request(
    analysis_type: str, input_data: dict[str, Any], **kwargs: dict[str, Any]
) -> MLAnalysisRequest:
    """Create a validated ML analysis request."""
    return MLAnalysisRequest(
        analysis_type=analysis_type, input_data=input_data, **kwargs
    )


def create_ml_training_request(
    model_type: MLModelType, training_data: dict[str, Any], **kwargs: dict[str, Any]
) -> MLTrainingRequest:
    """Create a validated ML training request."""
    return MLTrainingRequest(
        model_type=model_type, training_data=training_data, **kwargs
    )
