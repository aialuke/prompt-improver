"""Protocol interfaces for MLRepositoryFacade decomposition.

Domain-specific repository patterns following Clean Architecture principles.
Each repository handles a specific domain with focused responsibilities.
"""

from typing import Any, Dict, List, Optional, Protocol, runtime_checkable
from datetime import datetime
from uuid import UUID

from prompt_improver.core.domain.types import (
    GenerationAnalyticsData,
    GenerationBatchData,
    GenerationMethodPerformanceData,
    GenerationQualityAssessmentData,
    GenerationSessionData,
    MLModelPerformanceData,
    SyntheticDataSampleData,
    TrainingIterationData,
    TrainingPromptData,
    TrainingSessionData,
    TrainingSessionCreateData,
    TrainingSessionUpdateData,
)


@runtime_checkable
class TrainingRepositoryProtocol(Protocol):
    """Protocol for training session and iteration management."""
    
    async def create_training_session(
        self,
        session_data: TrainingSessionCreate
    ) -> TrainingSession:
        """Create a new training session.
        
        Args:
            session_data: Training session creation data
            
        Returns:
            Created training session
        """
        ...
    
    async def get_training_session(
        self,
        session_id: UUID
    ) -> Optional[TrainingSession]:
        """Get training session by ID.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Training session or None
        """
        ...
    
    async def update_training_session(
        self,
        session_id: UUID,
        update_data: TrainingSessionUpdate
    ) -> Optional[TrainingSession]:
        """Update training session.
        
        Args:
            session_id: Session identifier
            update_data: Update data
            
        Returns:
            Updated session or None
        """
        ...
    
    async def create_training_iteration(
        self,
        session_id: UUID,
        iteration_data: Dict[str, Any]
    ) -> TrainingIteration:
        """Create training iteration.
        
        Args:
            session_id: Parent session ID
            iteration_data: Iteration data
            
        Returns:
            Created iteration
        """
        ...
    
    async def get_training_iterations(
        self,
        session_id: UUID,
        limit: Optional[int] = None,
        offset: Optional[int] = None
    ) -> List[TrainingIteration]:
        """Get training iterations for a session.
        
        Args:
            session_id: Session identifier
            limit: Result limit
            offset: Result offset
            
        Returns:
            List of training iterations
        """
        ...
    
    async def get_active_training_sessions(
        self,
        limit: int = 10
    ) -> List[TrainingSession]:
        """Get active training sessions.
        
        Args:
            limit: Maximum number of sessions
            
        Returns:
            List of active sessions
        """
        ...
    
    async def get_training_metrics(
        self,
        session_id: UUID
    ) -> Dict[str, Any]:
        """Get training metrics for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Training metrics
        """
        ...
    
    async def cleanup_old_sessions(
        self,
        days_old: int = 30
    ) -> int:
        """Clean up old training sessions.
        
        Args:
            days_old: Age threshold in days
            
        Returns:
            Number of sessions cleaned
        """
        ...


@runtime_checkable
class ModelRepositoryProtocol(Protocol):
    """Protocol for model performance and versioning management."""
    
    async def create_model_performance(
        self,
        model_data: Dict[str, Any]
    ) -> MLModelPerformance:
        """Create model performance record.
        
        Args:
            model_data: Model performance data
            
        Returns:
            Created performance record
        """
        ...
    
    async def get_model_performance(
        self,
        model_id: UUID
    ) -> Optional[MLModelPerformance]:
        """Get model performance by ID.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Model performance or None
        """
        ...
    
    async def get_latest_model_version(
        self,
        model_type: str
    ) -> Optional[str]:
        """Get latest model version for a type.
        
        Args:
            model_type: Type of model
            
        Returns:
            Latest version string or None
        """
        ...
    
    async def get_model_versions(
        self,
        model_type: str,
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get model versions for a type.
        
        Args:
            model_type: Type of model
            limit: Maximum versions to return
            
        Returns:
            List of model versions with metadata
        """
        ...
    
    async def compare_model_performance(
        self,
        model_id1: UUID,
        model_id2: UUID
    ) -> Dict[str, Any]:
        """Compare performance of two models.
        
        Args:
            model_id1: First model ID
            model_id2: Second model ID
            
        Returns:
            Comparison metrics
        """
        ...
    
    async def update_model_metrics(
        self,
        model_id: UUID,
        metrics: Dict[str, float]
    ) -> bool:
        """Update model metrics.
        
        Args:
            model_id: Model identifier
            metrics: New metrics
            
        Returns:
            Success status
        """
        ...
    
    async def get_model_deployment_status(
        self,
        model_id: UUID
    ) -> Dict[str, Any]:
        """Get model deployment status.
        
        Args:
            model_id: Model identifier
            
        Returns:
            Deployment status and metadata
        """
        ...
    
    async def archive_old_models(
        self,
        days_old: int = 90
    ) -> int:
        """Archive old model versions.
        
        Args:
            days_old: Age threshold
            
        Returns:
            Number of models archived
        """
        ...


@runtime_checkable
class AnalyticsRepositoryProtocol(Protocol):
    """Protocol for generation analytics and metrics management."""
    
    async def create_generation_session(
        self,
        session_data: Dict[str, Any]
    ) -> GenerationSession:
        """Create generation session.
        
        Args:
            session_data: Session data
            
        Returns:
            Created session
        """
        ...
    
    async def create_generation_analytics(
        self,
        analytics_data: Dict[str, Any]
    ) -> GenerationAnalytics:
        """Create generation analytics record.
        
        Args:
            analytics_data: Analytics data
            
        Returns:
            Created analytics record
        """
        ...
    
    async def get_generation_analytics(
        self,
        session_id: UUID,
        time_range: Optional[tuple[datetime, datetime]] = None
    ) -> List[GenerationAnalytics]:
        """Get generation analytics for a session.
        
        Args:
            session_id: Session identifier
            time_range: Optional time range filter
            
        Returns:
            List of analytics records
        """
        ...
    
    async def create_quality_assessment(
        self,
        assessment_data: Dict[str, Any]
    ) -> GenerationQualityAssessment:
        """Create quality assessment.
        
        Args:
            assessment_data: Assessment data
            
        Returns:
            Created assessment
        """
        ...
    
    async def get_quality_assessments(
        self,
        generation_id: UUID
    ) -> List[GenerationQualityAssessment]:
        """Get quality assessments for a generation.
        
        Args:
            generation_id: Generation identifier
            
        Returns:
            List of quality assessments
        """
        ...
    
    async def create_synthetic_data_sample(
        self,
        sample_data: Dict[str, Any]
    ) -> SyntheticDataSample:
        """Create synthetic data sample.
        
        Args:
            sample_data: Sample data
            
        Returns:
            Created sample
        """
        ...
    
    async def get_synthetic_data_metrics(
        self,
        time_range: Optional[tuple[datetime, datetime]] = None
    ) -> Dict[str, Any]:
        """Get synthetic data generation metrics.
        
        Args:
            time_range: Optional time range
            
        Returns:
            Synthetic data metrics
        """
        ...
    
    async def aggregate_analytics(
        self,
        aggregation_type: str,
        filters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Aggregate analytics data.
        
        Args:
            aggregation_type: Type of aggregation
            filters: Optional filters
            
        Returns:
            Aggregated metrics
        """
        ...


@runtime_checkable
class MLRepositoryFacadeProtocol(Protocol):
    """Protocol for unified MLRepositoryFacade."""
    
    @property
    def training(self) -> TrainingRepositoryProtocol:
        """Get training repository."""
        ...
    
    @property
    def models(self) -> ModelRepositoryProtocol:
        """Get model repository."""
        ...
    
    @property
    def analytics(self) -> AnalyticsRepositoryProtocol:
        """Get analytics repository."""
        ...
    
    async def get_comprehensive_metrics(
        self,
        session_id: Optional[UUID] = None,
        model_id: Optional[UUID] = None
    ) -> Dict[str, Any]:
        """Get comprehensive metrics across repositories.
        
        Args:
            session_id: Optional session filter
            model_id: Optional model filter
            
        Returns:
            Comprehensive metrics
        """
        ...
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all repositories.
        
        Returns:
            Health status of each repository
        """
        ...
    
    async def cleanup_all(
        self,
        days_old: int = 30
    ) -> Dict[str, int]:
        """Cleanup old data across all repositories.
        
        Args:
            days_old: Age threshold
            
        Returns:
            Cleanup counts per repository
        """
        ...