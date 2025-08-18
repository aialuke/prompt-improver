"""ML Application Service

Orchestrates machine learning workflows including training, inference, pattern discovery,
and model deployment while managing complex transaction boundaries and resource coordination.
"""

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from prompt_improver.application.protocols.application_service_protocols import (
    MLApplicationServiceProtocol,
)
if TYPE_CHECKING:
    from prompt_improver.database.composition import DatabaseServices
from prompt_improver.core.di.ml_container import MLServiceContainer
from prompt_improver.repositories.protocols.session_manager_protocol import (
    SessionManagerProtocol,
)
from prompt_improver.repositories.protocols.apriori_repository_protocol import (
    AprioriRepositoryProtocol,
)
from prompt_improver.core.domain.types import (
    AprioriAnalysisRequestData,
    AprioriAnalysisResponseData,
    PatternDiscoveryRequestData,
    PatternDiscoveryResponseData,
)
from prompt_improver.ml.core import MLModelService  # Now points to facade
from prompt_improver.ml.learning.patterns.advanced_pattern_discovery import (
    AdvancedPatternDiscovery,
)
from prompt_improver.ml.learning.patterns.apriori_analyzer import AprioriAnalyzer
from prompt_improver.repositories.protocols.ml_repository_protocol import (
    MLRepositoryProtocol,
)

logger = logging.getLogger(__name__)


class MLApplicationService:
    """
    Application service for ML training and inference workflows.
    
    Orchestrates complex ML processes including:
    - Training workflow coordination and monitoring  
    - Pattern discovery and analysis orchestration
    - Model deployment and version management
    - Inference pipeline coordination
    - Resource management and scaling
    - Transaction boundary management across ML operations
    """

    def __init__(
        self,
        db_services: "DatabaseServices",
        ml_repository: MLRepositoryProtocol,
        ml_service_container: MLServiceContainer,
        ml_model_service: MLModelService,
        apriori_analyzer: AprioriAnalyzer,
        pattern_discovery: AdvancedPatternDiscovery,
    ):
        self.db_services = db_services
        self.ml_repository = ml_repository
        self.ml_service_container = ml_service_container
        self.ml_model_service = ml_model_service
        self.apriori_analyzer = apriori_analyzer
        self.pattern_discovery = pattern_discovery
        self.logger = logger

    async def initialize(self) -> None:
        """Initialize the ML application service."""
        self.logger.info("Initializing MLApplicationService")
        await self.ml_service_container.initialize()

    async def cleanup(self) -> None:
        """Clean up ML application service resources."""
        self.logger.info("Cleaning up MLApplicationService")
        await self.ml_service_container.cleanup()

    async def execute_training_workflow(
        self,
        training_config: Dict[str, Any],
        session_id: str | None = None,
    ) -> Dict[str, Any]:
        """
        Execute a complete ML training workflow.
        
        Orchestrates the entire ML training process:
        1. Validate training configuration
        2. Initialize training resources and data pipelines
        3. Execute training with monitoring and checkpointing
        4. Validate model performance and metrics
        5. Store model artifacts and metadata
        6. Clean up training resources
        
        Args:
            training_config: Configuration for training workflow
            session_id: Optional session identifier for tracking
            
        Returns:
            Dict containing training results and metadata
        """
        workflow_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        try:
            self.logger.info(f"Starting training workflow {workflow_id}")
            
            # 1. Validate configuration
            config_validation = await self._validate_training_config(training_config)
            if not config_validation["valid"]:
                return {
                    "status": "error",
                    "error": config_validation["error"],
                    "workflow_id": workflow_id,
                    "timestamp": start_time.isoformat(),
                }
            
            # 2. Transaction boundary for entire training workflow
            async with self.db_services.get_session() as db_session:
                try:
                    # 3. Initialize training context
                    training_context = await self._initialize_training_context(
                        workflow_id, training_config, session_id, db_session
                    )
                    
                    # 4. Execute training via ML model service
                    training_result = await self.ml_model_service.execute_training(
                        training_config=training_config,
                        session_context=training_context,
                        db_session=db_session,
                    )
                    
                    # 5. Store training results and artifacts
                    await self._store_training_artifacts(
                        workflow_id, training_result, db_session
                    )
                    
                    # 6. Update training metrics and status
                    await self._update_training_status(
                        workflow_id, "completed", training_result, db_session
                    )
                    
                    await db_session.commit()
                    
                    end_time = datetime.now(timezone.utc)
                    duration_seconds = (end_time - start_time).total_seconds()
                    
                    return {
                        "status": "success",
                        "workflow_id": workflow_id,
                        "session_id": session_id,
                        "training_results": {
                            "model_id": training_result.get("model_id"),
                            "performance_metrics": training_result.get("metrics", {}),
                            "training_duration_seconds": duration_seconds,
                            "model_artifacts": training_result.get("artifacts", []),
                        },
                        "workflow_metadata": {
                            "started_at": start_time.isoformat(),
                            "completed_at": end_time.isoformat(),
                            "duration_seconds": duration_seconds,
                            "configuration": training_config,
                        },
                        "timestamp": end_time.isoformat(),
                    }
                    
                except Exception as e:
                    await db_session.rollback()
                    await self._update_training_status(
                        workflow_id, "failed", {"error": str(e)}, db_session
                    )
                    raise
                    
        except Exception as e:
            self.logger.error(f"Training workflow {workflow_id} failed: {e}")
            return {
                "status": "error",
                "workflow_id": workflow_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def execute_pattern_discovery(
        self,
        request: PatternDiscoveryRequestData,
        session_id: str | None = None,
    ) -> PatternDiscoveryResponseData:
        """
        Execute comprehensive pattern discovery workflow.
        
        Orchestrates advanced pattern discovery using multiple algorithms:
        1. Traditional ML parameter analysis
        2. HDBSCAN clustering for density-based patterns
        3. FP-Growth for frequent pattern mining
        4. Apriori for association rule mining
        5. Semantic analysis for rule relationships
        
        Args:
            request: Pattern discovery configuration
            session_id: Optional session identifier
            
        Returns:
            PatternDiscoveryResponseData with comprehensive patterns
        """
        discovery_run_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        try:
            self.logger.info(f"Starting pattern discovery {discovery_run_id}")
            
            # Transaction boundary for pattern discovery workflow
            async with self.db_services.get_session() as db_session:
                try:
                    # Execute comprehensive pattern discovery via ML service
                    discovery_results = await self.ml_model_service.discover_patterns(
                        db_session=db_session,
                        min_effectiveness=request.min_effectiveness,
                        min_support=request.min_support,
                        use_advanced_discovery=request.use_advanced_discovery,
                        include_apriori=request.include_apriori,
                    )
                    
                    # Validate discovery results
                    if discovery_results.get("status") == "error":
                        return PatternDiscoveryResponseData(
                            status="error",
                            discovery_run_id=discovery_run_id,
                            error_message=discovery_results.get("error"),
                            traditional_patterns=None,
                            advanced_patterns=None,
                            apriori_patterns=None,
                        )
                    
                    # Store pattern discovery metadata
                    await self._store_pattern_discovery_metadata(
                        discovery_run_id, request, discovery_results, db_session
                    )
                    
                    await db_session.commit()
                    
                    end_time = datetime.now(timezone.utc)
                    duration_seconds = (end_time - start_time).total_seconds()
                    
                    return PatternDiscoveryResponseData(
                        status=discovery_results.get("status", "success"),
                        discovery_run_id=discovery_run_id,
                        traditional_patterns=discovery_results.get("traditional_patterns"),
                        advanced_patterns=discovery_results.get("advanced_patterns"),
                        apriori_patterns=discovery_results.get("apriori_patterns"),
                        cross_validation=discovery_results.get("cross_validation"),
                        unified_recommendations=discovery_results.get("unified_recommendations", []),
                        business_insights=discovery_results.get("business_insights", {}),
                        discovery_metadata={
                            **discovery_results.get("discovery_metadata", {}),
                            "execution_time_seconds": duration_seconds,
                            "started_at": start_time.isoformat(),
                            "completed_at": end_time.isoformat(),
                            "session_id": session_id,
                        },
                    )
                    
                except Exception as e:
                    await db_session.rollback()
                    self.logger.error(f"Pattern discovery {discovery_run_id} failed: {e}")
                    raise
                    
        except Exception as e:
            self.logger.error(f"Pattern discovery workflow failed: {e}")
            return PatternDiscoveryResponseData(
                status="error",
                discovery_run_id=discovery_run_id,
                error_message=str(e),
                traditional_patterns=None,
                advanced_patterns=None,
                apriori_patterns=None,
            )

    async def execute_apriori_analysis(
        self,
        request: AprioriAnalysisRequestData,
        session_id: str | None = None,
    ) -> AprioriAnalysisResponseData:
        """
        Execute Apriori association rule mining workflow.
        
        Orchestrates comprehensive Apriori analysis:
        1. Data preparation and transaction building
        2. Frequent itemset mining
        3. Association rule generation
        4. Rule validation and filtering
        5. Business insight generation
        
        Args:
            request: Apriori analysis configuration
            session_id: Optional session identifier
            
        Returns:
            AprioriAnalysisResponseData with discovered patterns
        """
        analysis_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        try:
            self.logger.info(f"Starting Apriori analysis {analysis_id}")
            
            # Transaction boundary for Apriori analysis
            async with self.db_services.get_session() as db_session:
                try:
                    # Execute Apriori analysis via repository (which orchestrates the analyzer)
                    analysis_result = await self.ml_repository.execute_apriori_analysis(
                        request=request,
                        session_id=session_id,
                        db_session=db_session,
                    )
                    
                    # Store analysis metadata
                    await self._store_apriori_metadata(
                        analysis_id, request, analysis_result, db_session
                    )
                    
                    await db_session.commit()
                    
                    end_time = datetime.now(timezone.utc)
                    duration_seconds = (end_time - start_time).total_seconds()
                    
                    # Enhance result with workflow metadata
                    if isinstance(analysis_result, dict):
                        analysis_result["workflow_metadata"] = {
                            "analysis_id": analysis_id,
                            "execution_time_seconds": duration_seconds,
                            "started_at": start_time.isoformat(),
                            "completed_at": end_time.isoformat(),
                            "session_id": session_id,
                        }
                    
                    return analysis_result
                    
                except Exception as e:
                    await db_session.rollback()
                    self.logger.error(f"Apriori analysis {analysis_id} failed: {e}")
                    raise
                    
        except Exception as e:
            self.logger.error(f"Apriori analysis workflow failed: {e}")
            return AprioriAnalysisResponseData(
                status="error",
                analysis_id=analysis_id,
                error_message=str(e),
                transaction_count=0,
                frequent_itemsets=[],
                association_rules=[],
                execution_time_seconds=(datetime.now(timezone.utc) - start_time).total_seconds(),
            )

    async def deploy_model(
        self,
        model_id: str,
        deployment_config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Deploy a trained model to production.
        
        Orchestrates model deployment workflow:
        1. Validate model and deployment configuration
        2. Prepare deployment environment
        3. Deploy model with proper versioning
        4. Validate deployment and run health checks
        5. Update model registry and routing
        
        Args:
            model_id: Model identifier to deploy
            deployment_config: Deployment configuration
            
        Returns:
            Dict containing deployment results
        """
        deployment_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        try:
            self.logger.info(f"Starting model deployment {deployment_id} for model {model_id}")
            
            # Transaction boundary for deployment
            async with self.db_services.get_session() as db_session:
                try:
                    # 1. Validate model exists and is ready for deployment
                    model_validation = await self._validate_model_for_deployment(
                        model_id, db_session
                    )
                    if not model_validation["valid"]:
                        return {
                            "status": "error",
                            "deployment_id": deployment_id,
                            "error": model_validation["error"],
                            "timestamp": start_time.isoformat(),
                        }
                    
                    # 2. Execute deployment via ML service
                    deployment_result = await self.ml_model_service.deploy_model(
                        model_id=model_id,
                        deployment_config=deployment_config,
                        db_session=db_session,
                    )
                    
                    # 3. Store deployment metadata
                    await self._store_deployment_metadata(
                        deployment_id, model_id, deployment_config, deployment_result, db_session
                    )
                    
                    await db_session.commit()
                    
                    end_time = datetime.now(timezone.utc)
                    duration_seconds = (end_time - start_time).total_seconds()
                    
                    return {
                        "status": "success",
                        "deployment_id": deployment_id,
                        "model_id": model_id,
                        "deployment_results": deployment_result,
                        "deployment_metadata": {
                            "started_at": start_time.isoformat(),
                            "completed_at": end_time.isoformat(),
                            "duration_seconds": duration_seconds,
                            "configuration": deployment_config,
                        },
                        "timestamp": end_time.isoformat(),
                    }
                    
                except Exception as e:
                    await db_session.rollback()
                    raise
                    
        except Exception as e:
            self.logger.error(f"Model deployment {deployment_id} failed: {e}")
            return {
                "status": "error",
                "deployment_id": deployment_id,
                "model_id": model_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    async def execute_inference(
        self,
        model_id: str,
        input_data: Dict[str, Any],
        inference_config: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Execute model inference with proper error handling.
        
        Orchestrates inference workflow:
        1. Validate model availability and input data
        2. Prepare input data and apply preprocessing
        3. Execute model inference
        4. Apply post-processing and validation
        5. Log inference metrics and results
        
        Args:
            model_id: Model identifier for inference
            input_data: Input data for inference
            inference_config: Optional inference configuration
            
        Returns:
            Dict containing inference results
        """
        inference_id = str(uuid.uuid4())
        start_time = datetime.now(timezone.utc)
        
        try:
            self.logger.info(f"Executing inference {inference_id} with model {model_id}")
            
            # Lightweight validation without database transaction for performance
            input_validation = await self._validate_inference_input(input_data)
            if not input_validation["valid"]:
                return {
                    "status": "error",
                    "inference_id": inference_id,
                    "error": input_validation["error"],
                    "timestamp": start_time.isoformat(),
                }
            
            # Execute inference (typically no database transaction needed)
            inference_result = await self.ml_model_service.execute_inference(
                model_id=model_id,
                input_data=input_data,
                config=inference_config or {},
            )
            
            # Log inference metrics asynchronously
            await self._log_inference_metrics(
                inference_id, model_id, input_data, inference_result
            )
            
            end_time = datetime.now(timezone.utc)
            duration_ms = (end_time - start_time).total_seconds() * 1000
            
            return {
                "status": "success",
                "inference_id": inference_id,
                "model_id": model_id,
                "predictions": inference_result.get("predictions"),
                "confidence_scores": inference_result.get("confidence_scores"),
                "inference_metadata": {
                    "processing_time_ms": duration_ms,
                    "model_version": inference_result.get("model_version"),
                    "timestamp": end_time.isoformat(),
                    "configuration": inference_config or {},
                },
            }
            
        except Exception as e:
            self.logger.error(f"Inference {inference_id} failed: {e}")
            return {
                "status": "error",
                "inference_id": inference_id,
                "model_id": model_id,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

    # Private helper methods

    async def _validate_training_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate training configuration."""
        try:
            required_fields = ["model_type", "training_data", "hyperparameters"]
            for field in required_fields:
                if field not in config:
                    return {"valid": False, "error": f"Missing required field: {field}"}
            return {"valid": True, "error": None}
        except Exception as e:
            return {"valid": False, "error": str(e)}

    async def _initialize_training_context(
        self, workflow_id: str, config: Dict[str, Any], session_id: str | None, db_session
    ) -> Dict[str, Any]:
        """Initialize training context and resources."""
        return {
            "workflow_id": workflow_id,
            "session_id": session_id,
            "training_config": config,
            "initialized_at": datetime.now(timezone.utc).isoformat(),
        }

    async def _store_training_artifacts(
        self, workflow_id: str, training_result: Dict[str, Any], db_session
    ) -> None:
        """Store training artifacts and metadata."""
        await self.ml_repository.store_training_artifacts(
            workflow_id=workflow_id,
            artifacts=training_result.get("artifacts", []),
            metadata=training_result.get("metadata", {}),
            db_session=db_session,
        )

    async def _update_training_status(
        self, workflow_id: str, status: str, result: Dict[str, Any], db_session
    ) -> None:
        """Update training workflow status."""
        await self.ml_repository.update_training_status(
            workflow_id=workflow_id,
            status=status,
            result=result,
            db_session=db_session,
        )

    async def _store_pattern_discovery_metadata(
        self, discovery_run_id: str, request: PatternDiscoveryRequestData, results: Dict[str, Any], db_session
    ) -> None:
        """Store pattern discovery metadata."""
        await self.ml_repository.store_pattern_discovery_metadata(
            discovery_run_id=discovery_run_id,
            request=request,
            results=results,
            db_session=db_session,
        )

    async def _store_apriori_metadata(
        self, analysis_id: str, request: AprioriAnalysisRequestData, results: Any, db_session
    ) -> None:
        """Store Apriori analysis metadata."""
        await self.ml_repository.store_apriori_metadata(
            analysis_id=analysis_id,
            request=request,
            results=results,
            db_session=db_session,
        )

    async def _validate_model_for_deployment(
        self, model_id: str, db_session
    ) -> Dict[str, Any]:
        """Validate model readiness for deployment."""
        try:
            model_exists = await self.ml_repository.check_model_exists(model_id, db_session)
            if not model_exists:
                return {"valid": False, "error": f"Model {model_id} not found"}
            return {"valid": True, "error": None}
        except Exception as e:
            return {"valid": False, "error": str(e)}

    async def _store_deployment_metadata(
        self, deployment_id: str, model_id: str, config: Dict[str, Any], result: Dict[str, Any], db_session
    ) -> None:
        """Store deployment metadata."""
        await self.ml_repository.store_deployment_metadata(
            deployment_id=deployment_id,
            model_id=model_id,
            config=config,
            result=result,
            db_session=db_session,
        )

    async def _validate_inference_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate inference input data."""
        try:
            if not input_data:
                return {"valid": False, "error": "Empty input data"}
            return {"valid": True, "error": None}
        except Exception as e:
            return {"valid": False, "error": str(e)}

    async def _log_inference_metrics(
        self, inference_id: str, model_id: str, input_data: Dict[str, Any], result: Dict[str, Any]
    ) -> None:
        """Log inference metrics asynchronously."""
        try:
            # This would typically be sent to a metrics collection system
            self.logger.info(
                f"Inference metrics: {inference_id}, model: {model_id}, "
                f"input_size: {len(str(input_data))}, success: {result.get('status') == 'success'}"
            )
        except Exception as e:
            self.logger.error(f"Failed to log inference metrics: {e}")