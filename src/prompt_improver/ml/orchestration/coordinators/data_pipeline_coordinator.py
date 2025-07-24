"""
Data Pipeline Coordinator for ML Pipeline Orchestration.

Coordinates data flow and preprocessing across ML components.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timezone

from ..events.event_types import EventType, MLEvent

@dataclass
class DataPipelineConfig:
    """Configuration for data pipeline workflows."""
    default_timeout: int = 1200  # 20 minutes
    max_batch_size: int = 10000
    data_validation_enabled: bool = True
    preprocessing_workers: int = 4

class DataPipelineCoordinator:
    """
    Coordinates data flow and preprocessing across ML components.
    
    Manages data ingestion, preprocessing, transformation, and validation
    across all tiers of the ML pipeline.
    """
    
    def __init__(self, config: DataPipelineConfig, event_bus=None, resource_manager=None):
        """Initialize the data pipeline coordinator."""
        self.config = config
        self.event_bus = event_bus
        self.resource_manager = resource_manager
        self.logger = logging.getLogger(__name__)
        
        # Active data pipelines
        self.active_pipelines: Dict[str, Dict[str, Any]] = {}
        
        # Data flow tracking
        self.data_flows: Dict[str, List[str]] = {}
        
    async def start_data_pipeline(self, pipeline_id: str, parameters: Dict[str, Any]) -> None:
        """Start a new data pipeline workflow."""
        self.logger.info(f"Starting data pipeline {pipeline_id}")
        
        # Validate pipeline parameters
        await self._validate_pipeline_parameters(parameters)
        
        # Register pipeline
        self.active_pipelines[pipeline_id] = {
            "status": "running",
            "started_at": datetime.now(timezone.utc),
            "parameters": parameters,
            "current_step": None,
            "data_processed": 0,
            "validation_results": {},
            "transformation_history": []
        }
        
        try:
            # Step 1: Data ingestion
            await self._execute_data_ingestion(pipeline_id, parameters)
            
            # Step 2: Data validation
            if self.config.data_validation_enabled:
                await self._execute_data_validation(pipeline_id, parameters)
            
            # Step 3: Data preprocessing
            await self._execute_data_preprocessing(pipeline_id, parameters)
            
            # Step 4: Data transformation
            await self._execute_data_transformation(pipeline_id, parameters)
            
            # Step 5: Data distribution to components
            await self._execute_data_distribution(pipeline_id, parameters)
            
            # Mark pipeline as completed
            self.active_pipelines[pipeline_id]["status"] = "completed"
            self.active_pipelines[pipeline_id]["completed_at"] = datetime.now(timezone.utc)
            
            if self.event_bus:
                await self.event_bus.emit(MLEvent(
                    event_type=EventType.DATA_PIPELINE_COMPLETED,
                    source="data_pipeline_coordinator",
                    data={
                        "pipeline_id": pipeline_id,
                        "data_processed": self.active_pipelines[pipeline_id]["data_processed"],
                        "duration": (datetime.now(timezone.utc) - self.active_pipelines[pipeline_id]["started_at"]).total_seconds()
                    }
                ))
            
            self.logger.info(f"Data pipeline {pipeline_id} completed successfully")
            
        except Exception as e:
            await self._handle_pipeline_failure(pipeline_id, e)
            raise
    
    async def _validate_pipeline_parameters(self, parameters: Dict[str, Any]) -> None:
        """Validate data pipeline parameters."""
        required_params = ["data_source", "target_components"]
        missing_params = [param for param in required_params if param not in parameters]
        
        if missing_params:
            raise ValueError(f"Missing required pipeline parameters: {missing_params}")
        
        # Validate data source
        data_source = parameters.get("data_source")
        if not isinstance(data_source, (str, dict)):
            raise ValueError("data_source must be a string path or dict configuration")
        
        # Validate target components
        target_components = parameters.get("target_components", [])
        if not isinstance(target_components, list) or not target_components:
            raise ValueError("target_components must be a non-empty list")
    
    async def _execute_data_ingestion(self, pipeline_id: str, parameters: Dict[str, Any]) -> None:
        """Execute data ingestion step."""
        self.logger.info(f"Executing data ingestion for pipeline {pipeline_id}")
        self.active_pipelines[pipeline_id]["current_step"] = "data_ingestion"
        
        if self.event_bus:
            await self.event_bus.emit(MLEvent(
                event_type=EventType.DATA_INGESTION_STARTED,
                source="data_pipeline_coordinator",
                data={
                    "pipeline_id": pipeline_id,
                    "data_source": parameters.get("data_source")
                }
            ))
        
        # Simulate data ingestion
        await asyncio.sleep(0.1)
        
        # Track ingested data size
        simulated_data_size = parameters.get("data_size", 5000)
        self.active_pipelines[pipeline_id]["data_processed"] = simulated_data_size
        
        self.logger.info(f"Data ingestion completed for pipeline {pipeline_id}")
    
    async def _execute_data_validation(self, pipeline_id: str, parameters: Dict[str, Any]) -> None:
        """Execute data validation step."""
        self.logger.info(f"Executing data validation for pipeline {pipeline_id}")
        self.active_pipelines[pipeline_id]["current_step"] = "data_validation"
        
        # Simulate data validation checks
        await asyncio.sleep(0.1)
        
        validation_results = {
            "schema_valid": True,
            "data_quality_score": 0.92,
            "missing_values": 0.05,
            "outliers_detected": 12,
            "validation_timestamp": datetime.now(timezone.utc)
        }
        
        self.active_pipelines[pipeline_id]["validation_results"] = validation_results
        
        # Check if validation passed
        if validation_results["data_quality_score"] < 0.8:
            raise Exception(f"Data quality score {validation_results['data_quality_score']} below threshold 0.8")
        
        if self.event_bus:
            await self.event_bus.emit(MLEvent(
                event_type=EventType.DATA_VALIDATION_COMPLETED,
                source="data_pipeline_coordinator",
                data={
                    "pipeline_id": pipeline_id,
                    "validation_results": validation_results
                }
            ))
        
        self.logger.info(f"Data validation completed for pipeline {pipeline_id}")
    
    async def _execute_data_preprocessing(self, pipeline_id: str, parameters: Dict[str, Any]) -> None:
        """Execute data preprocessing step."""
        self.logger.info(f"Executing data preprocessing for pipeline {pipeline_id}")
        self.active_pipelines[pipeline_id]["current_step"] = "data_preprocessing"
        
        # Simulate preprocessing operations
        preprocessing_steps = [
            "data_cleaning",
            "feature_extraction", 
            "normalization",
            "encoding"
        ]
        
        for step in preprocessing_steps:
            await asyncio.sleep(0.05)  # Simulate processing time
            
            step_result = {
                "step": step,
                "timestamp": datetime.now(timezone.utc),
                "processed_records": self.active_pipelines[pipeline_id]["data_processed"],
                "status": "completed"
            }
            
            self.active_pipelines[pipeline_id]["transformation_history"].append(step_result)
        
        self.logger.info(f"Data preprocessing completed for pipeline {pipeline_id}")
    
    async def _execute_data_transformation(self, pipeline_id: str, parameters: Dict[str, Any]) -> None:
        """Execute data transformation step."""
        self.logger.info(f"Executing data transformation for pipeline {pipeline_id}")
        self.active_pipelines[pipeline_id]["current_step"] = "data_transformation"
        
        # Simulate advanced transformations
        transformations = parameters.get("transformations", ["feature_scaling", "dimensionality_reduction"])
        
        for transformation in transformations:
            await asyncio.sleep(0.05)
            
            transformation_result = {
                "transformation": transformation,
                "timestamp": datetime.now(timezone.utc),
                "output_features": 150,  # Simulated feature count
                "compression_ratio": 0.75
            }
            
            self.active_pipelines[pipeline_id]["transformation_history"].append(transformation_result)
        
        self.logger.info(f"Data transformation completed for pipeline {pipeline_id}")
    
    async def _execute_data_distribution(self, pipeline_id: str, parameters: Dict[str, Any]) -> None:
        """Execute data distribution to target components."""
        self.logger.info(f"Executing data distribution for pipeline {pipeline_id}")
        self.active_pipelines[pipeline_id]["current_step"] = "data_distribution"
        
        target_components = parameters.get("target_components", [])
        
        # Track data flow to components
        self.data_flows[pipeline_id] = target_components
        
        for component in target_components:
            await asyncio.sleep(0.02)  # Simulate distribution time
            
            if self.event_bus:
                await self.event_bus.emit(MLEvent(
                    event_type=EventType.DATA_DISTRIBUTED,
                    source="data_pipeline_coordinator",
                    data={
                        "pipeline_id": pipeline_id,
                        "target_component": component,
                        "data_size": self.active_pipelines[pipeline_id]["data_processed"]
                    }
                ))
        
        self.logger.info(f"Data distribution completed for pipeline {pipeline_id}")
    
    async def _handle_pipeline_failure(self, pipeline_id: str, error: Exception) -> None:
        """Handle data pipeline failure."""
        self.logger.error(f"Data pipeline {pipeline_id} failed: {error}")
        
        self.active_pipelines[pipeline_id]["status"] = "failed"
        self.active_pipelines[pipeline_id]["error"] = str(error)
        self.active_pipelines[pipeline_id]["completed_at"] = datetime.now(timezone.utc)
        
        if self.event_bus:
            await self.event_bus.emit(MLEvent(
                event_type=EventType.DATA_PIPELINE_FAILED,
                source="data_pipeline_coordinator",
                data={
                    "pipeline_id": pipeline_id,
                    "error_message": str(error),
                    "current_step": self.active_pipelines[pipeline_id]["current_step"]
                }
            ))
    
    async def stop_pipeline(self, pipeline_id: str) -> None:
        """Stop a running data pipeline."""
        if pipeline_id not in self.active_pipelines:
            raise ValueError(f"Data pipeline {pipeline_id} not found")
        
        self.active_pipelines[pipeline_id]["status"] = "stopped"
        self.active_pipelines[pipeline_id]["completed_at"] = datetime.now(timezone.utc)
        
        self.logger.info(f"Data pipeline {pipeline_id} stopped")
    
    async def get_pipeline_status(self, pipeline_id: str) -> Dict[str, Any]:
        """Get the status of a data pipeline."""
        if pipeline_id not in self.active_pipelines:
            raise ValueError(f"Data pipeline {pipeline_id} not found")
        
        return self.active_pipelines[pipeline_id].copy()
    
    async def list_active_pipelines(self) -> List[str]:
        """List all active data pipelines."""
        return [
            pipe_id for pipe_id, pipe_data in self.active_pipelines.items()
            if pipe_data["status"] == "running"
        ]
    
    async def get_data_flow_map(self) -> Dict[str, List[str]]:
        """Get the data flow mapping for all pipelines."""
        return self.data_flows.copy()
    
    async def get_pipeline_metrics(self, pipeline_id: str) -> Optional[Dict[str, Any]]:
        """Get metrics for a specific pipeline."""
        if pipeline_id not in self.active_pipelines:
            return None
        
        pipeline_data = self.active_pipelines[pipeline_id]
        
        return {
            "data_processed": pipeline_data["data_processed"],
            "processing_time": (
                pipeline_data.get("completed_at", datetime.now(timezone.utc)) - 
                pipeline_data["started_at"]
            ).total_seconds(),
            "validation_results": pipeline_data.get("validation_results", {}),
            "transformation_count": len(pipeline_data.get("transformation_history", [])),
            "target_components": len(self.data_flows.get(pipeline_id, []))
        }