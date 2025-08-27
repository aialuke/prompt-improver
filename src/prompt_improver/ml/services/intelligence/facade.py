"""ML Intelligence Service Facade.

Unified interface for ML intelligence processing services.
Replaces the 1,319-line intelligence_processor.py god object with clean architecture.

Performance Target: <200ms for complete intelligence processing
Memory Target: <150MB for service coordination
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from prompt_improver.ml.services.intelligence.protocols.intelligence_service_protocols import (
    MLIntelligenceServiceFacadeProtocol,
    IntelligenceResult,
    MLCircuitBreakerServiceProtocol,
    RuleAnalysisServiceProtocol,
    PatternDiscoveryServiceProtocol,
    MLPredictionServiceProtocol,
    BatchProcessingServiceProtocol,
)
from prompt_improver.ml.services.intelligence.circuit_breaker_service import MLCircuitBreakerService
from prompt_improver.ml.services.intelligence.rule_analysis_service import RuleAnalysisService
from prompt_improver.ml.services.intelligence.pattern_discovery_service import PatternDiscoveryService
from prompt_improver.ml.services.intelligence.prediction_service import MLPredictionService
from prompt_improver.ml.services.intelligence.batch_processing_service import BatchProcessingService
from prompt_improver.repositories.protocols.ml_repository_protocol import MLRepositoryProtocol
from prompt_improver.ml.learning.patterns.advanced_pattern_discovery import AdvancedPatternDiscovery
from prompt_improver.performance.monitoring.metrics_registry import (
    StandardMetrics,
    get_metrics_registry,
)

logger = logging.getLogger(__name__)


class MLIntelligenceServiceFacade:
    """ML Intelligence Service Facade.
    
    Provides unified interface for ML intelligence processing with clean architecture.
    Orchestrates multiple focused services to replace the intelligence processor god object.
    """
    
    def __init__(
        self,
        ml_repository: MLRepositoryProtocol,
        pattern_discovery: AdvancedPatternDiscovery
    ):
        """Initialize ML intelligence service facade.
        
        Args:
            ml_repository: ML repository for data access
            pattern_discovery: Advanced pattern discovery component
        """
        self._ml_repository = ml_repository
        self._metrics_registry = get_metrics_registry()
        self._background_task: Optional[asyncio.Task] = None
        self._is_running = False
        
        # Initialize services with dependency injection
        self._circuit_breaker_service = MLCircuitBreakerService()
        
        self._rule_analysis_service = RuleAnalysisService(
            ml_repository=ml_repository,
            circuit_breaker_service=self._circuit_breaker_service,
            pattern_discovery=pattern_discovery
        )
        
        self._pattern_discovery_service = PatternDiscoveryService(
            pattern_discovery=pattern_discovery,
            ml_repository=ml_repository,
            circuit_breaker_service=self._circuit_breaker_service
        )
        
        self._prediction_service = MLPredictionService(
            ml_repository=ml_repository,
            circuit_breaker_service=self._circuit_breaker_service
        )
        
        self._batch_processing_service = BatchProcessingService(
            ml_repository=ml_repository,
            circuit_breaker_service=self._circuit_breaker_service
        )
        
        logger.info("MLIntelligenceServiceFacade initialized with all services")
    
    async def run_intelligence_processing(
        self,
        rule_ids: Optional[List[str]] = None,
        enable_patterns: bool = True,
        enable_predictions: bool = True,
        batch_size: int = 25
    ) -> IntelligenceResult:
        """Execute complete ML intelligence processing pipeline.
        
        Args:
            rule_ids: Specific rule IDs to process (None for all)
            enable_patterns: Whether to enable pattern discovery
            enable_predictions: Whether to enable ML predictions
            batch_size: Batch size for processing
            
        Returns:
            Intelligence result with comprehensive analysis
        """
        start_time = datetime.now(timezone.utc)
        
        try:
            # Setup circuit breakers for all components
            await self._setup_circuit_breakers()
            
            # Initialize results container
            intelligence_results = {
                "rule_intelligence": {},
                "combination_intelligence": {},
                "pattern_insights": {},
                "predictions": {},
                "processing_metadata": {
                    "started_at": start_time.isoformat(),
                    "rule_ids_requested": rule_ids,
                    "patterns_enabled": enable_patterns,
                    "predictions_enabled": enable_predictions,
                    "batch_size": batch_size
                }
            }
            
            # Execute intelligence processing phases in parallel
            processing_tasks = []
            
            # Phase 1: Rule Intelligence Analysis
            rule_intelligence_task = asyncio.create_task(
                self._rule_analysis_service.process_rule_intelligence(rule_ids)
            )
            processing_tasks.append(("rule_intelligence", rule_intelligence_task))
            
            # Phase 2: Combination Intelligence Analysis
            combination_task = asyncio.create_task(
                self._rule_analysis_service.process_combination_intelligence()
            )
            processing_tasks.append(("combination_intelligence", combination_task))
            
            # Phase 3: Pattern Discovery (if enabled)
            if enable_patterns:
                # Get data for pattern discovery
                batch_data = await self._get_pattern_discovery_data(rule_ids, batch_size)
                
                if batch_data:
                    pattern_task = asyncio.create_task(
                        self._pattern_discovery_service.discover_patterns(batch_data)
                    )
                    processing_tasks.append(("pattern_insights", pattern_task))
            
            # Execute all phases in parallel
            completed_tasks = await asyncio.gather(
                *[task for _, task in processing_tasks],
                return_exceptions=True
            )
            
            # Process results
            overall_confidence = 0.0
            successful_phases = 0
            total_processing_time = 0.0
            
            for i, (phase_name, _) in enumerate(processing_tasks):
                result = completed_tasks[i]
                
                if isinstance(result, Exception):
                    logger.error(f"Phase {phase_name} failed: {result}")
                    intelligence_results[phase_name] = {
                        "success": False,
                        "error": str(result),
                        "data": {}
                    }
                else:
                    intelligence_results[phase_name] = result.__dict__ if hasattr(result, '__dict__') else result
                    if result.success:
                        successful_phases += 1
                        overall_confidence += result.confidence
                        total_processing_time += result.processing_time_ms
            
            # Calculate overall metrics
            if successful_phases > 0:
                overall_confidence /= successful_phases
            
            # Phase 4: ML Predictions (if enabled and we have data)
            if enable_predictions and successful_phases > 0:
                prediction_data = self._prepare_prediction_data(intelligence_results)
                if prediction_data:
                    predictions_result = await self._prediction_service.generate_predictions_with_confidence(
                        prediction_data
                    )
                    intelligence_results["predictions"] = predictions_result.__dict__ if hasattr(predictions_result, '__dict__') else predictions_result
                    
                    if predictions_result.success:
                        overall_confidence = (overall_confidence + predictions_result.confidence) / 2
                        total_processing_time += predictions_result.processing_time_ms
            
            # Finalize processing metadata
            end_time = datetime.now(timezone.utc)
            total_elapsed_ms = (end_time - start_time).total_seconds() * 1000
            
            intelligence_results["processing_metadata"].update({
                "completed_at": end_time.isoformat(),
                "total_elapsed_ms": total_elapsed_ms,
                "successful_phases": successful_phases,
                "total_phases": len(processing_tasks) + (1 if enable_predictions else 0),
                "phase_processing_time_ms": total_processing_time,
                "coordination_overhead_ms": total_elapsed_ms - total_processing_time
            })
            
            # Record facade-level metrics
            self._record_processing_metrics(intelligence_results)
            
            return IntelligenceResult(
                success=successful_phases > 0,
                data=intelligence_results,
                confidence=overall_confidence,
                processing_time_ms=total_elapsed_ms,
                cache_hit=False
            )
            
        except Exception as e:
            processing_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            self._metrics_registry.increment(
                "ml_intelligence_facade_operations_total",
                tags={"result": "error"}
            )
            
            logger.error(f"Intelligence processing failed: {e}")
            
            return IntelligenceResult(
                success=False,
                data={"error": str(e), "processing_metadata": {"error_at_ms": processing_time}},
                confidence=0.0,
                processing_time_ms=processing_time,
                cache_hit=False,
                error_message=str(e)
            )
    
    async def _setup_circuit_breakers(self) -> None:
        """Setup circuit breakers for all ML components."""
        components = [
            "rule_analysis",
            "pattern_discovery", 
            "prediction_generation",
            "batch_processing"
        ]
        
        await self._circuit_breaker_service.setup_circuit_breakers(components)
    
    async def _get_pattern_discovery_data(
        self, 
        rule_ids: Optional[List[str]], 
        batch_size: int
    ) -> List[Dict[str, Any]]:
        """Get data for pattern discovery analysis.
        
        Args:
            rule_ids: Rule IDs to get data for
            batch_size: Batch size for data retrieval
            
        Returns:
            Pattern discovery data
        """
        try:
            # Use repository to get session data for pattern discovery
            if rule_ids:
                # Get sessions for specific rules
                sessions = await self._ml_repository.get_sessions_by_rule_ids(rule_ids, limit=batch_size)
            else:
                # Get recent sessions for general pattern discovery
                sessions = await self._ml_repository.get_recent_sessions(limit=batch_size)
            
            return sessions
            
        except Exception as e:
            logger.error(f"Failed to get pattern discovery data: {e}")
            return []
    
    def _prepare_prediction_data(self, intelligence_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare data for ML predictions based on intelligence results.
        
        Args:
            intelligence_results: Results from intelligence analysis
            
        Returns:
            Prepared prediction data
        """
        try:
            prediction_data = {
                "characteristics": {},
                "context": {},
                "intelligence_summary": {}
            }
            
            # Extract characteristics from rule intelligence
            rule_intelligence = intelligence_results.get("rule_intelligence", {})
            if rule_intelligence and rule_intelligence.get("success"):
                rule_data = rule_intelligence.get("data", {})
                prediction_data["characteristics"].update(rule_data.get("characteristics", {}))
                prediction_data["intelligence_summary"]["rule_effectiveness"] = rule_data.get("effectiveness_analysis", {})
            
            # Extract insights from pattern discovery
            pattern_insights = intelligence_results.get("pattern_insights", {})
            if pattern_insights and pattern_insights.get("success"):
                pattern_data = pattern_insights.get("data", {})
                prediction_data["context"]["pattern_insights"] = pattern_data.get("insights", {})
                prediction_data["intelligence_summary"]["discovered_patterns"] = len(pattern_data.get("patterns", []))
            
            # Extract combination intelligence
            combination_intelligence = intelligence_results.get("combination_intelligence", {})
            if combination_intelligence and combination_intelligence.get("success"):
                combination_data = combination_intelligence.get("data", {})
                prediction_data["context"]["rule_combinations"] = combination_data.get("combinations", [])
                prediction_data["intelligence_summary"]["combination_effectiveness"] = combination_data.get("effectiveness", 0.0)
            
            return prediction_data if prediction_data["characteristics"] else {}
            
        except Exception as e:
            logger.error(f"Failed to prepare prediction data: {e}")
            return {}
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get health status of all intelligence services.
        
        Returns:
            Service health information
        """
        try:
            # Get circuit breaker states
            circuit_states = await self._circuit_breaker_service.get_all_states()
            
            # Get batch processing status
            batch_status = await self._batch_processing_service.get_processing_status()
            
            # Get pattern discovery cache stats
            pattern_cache_stats = await self._pattern_discovery_service.get_cache_statistics()
            
            health_status = {
                "overall_status": "healthy",
                "circuit_breakers": {
                    name: {
                        "is_open": state.is_open,
                        "failure_count": state.failure_count,
                        "component_name": state.component_name
                    }
                    for name, state in circuit_states.items()
                },
                "batch_processing": batch_status,
                "pattern_cache": pattern_cache_stats,
                "facade_info": {
                    "is_background_running": self._is_running,
                    "background_task_status": str(self._background_task.done()) if self._background_task else "not_running"
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Determine overall health
            open_circuits = [name for name, state in circuit_states.items() if state.is_open]
            if open_circuits:
                health_status["overall_status"] = "degraded"
                health_status["issues"] = [f"Circuit breaker open: {circuit}" for circuit in open_circuits]
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "overall_status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
    
    async def start_background_processing(self) -> None:
        """Start background intelligence processing service."""
        if self._is_running:
            logger.warning("Background processing already running")
            return
        
        self._is_running = True
        self._background_task = asyncio.create_task(self._background_processing_loop())
        
        logger.info("Background intelligence processing started")
    
    async def stop_background_processing(self) -> None:
        """Stop background intelligence processing service."""
        if not self._is_running:
            return
        
        self._is_running = False
        
        if self._background_task and not self._background_task.done():
            self._background_task.cancel()
            try:
                await self._background_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Background intelligence processing stopped")
    
    async def _background_processing_loop(self) -> None:
        """Background processing loop for continuous intelligence analysis."""
        while self._is_running:
            try:
                # Run intelligence processing with default settings
                result = await self.run_intelligence_processing(
                    rule_ids=None,
                    enable_patterns=True,
                    enable_predictions=True,
                    batch_size=50
                )
                
                if result.success:
                    logger.debug("Background intelligence processing completed successfully")
                else:
                    logger.warning(f"Background intelligence processing failed: {result.error_message}")
                
                # Wait before next processing cycle (30 minutes)
                await asyncio.sleep(1800)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background processing error: {e}")
                # Wait before retrying (5 minutes)
                await asyncio.sleep(300)
    
    async def get_processing_metrics(self) -> Dict[str, Any]:
        """Get comprehensive processing metrics.
        
        Returns:
            Processing metrics from all services
        """
        try:
            # Collect metrics from all services
            service_health = await self.get_service_health()
            
            # Get additional metrics
            metrics = {
                "facade_metrics": {
                    "total_operations": self._metrics_registry.get_counter_value(
                        "ml_intelligence_facade_operations_total"
                    ),
                    "successful_operations": self._metrics_registry.get_counter_value(
                        "ml_intelligence_facade_operations_total",
                        tags={"result": "success"}
                    ),
                    "failed_operations": self._metrics_registry.get_counter_value(
                        "ml_intelligence_facade_operations_total", 
                        tags={"result": "error"}
                    )
                },
                "service_health": service_health,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Calculate success rate
            total_ops = metrics["facade_metrics"]["total_operations"] or 0
            successful_ops = metrics["facade_metrics"]["successful_operations"] or 0
            
            if total_ops > 0:
                metrics["facade_metrics"]["success_rate"] = successful_ops / total_ops
            else:
                metrics["facade_metrics"]["success_rate"] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get processing metrics: {e}")
            return {"error": str(e), "timestamp": datetime.now(timezone.utc).isoformat()}
    
    def _record_processing_metrics(self, results: Dict[str, Any]) -> None:
        """Record facade-level processing metrics.
        
        Args:
            results: Processing results to record metrics for
        """
        try:
            metadata = results.get("processing_metadata", {})
            
            # Record operation
            self._metrics_registry.increment(
                "ml_intelligence_facade_operations_total",
                tags={"result": "success"}
            )
            
            # Record timing metrics
            total_time = metadata.get("total_elapsed_ms", 0)
            self._metrics_registry.record_value(
                "ml_intelligence_facade_duration_ms",
                total_time
            )
            
            # Record coordination overhead
            coordination_overhead = metadata.get("coordination_overhead_ms", 0)
            self._metrics_registry.record_value(
                "ml_intelligence_facade_coordination_overhead_ms",
                coordination_overhead
            )
            
            # Record success metrics
            successful_phases = metadata.get("successful_phases", 0)
            total_phases = metadata.get("total_phases", 1)
            
            self._metrics_registry.set_gauge(
                "ml_intelligence_facade_phase_success_rate",
                successful_phases / total_phases if total_phases > 0 else 0
            )
            
        except Exception as e:
            logger.error(f"Failed to record processing metrics: {e}")


# Factory function for dependency injection
def create_ml_intelligence_service_facade(
    ml_repository: MLRepositoryProtocol,
    pattern_discovery: AdvancedPatternDiscovery
) -> MLIntelligenceServiceFacade:
    """Create ML intelligence service facade with dependencies.
    
    Args:
        ml_repository: ML repository for data access
        pattern_discovery: Advanced pattern discovery component
        
    Returns:
        Configured ML intelligence service facade
    """
    return MLIntelligenceServiceFacade(
        ml_repository=ml_repository,
        pattern_discovery=pattern_discovery
    )


