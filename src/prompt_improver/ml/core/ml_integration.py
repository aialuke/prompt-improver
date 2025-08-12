"""Direct Python ML integration service for Phase 3 continuous learning.
Replaces cross-language bridge architecture with direct Python function calls.
Performance improvement: 50-100ms → 1-5ms response times.

Enhanced with production model registry, alias-based deployment, and Apriori pattern discovery.

REFACTORED: This file now uses a facade pattern to maintain backward compatibility
while delegating to focused, single-responsibility services.
"""

import logging
from typing import Any

from .facade import MLModelServiceFacade

# Lazy import to avoid circular dependency
def _get_database_manager():
    """Lazy import of database manager to avoid circular imports."""
    from ...database import ManagerMode, get_database_services
    return get_database_services, ManagerMode


logger = logging.getLogger(__name__)

# Alias for backward compatibility with existing imports
# The actual implementation has been moved to focused services
class MLModelService:
    """Enhanced ML service with direct Python integration and production deployment capabilities.
    
    DEPRECATED: This class is now a thin wrapper around MLModelServiceFacade.
    The implementation has been refactored into focused, single-responsibility services:
    - MLModelRegistry: Model caching and memory management
    - MLTrainingService: Model training and optimization
    - MLInferenceService: Model predictions and inference
    - MLProductionService: Production deployment and monitoring
    - MLPatternDiscoveryService: Pattern discovery and analysis
    - MLOrchestrationAdapter: Orchestrator integration
    
    Use MLModelServiceFacade directly for new code.
    """

    def __init__(self, db_manager=None, orchestrator_event_bus=None):
        """Initialize ML service using facade pattern for backward compatibility."""
        get_database_services, ManagerMode = _get_database_manager()
        
        # Create facade with proper database manager
        if db_manager is None:
            db_manager = get_database_services(ManagerMode.ML_TRAINING)
            
        self._facade = MLModelServiceFacade(
            db_manager=db_manager,
            orchestrator_event_bus=orchestrator_event_bus
        )
        
        # Expose model_registry for backward compatibility
        self.model_registry = self._facade.model_registry
        
        logger.info("ML Model Service (facade) initialized for backward compatibility")

    # Delegate all methods to the facade for backward compatibility
    
    async def optimize_rules(self, training_data, db_session, rule_ids=None):
        """Delegate to facade."""
        return await self._facade.optimize_rules(training_data, db_session, rule_ids)
    
    async def predict_rule_effectiveness(self, model_id, rule_features):
        """Delegate to facade."""
        return await self._facade.predict_rule_effectiveness(model_id, rule_features)
    
    async def optimize_ensemble_rules(self, training_data, db_session):
        """Delegate to facade."""
        return await self._facade.optimize_ensemble_rules(training_data, db_session)
    
    async def discover_patterns(self, db_session, min_effectiveness=0.7, min_support=5, 
                               use_advanced_discovery=True, include_apriori=True):
        """Delegate to facade."""
        return await self._facade.discover_patterns(
            db_session, min_effectiveness, min_support, use_advanced_discovery, include_apriori
        )
    
    async def get_contextualized_patterns(self, context_items, db_session, min_confidence=0.6):
        """Delegate to facade."""
        return await self._facade.get_contextualized_patterns(context_items, db_session, min_confidence)
    
    async def enable_production_deployment(self, tracking_uri=None):
        """Delegate to facade."""
        return await self._facade.enable_production_deployment(tracking_uri)
    
    async def deploy_to_production(self, model_name, version, alias=None, strategy=None):
        """Delegate to facade."""
        return await self._facade.deploy_to_production(model_name, version, alias, strategy)
    
    async def rollback_production(self, model_name, alias=None, reason="Performance degradation detected"):
        """Delegate to facade."""
        return await self._facade.rollback_production(model_name, alias, reason)
    
    async def monitor_production_health(self, model_name, alias=None):
        """Delegate to facade."""
        return await self._facade.monitor_production_health(model_name, alias)
    
    async def get_production_model(self, model_name, alias=None):
        """Delegate to facade."""
        return await self._facade.get_production_model(model_name, alias)
    
    async def list_production_deployments(self):
        """Delegate to facade."""
        return await self._facade.list_production_deployments()
    
    async def get_model_cache_stats(self):
        """Delegate to facade."""
        return await self._facade.get_model_cache_stats()
    
    async def optimize_model_cache(self):
        """Delegate to facade."""
        return await self._facade.optimize_model_cache()
    
    async def send_training_batch(self, batch):
        """Delegate to facade."""
        return await self._facade.send_training_batch(batch)
    
    async def fetch_latest_model(self):
        """Delegate to facade."""
        return await self._facade.fetch_latest_model()
    
    async def run_orchestrated_analysis(self, config):
        """Delegate to facade."""
        return await self._facade.run_orchestrated_analysis(config)

    # Backward compatibility: delegate to facade services
    def __getattr__(self, name):
        """Forward any missing attributes to the facade for full compatibility."""
        return getattr(self._facade, name)


# Global service instance
_ml_service: MLModelService | None = None


async def get_ml_service() -> MLModelService:
    """Get or create global ML service instance with unified manager."""
    global _ml_service
    if _ml_service is None:
        get_database_services, ManagerMode = _get_database_manager()
        # Pass unified manager for async operations
        db_manager = get_database_services(ManagerMode.ML_TRAINING)
        _ml_service = MLModelService(db_manager=db_manager)
    return _ml_service

