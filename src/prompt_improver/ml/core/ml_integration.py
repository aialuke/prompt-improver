"""Direct Python ML integration service for Phase 3 continuous learning.
Replaces cross-language bridge architecture with direct Python function calls.
Performance improvement: 50-100ms → 1-5ms response times.

Enhanced with production model registry, alias-based deployment, and Apriori pattern discovery.

Modern ML architecture using focused, single-responsibility services:
- MLModelRegistry: Model caching and memory management
- MLTrainingService: Model training and optimization
- MLInferenceService: Model predictions and inference
- MLProductionService: Production deployment and monitoring
- MLPatternDiscoveryService: Pattern discovery and analysis
- MLOrchestrationAdapter: Orchestrator integration
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


# Global service instance
_ml_service: MLModelServiceFacade | None = None


async def get_ml_service() -> MLModelServiceFacade:
    """Get or create global ML service instance with unified manager."""
    global _ml_service
    if _ml_service is None:
        get_database_services, ManagerMode = _get_database_manager()
        # Pass unified manager for async operations
        db_manager = get_database_services(ManagerMode.ML_TRAINING)
        _ml_service = MLModelServiceFacade(db_manager=db_manager)
    return _ml_service

