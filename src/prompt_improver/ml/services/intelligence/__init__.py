"""ML Intelligence Services Package.

Decomposed ML intelligence processing services following clean architecture patterns.
Replaces the 1,319-line intelligence_processor.py god object with focused services.

Services:
- MLCircuitBreakerService: Circuit breaker protection for ML operations
- RuleAnalysisService: Rule effectiveness and combination analysis  
- PatternDiscoveryService: ML pattern discovery and caching
- MLPredictionService: ML predictions with confidence scoring
- BatchProcessingService: Parallel batch processing for large datasets
- MLIntelligenceServiceFacade: Unified interface for all ML intelligence services
"""

from prompt_improver.ml.services.intelligence.circuit_breaker_service import MLCircuitBreakerService
from prompt_improver.ml.services.intelligence.rule_analysis_service import RuleAnalysisService
from prompt_improver.ml.services.intelligence.pattern_discovery_service import PatternDiscoveryService
from prompt_improver.ml.services.intelligence.prediction_service import MLPredictionService
from prompt_improver.ml.services.intelligence.batch_processing_service import BatchProcessingService
from prompt_improver.ml.services.intelligence.facade import (
    MLIntelligenceServiceFacade,
    create_ml_intelligence_service_facade,
)

__all__ = [
    "MLCircuitBreakerService",
    "RuleAnalysisService", 
    "PatternDiscoveryService",
    "MLPredictionService",
    "BatchProcessingService",
    "MLIntelligenceServiceFacade",
    "create_ml_intelligence_service_facade",
]

from .circuit_breaker_service import MLCircuitBreakerService
from .rule_analysis_service import RuleAnalysisService
from .protocols.intelligence_service_protocols import (
    RuleAnalysisServiceProtocol,
    MLCircuitBreakerServiceProtocol,
    IntelligenceResult,
)

__all__ = [
    "RuleAnalysisService",
    "MLCircuitBreakerService",
    "RuleAnalysisServiceProtocol", 
    "MLCircuitBreakerServiceProtocol",
    "IntelligenceResult",
]