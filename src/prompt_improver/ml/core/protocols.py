"""Protocols and interfaces for ML service components.

Defines clean interfaces between ML service components to enable
loose coupling and proper dependency injection.
"""

from typing import Any, Dict, List, Protocol, TYPE_CHECKING

# Import heavy dependencies only for type checking to optimize startup performance
if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

class ModelRegistryProtocol(Protocol):
    """Protocol for model registry implementations."""
    
    def get_model(self, model_id: str) -> Any | None:
        """Get model from registry."""
        ...
        
    def add_model(
        self, 
        model_id: str, 
        model: Any, 
        model_type: str = "sklearn",
        ttl_minutes: int | None = None
    ) -> bool:
        """Add model to registry."""
        ...
        
    def remove_model(self, model_id: str) -> bool:
        """Remove model from registry."""
        ...
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        ...
        
    def cleanup_expired(self) -> int:
        """Clean up expired models."""
        ...


class TrainingServiceProtocol(Protocol):
    """Protocol for ML training services."""
    
    async def optimize_rules(
        self,
        training_data: Dict[str, List],
        db_session: "AsyncSession",
        rule_ids: List[str] | None = None
    ) -> Dict[str, Any]:
        """Optimize rule parameters using ML training."""
        ...
        
    async def optimize_ensemble_rules(
        self,
        training_data: Dict[str, List],
        db_session: "AsyncSession"
    ) -> Dict[str, Any]:
        """Optimize rules using ensemble methods."""
        ...
        
    async def send_training_batch(self, batch: List[Dict]) -> Dict[str, Any]:
        """Send training batch to storage."""
        ...
        
    async def fetch_latest_model(self) -> Dict[str, Any]:
        """Fetch the latest model from storage."""
        ...


class InferenceServiceProtocol(Protocol):
    """Protocol for ML inference services."""
    
    async def predict_rule_effectiveness(
        self,
        model_id: str,
        rule_features: List[float]
    ) -> Dict[str, Any]:
        """Predict rule effectiveness using trained model."""
        ...


class ProductionServiceProtocol(Protocol):
    """Protocol for production deployment services."""
    
    async def enable_production_deployment(
        self, 
        tracking_uri: str | None = None
    ) -> Dict[str, Any]:
        """Enable production deployment capabilities."""
        ...
        
    async def deploy_to_production(
        self,
        model_name: str,
        version: str,
        alias: Any = None,
        strategy: Any = None
    ) -> Dict[str, Any]:
        """Deploy model to production."""
        ...
        
    async def rollback_production(
        self,
        model_name: str,
        alias: Any = None,
        reason: str = "Performance degradation detected"
    ) -> Dict[str, Any]:
        """Rollback production deployment."""
        ...
        
    async def monitor_production_health(
        self,
        model_name: str,
        alias: Any = None
    ) -> Dict[str, Any]:
        """Monitor production model health."""
        ...
        
    async def get_production_model(
        self,
        model_name: str,
        alias: Any = None
    ) -> Any:
        """Load production model by alias."""
        ...
        
    async def list_production_deployments(self) -> List[Dict[str, Any]]:
        """List all production deployments."""
        ...


class PatternDiscoveryServiceProtocol(Protocol):
    """Protocol for pattern discovery services."""
    
    async def discover_patterns(
        self,
        db_session: "AsyncSession",
        min_effectiveness: float = 0.7,
        min_support: int = 5,
        use_advanced_discovery: bool = True,
        include_apriori: bool = True
    ) -> Dict[str, Any]:
        """Discover patterns using ML and Apriori methods."""
        ...
        
    async def get_contextualized_patterns(
        self,
        context_items: List[str],
        db_session: "AsyncSession",
        min_confidence: float = 0.6
    ) -> Dict[str, Any]:
        """Get patterns relevant to specific context."""
        ...


class OrchestrationAdapterProtocol(Protocol):
    """Protocol for orchestrator integration."""
    
    async def run_orchestrated_analysis(
        self, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Run orchestrator-compatible ML operations."""
        ...


class MLServiceProtocol(Protocol):
    """Main protocol for the ML service facade."""
    
    model_registry: ModelRegistryProtocol
    
    # Training methods
    async def optimize_rules(
        self,
        training_data: Dict[str, List],
        db_session: "AsyncSession",
        rule_ids: List[str] | None = None
    ) -> Dict[str, Any]:
        ...
        
    async def optimize_ensemble_rules(
        self,
        training_data: Dict[str, List],
        db_session: "AsyncSession"
    ) -> Dict[str, Any]:
        ...
        
    # Inference methods
    async def predict_rule_effectiveness(
        self,
        model_id: str,
        rule_features: List[float]
    ) -> Dict[str, Any]:
        ...
        
    # Pattern discovery methods
    async def discover_patterns(
        self,
        db_session: "AsyncSession",
        min_effectiveness: float = 0.7,
        min_support: int = 5,
        use_advanced_discovery: bool = True,
        include_apriori: bool = True
    ) -> Dict[str, Any]:
        ...
        
    async def get_contextualized_patterns(
        self,
        context_items: List[str],
        db_session: "AsyncSession",
        min_confidence: float = 0.6
    ) -> Dict[str, Any]:
        ...
        
    # Production methods  
    async def enable_production_deployment(
        self, 
        tracking_uri: str | None = None
    ) -> Dict[str, Any]:
        ...
        
    async def deploy_to_production(
        self,
        model_name: str,
        version: str,
        alias: Any = None,
        strategy: Any = None
    ) -> Dict[str, Any]:
        ...
        
    # Cache and utility methods
    async def get_model_cache_stats(self) -> Dict[str, Any]:
        ...
        
    async def optimize_model_cache(self) -> Dict[str, Any]:
        ...
        
    async def send_training_batch(self, batch: List[Dict]) -> Dict[str, Any]:
        ...
        
    async def fetch_latest_model(self) -> Dict[str, Any]:
        ...
        
    # Orchestrator integration
    async def run_orchestrated_analysis(
        self, 
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        ...