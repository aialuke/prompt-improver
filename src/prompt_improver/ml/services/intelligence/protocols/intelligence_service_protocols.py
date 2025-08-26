"""Protocol interfaces for ML Intelligence Services.

Clean architecture protocol-based interfaces for ML intelligence processing components.
Follows 2025 architectural standards with protocol-based dependency injection.
"""

from typing import Protocol, Any, Dict, List, Callable, Optional
from dataclasses import dataclass
from datetime import datetime


@dataclass
class CircuitBreakerState:
    """Circuit breaker state information."""
    is_open: bool
    failure_count: int
    last_failure_time: Optional[datetime]
    recovery_timeout: float
    component_name: str


@dataclass
class IntelligenceResult:
    """Result container for ML intelligence operations."""
    success: bool
    data: Dict[str, Any]
    confidence: float
    processing_time_ms: float
    cache_hit: bool = False
    error_message: Optional[str] = None


class MLCircuitBreakerServiceProtocol(Protocol):
    """Protocol for ML circuit breaker service."""
    
    async def call_with_breaker(self, component: str, operation: Callable[..., Any], *args, **kwargs) -> Any:
        """Execute operation with circuit breaker protection."""
        ...
    
    async def setup_circuit_breakers(self, components: List[str]) -> None:
        """Initialize circuit breakers for ML components."""
        ...
    
    async def get_circuit_state(self, component: str) -> CircuitBreakerState:
        """Get current circuit breaker state for component."""
        ...
    
    async def handle_state_change(self, component: str, is_success: bool) -> None:
        """Handle circuit breaker state transitions."""
        ...


class RuleAnalysisServiceProtocol(Protocol):
    """Protocol for rule analysis service."""
    
    async def process_rule_intelligence(self, rule_ids: Optional[List[str]] = None) -> IntelligenceResult:
        """Process individual rule effectiveness analysis."""
        ...
    
    async def process_combination_intelligence(self, combination_limit: int = 50) -> IntelligenceResult:
        """Process rule combination synergy analysis."""
        ...
    
    async def generate_intelligence_data(self, rule_characteristics: Dict[str, Any]) -> Dict[str, Any]:
        """Generate ML intelligence for specific rule characteristics."""
        ...
    
    async def analyze_rule_effectiveness(self, rule_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze effectiveness of specific rule."""
        ...


class PatternDiscoveryServiceProtocol(Protocol):
    """Protocol for ML pattern discovery service."""
    
    async def discover_patterns(self, batch_data: List[Dict[str, Any]]) -> IntelligenceResult:
        """Discover ML patterns in batch data."""
        ...
    
    async def analyze_pattern_insights(self, patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze discovered patterns for insights."""
        ...


class MLPredictionServiceProtocol(Protocol):
    """Protocol for ML prediction service."""
    
    async def generate_predictions_with_confidence(self, rule_data: Dict[str, Any]) -> IntelligenceResult:
        """Generate ML predictions with confidence scoring."""
        ...
    
    async def validate_prediction_quality(self, predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Validate quality and reliability of predictions."""
        ...
    
    async def calculate_confidence_metrics(self, prediction_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate confidence metrics for predictions."""
        ...
    
    async def analyze_prediction_trends(self, historical_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze prediction trends over time."""
        ...


class BatchProcessingServiceProtocol(Protocol):
    """Protocol for batch processing service."""
    
    async def process_parallel_batches(
        self, 
        data: List[Dict[str, Any]], 
        batch_size: int = 25,
        max_workers: int = 5
    ) -> IntelligenceResult:
        """Process data in parallel batches."""
        ...
    
    async def calculate_batch_ranges(self, total_items: int, batch_size: int) -> List[tuple[int, int]]:
        """Calculate optimal batch ranges for processing."""
        ...
    
    async def manage_parallel_workers(self, tasks: List[Callable], max_workers: int) -> List[Any]:
        """Manage parallel worker execution."""
        ...
    
    async def optimize_batch_size(self, data_size: int, memory_limit_mb: int) -> int:
        """Calculate optimal batch size based on data size and memory constraints."""
        ...


class MLIntelligenceServiceFacadeProtocol(Protocol):
    """Protocol for ML Intelligence Service Facade."""
    
    async def run_intelligence_processing(
        self,
        rule_ids: Optional[List[str]] = None,
        enable_patterns: bool = True,
        enable_predictions: bool = True,
        batch_size: int = 25
    ) -> IntelligenceResult:
        """Execute complete ML intelligence processing pipeline."""
        ...
    
    async def get_service_health(self) -> Dict[str, Any]:
        """Get health status of all intelligence services."""
        ...
    
    async def start_background_processing(self) -> None:
        """Start background intelligence processing service."""
        ...
    
    async def stop_background_processing(self) -> None:
        """Stop background intelligence processing service."""
        ...
    
    async def get_processing_metrics(self) -> Dict[str, Any]:
        """Get comprehensive processing metrics."""
        ...