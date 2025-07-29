"""
Shared component types for ML orchestration to avoid circular imports.

This module contains common types and enums used by both component_registry 
and component_definitions modules, following 2025 best practices for 
dependency management.
"""

from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional


class ComponentTier(Enum):
    """Component tier classification for orchestration priority."""
    TIER_1 = "tier_1"  # Critical path components
    TIER_2 = "tier_2"  # Important but not critical
    TIER_3 = "tier_3"  # Optional/experimental components


class ComponentCapability(Enum):
    """Component capability types for orchestration."""
    DATA_PROCESSING = "data_processing"
    MODEL_TRAINING = "model_training"
    FEATURE_ENGINEERING = "feature_engineering"
    VALIDATION = "validation"
    OPTIMIZATION = "optimization"
    MONITORING = "monitoring"
    DEPLOYMENT = "deployment"


@dataclass
class ComponentInfo:
    """Information about a component in the orchestration system."""
    name: str
    tier: ComponentTier
    capabilities: List[ComponentCapability]
    dependencies: List[str] = None
    config: Dict[str, Any] = None
    enabled: bool = True
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.config is None:
            self.config = {}


@dataclass
class ComponentStatus:
    """Runtime status of a component."""
    name: str
    is_running: bool = False
    is_healthy: bool = True
    last_check: Optional[str] = None
    error_message: Optional[str] = None
    metrics: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = {}


@dataclass
class ComponentMetrics:
    """Performance metrics for a component."""
    execution_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    success_rate: float = 100.0
    error_count: int = 0
    last_execution: Optional[str] = None
