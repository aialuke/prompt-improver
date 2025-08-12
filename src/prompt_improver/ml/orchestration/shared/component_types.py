"""
Shared component types for ML orchestration to avoid circular imports.

This module contains common types and enums used by both component_registry 
and component_definitions modules, following 2025 best practices for 
dependency management.
"""
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional
if TYPE_CHECKING:
    from datetime import datetime

class ComponentTier(Enum):
    """Component tier classification for orchestration priority."""
    TIER_1 = 'tier_1'
    TIER_2 = 'tier_2'
    TIER_3 = 'tier_3'

class ComponentCapability(Enum):
    """Component capability types for orchestration."""
    DATA_PROCESSING = 'data_processing'
    MODEL_TRAINING = 'model_training'
    FEATURE_ENGINEERING = 'feature_engineering'
    VALIDATION = 'validation'
    OPTIMIZATION = 'optimization'
    MONITORING = 'monitoring'
    DEPLOYMENT = 'deployment'

@dataclass
class ComponentInfo:
    """Information about a component in the orchestration system."""
    name: str
    tier: ComponentTier
    capabilities: list[ComponentCapability]
    dependencies: list[str] = None
    config: dict[str, Any] = None
    enabled: bool = True
    status: str | None = None
    last_health_check: Optional['datetime'] = None
    error_message: str | None = None
    health_check_endpoint: str | None = None
    registered_at: Optional['datetime'] = None

    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.config is None:
            self.config = {}
        if self.registered_at is None:
            from datetime import datetime, timezone
            self.registered_at = datetime.now(timezone.utc)

@dataclass
class ComponentStatus:
    """Runtime status of a component."""
    name: str
    is_running: bool = False
    is_healthy: bool = True
    last_check: str | None = None
    error_message: str | None = None
    metrics: dict[str, Any] = None

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
    last_execution: str | None = None
