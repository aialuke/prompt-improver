"""
Configuration management for ML Pipeline Orchestrator.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional

@dataclass
class OrchestratorConfig:
    """Configuration for ML Pipeline Orchestrator."""
    
    # Resource limits
    max_concurrent_workflows: int = 10
    gpu_allocation_timeout: int = 300  # seconds
    memory_limit_gb: float = 16.0
    cpu_limit_cores: int = 8
    
    # Component timeouts
    training_timeout: int = 3600  # 1 hour
    evaluation_timeout: int = 1800  # 30 minutes
    deployment_timeout: int = 600  # 10 minutes
    optimization_timeout: int = 7200  # 2 hours
    
    # Health check intervals
    component_health_check_interval: int = 30  # seconds
    pipeline_status_update_interval: int = 10  # seconds
    orchestrator_health_check_interval: int = 60  # seconds
    
    # Event system configuration
    event_bus_buffer_size: int = 1000
    event_handler_timeout: int = 30  # seconds
    event_history_size: int = 1000
    
    # Workflow execution settings
    workflow_retry_attempts: int = 3
    workflow_retry_delay: int = 5  # seconds
    parallel_execution_limit: int = 5
    
    # Monitoring and alerting
    metrics_collection_interval: int = 30  # seconds
    alert_threshold_cpu: float = 0.8  # 80%
    alert_threshold_memory: float = 0.9  # 90%
    alert_threshold_error_rate: float = 0.05  # 5%
    
    # Database settings
    db_connection_pool_size: int = 20
    db_connection_timeout: int = 30
    db_query_timeout: int = 60
    
    # Cache settings
    redis_connection_pool_size: int = 10
    cache_default_ttl: int = 3600  # 1 hour
    cache_max_memory: str = "2gb"
    
    # API settings
    api_timeout: int = 30
    api_max_concurrent_requests: int = 100
    api_rate_limit_per_minute: int = 1000
    
    # Security settings
    enable_authentication: bool = True
    enable_authorization: bool = True
    enable_audit_logging: bool = True
    enable_input_validation: bool = True
    
    # Development/debugging settings
    debug_mode: bool = False
    verbose_logging: bool = False
    enable_performance_profiling: bool = False
    
    # Component-specific settings
    component_startup_timeout: int = 120  # seconds
    component_shutdown_timeout: int = 60  # seconds
    component_discovery_interval: int = 300  # 5 minutes
    
    # Tier-specific configurations
    tier_configs: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "tier1_core": {
            "priority": 1,
            "resource_weight": 1.0,
            "timeout_multiplier": 1.0
        },
        "tier2_optimization": {
            "priority": 2,
            "resource_weight": 0.8,
            "timeout_multiplier": 1.5
        },
        "tier3_evaluation": {
            "priority": 3,
            "resource_weight": 0.6,
            "timeout_multiplier": 1.2
        },
        "tier4_performance": {
            "priority": 4,
            "resource_weight": 0.4,
            "timeout_multiplier": 1.0
        },
        "tier5_infrastructure": {
            "priority": 5,
            "resource_weight": 0.8,
            "timeout_multiplier": 0.8
        },
        "tier6_security": {
            "priority": 6,
            "resource_weight": 0.3,
            "timeout_multiplier": 2.0
        }
    })
    
    # Feature flags
    enable_auto_scaling: bool = True
    enable_resource_optimization: bool = True
    enable_predictive_scheduling: bool = False
    enable_cross_tier_coordination: bool = True
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OrchestratorConfig":
        """Create config from dictionary."""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    def get_tier_config(self, tier_name: str) -> Dict[str, Any]:
        """Get configuration for a specific tier."""
        return self.tier_configs.get(tier_name, {})
    
    def update_tier_config(self, tier_name: str, config_updates: Dict[str, Any]) -> None:
        """Update configuration for a specific tier."""
        if tier_name not in self.tier_configs:
            self.tier_configs[tier_name] = {}
        
        self.tier_configs[tier_name].update(config_updates)
    
    def validate(self) -> List[str]:
        """
        Validate configuration settings.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate resource limits
        if self.max_concurrent_workflows <= 0:
            errors.append("max_concurrent_workflows must be positive")
        
        if self.memory_limit_gb <= 0:
            errors.append("memory_limit_gb must be positive")
        
        if self.cpu_limit_cores <= 0:
            errors.append("cpu_limit_cores must be positive")
        
        # Validate timeouts
        if self.training_timeout <= 0:
            errors.append("training_timeout must be positive")
        
        if self.evaluation_timeout <= 0:
            errors.append("evaluation_timeout must be positive")
        
        # Validate intervals
        if self.component_health_check_interval <= 0:
            errors.append("component_health_check_interval must be positive")
        
        # Validate event system
        if self.event_bus_buffer_size <= 0:
            errors.append("event_bus_buffer_size must be positive")
        
        # Validate thresholds
        if not (0 <= self.alert_threshold_cpu <= 1):
            errors.append("alert_threshold_cpu must be between 0 and 1")
        
        if not (0 <= self.alert_threshold_memory <= 1):
            errors.append("alert_threshold_memory must be between 0 and 1")
        
        return errors