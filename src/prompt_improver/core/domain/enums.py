"""Domain enums for status and configuration values.

These enums represent domain-specific states and configurations
that are used across the application.
"""

from enum import Enum, auto


class SessionStatus(Enum):
    """Status of improvement or training sessions."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class AnalysisStatus(Enum):
    """Status of analysis operations."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class HealthStatus(Enum):
    """Health status of services and components."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


class ModelStatus(Enum):
    """Status of ML models."""
    TRAINING = "training"
    READY = "ready"
    DEPLOYED = "deployed"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class TrainingStatus(Enum):
    """Status of training operations."""
    INITIALIZED = "initialized"
    TRAINING = "training"
    VALIDATING = "validating"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"


class ValidationLevel(Enum):
    """Level of validation to apply."""
    BASIC = "basic"
    STANDARD = "standard"
    STRICT = "strict"
    CUSTOM = "custom"


class CacheLevel(Enum):
    """Cache levels for multi-level caching."""
    L1 = "l1_memory"
    L2 = "l2_redis" 
    L3 = "l3_database"


class SecurityLevel(Enum):
    """Security levels for operations."""
    PUBLIC = "public"
    AUTHENTICATED = "authenticated"
    AUTHORIZED = "authorized" 
    ADMIN = "admin"


class PerformanceLevel(Enum):
    """Performance optimization levels."""
    DISABLED = "disabled"
    BASIC = "basic"
    OPTIMIZED = "optimized"
    AGGRESSIVE = "aggressive"


class MonitoringLevel(Enum):
    """Monitoring detail levels."""
    MINIMAL = "minimal"
    BASIC = "basic"
    DETAILED = "detailed"
    COMPREHENSIVE = "comprehensive"