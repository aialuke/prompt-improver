"""Domain-specific type definitions for the prompt-improver system.

This module contains types that are specific to the prompt improvement
domain and business logic.
"""

from pydantic import BaseModel, Field

# Import shared infrastructure types
from prompt_improver.shared.types import ConfigDict


class ModelConfig(BaseModel):
    """ML model configuration."""

    name: str
    version: str
    parameters: ConfigDict
    timeout: float = Field(default=120.0)
    batch_size: int = Field(default=32)


class FeatureVector(BaseModel):
    """Feature vector for ML processing."""

    features: list[float]
    feature_metadata: ConfigDict
    timestamp: str | None = Field(default=None)


class SecurityContext(BaseModel):
    """Security context for authenticated operations."""
    
    user_id: str | None = Field(default=None)
    session_id: str | None = Field(default=None)
    correlation_id: str | None = Field(default=None)
    permissions: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    authenticated: bool = Field(default=False)
    auth_method: str | None = Field(default=None)
