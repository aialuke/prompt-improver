"""Security Configuration Module.

Unified security configuration consolidating all security-related settings
with environment-specific profiles and comprehensive validation.
"""

import os
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class SecurityProfile(StrEnum):
    """Security profiles for different deployment scenarios."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"
    HIGH_SECURITY = "high_security"


class AuthenticationMode(StrEnum):
    """Authentication modes supported by unified manager."""

    API_KEY_ONLY = "api_key_only"
    SESSION_TOKEN_ONLY = "session_token_only"
    BOTH_METHODS = "both_methods"
    SYSTEM_TOKEN_ONLY = "system_token_only"


class AuthenticationConfig(BaseModel):
    """Authentication system configuration."""

    authentication_mode: AuthenticationMode = Field(
        default=AuthenticationMode.BOTH_METHODS,
        description="Authentication methods to enable",
    )
    enable_api_keys: bool = Field(
        default=True, description="Enable API key authentication"
    )
    enable_session_tokens: bool = Field(
        default=True, description="Enable session token authentication"
    )
    api_key_length_bytes: int = Field(
        default=32, ge=16, le=64, description="API key length in bytes"
    )
    api_key_prefix: str = Field(
        default="pi_key_", min_length=1, max_length=20, description="API key prefix"
    )
    session_default_expires_minutes: int = Field(
        default=60, gt=0, le=1440, description="Default session expiration in minutes"
    )
    session_max_expires_minutes: int = Field(
        default=480, gt=0, le=1440, description="Maximum session expiration"
    )
    fail_secure_enabled: bool = Field(
        default=True, description="Enable fail-secure security policy"
    )
    zero_trust_mode: bool = Field(
        default=True, description="Enable zero-trust authentication mode"
    )
    max_failed_attempts_per_hour: int = Field(
        default=10, gt=0, le=1000, description="Maximum failed authentication attempts"
    )
    lockout_duration_minutes: int = Field(
        default=15, gt=0, le=1440, description="Account lockout duration"
    )


class ValidationConfig(BaseModel):
    """Input/output validation configuration."""

    validation_timeout_ms: int = Field(
        default=10, gt=0, le=10000, description="Input validation timeout"
    )
    enable_owasp_validation: bool = Field(
        default=True, description="Enable OWASP-compliant input validation"
    )
    enable_ml_threat_detection: bool = Field(
        default=True, description="Enable ML-based threat detection"
    )
    min_threat_score_to_block: float = Field(
        default=0.7, ge=0.0, le=1.0, description="Minimum threat score to block"
    )
    max_input_length: int = Field(
        default=10240, gt=0, le=1048576, description="Maximum input length"
    )
    enable_prompt_injection_detection: bool = Field(
        default=True, description="Enable prompt injection detection"
    )
    enable_xss_detection: bool = Field(
        default=True, description="Enable XSS attack detection"
    )
    enable_sql_injection_detection: bool = Field(
        default=True, description="Enable SQL injection detection"
    )


class RateLimitingConfig(BaseModel):
    """Rate limiting configuration."""

    basic_tier_rate_limit: int = Field(
        default=60, gt=0, description="Basic tier rate limit per minute"
    )
    basic_tier_burst_capacity: int = Field(
        default=90, gt=0, description="Basic tier burst capacity"
    )
    professional_tier_rate_limit: int = Field(
        default=300, gt=0, description="Professional tier rate limit"
    )
    professional_tier_burst_capacity: int = Field(
        default=450, gt=0, description="Professional tier burst capacity"
    )
    enterprise_tier_rate_limit: int = Field(
        default=1000, gt=0, description="Enterprise tier rate limit"
    )
    enterprise_tier_burst_capacity: int = Field(
        default=1500, gt=0, description="Enterprise tier burst capacity"
    )
    window_size_seconds: int = Field(
        default=60, gt=0, description="Rate limiting window size"
    )
    enable_sliding_window: bool = Field(
        default=True, description="Enable sliding window rate limiting"
    )


class CryptographyConfig(BaseModel):
    """Cryptographic operations configuration."""

    hash_algorithm: str = Field(default="sha256", description="Default hash algorithm")
    encryption_algorithm: str = Field(
        default="AES-256-GCM", description="Default encryption algorithm"
    )
    key_derivation_iterations: int = Field(
        default=600000, ge=100000, description="PBKDF2 iterations"
    )
    key_rotation_interval_hours: int = Field(
        default=24, gt=0, description="Key rotation interval"
    )
    max_key_age_hours: int = Field(default=72, gt=0, description="Maximum key age")
    auto_key_rotation_enabled: bool = Field(
        default=True, description="Enable automatic key rotation"
    )


class SecurityConfig(BaseSettings):
    """Main security configuration with all security components."""

    # Security Profile
    security_profile: SecurityProfile = Field(
        default=SecurityProfile.PRODUCTION,
        description="Security profile for deployment environment",
    )

    # Core Security Settings
    secret_key: str = Field(
        ..., min_length=32, description="Application secret key (min 32 chars)"
    )
    encryption_key: str | None = Field(
        default=None, description="Encryption key for sensitive data"
    )
    token_expiry_seconds: int = Field(
        default=3600, ge=1, le=86400, description="Token expiry time in seconds"
    )
    hash_rounds: int = Field(
        default=12, ge=4, le=20, description="Password hash rounds"
    )
    max_login_attempts: int = Field(
        default=5, ge=1, le=20, description="Maximum login attempts"
    )

    # Component Configurations
    authentication: AuthenticationConfig = Field(default_factory=AuthenticationConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    rate_limiting: RateLimitingConfig = Field(default_factory=RateLimitingConfig)
    cryptography: CryptographyConfig = Field(default_factory=CryptographyConfig)

    # Feature Toggles
    enable_unified_audit_logging: bool = Field(
        default=True, description="Enable unified security audit logging"
    )
    enable_opentelemetry_integration: bool = Field(
        default=True, description="Enable OpenTelemetry observability integration"
    )
    enable_performance_monitoring: bool = Field(
        default=True, description="Enable security performance monitoring"
    )
    global_operation_timeout_ms: int = Field(
        default=30000, gt=0, le=300000, description="Global security operation timeout"
    )

    @field_validator("secret_key")
    @classmethod
    def validate_secret_key(cls, v):
        """Validate secret key requirements."""
        if not v or len(v.strip()) == 0:
            raise ValueError("secret_key is required")
        if len(v) < 32:
            raise ValueError(f"secret_key must be at least 32 characters long, got {len(v)}")

        # Check for development keys in production
        env = os.getenv("ENVIRONMENT", "development").lower()
        if ("dev-secret-key" in v.lower() or "development" in v.lower()) and env == "production":
            raise ValueError("Development secret key detected in production environment")
        return v

    def apply_profile_settings(self) -> None:
        """Apply profile-specific security settings."""
        if self.security_profile == SecurityProfile.DEVELOPMENT:
            self.authentication.fail_secure_enabled = False
            self.authentication.max_failed_attempts_per_hour = 100
            self.validation.min_threat_score_to_block = 0.9
            self.rate_limiting.basic_tier_rate_limit = 1000
            self.cryptography.auto_key_rotation_enabled = False

        elif self.security_profile == SecurityProfile.TESTING:
            self.authentication.max_failed_attempts_per_hour = 50
            self.validation.min_threat_score_to_block = 0.8
            self.rate_limiting.basic_tier_rate_limit = 300

        elif self.security_profile == SecurityProfile.STAGING:
            self.authentication.max_failed_attempts_per_hour = 20
            self.validation.min_threat_score_to_block = 0.75
            self.rate_limiting.basic_tier_rate_limit = 120

        elif self.security_profile == SecurityProfile.HIGH_SECURITY:
            self.authentication.max_failed_attempts_per_hour = 5
            self.authentication.lockout_duration_minutes = 60
            self.validation.min_threat_score_to_block = 0.6
            self.rate_limiting.basic_tier_rate_limit = 30
            self.cryptography.key_rotation_interval_hours = 12

    def model_post_init(self, __context: Any) -> None:
        """Apply profile settings after initialization."""
        self.apply_profile_settings()

    model_config = {
        "env_prefix": "SECURITY_",
        "case_sensitive": False,
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "validate_assignment": True,
    }
