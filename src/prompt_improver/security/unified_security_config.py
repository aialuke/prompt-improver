"""
Unified Security Configuration Manager

Centralizes all security configuration across the unified security infrastructure.
Provides type-safe, validated configuration for all security components with
environment-specific settings and comprehensive validation.

Following 2025 Security Configuration Best Practices:
- Secure by default configuration
- Environment-specific security profiles
- Comprehensive validation and audit logging
- Integration with existing configuration infrastructure
"""

import os
import logging
from typing import Dict, Any, Optional, List
from enum import Enum
from dataclasses import dataclass, field
from datetime import timedelta

# Removed AppConfig dependency to fix circular import - security must be foundational
from .key_manager import SecurityLevel, KeyDerivationMethod

logger = logging.getLogger(__name__)

class SecurityProfile(str, Enum):
    """Security profiles for different deployment scenarios."""
    DEVELOPMENT = "development"      # Relaxed security for development
    TESTING = "testing"             # Balanced security for testing
    STAGING = "staging"             # Production-like security for staging
    PRODUCTION = "production"        # Maximum security for production
    HIGH_SECURITY = "high_security" # Enhanced security for sensitive operations

class AuthenticationMode(str, Enum):
    """Authentication modes supported by unified manager."""
    API_KEY_ONLY = "api_key_only"
    SESSION_TOKEN_ONLY = "session_token_only"
    BOTH_METHODS = "both_methods"
    SYSTEM_TOKEN_ONLY = "system_token_only"

@dataclass
class UnifiedAuthenticationConfig:
    """Configuration for UnifiedAuthenticationManager."""
    
    # Authentication modes and methods
    authentication_mode: AuthenticationMode = AuthenticationMode.BOTH_METHODS
    enable_api_keys: bool = True
    enable_session_tokens: bool = True
    enable_system_tokens: bool = False
    
    # API key configuration
    api_key_length_bytes: int = 32  # 256 bits
    api_key_prefix: str = "pi_key_"
    api_key_default_tier: str = "basic"
    api_key_default_expires_hours: Optional[int] = None
    
    # Session token configuration
    session_token_length_bytes: int = 32  # 256 bits
    session_token_prefix: str = "session_"
    session_default_expires_minutes: int = 60
    session_max_expires_minutes: int = 480  # 8 hours
    session_cleanup_interval_minutes: int = 30
    
    # Security policies
    fail_secure_enabled: bool = True
    zero_trust_mode: bool = True
    max_failed_attempts_per_hour: int = 10
    lockout_duration_minutes: int = 15
    enable_comprehensive_audit_logging: bool = True
    
    # Performance settings
    authentication_timeout_ms: int = 5000
    validation_cache_ttl_seconds: int = 300
    memory_cache_enabled: bool = True
    memory_cache_max_size: int = 10000

@dataclass
class UnifiedValidationConfig:
    """Configuration for UnifiedValidationManager."""
    
    # Validation modes and thresholds
    validation_timeout_ms: int = 10
    enable_owasp_validation: bool = True
    enable_ml_threat_detection: bool = True
    enable_context_aware_validation: bool = True
    
    # Threat detection configuration
    threat_detection_enabled: bool = True
    min_threat_score_to_block: float = 0.7
    enable_advanced_threat_detection: bool = True
    enable_typoglycemia_detection: bool = True
    enable_encoding_attack_detection: bool = True
    
    # Input validation settings
    max_input_length: int = 10240
    enable_prompt_injection_detection: bool = True
    enable_html_injection_detection: bool = True
    enable_sql_injection_detection: bool = True
    enable_xss_detection: bool = True
    
    # Output validation settings
    enable_output_validation: bool = True
    enable_credential_leakage_detection: bool = True
    enable_system_prompt_leakage_detection: bool = True
    enable_internal_data_exposure_detection: bool = True
    
    # Performance and caching
    enable_validation_caching: bool = True
    validation_cache_ttl_seconds: int = 300
    compiled_regex_cache_enabled: bool = True
    max_validation_cache_size: int = 5000

@dataclass
class UnifiedSecurityStackConfig:
    """Configuration for UnifiedSecurityStack middleware."""
    
    # Security stack modes and layers
    enable_authentication_layer: bool = True
    enable_rate_limiting_layer: bool = True
    enable_validation_layer: bool = True
    enable_audit_logging_layer: bool = True
    enable_error_handling_layer: bool = True
    enable_performance_monitoring_layer: bool = True
    
    # Circuit breaker configuration
    enable_circuit_breakers: bool = True
    circuit_breaker_failure_threshold: int = 5
    circuit_breaker_recovery_timeout_seconds: int = 30
    circuit_breaker_half_open_max_calls: int = 3
    
    # Performance settings
    middleware_timeout_ms: int = 30000
    enable_middleware_caching: bool = True
    enable_parallel_processing: bool = True
    max_concurrent_operations: int = 10
    
    # Security monitoring
    enable_real_time_monitoring: bool = True
    enable_security_incident_detection: bool = True
    enable_performance_degradation_detection: bool = True
    security_incident_threshold: int = 3

@dataclass 
class UnifiedRateLimitingConfig:
    """Configuration for UnifiedRateLimiter."""
    
    # Rate limiting tiers (requests per minute)
    basic_tier_rate_limit: int = 60
    basic_tier_burst_capacity: int = 90
    professional_tier_rate_limit: int = 300
    professional_tier_burst_capacity: int = 450
    enterprise_tier_rate_limit: int = 1000
    enterprise_tier_burst_capacity: int = 1500
    
    # Sliding window configuration
    window_size_seconds: int = 60
    bucket_size_seconds: int = 10
    enable_sliding_window: bool = True
    
    # Security policies
    enable_fail_secure_rate_limiting: bool = True
    enable_distributed_rate_limiting: bool = True
    enable_agent_based_rate_limiting: bool = True
    
    # Performance settings
    rate_limit_check_timeout_ms: int = 100
    enable_rate_limit_caching: bool = True
    rate_limit_cache_ttl_seconds: int = 60

@dataclass
class UnifiedCryptoConfig:
    """Configuration for unified cryptographic operations."""
    
    # Security levels and algorithms
    default_security_level: SecurityLevel = SecurityLevel.enhanced
    default_key_derivation_method: KeyDerivationMethod = KeyDerivationMethod.scrypt
    enable_hsm_integration: bool = False
    enable_cloud_kms_integration: bool = False
    
    # Key management
    key_rotation_interval_hours: int = 24
    max_key_age_hours: int = 72
    key_version_limit: int = 5
    auto_key_rotation_enabled: bool = True
    
    # Cryptographic operations
    hash_algorithm: str = "sha256"
    encryption_algorithm: str = "AES-256-GCM"
    key_derivation_iterations: int = 600000  # PBKDF2
    scrypt_n: int = 32768
    scrypt_r: int = 8
    scrypt_p: int = 1
    
    # Performance settings
    crypto_operation_timeout_ms: int = 1000
    enable_crypto_caching: bool = True
    crypto_cache_ttl_seconds: int = 300

@dataclass
class UnifiedSecurityConfig:
    """
    Master configuration for all unified security components.
    
    Provides centralized, validated configuration management for the entire
    unified security infrastructure with environment-specific profiles.
    """
    
    # Security profile and environment
    security_profile: SecurityProfile = SecurityProfile.PRODUCTION
    environment: str = "production"
    debug_mode: bool = False
    
    # Component configurations
    authentication: UnifiedAuthenticationConfig = field(default_factory=UnifiedAuthenticationConfig)
    validation: UnifiedValidationConfig = field(default_factory=UnifiedValidationConfig)
    security_stack: UnifiedSecurityStackConfig = field(default_factory=UnifiedSecurityStackConfig)
    rate_limiting: UnifiedRateLimitingConfig = field(default_factory=UnifiedRateLimitingConfig)
    cryptography: UnifiedCryptoConfig = field(default_factory=UnifiedCryptoConfig)
    
    # Global security settings
    enable_unified_audit_logging: bool = True
    enable_opentelemetry_integration: bool = True
    enable_performance_monitoring: bool = True
    enable_security_incident_response: bool = True
    
    # Integration settings
    enable_database_security_integration: bool = True
    enable_connection_manager_integration: bool = True
    enable_session_store_integration: bool = True
    
    # Performance and monitoring
    global_operation_timeout_ms: int = 30000
    enable_health_monitoring: bool = True
    health_check_interval_seconds: int = 60
    enable_metrics_collection: bool = True

class UnifiedSecurityConfigManager:
    """
    Unified security configuration manager with environment-specific profiles
    and comprehensive validation.
    """
    
    def __init__(self, 
                 security_profile: Optional[SecurityProfile] = None):
        """Initialize security configuration manager."""
        self.security_profile = security_profile or self._detect_security_profile()
        self._config_cache: Dict[str, UnifiedSecurityConfig] = {}
        
        logger.info(f"Initialized UnifiedSecurityConfigManager with profile: {self.security_profile}")
    
    def get_security_config(self, 
                          profile: Optional[SecurityProfile] = None) -> UnifiedSecurityConfig:
        """Get validated security configuration for specified profile."""
        profile = profile or self.security_profile
        
        if profile.value in self._config_cache:
            return self._config_cache[profile.value]
        
        # Create base configuration
        config = UnifiedSecurityConfig(
            security_profile=profile,
            environment=os.getenv("ENVIRONMENT", "production"),
            debug_mode=os.getenv("DEBUG", "false").lower() == "true"
        )
        
        # Apply profile-specific configurations
        config = self._apply_profile_settings(config, profile)
        
        # Apply environment variable overrides
        config = self._apply_environment_overrides(config)
        
        # Validate configuration
        self._validate_security_config(config)
        
        # Cache configuration
        self._config_cache[profile.value] = config
        
        logger.info(f"Created security configuration for profile: {profile}")
        return config
    
    def _detect_security_profile(self) -> SecurityProfile:
        """Auto-detect security profile based on environment."""
        env = os.getenv("ENVIRONMENT", "production").lower()
        
        profile_mapping = {
            "development": SecurityProfile.DEVELOPMENT,
            "dev": SecurityProfile.DEVELOPMENT,
            "testing": SecurityProfile.TESTING,
            "test": SecurityProfile.TESTING,
            "staging": SecurityProfile.STAGING,
            "stage": SecurityProfile.STAGING,
            "production": SecurityProfile.PRODUCTION,
            "prod": SecurityProfile.PRODUCTION,
            "high_security": SecurityProfile.HIGH_SECURITY,
            "secure": SecurityProfile.HIGH_SECURITY
        }
        
        detected_profile = profile_mapping.get(env, SecurityProfile.PRODUCTION)
        logger.info(f"Auto-detected security profile: {detected_profile} (from env: {env})")
        return detected_profile
    
    def _apply_profile_settings(self, 
                              config: UnifiedSecurityConfig, 
                              profile: SecurityProfile) -> UnifiedSecurityConfig:
        """Apply profile-specific security settings."""
        
        if profile == SecurityProfile.DEVELOPMENT:
            # Relaxed security for development
            config.authentication.fail_secure_enabled = False
            config.authentication.max_failed_attempts_per_hour = 100
            config.validation.min_threat_score_to_block = 0.9
            config.security_stack.enable_circuit_breakers = False
            config.rate_limiting.basic_tier_rate_limit = 1000
            config.cryptography.auto_key_rotation_enabled = False
            
        elif profile == SecurityProfile.TESTING:
            # Balanced security for testing
            config.authentication.max_failed_attempts_per_hour = 50
            config.validation.min_threat_score_to_block = 0.8
            config.rate_limiting.basic_tier_rate_limit = 300
            config.cryptography.key_rotation_interval_hours = 168  # 1 week
            
        elif profile == SecurityProfile.STAGING:
            # Production-like security for staging
            config.authentication.max_failed_attempts_per_hour = 20
            config.validation.min_threat_score_to_block = 0.75
            config.rate_limiting.basic_tier_rate_limit = 120
            config.cryptography.key_rotation_interval_hours = 48
            
        elif profile == SecurityProfile.PRODUCTION:
            # Standard production security (defaults are appropriate)
            pass
            
        elif profile == SecurityProfile.HIGH_SECURITY:
            # Enhanced security for sensitive operations
            config.authentication.max_failed_attempts_per_hour = 5
            config.authentication.lockout_duration_minutes = 60
            config.validation.min_threat_score_to_block = 0.6
            config.security_stack.circuit_breaker_failure_threshold = 3
            config.rate_limiting.basic_tier_rate_limit = 30
            config.cryptography.default_security_level = SecurityLevel.CRITICAL
            config.cryptography.key_rotation_interval_hours = 12
            config.cryptography.enable_hsm_integration = True
        
        return config
    
    def _apply_environment_overrides(self, config: UnifiedSecurityConfig) -> UnifiedSecurityConfig:
        """Apply environment variable overrides to security configuration."""
        
        # Authentication overrides
        if os.getenv("AUTH_FAIL_SECURE"):
            config.authentication.fail_secure_enabled = os.getenv("AUTH_FAIL_SECURE").lower() == "true"
        
        if os.getenv("AUTH_MAX_FAILED_ATTEMPTS"):
            config.authentication.max_failed_attempts_per_hour = int(os.getenv("AUTH_MAX_FAILED_ATTEMPTS"))
        
        if os.getenv("AUTH_API_KEY_EXPIRES_HOURS"):
            config.authentication.api_key_default_expires_hours = int(os.getenv("AUTH_API_KEY_EXPIRES_HOURS"))
        
        # Validation overrides
        if os.getenv("VALIDATION_TIMEOUT_MS"):
            config.validation.validation_timeout_ms = int(os.getenv("VALIDATION_TIMEOUT_MS"))
        
        if os.getenv("VALIDATION_THREAT_THRESHOLD"):
            config.validation.min_threat_score_to_block = float(os.getenv("VALIDATION_THREAT_THRESHOLD"))
        
        # Rate limiting overrides
        if os.getenv("RATE_LIMIT_BASIC_TIER"):
            config.rate_limiting.basic_tier_rate_limit = int(os.getenv("RATE_LIMIT_BASIC_TIER"))
        
        if os.getenv("RATE_LIMIT_PROFESSIONAL_TIER"):
            config.rate_limiting.professional_tier_rate_limit = int(os.getenv("RATE_LIMIT_PROFESSIONAL_TIER"))
        
        # Cryptography overrides
        if os.getenv("CRYPTO_KEY_ROTATION_HOURS"):
            config.cryptography.key_rotation_interval_hours = int(os.getenv("CRYPTO_KEY_ROTATION_HOURS"))
        
        if os.getenv("CRYPTO_HSM_ENABLED"):
            config.cryptography.enable_hsm_integration = os.getenv("CRYPTO_HSM_ENABLED").lower() == "true"
        
        return config
    
    def _validate_security_config(self, config: UnifiedSecurityConfig) -> None:
        """Validate security configuration for consistency and security."""
        
        validation_errors = []
        
        # Authentication validation
        if config.authentication.api_key_length_bytes < 16:
            validation_errors.append("API key length must be at least 16 bytes (128 bits)")
        
        if config.authentication.max_failed_attempts_per_hour < 1:
            validation_errors.append("Max failed attempts must be at least 1")
        
        if config.authentication.lockout_duration_minutes < 1:
            validation_errors.append("Lockout duration must be at least 1 minute")
        
        # Validation configuration validation
        if config.validation.validation_timeout_ms < 1:
            validation_errors.append("Validation timeout must be at least 1ms")
        
        if not (0.0 <= config.validation.min_threat_score_to_block <= 1.0):
            validation_errors.append("Threat score threshold must be between 0.0 and 1.0")
        
        # Rate limiting validation
        if config.rate_limiting.basic_tier_rate_limit < 1:
            validation_errors.append("Basic tier rate limit must be at least 1")
        
        if config.rate_limiting.basic_tier_burst_capacity < config.rate_limiting.basic_tier_rate_limit:
            validation_errors.append("Burst capacity must be >= rate limit")
        
        # Cryptography validation
        if config.cryptography.key_rotation_interval_hours < 1:
            validation_errors.append("Key rotation interval must be at least 1 hour")
        
        if config.cryptography.key_derivation_iterations < 100000:
            validation_errors.append("PBKDF2 iterations should be at least 100,000 for security")
        
        # Global settings validation
        if config.global_operation_timeout_ms < 1000:
            validation_errors.append("Global operation timeout should be at least 1000ms")
        
        if validation_errors:
            error_message = "Security configuration validation failed:\n" + "\n".join(f"- {error}" for error in validation_errors)
            raise ValueError(error_message)
        
        logger.info("Security configuration validation passed")
    
    def get_component_config(self, 
                           component: str, 
                           profile: Optional[SecurityProfile] = None) -> Any:
        """Get configuration for specific security component."""
        config = self.get_security_config(profile)
        
        component_map = {
            "authentication": config.authentication,
            "validation": config.validation,
            "security_stack": config.security_stack,
            "rate_limiting": config.rate_limiting,
            "cryptography": config.cryptography
        }
        
        if component not in component_map:
            raise ValueError(f"Unknown security component: {component}")
        
        return component_map[component]
    
    def export_config_dict(self, 
                         profile: Optional[SecurityProfile] = None) -> Dict[str, Any]:
        """Export security configuration as dictionary for serialization."""
        config = self.get_security_config(profile)
        
        return {
            "security_profile": config.security_profile.value,
            "environment": config.environment,
            "debug_mode": config.debug_mode,
            "authentication": {
                "authentication_mode": config.authentication.authentication_mode.value,
                "enable_api_keys": config.authentication.enable_api_keys,
                "enable_session_tokens": config.authentication.enable_session_tokens,
                "api_key_length_bytes": config.authentication.api_key_length_bytes,
                "session_default_expires_minutes": config.authentication.session_default_expires_minutes,
                "fail_secure_enabled": config.authentication.fail_secure_enabled,
                "zero_trust_mode": config.authentication.zero_trust_mode,
                "max_failed_attempts_per_hour": config.authentication.max_failed_attempts_per_hour
            },
            "validation": {
                "validation_timeout_ms": config.validation.validation_timeout_ms,
                "enable_owasp_validation": config.validation.enable_owasp_validation,
                "min_threat_score_to_block": config.validation.min_threat_score_to_block,
                "max_input_length": config.validation.max_input_length,
                "enable_prompt_injection_detection": config.validation.enable_prompt_injection_detection
            },
            "rate_limiting": {
                "basic_tier_rate_limit": config.rate_limiting.basic_tier_rate_limit,
                "professional_tier_rate_limit": config.rate_limiting.professional_tier_rate_limit,
                "enterprise_tier_rate_limit": config.rate_limiting.enterprise_tier_rate_limit,
                "enable_fail_secure_rate_limiting": config.rate_limiting.enable_fail_secure_rate_limiting
            },
            "cryptography": {
                "default_security_level": config.cryptography.default_security_level.value,
                "key_rotation_interval_hours": config.cryptography.key_rotation_interval_hours,
                "auto_key_rotation_enabled": config.cryptography.auto_key_rotation_enabled,
                "enable_hsm_integration": config.cryptography.enable_hsm_integration
            }
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get health status of security configuration manager."""
        return {
            "status": "healthy",
            "security_profile": self.security_profile.value,
            "cached_configs": len(self._config_cache),
            "environment": os.getenv("ENVIRONMENT", "production"),
            "configuration_valid": True
        }

# Global configuration manager instance
_unified_security_config_manager: Optional[UnifiedSecurityConfigManager] = None

def get_unified_security_config_manager(
    security_profile: Optional[SecurityProfile] = None
) -> UnifiedSecurityConfigManager:
    """Get global unified security configuration manager instance."""
    global _unified_security_config_manager
    if _unified_security_config_manager is None:
        _unified_security_config_manager = UnifiedSecurityConfigManager(
            security_profile=security_profile
        )
    return _unified_security_config_manager

def get_security_config(profile: Optional[SecurityProfile] = None) -> UnifiedSecurityConfig:
    """Get unified security configuration for specified profile."""
    manager = get_unified_security_config_manager()
    return manager.get_security_config(profile)

def get_production_security_config() -> UnifiedSecurityConfig:
    """Get production security configuration."""
    return get_security_config(SecurityProfile.PRODUCTION)

def get_high_security_config() -> UnifiedSecurityConfig:
    """Get high security configuration."""
    return get_security_config(SecurityProfile.HIGH_SECURITY)

def get_development_security_config() -> UnifiedSecurityConfig:
    """Get development security configuration."""
    return get_security_config(SecurityProfile.DEVELOPMENT)

# Convenience functions for component-specific configuration
def get_authentication_config(profile: Optional[SecurityProfile] = None) -> UnifiedAuthenticationConfig:
    """Get authentication configuration for specified profile."""
    manager = get_unified_security_config_manager()
    return manager.get_component_config("authentication", profile)

def get_validation_config(profile: Optional[SecurityProfile] = None) -> UnifiedValidationConfig:
    """Get validation configuration for specified profile."""
    manager = get_unified_security_config_manager()
    return manager.get_component_config("validation", profile)

def get_rate_limiting_config(profile: Optional[SecurityProfile] = None) -> UnifiedRateLimitingConfig:
    """Get rate limiting configuration for specified profile."""  
    manager = get_unified_security_config_manager()
    return manager.get_component_config("rate_limiting", profile)

def get_crypto_config(profile: Optional[SecurityProfile] = None) -> UnifiedCryptoConfig:
    """Get cryptography configuration for specified profile."""
    manager = get_unified_security_config_manager()
    return manager.get_component_config("cryptography", profile)